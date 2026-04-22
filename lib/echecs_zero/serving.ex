defmodule EchecsZero.Serving do
  @moduledoc """
  Serving process for EchecsZero model inference.
  It encapsulates the Axon model and provides a batched inference API.
  """

  @doc """
  Returns a child spec to start the serving under a supervision tree.
  """
  def child_spec(opts) do
    name = Keyword.get(opts, :name, __MODULE__)
    batch_size = Keyword.get(opts, :batch_size, 8)
    batch_timeout = Keyword.get(opts, :batch_timeout, 100)

    serving = build_serving(batch_size)

    %{
      id: name,
      start:
        {Nx.Serving, :start_link,
         [[serving: serving, name: name, batch_size: batch_size, batch_timeout: batch_timeout]]}
    }
  end

  defp build_serving(batch_size) do
    Nx.Serving.new(fn _opts ->
      model = EchecsZero.Model.build()
      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

      # Template for a single batch
      template = Nx.template({batch_size, 119, 8, 8}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      # Compile the prediction function upfront
      predict_fn = Nx.Defn.compile(predict_fn, [params, template], compiler: EXLA)

      fn inputs ->
        # Pad the inputs to the expected batch_size
        inputs = Nx.Batch.pad(inputs, batch_size - inputs.size)
        predict_fn.(params, inputs)
      end
    end)
    |> Nx.Serving.client_preprocessing(fn input ->
      # Convert single tensor input to a batch of size 1
      {Nx.Batch.stack([input]), :client_info}
    end)
    |> Nx.Serving.client_postprocessing(fn {result, _info}, _metadata ->
      # Extract the single result out of the batch dimension
      %{
        policy: result.policy[0],
        value: result.value[0]
      }
    end)
  end
end
