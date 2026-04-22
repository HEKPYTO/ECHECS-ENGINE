defmodule EchecsEngine.Serving do
  @moduledoc """
  Manages the application boundary and distributed processing 
  of `Axon` neural network inferences via `Nx.Serving`.
  """

  @doc """
  Configures the child specification to deploy `Nx.Serving` within
  a supervision tree. Instantiates the AlphaZero network and builds 
  the pre/post processing batch pipelines automatically.
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

  @doc false
  defp build_serving(batch_size) do
    Nx.Serving.new(fn _opts ->
      model = EchecsEngine.Model.ViT.build()
      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

      template = Nx.template({batch_size, 119, 8, 8}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())

      predict_fn = Nx.Defn.compile(predict_fn, [params, template], compiler: EXLA)

      fn inputs ->
        inputs = Nx.Batch.pad(inputs, batch_size - inputs.size)
        predict_fn.(params, inputs)
      end
    end)
    |> Nx.Serving.client_preprocessing(fn input ->
      {Nx.Batch.stack([input]), :client_info}
    end)
    |> Nx.Serving.client_postprocessing(fn {result, _info}, _metadata ->
      %{
        policy: result.policy[0],
        value: result.value[0]
      }
    end)
  end
end
