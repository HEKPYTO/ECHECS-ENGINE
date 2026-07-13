defmodule EchecsEngine.Serving do
  @moduledoc """
  Manages the application boundary and distributed processing 
  of `Axon` neural network inferences via `Nx.Serving`.
  """

  require Logger

  @doc """
  Configures the child specification to deploy `Nx.Serving` within
  a supervision tree. Instantiates the legacy frameworks network and builds 
  the pre/post processing batch pipelines automatically.
  """
  def child_spec(opts) do
    name = Keyword.get(opts, :name, __MODULE__)
    batch_size = Keyword.get(opts, :batch_size, 8)
    batch_timeout = Keyword.get(opts, :batch_timeout, 100)
    model_paths = Keyword.get(opts, :model_paths, EchecsEngine.Checkpoint.default_model_paths())

    serving = build_serving(batch_size, model_paths)

    %{
      id: name,
      start:
        {Nx.Serving, :start_link,
         [[serving: serving, name: name, batch_size: batch_size, batch_timeout: batch_timeout]]}
    }
  end

  @doc false
  defp build_serving(batch_size, model_paths) do
    Nx.Serving.new(fn _opts ->
      model = EchecsEngine.Model.ViT.build()
      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

      template = Nx.template({batch_size, 119, 8, 8}, :f32)
      params = load_or_init_params(init_fn, template, model_paths)

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
      wdl = result.wdl[0]

      %{
        policy: result.policy[0],
        wdl: wdl,
        moves_left: result.moves_left[0],
        value: Nx.tensor([EchecsEngine.Value.wdl_to_q(wdl)], type: :f32)
      }
    end)
  end

  defp load_or_init_params(init_fn, template, model_paths) do
    case EchecsEngine.Checkpoint.load_model_state(model_paths) do
      {:ok, params} ->
        Logger.info("Loaded model parameters from checkpoint for serving.")
        params

      {:error, :enoent} ->
        Logger.warning("No model checkpoint found for serving. Initializing random weights.")
        init_fn.(template, Axon.ModelState.empty())

      {:error, {:incompatible_schema, loaded, expected}} ->
        Logger.warning(
          "Model checkpoint schema v#{loaded} is incompatible with current v#{expected}. " <>
            "Initializing random weights; retrain to refresh the checkpoint."
        )

        init_fn.(template, Axon.ModelState.empty())

      {:error, reason} ->
        raise "failed to load serving model parameters: #{inspect(reason)}"
    end
  end
end
