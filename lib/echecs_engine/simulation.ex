defmodule EchecsEngine.Simulation do
  @moduledoc """
  Provides the supervised JSONL training loop for ECHECS-ENGINE.

  This module sets up the ViT policy/WDL/moves-left model, legal-move masked
  losses, optimizer, full loop-state checkpointing, and supervised FEN/eval
  dataset loading.
  """

  require Logger

  @doc """
  Runs supervised training.

  Ensures necessary applications are started, configures `EXLA`, compiles the
  ViT model, and trains on JSONL records containing FEN positions, legal UCI
  moves, WDL/result/eval labels, and optional moves-left labels.
  """
  def run(opts \\ []) do
    Application.ensure_all_started(:exla)
    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:axon)

    Nx.global_default_backend(EXLA.Backend)
    Nx.Defn.global_default_options(compiler: EXLA)

    Logger.info("Setting up ECHECS-ENGINE Training Simulation on GPU...")

    model = EchecsEngine.Model.ViT.build()

    loss = loss_fn()

    optimizer = Polaris.Optimizers.adam(learning_rate: 1.0e-4)

    batch_size = Keyword.get(opts, :batch_size, 64)
    dataset_path = dataset_path!(opts)

    Logger.info(
      "Loading supervised training data from #{dataset_path} (batch size #{batch_size})..."
    )

    data = EchecsEngine.Dataset.batches_from_jsonl!(dataset_path, batch_size: batch_size)

    Logger.info("Starting training loop...")

    checkpoint_path = EchecsEngine.Checkpoint.latest_path()

    loop =
      Axon.Loop.trainer(model, loss, optimizer)
      |> Axon.Loop.handle_event(:epoch_completed, fn loop_state ->
        EchecsEngine.Checkpoint.save_training_state!(checkpoint_path, loop_state)
        Logger.info("Model weights auto-saved to #{checkpoint_path} after epoch.")
        {:continue, loop_state}
      end)

    loop =
      case EchecsEngine.Checkpoint.load_training_state(checkpoint_path) do
        {:ok, training_state} ->
          Logger.info(
            "Found existing checkpoint at #{checkpoint_path}. Loading full training state to resume..."
          )

          Axon.Loop.from_state(loop, training_state)

        {:error, :enoent} ->
          Logger.info("No checkpoint found. Training from scratch...")
          loop

        {:error, reason} ->
          raise "failed to load training checkpoint: #{inspect(reason)}"
      end

    trained_model_state =
      Axon.Loop.run(loop, data, %{}, epochs: training_epochs(opts), compiler: EXLA)

    Logger.info("Training simulation completed successfully.")

    production_path = EchecsEngine.Checkpoint.production_path()
    File.mkdir_p!(Path.dirname(production_path))
    EchecsEngine.Checkpoint.save_model_state!(production_path, trained_model_state)
    Logger.info("Exported final production weights to #{production_path}")
  end

  @doc false
  @spec training_epochs(keyword()) :: pos_integer()
  def training_epochs(opts) do
    opts
    |> Keyword.get(:epochs, 2)
    |> max(1)
  end

  @doc false
  def loss_fn do
    fn %{
         policy: target_p,
         policy_mask: policy_mask,
         wdl: target_wdl,
         moves_left: target_moves_left
       },
       %{policy: pred_p, wdl: pred_wdl, moves_left: pred_moves_left} ->
      masked_policy = Nx.select(policy_mask, pred_p, Nx.broadcast(-1.0e9, Nx.shape(pred_p)))

      p_loss =
        Axon.Losses.categorical_cross_entropy(target_p, masked_policy,
          reduction: :mean,
          from_logits: true
        )

      wdl_loss =
        Axon.Losses.categorical_cross_entropy(target_wdl, pred_wdl,
          reduction: :mean,
          from_logits: false
        )

      moves_left_loss =
        Axon.Losses.mean_squared_error(target_moves_left, pred_moves_left, reduction: :mean)

      p_loss
      |> Nx.add(wdl_loss)
      |> Nx.add(Nx.multiply(moves_left_loss, 0.01))
    end
  end

  defp dataset_path!(opts) do
    Keyword.get(opts, :dataset_path) || System.get_env("ENGINE_DATASET") ||
      raise ArgumentError,
            "missing supervised dataset path; pass dataset_path: path or set ENGINE_DATASET"
  end
end
