defmodule Mix.Tasks.Engine.Export do
  @moduledoc """
  Exports the trained neural network model state (weights and biases)
  from a continuous training checkpoint, stripping out optimizer state.

  Usage:
      mix engine.export
  """

  use Mix.Task
  require Logger

  @shortdoc "Exports raw model weights from a training checkpoint"

  def run(_args) do
    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:axon)

    checkpoint_path = EchecsEngine.Checkpoint.latest_path()
    export_path = EchecsEngine.Checkpoint.production_path()

    if File.exists?(checkpoint_path) do
      Logger.info("Reading training checkpoint from #{checkpoint_path}...")

      {:ok, model_state} = EchecsEngine.Checkpoint.load_model_state(checkpoint_path)

      EchecsEngine.Checkpoint.save_model_state!(export_path, model_state)

      Logger.info("Successfully exported production model weights to #{export_path}")
    else
      Logger.error("No training checkpoint found at #{checkpoint_path}. Run training first.")
    end
  end
end
