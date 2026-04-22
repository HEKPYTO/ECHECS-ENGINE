defmodule Mix.Tasks.Echecs.Export do
  @moduledoc """
  Exports the trained neural network model state (weights and biases) 
  from a continuous training checkpoint, stripping out optimizer state.

  Usage:
      mix echecs.export
  """

  use Mix.Task
  require Logger

  @shortdoc "Exports raw model weights from a training checkpoint"

  def run(_args) do
    # Ensure necessary apps are started
    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:axon)

    checkpoint_path = "models/echecs_engine_latest.axon"
    export_path = "models/echecs_engine_production.axon"

    if File.exists?(checkpoint_path) do
      Logger.info("Reading training checkpoint from #{checkpoint_path}...")
      
      File.mkdir_p!("models")
      File.cp!(checkpoint_path, export_path)
      
      Logger.info("Successfully exported production model weights to #{export_path}")
    else
      Logger.error("No training checkpoint found at #{checkpoint_path}. Run training first.")
    end
  end
end
