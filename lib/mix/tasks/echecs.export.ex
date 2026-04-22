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

    checkpoint_path = "models/echecs_engine_latest.ckpt"
    export_path = "models/echecs_engine_production.axon"

    if File.exists?(checkpoint_path) do
      Logger.info("Reading training checkpoint from #{checkpoint_path}...")

      # The loop state is an Axon.Loop.State struct containing:
      # :step_state -> typically contains a tuple of {model_state, optimizer_state}
      loop_state = :erlang.binary_to_term(File.read!(checkpoint_path))

      model_state =
        case loop_state do
          %{step_state: %{model_state: ms}} ->
            ms

          %{step_state: {ms, _opt_state}} ->
            ms

          _ ->
            Logger.warning(
              "Could not automatically infer model_state from loop. Saving raw step_state."
            )

            loop_state.step_state
        end

      File.mkdir_p!("models")

      # We serialize the pure model state using Axon's serializer
      # Note: For Axon ~> 0.6, erlang term serialization of the model_state map is stable.
      File.write!(export_path, :erlang.term_to_binary(model_state))

      Logger.info("Successfully exported production model weights to #{export_path}")
    else
      Logger.error("No training checkpoint found at #{checkpoint_path}. Run training first.")
    end
  end
end
