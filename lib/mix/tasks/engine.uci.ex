defmodule Mix.Tasks.Engine.Uci do
  @moduledoc """
  Runs ECHECS-ENGINE as a UCI-compatible process.

  Usage:

      mix engine.uci
  """

  use Mix.Task

  @shortdoc "Runs the engine over the UCI protocol"

  @impl Mix.Task
  def run(_args) do
    Application.ensure_all_started(:echecs_engine)
    EchecsEngine.UCI.run()
  end
end
