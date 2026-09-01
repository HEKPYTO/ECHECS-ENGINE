defmodule Mix.Tasks.Engine.Seed do
  @moduledoc """
  Writes the deterministic integer evaluator to `priv/echecs.nnue`.

  ## Usage

      mix engine.seed

  Generates `EchecsEngine.Eval.seed_weights/0` (material-aware PSQT, zeroed
  dense tail) and persists it via `EchecsEngine.Eval.dump!/2`. The file is
  tracked so the engine is usable without prior training.
  """
  use Mix.Task

  @shortdoc "Writes the deterministic integer evaluator"

  @impl Mix.Task
  def run(_args) do
    path = Path.join(File.cwd!(), "priv/echecs.nnue")
    File.mkdir_p!(Path.dirname(path))
    :ok = EchecsEngine.Eval.dump!(path, EchecsEngine.Eval.seed_weights())
  end
end
