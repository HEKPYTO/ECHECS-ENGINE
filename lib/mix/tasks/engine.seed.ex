defmodule Mix.Tasks.Engine.Seed do
  @moduledoc false
  use Mix.Task

  @shortdoc "Writes the deterministic integer evaluator"

  @impl Mix.Task
  def run(_args) do
    path = Path.join(File.cwd!(), "priv/echecs.nnue")
    File.mkdir_p!(Path.dirname(path))
    :ok = EchecsEngine.Eval.dump!(path, EchecsEngine.Eval.seed_weights())
  end
end
