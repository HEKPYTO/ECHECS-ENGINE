defmodule Mix.Tasks.Engine.TrainEvaluator do
  @moduledoc false
  use Mix.Task
  @shortdoc "Trains a version-one integer evaluator artifact"

  @impl Mix.Task
  def run([input, output]), do: print(EchecsEngine.Trainer.train!(input, output))

  def run([input]),
    do: print(EchecsEngine.Trainer.train!(input, Path.join(File.cwd!(), "priv/echecs.nnue")))

  def run(_), do: Mix.raise("usage: mix engine.train_evaluator train.jsonl [output.nnue]")

  defp print(result) do
    IO.puts("saved evaluator: #{result.output}")

    IO.puts(
      "rows=#{result.rows} train_loss=#{result.initial_training_loss}->#{result.training_loss} " <>
        "validation_loss=#{result.initial_validation_loss}->#{result.validation_loss} " <>
        "active_updates=#{result.active_updates} changed=#{inspect(result.changed_tensors)}"
    )
  end
end
