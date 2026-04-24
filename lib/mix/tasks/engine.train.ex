defmodule Mix.Tasks.Engine.Train do
  @moduledoc """
  Trains ECHECS-ENGINE from supervised JSONL data.

  Usage:

      mix engine.train path/to/train.jsonl
  """

  use Mix.Task

  @shortdoc "Trains from supervised FEN/eval JSONL data"

  @impl Mix.Task
  def run([dataset_path | args]) do
    Application.ensure_all_started(:echecs_engine)

    EchecsEngine.Simulation.run(
      dataset_path: dataset_path,
      batch_size: batch_size(args)
    )
  end

  def run(_args), do: Mix.raise("usage: mix engine.train path/to/train.jsonl")

  defp batch_size(args) do
    args
    |> Enum.chunk_every(2)
    |> Enum.find_value(64, fn
      ["--batch-size", value] -> String.to_integer(value)
      _other -> nil
    end)
  end
end
