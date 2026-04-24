defmodule Mix.Tasks.Engine.Pgn do
  @moduledoc """
  Converts a PGN file into supervised JSONL records.

  Usage:

      mix engine.pgn path/to/game.pgn path/to/output.jsonl
  """

  use Mix.Task

  @shortdoc "Converts PGN games into supervised JSONL"

  @impl Mix.Task
  def run([pgn_path, output_path]) do
    pgn_path
    |> File.read!()
    |> EchecsEngine.PGNDataset.records_from_pgn!()
    |> then(fn records ->
      EchecsEngine.PGNDataset.write_jsonl!(output_path, records)
      IO.puts("records: #{length(records)}")
      IO.puts("output: #{output_path}")
    end)
  end

  def run(_args), do: Mix.raise("usage: mix engine.pgn path/to/game.pgn path/to/output.jsonl")
end
