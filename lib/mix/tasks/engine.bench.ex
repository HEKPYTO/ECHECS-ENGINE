defmodule Mix.Tasks.Engine.Bench do
  @moduledoc """
  Runs a JSONL best-move benchmark suite.

  Usage:

      mix engine.bench path/to/suite.jsonl
  """

  use Mix.Task

  @shortdoc "Runs a FEN best-move benchmark suite"

  @impl Mix.Task
  def run(["--sprt", wins, losses, draws | args]) do
    opts = sprt_opts(args)

    %{
      decision: decision,
      llr: llr,
      lower_bound: lower,
      upper_bound: upper
    } =
      EchecsEngine.Benchmark.sprt(
        %{
          wins: String.to_integer(wins),
          losses: String.to_integer(losses),
          draws: String.to_integer(draws)
        },
        opts
      )

    IO.puts("sprt:")
    IO.puts("decision: #{decision}")
    IO.puts("llr: #{Float.round(llr, 4)}")
    IO.puts("lower: #{Float.round(lower, 4)}")
    IO.puts("upper: #{Float.round(upper, 4)}")
  end

  def run(["--match", path | args]) do
    Application.ensure_all_started(:echecs_engine)

    plies =
      case args do
        ["--plies", value | _rest] -> String.to_integer(value)
        _other -> 80
      end

    path
    |> EchecsEngine.Benchmark.run_match_jsonl!(plies: plies)
    |> print_match_summary()
  end

  def run([path | _args]) do
    Application.ensure_all_started(:echecs_engine)

    path
    |> EchecsEngine.Benchmark.run_jsonl!()
    |> print_summary()
  end

  def run(_args), do: Mix.raise("usage: mix engine.bench path/to/suite.jsonl")

  defp print_summary(result) do
    IO.puts("positions: #{result.total}")
    IO.puts("correct: #{result.correct}")
    IO.puts("accuracy: #{Float.round(result.accuracy * 100.0, 2)}%")

    total_time = Enum.reduce(result.positions, 0, fn position, acc -> acc + position.time_us end)
    avg_time = if result.total == 0, do: 0.0, else: total_time / result.total / 1000.0
    IO.puts("avg_ms: #{Float.round(avg_time, 3)}")
  end

  defp print_match_summary(result) do
    IO.puts("games: #{result.games}")

    result.results
    |> Enum.with_index(1)
    |> Enum.each(fn {game, idx} ->
      IO.puts("game #{idx}:")
      IO.puts("moves: #{Enum.join(game.moves, " ")}")
      IO.puts("status: #{inspect(game.status)}")
    end)
  end

  defp sprt_opts(args) do
    args
    |> Enum.chunk_every(2)
    |> Enum.flat_map(fn
      ["--elo0", elo] -> [elo0: parse_number(elo)]
      ["--elo1", elo] -> [elo1: parse_number(elo)]
      ["--alpha", alpha] -> [alpha: parse_number(alpha)]
      ["--beta", beta] -> [beta: parse_number(beta)]
      _other -> []
    end)
  end

  defp parse_number(value) do
    case Float.parse(value) do
      {number, ""} -> number
      _other -> raise ArgumentError, "invalid numeric option: #{value}"
    end
  end
end
