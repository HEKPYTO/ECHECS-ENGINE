defmodule Mix.Tasks.Engine.Best do
  @moduledoc """
  Prints the best move for a FEN position.

  Usage:

      mix engine.best "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      mix engine.best --allow-zero-evaluator "8/8/8/8/8/8/4P3/4K2k w - - 0 1"
      mix engine.best --no-default-evaluator "8/8/8/8/8/8/4P3/4K2k w - - 0 1"
  """

  use Mix.Task

  @shortdoc "Prints the best UCI move for a FEN"

  @impl Mix.Task
  def run(args) do
    Application.ensure_all_started(:echecs_engine)

    {search_opts, fen_args} = parse_args(args)

    fen_args
    |> Enum.join(" ")
    |> EchecsEngine.best_move(search_opts)
    |> print_result()
  end

  defp print_result({:ok, move}), do: IO.puts(move)
  defp print_result({:terminal, status}), do: IO.puts("terminal:#{status}")

  defp print_result({:error, %_exception{} = reason}),
    do: Mix.raise("invalid FEN: #{Exception.message(reason)}")

  defp print_result({:error, reason}), do: Mix.raise("engine search failed: #{inspect(reason)}")

  defp parse_args(["--allow-zero-evaluator" | rest]),
    do: {[allow_zero_evaluator: true], rest}

  defp parse_args(["--no-default-evaluator" | rest]),
    do: {[evaluator_paths: []], rest}

  defp parse_args(args), do: {[], args}
end
