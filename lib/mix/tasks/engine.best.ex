defmodule Mix.Tasks.Engine.Best do
  @moduledoc false
  use Mix.Task

  @shortdoc "Prints the best UCI move for a FEN"

  @impl Mix.Task
  def run(args) do
    {opts, fen} = parse(args, [])

    case EchecsEngine.best_move(Enum.join(fen, " "), opts) do
      {:ok, move} -> IO.puts(move)
      {:terminal, status} -> IO.puts("terminal:#{status}")
      {:error, reason} -> Mix.raise("engine search failed: #{inspect(reason)}")
    end
  end

  defp parse(["--depth", value | rest], opts), do: parse(rest, [{:depth, integer!(value)} | opts])
  defp parse(["--nodes", value | rest], opts), do: parse(rest, [{:nodes, integer!(value)} | opts])

  defp parse(["--movetime", value | rest], opts),
    do: parse(rest, [{:movetime, integer!(value)} | opts])

  defp parse([flag | _] = args, opts) do
    if String.starts_with?(flag, "--"),
      do: Mix.raise("unsupported option #{flag}"),
      else: {Enum.reverse(opts), args}
  end

  defp parse([], opts), do: {Enum.reverse(opts), []}

  defp integer!(value) do
    case Integer.parse(value) do
      {number, ""} -> number
      _ -> Mix.raise("invalid integer #{value}")
    end
  end
end
