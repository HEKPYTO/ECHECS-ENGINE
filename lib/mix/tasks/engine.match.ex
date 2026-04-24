defmodule Mix.Tasks.Engine.Match do
  @moduledoc """
  Runs or prints an external UCI engine match command.

  Usage:

      mix engine.match --dry-run --engine-a "mix engine.uci" --engine-b stockfish
      mix engine.match --runner fastchess --engine-a "mix engine.uci" --engine-b stockfish --games 200
  """

  use Mix.Task

  @shortdoc "Runs external cutechess/fastchess match gates"

  @impl Mix.Task
  def run(args) do
    opts = parse_args(args)

    if Keyword.get(opts, :dry_run, false) do
      IO.puts(EchecsEngine.MatchRunner.shell_command(opts))
    else
      case EchecsEngine.MatchRunner.run(opts) do
        {:ok, %{status: status, output: output}} ->
          IO.write(output)
          if status != 0, do: Mix.raise("match runner exited with status #{status}")

        {:error, {:runner_not_found, runner}} ->
          Mix.raise("match runner not found: #{runner}")
      end
    end
  end

  defp parse_args(args) do
    {opts, rest} = do_parse(args, [])

    if rest != [] do
      Mix.raise("unsupported engine.match arguments: #{Enum.join(rest, " ")}")
    end

    opts
    |> Keyword.put_new_lazy(:runner, fn ->
      if Keyword.get(opts, :docker, false), do: "fastchess", else: "cutechess-cli"
    end)
    |> Keyword.put_new(:engine_a, %{name: "new", command: "mix engine.uci"})
    |> Keyword.put_new(:engine_b, %{name: "base", command: "stockfish"})
  end

  defp do_parse([], opts), do: {Enum.reverse(opts), []}
  defp do_parse(["--dry-run" | rest], opts), do: do_parse(rest, [{:dry_run, true} | opts])
  defp do_parse(["--docker" | rest], opts), do: do_parse(rest, [{:docker, true} | opts])
  defp do_parse(["--runner", value | rest], opts), do: do_parse(rest, [{:runner, value} | opts])

  defp do_parse(["--docker-service", value | rest], opts),
    do: do_parse(rest, [{:docker_service, value} | opts])

  defp do_parse(["--engine-a", value | rest], opts),
    do: do_parse(rest, [{:engine_a, %{name: "new", command: value}} | opts])

  defp do_parse(["--engine-b", value | rest], opts),
    do: do_parse(rest, [{:engine_b, %{name: "base", command: value}} | opts])

  defp do_parse(["--games", value | rest], opts),
    do: do_parse(rest, [{:games, String.to_integer(value)} | opts])

  defp do_parse(["--concurrency", value | rest], opts),
    do: do_parse(rest, [{:concurrency, String.to_integer(value)} | opts])

  defp do_parse(["--tc", value | rest], opts), do: do_parse(rest, [{:tc, value} | opts])

  defp do_parse(["--openings", value | rest], opts),
    do: do_parse(rest, [{:openings, value} | opts])

  defp do_parse(["--sprt", elo0, elo1 | rest], opts) do
    sprt = %{elo0: parse_number(elo0), elo1: parse_number(elo1), alpha: 0.05, beta: 0.05}
    do_parse(rest, [{:sprt, sprt} | opts])
  end

  defp do_parse(rest, opts), do: {Enum.reverse(opts), rest}

  defp parse_number(value) do
    case Float.parse(value) do
      {number, ""} -> number
      _other -> Mix.raise("invalid numeric value: #{value}")
    end
  end
end
