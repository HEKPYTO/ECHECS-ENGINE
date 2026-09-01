defmodule Mix.Tasks.Engine.Bench do
  @moduledoc """
  Runs deterministic benchmarks or a paired fastchess match.

  ## Usage

      mix engine.bench
      mix engine.bench jsonl PATH
      mix engine.bench match --base-cmd CMD --base-args ARGS \\
        --candidate-cmd CMD --candidate-args ARGS --book BOOK \\
        [--rounds N --tc TC --concurrency N --elo0 F --elo1 F --dir DIR]

  * `mix engine.bench` — `EchecsEngine.Bench.smoke/0` (depth 2, fast).
  * `mix engine.bench jsonl PATH` — `EchecsEngine.Bench.run_jsonl!/2` over a
    `fen`/`bestmove` JSONL fixture.
  * `mix engine.bench match ...` — shells out to `fastchess` via
    `EchecsEngine.Bench.run_fastchess!/1` and prints the parsed SPRT result.
    Requires `fastchess` on `PATH` and an opening book (`.pgn` / `.epd`).
    Quote each `--*-args` value as a single shell string; it is forwarded as
    `args=` to fastchess.
  """
  use Mix.Task

  @shortdoc "Runs deterministic benchmarks or a paired fastchess match"

  @switches [
    base_cmd: :string,
    base_args: :string,
    candidate_cmd: :string,
    candidate_args: :string,
    book: :string,
    rounds: :integer,
    tc: :string,
    concurrency: :integer,
    elo0: :float,
    elo1: :float,
    dir: :string
  ]

  @impl Mix.Task
  def run(["match" | args]) do
    {opts, rest, invalid} = OptionParser.parse(args, strict: @switches)

    if rest != [] or invalid != [] do
      usage!()
    end

    required = [:base_cmd, :base_args, :candidate_cmd, :candidate_args, :book]

    unless Enum.all?(required, &Keyword.has_key?(opts, &1)) do
      usage!()
    end

    match =
      opts
      |> Keyword.put(:base_args, String.split(opts[:base_args]))
      |> Keyword.put(:candidate_args, String.split(opts[:candidate_args]))
      |> Keyword.put_new(:dir, File.cwd!())

    EchecsEngine.Bench.run_fastchess!(match) |> inspect() |> IO.puts()
  end

  def run(["jsonl", path]), do: EchecsEngine.Bench.run_jsonl!(path) |> inspect() |> IO.puts()
  def run([]), do: EchecsEngine.Bench.smoke() |> inspect() |> IO.puts()
  def run(_), do: usage!()

  @spec usage!() :: no_return()
  defp usage! do
    Mix.raise(
      "usage: mix engine.bench | mix engine.bench jsonl PATH | " <>
        "mix engine.bench match --base-cmd CMD --base-args ARGS --candidate-cmd CMD " <>
        "--candidate-args ARGS --book BOOK [--rounds N --tc TC --concurrency N --elo0 F --elo1 F]"
    )
  end
end
