defmodule EchecsEngine.BenchmarkTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Benchmark

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "runs a JSONL benchmark suite and scores expected best moves" do
    path =
      Path.join(System.tmp_dir!(), "echecs_benchmark_#{System.unique_integer([:positive])}.jsonl")

    records = [
      Jason.encode!(%{"id" => "start", "fen" => @start_fen, "best" => ["e2e4", "d2d4"]}),
      Jason.encode!(%{"id" => "miss", "fen" => @start_fen, "best" => ["g1f3"]})
    ]

    File.write!(path, Enum.join(records, "\n"))
    on_exit(fn -> File.rm(path) end)

    start_fen = @start_fen
    best_move = fn ^start_fen, _opts -> {:ok, "e2e4"} end

    result = Benchmark.run_jsonl!(path, best_move: best_move)

    assert result.total == 2
    assert result.correct == 1
    assert result.accuracy == 0.5
    assert length(result.positions) == 2
    assert Enum.all?(result.positions, &is_integer(&1.time_us))
  end

  test "computes SPRT-style accept/reject/continue decisions" do
    assert %{decision: :accept} = Benchmark.sprt(%{wins: 8, losses: 1, draws: 1}, elo1: 300)
    assert %{decision: :reject} = Benchmark.sprt(%{wins: 1, losses: 8, draws: 1}, elo1: 300)
    assert %{decision: :continue} = Benchmark.sprt(%{wins: 2, losses: 2, draws: 6})
  end

  test "runs a simple engine-vs-engine match over opening positions" do
    opening_path =
      Path.join(System.tmp_dir!(), "echecs_match_#{System.unique_integer([:positive])}.jsonl")

    File.write!(
      opening_path,
      IO.iodata_to_binary(Jason.encode!(%{"fen" => @start_fen})) <> "\n"
    )

    on_exit(fn -> File.rm(opening_path) end)

    white = fn _fen, _opts -> {:ok, "e2e4"} end
    black = fn _fen, _opts -> {:ok, "e7e5"} end

    result = Benchmark.run_match_jsonl!(opening_path, white: white, black: black, plies: 2)

    assert result.games == 1
    assert length(result.results) == 1
    assert hd(result.results).moves == ["e2e4", "e7e5"]
  end
end
