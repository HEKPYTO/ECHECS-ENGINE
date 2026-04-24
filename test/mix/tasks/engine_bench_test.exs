defmodule Mix.Tasks.EngineBenchTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "prints benchmark summary" do
    path =
      Path.join(
        System.tmp_dir!(),
        "echecs_bench_task_#{System.unique_integer([:positive])}.jsonl"
      )

    File.write!(
      path,
      :json.encode(%{"id" => "start", "fen" => @start_fen, "best" => ["e2e4", "d2d4"]})
    )

    on_exit(fn -> File.rm(path) end)

    output =
      capture_io(fn ->
        Mix.Tasks.Engine.Bench.run([path])
      end)

    assert output =~ "positions:"
    assert output =~ "accuracy:"
  end

  test "prints SPRT summary from aggregate results" do
    output =
      capture_io(fn ->
        Mix.Tasks.Engine.Bench.run(["--sprt", "8", "1", "1", "--elo1", "300"])
      end)

    assert output =~ "sprt:"
    assert output =~ "decision: accept"
  end

  test "prints match summary from an opening suite" do
    path =
      Path.join(
        System.tmp_dir!(),
        "echecs_match_task_#{System.unique_integer([:positive])}.jsonl"
      )

    File.write!(path, IO.iodata_to_binary(:json.encode(%{"fen" => @start_fen})) <> "\n")
    on_exit(fn -> File.rm(path) end)

    output =
      capture_io(fn ->
        Mix.Tasks.Engine.Bench.run(["--match", path, "--plies", "2"])
      end)

    assert output =~ "games:"
    assert output =~ "moves:"
  end
end
