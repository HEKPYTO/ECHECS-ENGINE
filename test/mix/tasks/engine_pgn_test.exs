defmodule Mix.Tasks.EnginePgnTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  test "converts a PGN file into supervised JSONL" do
    pgn_path =
      Path.join(System.tmp_dir!(), "echecs_pgn_#{System.unique_integer([:positive])}.pgn")

    jsonl_path =
      Path.join(System.tmp_dir!(), "echecs_pgn_#{System.unique_integer([:positive])}.jsonl")

    File.write!(
      pgn_path,
      """
      [Event "Example"]
      [Result "1-0"]

      1. e4 e5 2. Nf3 Nc6 1-0
      """
    )

    on_exit(fn ->
      File.rm(pgn_path)
      File.rm(jsonl_path)
    end)

    output =
      capture_io(fn ->
        Mix.Tasks.Engine.Pgn.run([pgn_path, jsonl_path])
      end)

    assert output =~ "records:"
    assert File.exists?(jsonl_path)
    assert File.read!(jsonl_path) =~ "\"move\":\"e2e4\""
  end
end
