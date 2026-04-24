defmodule EchecsEngine.PGNDatasetTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.PGNDataset

  test "converts a PGN game into supervised JSONL-ready records" do
    pgn = """
    [Event "Example"]
    [Result "1-0"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
    """

    assert records = PGNDataset.records_from_pgn!(pgn)
    assert length(records) == 6

    first = hd(records)
    assert first["fen"] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert first["move"] == "e2e4"
    assert first["result"] == "1-0"
    assert first["history_fens"] == []

    second = Enum.at(records, 1)
    assert second["move"] == "e7e5"
    assert length(second["history_fens"]) == 1
  end

  test "converts every game in a multi-game PGN corpus" do
    pgn = """
    [Event "One"]
    [Result "1-0"]

    1. e4 e5 1-0

    [Event "Two"]
    [Result "0-1"]

    1. d4 d5 0-1
    """

    records = PGNDataset.records_from_pgn!(pgn)

    assert Enum.map(records, & &1["move"]) == ["e2e4", "e7e5", "d2d4", "d7d5"]
    assert Enum.map(records, & &1["result"]) == ["1-0", "1-0", "0-1", "0-1"]
  end

  test "writes JSONL incrementally while consuming the enumerable" do
    path =
      System.tmp_dir!()
      |> Path.join("pgn_dataset_stream_#{System.unique_integer([:positive])}.jsonl")

    record1 = %{"fen" => "fen-1", "move" => "e2e4"}
    record2 = %{"fen" => "fen-2", "move" => "d2d4"}

    records =
      Stream.transform([record1, record2], 0, fn record, index ->
        if index == 1 do
          assert {:ok, contents} = File.read(path)
          assert contents == IO.iodata_to_binary([Jason.encode!(record1), "\n"])
        end

        {[record], index + 1}
      end)

    on_exit(fn -> File.rm(path) end)

    :ok = PGNDataset.write_jsonl!(path, records)

    assert File.read!(path) ==
             IO.iodata_to_binary([
               Jason.encode!(record1),
               "\n",
               Jason.encode!(record2),
               "\n"
             ])
  end
end
