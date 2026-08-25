defmodule EchecsEngine.BenchTest do
  use ExUnit.Case, async: false

  alias EchecsEngine.Bench

  test "fixed depth signatures retain deterministic chess values" do
    one = Bench.signature(depth: 3)
    two = Bench.signature(depth: 3)

    assert Map.take(one, [:positions, :nodes, :score, :pv]) ==
             Map.take(two, [:positions, :nodes, :score, :pv])

    assert one.nps >= 0
    assert one.engine_version
    assert one.tt_hits > 0
    assert one.tt_hit_rate > 0 and one.tt_hit_rate <= 1
    assert one.tt_slots > 0
    assert one.tt_entries > 0
    assert is_integer(one.memory_bytes_delta)
    assert is_map(one.gc)
  end

  test "parses the last official fastchess normalized-Elo SPRT report" do
    output = """
    Ptnml(0-2): [9, 8, 7, 6, 5]
    LLR: -3.2 (108.8%) (-2.94, 2.94) [0.00, 5.00]
    Ptnml(0-2): [1, 7, 99, 7, 1]
    LLR: 3.01 (102.2%) (-2.94, 2.94) [0.00, 5.00]
    """

    assert %{decision: :accept, buckets: [1, 7, 99, 7, 1], llr: 3.01} =
             Bench.parse_fastchess_report!(output)
  end

  test "derives reject and continue only from fastchess-reported SPRT bounds" do
    assert %{decision: :reject, buckets: [45, 3, 900, 3, 1]} =
             Bench.parse_fastchess_report!(
               "Ptnml(0-2): [45, 3, 900, 3, 1]\nLLR: -3.0 (101.9%) (-2.94, 2.94) [0, 5]"
             )

    assert %{decision: :continue, buckets: [0, 0, 1_000, 0, 0]} =
             Bench.parse_fastchess_report!(
               "Ptnml(0-2): [0, 0, 1000, 0, 0]\nLLR: 0.0 (0.0%) (-2.94, 2.94) [0, 5]"
             )
  end

  test "rejects malformed fastchess reports instead of estimating normalized Elo" do
    for output <- [
          "Ptnml(0-2): [1, 2, 3, 4]\nLLR: 0.0 (0.0%) (-2.94, 2.94) [0, 5]",
          "Ptnml(0-2): [1, -2, 3, 4, 5]\nLLR: 0.0 (0.0%) (-2.94, 2.94) [0, 5]",
          "Ptnml(0-2): [1, 2, 3, 4, 5]"
        ] do
      assert_raise ArgumentError, ~r/fastchess/, fn -> Bench.parse_fastchess_report!(output) end
    end
  end

  test "constructs a paired fastchess argv and parses runner stdout" do
    parent = self()
    report = "Ptnml(0-2): [0, 1, 2, 3, 4]\nLLR: 0.5 (17.0%) (-2.94, 2.94) [0, 5]"

    runner = fn executable, argv, options ->
      send(parent, {:fastchess, executable, argv, options})
      {report, 0}
    end

    assert %{decision: :continue, buckets: [0, 1, 2, 3, 4]} =
             Bench.run_fastchess!(
               base_cmd: "mix",
               base_args: ["engine.uci"],
               candidate_cmd: "mix",
               candidate_args: ["engine.uci"],
               book: "book.pgn",
               rounds: 12,
               tc: "10+0.1",
               concurrency: 2,
               elo0: 0.0,
               elo1: 5.0,
               dir: "/tmp",
               runner: runner
             )

    assert_receive {:fastchess, "fastchess", argv, command_options}
    assert command_options[:cd] == "/tmp"
    assert command_options[:stderr_to_stdout]

    assert ["-rounds", "12"] =
             Enum.chunk_every(argv, 2, 1, :discard) |> Enum.find(&(&1 == ["-rounds", "12"]))

    assert "-repeat" in argv
    refute Enum.member?(Enum.chunk_every(argv, 2, 1, :discard), ["-repeat", "2"])

    assert ["-engine", "name=base", "cmd=mix", "args=engine.uci"] == Enum.take(argv, 4)
    assert ["-engine", "name=candidate", "cmd=mix", "args=engine.uci"] == Enum.slice(argv, 4, 4)

    assert ["-report", "penta=true"] =
             Enum.chunk_every(argv, 2, 1, :discard)
             |> Enum.find(&(&1 == ["-report", "penta=true"]))

    assert Enum.member?(argv, "model=normalized")
    assert Enum.member?(argv, "format=pgn")
  end

  test "selects fastchess opening format from an EPD book extension" do
    parent = self()
    report = "Ptnml(0-2): [0, 0, 1, 0, 0]\nLLR: 0.0 (0.0%) (-2.94, 2.94) [0, 5]"

    runner = fn _executable, argv, _options ->
      send(parent, {:fastchess_argv, argv})
      {report, 0}
    end

    Bench.run_fastchess!(
      base_cmd: "mix",
      base_args: ["engine.uci"],
      candidate_cmd: "mix",
      candidate_args: ["engine.uci"],
      book: "openings.epd",
      runner: runner
    )

    assert_receive {:fastchess_argv, argv}
    assert "format=epd" in argv
  end

  test "includes fastchess output when the paired runner exits unsuccessfully" do
    runner = fn _executable, _argv, _options -> {"opening book failed", 7} end

    assert_raise ArgumentError, ~r/fastchess exited 7: opening book failed/, fn ->
      Bench.run_fastchess!(
        base_cmd: "mix",
        base_args: ["engine.uci"],
        candidate_cmd: "mix",
        candidate_args: ["engine.uci"],
        book: "book.pgn",
        runner: runner
      )
    end
  end

  test "JSONL reports malformed rows with line numbers" do
    path = Path.join(System.tmp_dir!(), "bad-bench-#{System.unique_integer([:positive])}.jsonl")
    File.write!(path, "{bad}\n")
    assert_raise ArgumentError, ~r/line 1/, fn -> Bench.run_jsonl!(path, depth: 1) end
  end
end
