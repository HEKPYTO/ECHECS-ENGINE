defmodule EchecsEngine.MatchRunnerTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.MatchRunner

  test "builds a cutechess-cli command with SPRT gating" do
    assert {exe, args} =
             MatchRunner.command(
               runner: "cutechess-cli",
               engine_a: %{name: "new", command: "mix engine.uci"},
               engine_b: %{name: "base", command: "stockfish"},
               games: 200,
               concurrency: 4,
               tc: "10+0.1",
               openings: "openings.epd",
               sprt: %{elo0: 0, elo1: 5, alpha: 0.05, beta: 0.05}
             )

    assert exe == "cutechess-cli"
    assert "-engine" in args
    assert "name=new" in args
    assert "cmd=mix engine.uci" in args
    assert "-openings" in args
    assert "file=openings.epd" in args
    assert "-sprt" in args
    assert "elo0=0" in args
    assert "elo1=5" in args
  end

  test "builds a fastchess command with common match options" do
    assert {"fastchess", args} =
             MatchRunner.command(
               runner: "fastchess",
               engine_a: %{name: "new", command: "mix engine.uci"},
               engine_b: %{name: "base", command: "stockfish"},
               games: 20,
               concurrency: 2,
               tc: "5+0.05"
             )

    assert "-engine" in args
    assert "cmd=mix engine.uci" in args
    assert "-each" in args
    assert "tc=5+0.05" in args
    assert "-rounds" in args
    assert "10" in args
  end

  test "returns a clear error when the external runner is missing" do
    assert {:error, {:runner_not_found, "missing-runner"}} =
             MatchRunner.run(
               runner: "missing-runner",
               engine_a: %{name: "new", command: "mix engine.uci"},
               engine_b: %{name: "base", command: "stockfish"}
             )
  end

  test "wraps fastchess in docker compose for containerized validation" do
    assert {"docker", args} =
             MatchRunner.command(
               docker: true,
               engine_a: %{name: "new", command: "mix engine.uci"},
               engine_b: %{name: "base", command: "stockfish"},
               games: 20,
               concurrency: 2,
               tc: "5+0.05"
             )

    assert Enum.take(args, 5) == ["compose", "run", "--rm", "--no-deps", "engine-match"]
    assert "fastchess" in args
    assert "cmd=mix engine.uci" in args
    assert "-rounds" in args
  end

  test "rejects cutechess-cli for dockerized validation" do
    assert_raise ArgumentError, ~r/docker match runner supports only fastchess/, fn ->
      MatchRunner.command(
        docker: true,
        runner: "cutechess-cli",
        engine_a: %{name: "new", command: "mix engine.uci"},
        engine_b: %{name: "base", command: "stockfish"}
      )
    end
  end
end
