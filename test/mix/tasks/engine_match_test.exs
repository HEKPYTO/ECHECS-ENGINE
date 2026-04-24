defmodule Mix.Tasks.EngineMatchTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  test "prints the external match command in dry-run mode" do
    output =
      capture_io(fn ->
        Mix.Tasks.Engine.Match.run([
          "--dry-run",
          "--runner",
          "cutechess-cli",
          "--engine-a",
          "mix engine.uci",
          "--engine-b",
          "stockfish",
          "--games",
          "40",
          "--concurrency",
          "2",
          "--sprt",
          "0",
          "5"
        ])
      end)

    assert output =~ "cutechess-cli"
    assert output =~ "cmd=mix engine.uci"
    assert output =~ "-sprt"
  end

  test "prints a dockerized fastchess command in dry-run mode" do
    output =
      capture_io(fn ->
        Mix.Tasks.Engine.Match.run([
          "--dry-run",
          "--docker",
          "--engine-a",
          "mix engine.uci",
          "--engine-b",
          "stockfish",
          "--games",
          "40",
          "--concurrency",
          "2",
          "--sprt",
          "0",
          "5"
        ])
      end)

    assert output =~ "docker compose run --rm --no-deps engine-match fastchess"
    assert output =~ "cmd=mix engine.uci"
    assert output =~ "-sprt"
  end
end
