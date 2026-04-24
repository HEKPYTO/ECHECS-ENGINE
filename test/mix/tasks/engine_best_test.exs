defmodule Mix.Tasks.EngineBestTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureIO

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "prints the best move returned by the engine" do
    output =
      capture_io(fn ->
        Mix.Tasks.Engine.Best.run(["--allow-zero-evaluator", @start_fen])
      end)

    assert output =~ ~r/^[a-h][1-8][a-h][1-8][qrbn]?\n$/
  end

  test "fails loudly without an evaluator or explicit smoke-test fallback" do
    assert_raise Mix.Error, ~r/missing_evaluator/, fn ->
      Mix.Tasks.Engine.Best.run(["--no-default-evaluator", @start_fen])
    end
  end
end
