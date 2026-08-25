defmodule Mix.Tasks.EngineBestTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO
  alias Mix.Tasks.Engine.Best

  test "prints a seed-evaluator move" do
    output =
      capture_io(fn ->
        Best.run([
          "--depth",
          "1",
          "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ])
      end)

    assert output =~ ~r/^[a-h][1-8][a-h][1-8][qrbn]?\n$/
  end
end
