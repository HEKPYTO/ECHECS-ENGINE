defmodule Mix.Tasks.EngineUCITest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  test "runs a small UCI session" do
    input = "uci\nisready\nquit\n"

    output =
      capture_io(input, fn ->
        Mix.Tasks.Engine.Uci.run([])
      end)

    assert output =~ "id name ECHECS-ENGINE"
    assert output =~ "uciok"
    assert output =~ "readyok"
  end
end
