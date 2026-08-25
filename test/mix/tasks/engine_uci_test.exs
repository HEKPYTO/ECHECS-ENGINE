defmodule Mix.Tasks.EngineUCITest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO
  alias Mix.Tasks.Engine.Uci

  test "runs a small UCI session" do
    input = "uci\nisready\nquit\n"

    output =
      capture_io(input, fn ->
        Uci.run([])
      end)

    assert output =~ "id name ECHECS-ENGINE"
    assert output =~ "uciok"
    assert output =~ "readyok"
  end
end
