defmodule EchecsEngineTest do
  use ExUnit.Case, async: true

  test "public API searches without evaluator options" do
    assert {:ok, move} =
             EchecsEngine.best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
               depth: 1
             )

    assert move =~ ~r/^[a-h][1-8][a-h][1-8][qrbn]?$/
  end

  test "public API rejects unknown options" do
    assert {:error, {:unknown_option, :bogus}} =
             EchecsEngine.best_move("8/8/8/8/8/8/4P3/4K2k w - - 0 1", bogus: true)
  end
end
