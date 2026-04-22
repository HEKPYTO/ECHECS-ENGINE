defmodule EchecsEngine.Search.OrchestratorTest do
  use ExUnit.Case

  test "hybrid search successfully evaluates a board state and picks a move" do
    game = Echecs.new_game()

    result = EchecsEngine.Search.Orchestrator.search(game)

    case result do
      {move, score} ->
        assert %Echecs.Move{} = move
        assert is_float(score)

      other ->
        flunk("Expected {move, score}, got #{inspect(other)}")
    end
  end
end
