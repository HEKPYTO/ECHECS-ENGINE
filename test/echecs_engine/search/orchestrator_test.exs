defmodule EchecsEngine.Search.OrchestratorTest do
  use ExUnit.Case

  test "hybrid search successfully evaluates a board state and picks a move" do
    # We must ensure the application (specifically the Serving) is running
    # but since ExUnit starts applications, EchecsEngine.Serving should be available.

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
