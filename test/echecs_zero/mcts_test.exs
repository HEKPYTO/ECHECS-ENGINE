defmodule EchecsZero.MCTSTest do
  use ExUnit.Case

  alias EchecsZero.MCTS
  alias EchecsZero.MCTS.Node

  # The serving might take a bit to spin up or evaluate
  @tag timeout: 120_000
  test "search/2 performs MCTS and returns an updated root node" do
    game = Echecs.new_game()
    iterations = 5

    root_node = MCTS.search(game, iterations)

    assert %Node{} = root_node
    assert root_node.visits == iterations
    assert root_node.game == game
    assert map_size(root_node.children) > 0

    children_visits =
      Enum.map(root_node.children, fn {_, child} -> child.visits end) |> Enum.sum()

    # The first iteration expands the root, subsequent iterations expand children
    assert children_visits == iterations - 1
  end
end
