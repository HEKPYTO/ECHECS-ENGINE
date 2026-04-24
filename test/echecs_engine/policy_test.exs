defmodule EchecsEngine.PolicyTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Policy

  test "legal_move_priors/3 gives the highest probability to the targeted legal move" do
    game = Echecs.new_game()
    target_move = Enum.find(Echecs.legal_moves(game), &(&1.from == 52 and &1.to == 36))

    policy = Nx.broadcast(-10.0, {4672})
    policy = Nx.put_slice(policy, [Policy.move_index(game, target_move)], Nx.tensor([9.0]))

    priors = Policy.legal_move_priors(game, Echecs.legal_moves(game), policy)

    {best_move, best_prior} = Enum.max_by(priors, fn {_move, prior} -> prior end)

    assert best_move == target_move
    assert best_prior > 0.9
  end
end
