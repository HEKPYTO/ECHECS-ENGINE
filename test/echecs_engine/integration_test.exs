defmodule EchecsEngine.IntegrationTest do
  use ExUnit.Case, async: false

  alias EchecsEngine.Eval

  test "Echecs perft and incremental evaluator stay correct through a deterministic playout" do
    game = Echecs.new_game()
    assert perft(game, 1) == 20
    assert perft(game, 2) == 400
    assert perft(game, 3) == 8_902

    weights = Eval.seed_weights()
    :rand.seed(:exsss, {17, 23, 41})

    Enum.reduce_while(1..48, {game, Eval.refresh(game, weights)}, fn _ply, {position, acc} ->
      moves = Echecs.MoveGen.legal_moves_int(position)

      case moves do
        [] ->
          {:halt, {position, acc}}

        _ ->
          move = Enum.at(moves, :rand.uniform(length(moves)) - 1)
          next = Echecs.Game.make_move_int(position, move)
          updated = Eval.update(acc, position, move, next, weights)

          assert updated == Eval.refresh(next, weights)

          assert Echecs.FEN.to_string(Echecs.new_game(Echecs.FEN.to_string(next))) ==
                   Echecs.FEN.to_string(next)

          {:cont, {next, updated}}
      end
    end)
  end

  defp perft(_game, 0), do: 1

  defp perft(game, depth) do
    game
    |> Echecs.MoveGen.legal_moves_int()
    |> Enum.reduce(0, fn move, nodes ->
      nodes + perft(Echecs.Game.make_move_int(game, move), depth - 1)
    end)
  end
end
