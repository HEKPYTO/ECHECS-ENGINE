defmodule EchecsEngine.NNUE.FeaturesTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.NNUE.Features

  test "extracts two-perspective HalfKP-like active feature indices" do
    game = Echecs.new_game()

    features = Features.active_indices(game)

    assert map_size(features) == 2
    assert length(features.white) == 30
    assert length(features.black) == 30
    assert Enum.all?(features.white ++ features.black, &is_integer/1)
    assert Enum.all?(features.white, &(&1 in 0..40_959))
    assert Enum.all?(features.black, &(&1 in 40_960..81_919))
  end

  test "king movement changes the perspective feature bucket" do
    before = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    after_king_move = Echecs.new_game("8/8/8/8/8/8/4P3/5K1k w - - 0 1")

    assert Features.active_indices(before).white != Features.active_indices(after_king_move).white
  end

  test "computes added and removed feature deltas between positions" do
    before = Echecs.new_game()

    {:ok, after_e4} =
      Echecs.make_move(before, Echecs.Board.to_index("e2"), Echecs.Board.to_index("e4"), nil)

    delta = Features.delta_indices(before, after_e4)

    assert delta.added != []
    assert delta.removed != []
    assert Enum.all?(delta.added ++ delta.removed, &is_integer/1)
  end

  test "computes move-aware feature deltas without requiring a next position" do
    before = Echecs.new_game()
    move = Enum.find(Echecs.legal_moves(before), &(&1.from == 52 and &1.to == 36))
    {:ok, after_e4} = Echecs.make_move(before, move.from, move.to, move.promotion)

    assert Features.delta_indices(before, move) == Features.delta_indices(before, after_e4)
  end

  test "move-aware feature deltas handle captures and promotions" do
    capture = Echecs.new_game("8/8/8/3p4/4P3/8/8/4K2k w - - 0 1")
    capture_move = Enum.find(Echecs.legal_moves(capture), &(&1.from == 36 and &1.to == 27))
    {:ok, after_capture} = Echecs.make_move(capture, 36, 27)

    assert Features.delta_indices(capture, capture_move) ==
             Features.delta_indices(capture, after_capture)

    promotion = Echecs.new_game("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    promotion_move = Enum.find(Echecs.legal_moves(promotion), &(&1.promotion == :queen))

    {:ok, after_promotion} =
      Echecs.make_move(promotion, promotion_move.from, promotion_move.to, :queen)

    assert Features.delta_indices(promotion, promotion_move) ==
             Features.delta_indices(promotion, after_promotion)
  end

  test "move-aware feature deltas fall back when king perspective changes" do
    before = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    king_move = Enum.find(Echecs.legal_moves(before), &(&1.from == 60 and &1.to == 59))
    {:ok, after_king_move} = Echecs.make_move(before, king_move.from, king_move.to, nil)

    assert Features.delta_indices(before, king_move) ==
             Features.delta_indices(before, after_king_move)
  end
end
