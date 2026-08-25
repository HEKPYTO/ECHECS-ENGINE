defmodule EchecsEngine.EvalTest do
  use ExUnit.Case, async: true
  require Echecs.Move

  alias EchecsEngine.Eval

  test "seed artifact round-trips byte-for-byte" do
    path =
      Path.join(System.tmp_dir!(), "echecs-engine-#{System.unique_integer([:positive])}.nnue")

    weights = Eval.seed_weights()
    :ok = Eval.dump!(path, weights)
    assert Eval.dump_binary(Eval.load!(path)) == Eval.dump_binary(weights)
  end

  test "rejects an incompatible artifact" do
    path =
      Path.join(System.tmp_dir!(), "echecs-engine-#{System.unique_integer([:positive])}.nnue")

    File.write!(path, "ETNN" <> <<2::unsigned-16>>)

    assert_raise ArgumentError, ~r/unsupported evaluator artifact/, fn -> Eval.load!(path) end
  end

  test "rejects nonpositive scales while dumping and loading" do
    weights = Eval.seed_weights()

    for scale <- [0, -1] do
      assert_raise ArgumentError, ~r/invalid evaluator weights/, fn ->
        Eval.dump_binary(%{weights | scale: scale})
      end

      binary = Eval.dump_binary(weights)
      <<prefix::binary-size(20), _old_scale::signed-32, rest::binary>> = binary
      path = Path.join(System.tmp_dir!(), "echecs-engine-scale-#{scale}.nnue")
      File.write!(path, <<prefix::binary, scale::signed-32, rest::binary>>)

      assert_raise ArgumentError, ~r/invalid evaluator weights/, fn -> Eval.load!(path) end
    end
  end

  test "seed network values material for the side to move" do
    weights = Eval.seed_weights()
    white = Echecs.new_game("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    black = Echecs.new_game("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")

    assert Eval.evaluate(white, Eval.refresh(white, weights), weights) > 800
    assert Eval.evaluate(black, Eval.refresh(black, weights), weights) < -800
  end

  test "incremental accumulator equals refresh after every legal start-position move" do
    weights = Eval.seed_weights()
    game = Echecs.new_game()
    acc = Eval.refresh(game, weights)

    Enum.each(Echecs.MoveGen.legal_moves_int(game), fn move ->
      next = Echecs.Game.make_move_int(game, move)
      assert Eval.update(acc, game, move, next, weights) == Eval.refresh(next, weights)
    end)
  end

  test "packed special moves update incrementally" do
    weights = Eval.seed_weights()

    for {fen, uci} <- [
          {"4k3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1", "d4d5"},
          {"4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"},
          {"4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"},
          {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"}
        ] do
      game = Echecs.new_game(fen)
      move = packed_move(game, uci)
      next = Echecs.Game.make_move_int(game, move)

      assert Eval.update(Eval.refresh(game, weights), game, move, next, weights) ==
               Eval.refresh(next, weights)
    end
  end

  test "king bucket transition matches a full refresh" do
    weights = Eval.seed_weights()
    game = Echecs.new_game("4k3/8/8/8/8/8/4K3/8 w - - 0 1")
    move = packed_move(game, "e2e3")
    next = Echecs.Game.make_move_int(game, move)

    assert Eval.update(Eval.refresh(game, weights), game, move, next, weights) ==
             Eval.refresh(next, weights)
  end

  test "integer dense tail affects the runtime score" do
    game = Echecs.new_game()
    weights = Eval.seed_weights()
    acc = Eval.refresh(game, weights)

    dense = %{
      weights
      | w1: :binary.copy(<<1::signed-8>>, 128 * 16),
        w2: :binary.copy(<<1::signed-8>>, 16),
        b1: :binary.copy(<<64::signed-32>>, 16),
        scale: 1
    }

    refute Eval.evaluate(game, acc, weights) == Eval.evaluate(game, acc, dense)
  end

  defp packed_move(game, uci) do
    <<from::binary-size(2), to::binary-size(2), suffix::binary>> = uci
    promotion = %{"" => nil, "q" => :queen}[suffix]

    Enum.find(Echecs.MoveGen.legal_moves_int(game), fn move ->
      Echecs.Board.to_algebraic(Echecs.Move.unpack_from(move)) == from and
        Echecs.Board.to_algebraic(Echecs.Move.unpack_to(move)) == to and
        Echecs.Move.unpack_promotion(move) == promotion
    end)
  end
end
