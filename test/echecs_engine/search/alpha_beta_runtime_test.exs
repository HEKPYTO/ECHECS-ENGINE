defmodule EchecsEngine.Search.AlphaBetaRuntimeTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Search.AlphaBeta

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "transposition keys include history for history-dependent neural evaluation" do
    game = Echecs.new_game(@start_fen)

    {:ok, history_game} = Echecs.make_move(game, 52, 36)

    assert AlphaBeta.transposition_key(game, [], []) !=
             AlphaBeta.transposition_key(game, [history_game], [])
  end

  test "transposition keys include evaluator closure environment" do
    game = Echecs.new_game(@start_fen)

    assert AlphaBeta.transposition_key(game, [], evaluator_fn: closure_evaluator(1.0)) !=
             AlphaBeta.transposition_key(game, [], evaluator_fn: closure_evaluator(-1.0))
  end

  test "transposition keys include inference closure environment" do
    game = Echecs.new_game(@start_fen)

    assert AlphaBeta.transposition_key(game, [], inference: closure_inference([0.7, 0.2, 0.1])) !=
             AlphaBeta.transposition_key(game, [], inference: closure_inference([0.1, 0.2, 0.7]))
  end

  test "search/2 returns a best move with real stats and principal variation" do
    game = Echecs.new_game(@start_fen)
    target_move = Enum.find(Echecs.legal_moves(game), &(&1.from == 52 and &1.to == 36))
    other_move = Enum.find(Echecs.legal_moves(game), &(&1.from == 51 and &1.to == 35))

    {:ok, target_game} =
      Echecs.make_move(game, target_move.from, target_move.to, target_move.promotion)

    {:ok, other_game} =
      Echecs.make_move(game, other_move.from, other_move.to, other_move.promotion)

    target_fen = Echecs.FEN.to_string(target_game)
    other_fen = Echecs.FEN.to_string(other_game)

    inference = fn _tensor ->
      policy = Nx.broadcast(-10.0, {4672})

      policy =
        Nx.put_slice(
          policy,
          [EchecsEngine.Policy.move_index(game, other_move)],
          Nx.tensor([10.0])
        )

      %{policy: policy, wdl: Nx.tensor([0.4, 0.4, 0.2]), moves_left: Nx.tensor([30.0])}
    end

    evaluator_fn = fn position ->
      case Echecs.FEN.to_string(position) do
        ^target_fen -> -1.0
        ^other_fen -> 1.0
        _other -> 0.0
      end
    end

    assert %{best_move: ^target_move, depth: 1, nodes: nodes, pv: ["e2e4" | _rest], score: score} =
             AlphaBeta.search(game,
               depth: 1,
               inference: inference,
               evaluator_fn: evaluator_fn
             )

    assert is_integer(nodes)
    assert nodes > 2
    assert is_float(score)
  end

  test "neural policy ordering is root-only by default" do
    game = Echecs.new_game(@start_fen)
    parent = self()

    inference = fn _tensor ->
      send(parent, :inference_call)

      %{
        policy: Nx.broadcast(0.0, {4672}),
        wdl: Nx.tensor([0.4, 0.4, 0.2]),
        moves_left: Nx.tensor([30.0])
      }
    end

    _result =
      AlphaBeta.search(game,
        depth: 2,
        inference: inference,
        evaluator_fn: fn _position -> 0.0 end
      )

    assert_receive :inference_call
    assert_receive :inference_call
    refute_receive :inference_call, 20
  end

  test "infinite alpha-beta search is not capped to the default depth" do
    game = Echecs.new_game(@start_fen)

    result =
      AlphaBeta.search(game,
        infinite: true,
        nodes: 500,
        evaluator_fn: fn _position -> 0.0 end
      )

    assert result.depth > 2
    assert result.nodes >= 500
  end

  test "sparse evaluator scores are interpreted from the side to move" do
    white = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    black = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k b - - 0 1")

    feature_table =
      [white, black]
      |> Enum.flat_map(fn game ->
        game
        |> EchecsEngine.NNUE.Features.active_indices()
        |> Map.values()
        |> List.flatten()
      end)
      |> Enum.uniq()
      |> Map.new(fn feature_idx ->
        {feature_idx, Nx.broadcast(0.0, {3072}) |> Nx.put_slice([0], Nx.tensor([1.0]))}
      end)

    weights = %{
      feature_table: feature_table,
      w1: Nx.broadcast(0.0, {3072, 32}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      b1: Nx.broadcast(0.0, {32}),
      w2: Nx.broadcast(0.0, {32, 1}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      b2: Nx.broadcast(0.0, {1})
    }

    white_result = AlphaBeta.search(white, depth: 1, evaluator_weights: weights)
    black_result = AlphaBeta.search(black, depth: 1, evaluator_weights: weights)

    assert white_result.score > 0.0
    assert black_result.score < 0.0
  end

  defp closure_evaluator(score) do
    fn _game -> score end
  end

  defp closure_inference(wdl) do
    fn _tensor ->
      %{policy: Nx.broadcast(0.0, {4672}), wdl: Nx.tensor(wdl), moves_left: Nx.tensor([30.0])}
    end
  end
end
