defmodule EchecsEngine.BestMoveTest do
  use ExUnit.Case, async: true

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "best_move/2 returns a UCI string for the best move from FEN" do
    game = Echecs.new_game(@start_fen)
    target_move = Enum.find(Echecs.legal_moves(game), &(&1.from == 52 and &1.to == 36))

    inference = fn _tensor ->
      policy = Nx.broadcast(-10.0, {4672})

      policy =
        Nx.put_slice(
          policy,
          [EchecsEngine.Policy.move_index(game, target_move)],
          Nx.tensor([10.0])
        )

      %{policy: policy, value: Nx.tensor([0.2])}
    end

    assert {:ok, "e2e4"} =
             EchecsEngine.best_move(@start_fen,
               inference: inference,
               candidate_limit: 1
             )
  end

  test "best_move/2 includes promotion suffix" do
    fen = "8/P7/8/8/8/8/8/k6K w - - 0 1"
    game = Echecs.new_game(fen)

    target_move =
      Enum.find(
        Echecs.legal_moves(game),
        &(&1.from == 8 and &1.to == 0 and &1.promotion == :queen)
      )

    inference = fn _tensor ->
      policy = Nx.broadcast(-10.0, {4672})

      policy =
        Nx.put_slice(
          policy,
          [EchecsEngine.Policy.move_index(game, target_move)],
          Nx.tensor([10.0])
        )

      %{policy: policy, value: Nx.tensor([0.2])}
    end

    assert {:ok, "a7a8q"} =
             EchecsEngine.best_move(fen,
               inference: inference,
               candidate_limit: 1
             )
  end

  test "best_move/2 returns terminal status for positions without legal moves" do
    fen = "7k/5Q2/7K/8/8/8/8/8 b - - 0 1"

    assert {:terminal, :stalemate} = EchecsEngine.best_move(fen)
  end

  test "best_move/2 returns an error for invalid FEN" do
    assert {:error, _reason} = EchecsEngine.best_move("not a fen")
  end
end
