defmodule EchecsEngine.TensorTest do
  use ExUnit.Case
  doctest EchecsEngine.Tensor

  alias EchecsEngine.Tensor

  describe "to_tensor/1" do
    test "converts standard starting board to tensor of shape {1, 119, 8, 8}" do
      game = Echecs.new_game()
      tensor = Tensor.to_tensor(game)

      assert Nx.shape(tensor) == {1, 119, 8, 8}
      assert Nx.type(tensor) == {:u, 8}

      white_pawns = tensor[0][0]
      row_6 = white_pawns[6] |> Nx.to_flat_list()
      assert row_6 == [1, 1, 1, 1, 1, 1, 1, 1]

      black_pawns = tensor[0][6]
      row_1 = black_pawns[1] |> Nx.to_flat_list()
      assert row_1 == [1, 1, 1, 1, 1, 1, 1, 1]
    end

    test "handles board as struct" do
      game = Echecs.new_game()
      struct_board = Echecs.Board.to_struct(game.board)
      game = %{game | board: struct_board}

      tensor = Tensor.to_tensor(game)
      assert Nx.shape(tensor) == {1, 119, 8, 8}
    end

    test "encodes side to move, castling, and counters in aux planes" do
      white = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w KQ - 7 12")
      black = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k b - - 0 1")

      white_tensor = Tensor.to_tensor(white)
      black_tensor = Tensor.to_tensor(black)

      assert white_tensor[0][112] |> Nx.sum() |> Nx.to_number() == 64
      assert black_tensor[0][112] |> Nx.sum() |> Nx.to_number() == 0

      assert white_tensor[0][113][0][0] |> Nx.to_number() > 0
      assert white_tensor[0][114] |> Nx.sum() |> Nx.to_number() == 64
      assert white_tensor[0][115] |> Nx.sum() |> Nx.to_number() == 64
      assert black_tensor[0][114] |> Nx.sum() |> Nx.to_number() == 0

      assert white_tensor[0][118][0][0] |> Nx.to_number() > 0
    end

    test "encodes previous positions and repetition planes in the 8-position stack" do
      start = Echecs.new_game()

      {:ok, after_e4} =
        Echecs.make_move(start, Echecs.Board.to_index("e2"), Echecs.Board.to_index("e4"), nil)

      {:ok, after_e5} =
        Echecs.make_move(after_e4, Echecs.Board.to_index("e7"), Echecs.Board.to_index("e5"), nil)

      repeated = %{
        after_e5
        | history: [after_e5.zobrist_hash, after_e5.zobrist_hash | after_e5.history],
          halfmove: 6
      }

      tensor = Tensor.to_tensor(repeated, [after_e4, start])

      assert tensor[0][12] |> Nx.sum() |> Nx.to_number() == 64
      assert tensor[0][13] |> Nx.sum() |> Nx.to_number() == 64

      assert tensor[0][14 + 0][4][4] |> Nx.to_number() == 1
      assert tensor[0][28 + 0][6][4] |> Nx.to_number() == 1
    end
  end
end
