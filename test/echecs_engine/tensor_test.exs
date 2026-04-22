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

      # For the starting position, check white pawns
      white_pawns = tensor[0][0]
      row_6 = white_pawns[6] |> Nx.to_flat_list()
      assert row_6 == [1, 1, 1, 1, 1, 1, 1, 1]

      black_pawns = tensor[0][6]
      row_1 = black_pawns[1] |> Nx.to_flat_list()
      assert row_1 == [1, 1, 1, 1, 1, 1, 1, 1]
    end

    test "handles board as struct" do
      game = Echecs.new_game()
      # Force board to be a struct
      struct_board = Echecs.Board.to_struct(game.board)
      game = %{game | board: struct_board}

      tensor = Tensor.to_tensor(game)
      assert Nx.shape(tensor) == {1, 119, 8, 8}
    end
  end
end
