defmodule EchecsEngine.Tensor do
  @moduledoc """
  Adapter for converting Echecs.Game structs to Nx.Tensors.
  """

  import Bitwise

  @doc """
  Converts an Echecs.Game to an Nx.Tensor.
  Returns a tensor of shape {1, 119, 8, 8}.
  """
  def to_tensor(%Echecs.Game{board: board}) do
    board_tuple =
      if is_tuple(board) do
        board
      else
        Echecs.Board.from_struct(board)
      end

    # Extract the 12 piece bitboards
    bitboards =
      Tuple.to_list(board_tuple)
      |> Enum.take(12)

    # Convert bitboards to binary representations of 64 bytes each
    layers_bin =
      bitboards
      |> Enum.map(&bitboard_to_binary/1)
      |> Enum.join()

    # Pad with zeros up to 119 layers
    padding = :binary.copy(<<0>>, (119 - 12) * 64)
    full_bin = layers_bin <> padding

    Nx.from_binary(full_bin, {:u, 8})
    |> Nx.reshape({1, 119, 8, 8})
  end

  defp bitboard_to_binary(val) do
    for i <- 0..63, into: <<>>, do: <<val >>> i &&& 1::8>>
  end
end
