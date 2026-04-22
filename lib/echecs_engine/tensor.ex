defmodule EchecsEngine.Tensor do
  @moduledoc """
  Provides conversion adapters bridging `Echecs.Game` structures
  and `Nx.Tensor` formats for neural network ingestion.
  """

  import Bitwise

  @doc """
  Converts an `Echecs.Game` state to a uniform `Nx.Tensor`.

  Returns a tensor of shape `{1, 119, 8, 8}` formatted as `:u8`, where
  the first 12 planes represent the isolated piece-bitboards and the
  remaining planes act as padded context layers to match standard
  AlphaZero tensor dimension schemas.
  """
  def to_tensor(%Echecs.Game{board: board}) do
    board_tuple =
      if is_tuple(board) do
        board
      else
        Echecs.Board.from_struct(board)
      end

    bitboards =
      Tuple.to_list(board_tuple)
      |> Enum.take(12)

    layers_bin =
      bitboards
      |> Enum.map(&bitboard_to_binary/1)
      |> Enum.join()

    padding = :binary.copy(<<0>>, (119 - 12) * 64)
    full_bin = layers_bin <> padding

    Nx.from_binary(full_bin, {:u, 8})
    |> Nx.reshape({1, 119, 8, 8})
  end

  @doc false
  defp bitboard_to_binary(val) do
    for i <- 0..63, into: <<>>, do: <<val >>> i &&& 1::8>>
  end
end
