defmodule EchecsEngine.Tensor do
  @moduledoc """
  Provides conversion adapters bridging `Echecs.Game` structures
  and `Nx.Tensor` formats for neural network ingestion.
  """

  import Bitwise

  @history_length 8
  @planes_per_position 14
  @aux_planes 7

  @doc """
  Converts an `Echecs.Game` state to a uniform `Nx.Tensor`.

  Returns a tensor of shape `{1, 119, 8, 8}` formatted as `:u8`, where
  the first 112 planes represent an 8-position history stack with 14 planes
  per position (12 piece planes plus 2 repetition planes) and the final 7
  planes encode current side-to-move, move count, castling rights, and the
  halfmove clock.
  """
  def to_tensor(%Echecs.Game{} = game, history_games \\ []) do
    positions =
      [game | Enum.take(history_games, @history_length - 1)]
      |> Enum.take(@history_length)
      |> pad_positions()

    history_bin =
      positions
      |> Enum.flat_map(&position_layers/1)
      |> Enum.map(&layer_to_binary/1)
      |> Enum.join()

    aux_bin =
      game
      |> aux_layers()
      |> Enum.map(&layer_to_binary/1)
      |> Enum.join()

    Nx.from_binary(history_bin <> aux_bin, {:u, 8})
    |> Nx.reshape({1, @history_length * @planes_per_position + @aux_planes, 8, 8})
  end

  defp pad_positions(positions) do
    positions ++ List.duplicate(nil, @history_length - length(positions))
  end

  defp position_layers(nil), do: List.duplicate(constant_layer(0), @planes_per_position)

  defp position_layers(%Echecs.Game{board: board} = game) do
    board_tuple = if is_tuple(board), do: board, else: Echecs.Board.from_struct(board)

    board_tuple
    |> Tuple.to_list()
    |> Enum.take(12)
    |> Enum.map(&bitboard_to_list/1)
    |> Kernel.++([
      constant_layer(if(repetition_count(game) >= 2, do: 1, else: 0)),
      constant_layer(if(repetition_count(game) >= 3, do: 1, else: 0))
    ])
  end

  defp aux_layers(game) do
    [
      constant_layer(if(game.turn == :white, do: 1, else: 0)),
      constant_layer(min(game.fullmove, 255)),
      constant_layer(if((game.castling &&& 1) != 0, do: 1, else: 0)),
      constant_layer(if((game.castling &&& 2) != 0, do: 1, else: 0)),
      constant_layer(if((game.castling &&& 4) != 0, do: 1, else: 0)),
      constant_layer(if((game.castling &&& 8) != 0, do: 1, else: 0)),
      constant_layer(min(game.halfmove, 255))
    ]
  end

  defp constant_layer(value), do: List.duplicate(value, 64)

  defp bitboard_to_list(val), do: for(i <- 0..63, do: val >>> i &&& 1)

  defp repetition_count(%{history: history, zobrist_hash: zobrist_hash, halfmove: halfmove}) do
    history
    |> Enum.take(halfmove + 1)
    |> Enum.count(&(&1 == zobrist_hash))
  end

  defp layer_to_binary(layer) do
    for value <- layer, into: <<>>, do: <<value::8>>
  end
end
