defmodule EchecsEngine.NNUE.Features do
  @moduledoc """
  HalfKP-like feature extraction for NNUE-style sparse evaluators.
  """

  @perspective_size 64 * 10 * 64

  @type active_indices :: %{white: [non_neg_integer()], black: [non_neg_integer()]}

  @type delta_indices :: %{added: [non_neg_integer()], removed: [non_neg_integer()]}

  @spec active_indices(Echecs.Game.t()) :: active_indices()
  def active_indices(%Echecs.Game{board: board, king_pos: {white_king, black_king}}) do
    board_tuple = ensure_tuple(board)

    piece_features =
      for square <- 0..63,
          piece = Echecs.Board.at_tuple(board_tuple, square),
          include_piece?(piece) do
        {square, piece_index(piece)}
      end

    %{
      white:
        Enum.map(piece_features, fn {square, piece_idx} ->
          feature_index(white_king, piece_idx, square, 0)
        end),
      black:
        Enum.map(piece_features, fn {square, piece_idx} ->
          feature_index(black_king, piece_idx, square, @perspective_size)
        end)
    }
  end

  @spec delta_indices(Echecs.Game.t(), Echecs.Game.t()) :: delta_indices()
  def delta_indices(%Echecs.Game{} = before, %Echecs.Game{} = after_game) do
    before_set = active_indices(before) |> flatten_feature_set()
    after_set = active_indices(after_game) |> flatten_feature_set()

    %{
      added: MapSet.difference(after_set, before_set) |> MapSet.to_list() |> Enum.sort(),
      removed: MapSet.difference(before_set, after_set) |> MapSet.to_list() |> Enum.sort()
    }
  end

  def delta_indices(%Echecs.Game{} = before, %Echecs.Move{} = move) do
    board_tuple = ensure_tuple(before.board)

    case Echecs.Board.at_tuple(board_tuple, move.from) do
      {_color, :king} ->
        full_delta_after_move(before, move)

      nil ->
        full_delta_after_move(before, move)

      mover ->
        if move.special in [:kingside_castle, :queenside_castle] do
          full_delta_after_move(before, move)
        else
          move_delta(before, board_tuple, mover, move)
        end
    end
  end

  defp full_delta_after_move(before, move) do
    {:ok, after_game} = Echecs.make_move(before, move.from, move.to, move.promotion)
    delta_indices(before, after_game)
  end

  defp move_delta(%Echecs.Game{king_pos: kings, turn: turn}, board_tuple, mover, move) do
    moved_piece = promotion_piece(mover, move.promotion)

    removed =
      piece_feature_indices(kings, mover, move.from) ++
        captured_feature_indices(kings, board_tuple, turn, move)

    added = piece_feature_indices(kings, moved_piece, move.to)

    %{
      added: Enum.sort(added),
      removed: Enum.sort(removed)
    }
  end

  defp promotion_piece({color, type}, nil), do: {color, type}
  defp promotion_piece({color, _type}, promotion), do: {color, promotion}

  defp captured_feature_indices(kings, _board_tuple, turn, %{special: :en_passant, to: to}) do
    capture_square = if turn == :white, do: to + 8, else: to - 8
    piece_feature_indices(kings, opponent_pawn(turn), capture_square)
  end

  defp captured_feature_indices(kings, board_tuple, _turn, %{to: to}) do
    case Echecs.Board.at_tuple(board_tuple, to) do
      nil -> []
      {_color, :king} -> []
      captured -> piece_feature_indices(kings, captured, to)
    end
  end

  defp opponent_pawn(:white), do: {:black, :pawn}
  defp opponent_pawn(:black), do: {:white, :pawn}

  defp piece_feature_indices(_kings, {_color, :king}, _square), do: []

  defp piece_feature_indices({white_king, black_king}, piece, square) do
    piece_idx = piece_index(piece)

    [
      feature_index(white_king, piece_idx, square, 0),
      feature_index(black_king, piece_idx, square, @perspective_size)
    ]
  end

  defp ensure_tuple(board) when is_tuple(board), do: board
  defp ensure_tuple(board), do: Echecs.Board.from_struct(board)

  defp flatten_feature_set(features) do
    features
    |> Map.values()
    |> List.flatten()
    |> MapSet.new()
  end

  defp include_piece?(nil), do: false
  defp include_piece?({_color, :king}), do: false
  defp include_piece?({_color, _type}), do: true

  defp feature_index(king_square, piece_idx, piece_square, perspective_offset) do
    perspective_offset + (king_square * 10 + piece_idx) * 64 + piece_square
  end

  defp piece_index({color, type}) do
    type_offset(type) * 2 + color_offset(color)
  end

  defp type_offset(:pawn), do: 0
  defp type_offset(:knight), do: 1
  defp type_offset(:bishop), do: 2
  defp type_offset(:rook), do: 3
  defp type_offset(:queen), do: 4

  defp color_offset(:white), do: 0
  defp color_offset(:black), do: 1
end
