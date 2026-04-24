defmodule EchecsEngine.Policy do
  @moduledoc """
  Maps legal chess moves onto the 8x8x73 policy head used by the models.
  """

  alias Echecs.Move

  @plane_count 73
  @queen_directions [
    {-1, 0},
    {1, 0},
    {0, 1},
    {0, -1},
    {-1, 1},
    {-1, -1},
    {1, 1},
    {1, -1}
  ]
  @knight_directions [
    {-2, -1},
    {-2, 1},
    {-1, -2},
    {-1, 2},
    {1, -2},
    {1, 2},
    {2, -1},
    {2, 1}
  ]
  @underpromotion_promotions [:knight, :bishop, :rook]

  @spec move_index(Echecs.Game.t(), Move.t()) :: non_neg_integer()
  def move_index(game, %Move{} = move) do
    plane = move_plane(game, move)
    move.from * @plane_count + plane
  end

  @spec legal_move_priors(Echecs.Game.t(), [Move.t()], Nx.Tensor.t()) :: [{Move.t(), float()}]
  def legal_move_priors(game, legal_moves, policy_tensor) do
    logits =
      Enum.map(legal_moves, fn move ->
        index = move_index(game, move)
        {move, policy_tensor[index] |> Nx.to_number()}
      end)

    softmax_pairs(logits)
  end

  defp move_plane(game, %Move{} = move) do
    {from_row, from_col} = div_rem(move.from, 8)
    {to_row, to_col} = div_rem(move.to, 8)
    delta = {to_row - from_row, to_col - from_col}

    case move.promotion do
      promotion when promotion in @underpromotion_promotions ->
        underpromotion_plane(game.turn, delta, promotion)

      _ ->
        sliding_or_knight_plane(delta)
    end
  end

  defp sliding_or_knight_plane({dr, dc} = delta) do
    case queen_plane(delta) do
      {:ok, plane} ->
        plane

      :error ->
        case Enum.find_index(@knight_directions, &(&1 == {dr, dc})) do
          nil -> raise ArgumentError, "unsupported move delta for policy head: #{inspect(delta)}"
          idx -> 56 + idx
        end
    end
  end

  defp queen_plane({dr, dc}) do
    direction =
      Enum.find_index(@queen_directions, fn {dir_r, dir_c} ->
        direction_match?(dr, dc, dir_r, dir_c)
      end)

    case direction do
      nil ->
        :error

      dir_idx ->
        distance = max(abs(dr), abs(dc))

        if distance in 1..7 do
          {:ok, dir_idx * 7 + (distance - 1)}
        else
          :error
        end
    end
  end

  defp direction_match?(dr, dc, dir_r, dir_c) do
    cond do
      {dr, dc} == {0, 0} ->
        false

      dir_r == 0 ->
        dr == 0 and sign(dc) == dir_c

      dir_c == 0 ->
        dc == 0 and sign(dr) == dir_r

      abs(dr) == abs(dc) ->
        sign(dr) == dir_r and sign(dc) == dir_c

      true ->
        false
    end
  end

  defp underpromotion_plane(turn, {dr, dc}, promotion) do
    forward = if turn == :white, do: -1, else: 1

    direction_offset =
      case {dr, dc} do
        {^forward, -1} -> 0
        {^forward, 0} -> 3
        {^forward, 1} -> 6
        other -> raise ArgumentError, "unsupported underpromotion delta: #{inspect(other)}"
      end

    promotion_offset = Enum.find_index(@underpromotion_promotions, &(&1 == promotion))
    64 + direction_offset + promotion_offset
  end

  defp softmax_pairs([]), do: []

  defp softmax_pairs(logits) do
    max_logit = logits |> Enum.map(&elem(&1, 1)) |> Enum.max()

    weights =
      Enum.map(logits, fn {move, logit} ->
        {move, :math.exp(logit - max_logit)}
      end)

    total = Enum.reduce(weights, 0.0, fn {_move, weight}, acc -> acc + weight end)

    Enum.map(weights, fn {move, weight} ->
      {move, weight / total}
    end)
  end

  defp div_rem(value, divisor), do: {div(value, divisor), rem(value, divisor)}

  defp sign(value) when value < 0, do: -1
  defp sign(value) when value > 0, do: 1
  defp sign(_value), do: 0
end
