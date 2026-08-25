defmodule EchecsEngine.Eval do
  @moduledoc "The versioned, integer evaluator used by the engine."

  require Echecs.Move

  @magic "ETNN"
  @version 1
  @buckets 16
  @kinds 10
  @squares 64
  @neurons 64
  @hidden 16
  @output_scale 1_024
  @feature_bytes @buckets * @kinds * @squares * @neurons * 2
  @psqt_bytes @buckets * @kinds * @squares * 2
  @w1_bytes 128 * @hidden
  @b1_bytes @hidden * 4
  @w2_bytes @hidden
  @b2_bytes 4

  @type weights :: %{
          feature: binary(),
          psqt: binary(),
          w1: binary(),
          b1: binary(),
          w2: binary(),
          b2: binary(),
          scale: integer()
        }
  @type accumulator :: %{
          white: tuple(),
          black: tuple(),
          white_psqt: integer(),
          black_psqt: integer(),
          psqt: integer(),
          white_bucket: non_neg_integer(),
          black_bucket: non_neg_integer()
        }

  @spec load!(Path.t()) :: weights()
  def load!(path), do: path |> File.read!() |> load_binary!()

  @spec dump!(Path.t(), weights()) :: :ok
  def dump!(path, weights) do
    File.write!(path, dump_binary(weights))
    :ok
  end

  @spec dump_binary(weights()) :: binary()
  def dump_binary(weights) do
    validate_weights!(weights)

    <<@magic::binary, @version::unsigned-16, @buckets::unsigned-16, @kinds::unsigned-16,
      @squares::unsigned-16, @neurons::unsigned-16, 128::unsigned-16, @hidden::unsigned-16,
      1::unsigned-16, weights.scale::signed-32, weights.feature::binary, weights.psqt::binary,
      weights.w1::binary, weights.b1::binary, weights.w2::binary, weights.b2::binary>>
  end

  @spec seed_weights() :: weights()
  def seed_weights do
    %{
      feature: seed_features(),
      psqt: seed_psqt(),
      w1: :binary.copy(<<0::signed-8>>, @w1_bytes),
      b1: :binary.copy(<<0::signed-32>>, @hidden),
      w2: :binary.copy(<<0::signed-8>>, @w2_bytes),
      b2: <<0::signed-32>>,
      scale: @output_scale
    }
  end

  @spec refresh(Echecs.Game.t(), weights()) :: accumulator()
  def refresh(game, weights) do
    {white_bucket, black_bucket} = buckets(game)

    white_psqt = psqt_for_color(game, :white, white_bucket, weights)
    black_psqt = psqt_for_color(game, :black, black_bucket, weights)

    %{
      white: accumulator_for(game, :white, white_bucket, weights),
      black: accumulator_for(game, :black, black_bucket, weights),
      white_psqt: white_psqt,
      black_psqt: black_psqt,
      psqt: white_psqt + black_psqt,
      white_bucket: white_bucket,
      black_bucket: black_bucket
    }
  end

  @spec update(accumulator(), Echecs.Game.t(), integer(), Echecs.Game.t(), weights()) ::
          accumulator()
  def update(accumulator, before, move, after_game, weights) do
    {white_bucket, black_bucket} = buckets(after_game)
    changes = move_changes(before, move, after_game)

    {white, white_refreshed?} =
      if white_bucket == accumulator.white_bucket do
        {apply_changes(accumulator.white, changes, :white, white_bucket, weights), false}
      else
        {accumulator_for(after_game, :white, white_bucket, weights), true}
      end

    {black, black_refreshed?} =
      if black_bucket == accumulator.black_bucket do
        {apply_changes(accumulator.black, changes, :black, black_bucket, weights), false}
      else
        {accumulator_for(after_game, :black, black_bucket, weights), true}
      end

    white_psqt =
      if white_refreshed? do
        psqt_for_color(after_game, :white, white_bucket, weights)
      else
        apply_psqt_changes(
          accumulator.white_psqt,
          changes,
          :white,
          accumulator.white_bucket,
          weights
        )
      end

    black_psqt =
      if black_refreshed? do
        psqt_for_color(after_game, :black, black_bucket, weights)
      else
        apply_psqt_changes(
          accumulator.black_psqt,
          changes,
          :black,
          accumulator.black_bucket,
          weights
        )
      end

    %{
      white: white,
      black: black,
      white_psqt: white_psqt,
      black_psqt: black_psqt,
      psqt: white_psqt + black_psqt,
      white_bucket: white_bucket,
      black_bucket: black_bucket
    }
  end

  @spec evaluate(Echecs.Game.t(), accumulator(), weights()) :: integer()
  def evaluate(game, accumulator, weights) do
    {us, them} =
      if game.turn == :white,
        do: {accumulator.white, accumulator.black},
        else: {accumulator.black, accumulator.white}

    score = accumulator.psqt + dense_score(us, them, weights)
    if game.turn == :white, do: score, else: -score
  end

  @doc false
  @spec training_features(Echecs.Game.t()) :: %{
          white_rows: [non_neg_integer()],
          black_rows: [non_neg_integer()],
          feature_entries: [non_neg_integer()],
          psqt_entries: [non_neg_integer()]
        }
  def training_features(game) do
    {white_bucket, black_bucket} = buckets(game)

    {white_rows, black_rows, psqt_entries} =
      Enum.reduce(0..63, {[], [], []}, fn square, {white_rows, black_rows, psqt_entries} ->
        case training_piece_rows(
               Echecs.Board.at_tuple(game.board, square),
               square,
               white_bucket,
               black_bucket
             ) do
          {white_row, black_row, psqt_entry} ->
            {[white_row | white_rows], [black_row | black_rows], [psqt_entry | psqt_entries]}

          nil ->
            {white_rows, black_rows, psqt_entries}
        end
      end)

    white_rows = Enum.reverse(white_rows)
    black_rows = Enum.reverse(black_rows)

    %{
      white_rows: white_rows,
      black_rows: black_rows,
      feature_entries:
        Enum.flat_map(white_rows ++ black_rows, fn row ->
          for neuron <- 0..(@neurons - 1), do: row * @neurons + neuron
        end),
      psqt_entries: Enum.reverse(psqt_entries)
    }
  end

  defp load_binary!(
         <<"ETNN", @version::unsigned-16, @buckets::unsigned-16, @kinds::unsigned-16,
           @squares::unsigned-16, @neurons::unsigned-16, 128::unsigned-16, @hidden::unsigned-16,
           1::unsigned-16, scale::signed-32, feature::binary-size(@feature_bytes),
           psqt::binary-size(@psqt_bytes), w1::binary-size(@w1_bytes), b1::binary-size(@b1_bytes),
           w2::binary-size(@w2_bytes), b2::binary-size(@b2_bytes)>>
       ) do
    weights = %{feature: feature, psqt: psqt, w1: w1, b1: b1, w2: w2, b2: b2, scale: scale}
    validate_weights!(weights)
    weights
  end

  defp load_binary!(_), do: raise(ArgumentError, "unsupported evaluator artifact")

  defp validate_weights!(weights) do
    sizes = [
      {:feature, @feature_bytes},
      {:psqt, @psqt_bytes},
      {:w1, @w1_bytes},
      {:b1, @b1_bytes},
      {:w2, @w2_bytes},
      {:b2, @b2_bytes}
    ]

    unless is_map(weights) and is_integer(weights[:scale]) and weights[:scale] > 0 and
             Enum.all?(sizes, fn {key, bytes} ->
               is_binary(weights[key]) and byte_size(weights[key]) == bytes
             end) do
      raise ArgumentError, "invalid evaluator weights"
    end
  end

  defp seed_psqt do
    for bucket <- 0..(@buckets - 1), kind <- 0..(@kinds - 1), square <- 0..63, into: <<>> do
      <<psqt_seed(bucket, kind, square)::signed-16>>
    end
  end

  defp seed_features do
    for bucket <- 0..(@buckets - 1),
        kind <- 0..(@kinds - 1),
        square <- 0..63,
        neuron <- 0..(@neurons - 1),
        into: <<>> do
      <<1 + rem(bucket * 3 + kind * 5 + square + neuron, 4)::signed-16>>
    end
  end

  defp psqt_seed(_bucket, kind, square) do
    material = [100, 320, 330, 500, 900] |> Enum.at(rem(kind, 5))
    color = if kind < 5, do: 1, else: -1
    rank = div(square, 8)
    advance = if color == 1, do: rank, else: 7 - rank
    color * (material + if(rem(kind, 5) == 0, do: advance * 2, else: 0))
  end

  defp psqt_for_color(game, color, bucket, weights) do
    Enum.reduce(0..63, 0, fn square, score ->
      case Echecs.Board.at_tuple(game.board, square) do
        {^color, type} when type != :king ->
          score + psqt_at(weights.psqt, bucket, kind(color, type), square)

        _ ->
          score
      end
    end)
  end

  defp psqt_at(psqt, bucket, kind, square) do
    offset = psqt_index(bucket, kind, square) * 2
    <<value::signed-16>> = binary_part(psqt, offset, 2)
    value
  end

  defp psqt_index(bucket, kind, square),
    do: bucket * @kinds * @squares + kind * @squares + square

  defp training_piece_rows({color, type}, square, white_bucket, black_bucket)
       when type != :king do
    white_row = feature_row(white_bucket, kind(color, type), square)
    black_row = feature_row(black_bucket, kind(opposite(color), type), 63 - square)
    psqt_bucket = if color == :white, do: white_bucket, else: black_bucket
    {white_row, black_row, psqt_index(psqt_bucket, kind(color, type), square)}
  end

  defp training_piece_rows(_, _, _, _), do: nil

  defp king_bucket(square, :white), do: div(div(square, 8), 2) * 4 + div(rem(square, 8), 2)
  defp king_bucket(square, :black), do: king_bucket(63 - square, :white)

  defp buckets(game),
    do: {king_bucket(elem(game.king_pos, 0), :white), king_bucket(elem(game.king_pos, 1), :black)}

  defp accumulator_for(game, perspective, bucket, weights) do
    Enum.reduce(0..63, zero_accumulator(), fn square, accumulator ->
      case Echecs.Board.at_tuple(game.board, square) do
        {color, type} when type != :king ->
          add_feature(accumulator, perspective, bucket, color, type, square, 1, weights)

        _ ->
          accumulator
      end
    end)
  end

  defp move_changes(before, move, after_game) do
    from = Echecs.Move.unpack_from(move)
    to = Echecs.Move.unpack_to(move)
    promotion = Echecs.Move.unpack_promotion(move)

    case Echecs.Move.unpack_special(move) do
      :en_passant ->
        captured = to + if(before.turn == :white, do: 8, else: -8)

        [remove(before, from), remove(before, captured), add(after_game, to)]
        |> Enum.reject(&is_nil/1)

      :kingside_castle ->
        [
          remove(before, rook_from(before.turn, :kingside)),
          add(after_game, rook_to(before.turn, :kingside))
        ]
        |> Enum.reject(&is_nil/1)

      :queenside_castle ->
        [
          remove(before, rook_from(before.turn, :queenside)),
          add(after_game, rook_to(before.turn, :queenside))
        ]
        |> Enum.reject(&is_nil/1)

      _ ->
        [remove(before, from), remove(before, to), add(after_game, to, promotion)]
        |> Enum.reject(&is_nil/1)
    end
  end

  defp rook_from(:white, :kingside), do: 63
  defp rook_from(:white, :queenside), do: 56
  defp rook_from(:black, :kingside), do: 7
  defp rook_from(:black, :queenside), do: 0
  defp rook_to(:white, :kingside), do: 61
  defp rook_to(:white, :queenside), do: 59
  defp rook_to(:black, :kingside), do: 5
  defp rook_to(:black, :queenside), do: 3

  defp remove(game, square) do
    case Echecs.Board.at_tuple(game.board, square) do
      {color, type} when type != :king -> {-1, color, type, square}
      _ -> nil
    end
  end

  defp add(game, square, promotion \\ nil) do
    case Echecs.Board.at_tuple(game.board, square) do
      {color, type} when type != :king -> {1, color, promotion || type, square}
      _ -> nil
    end
  end

  defp apply_changes(accumulator, changes, perspective, bucket, weights) do
    Enum.reduce(changes, accumulator, fn {sign, color, type, square}, current ->
      add_feature(current, perspective, bucket, color, type, square, sign, weights)
    end)
  end

  defp apply_psqt_changes(score, changes, owner, bucket, weights) do
    Enum.reduce(changes, score, fn {sign, color, type, square}, total ->
      if color == owner,
        do: total + sign * psqt_at(weights.psqt, bucket, kind(color, type), square),
        else: total
    end)
  end

  defp zero_accumulator, do: List.duplicate(0, @neurons) |> List.to_tuple()

  defp add_feature(accumulator, perspective, bucket, color, type, square, sign, weights) do
    {kind, square} = normalize_feature(perspective, color, type, square)
    row = feature_offset(bucket, kind, square)

    0..(@neurons - 1)
    |> Enum.reduce(accumulator, fn neuron, current ->
      <<value::signed-16>> = binary_part(weights.feature, row + neuron * 2, 2)
      put_elem(current, neuron, elem(current, neuron) + sign * value)
    end)
  end

  defp normalize_feature(:white, color, type, square), do: {kind(color, type), square}

  defp normalize_feature(:black, color, type, square),
    do: {kind(opposite(color), type), 63 - square}

  defp opposite(:white), do: :black
  defp opposite(:black), do: :white

  defp feature_offset(bucket, kind, square),
    do: feature_row(bucket, kind, square) * @neurons * 2

  defp feature_row(bucket, kind, square),
    do: bucket * @kinds * @squares + kind * @squares + square

  defp dense_score(us, them, weights) do
    inputs = Tuple.to_list(us) ++ Tuple.to_list(them)

    hidden =
      for neuron <- 0..(@hidden - 1) do
        bias = signed32(weights.b1, neuron * 4)

        total =
          Enum.with_index(inputs)
          |> Enum.reduce(bias, fn {value, input}, sum ->
            sum + transformed(value) * signed8(weights.w1, input * @hidden + neuron)
          end)

        clip(div(total, 64), 0, 127)
      end

    output =
      signed32(weights.b2, 0) +
        (Enum.with_index(hidden)
         |> Enum.reduce(0, fn {value, neuron}, sum ->
           sum + value * signed8(weights.w2, neuron)
         end))

    div(output, weights.scale)
  end

  defp transformed(value) do
    clipped = clip(div(value, 32), 0, 127)
    clipped + div(clipped * clipped, 127)
  end

  defp clip(value, low, high), do: min(max(value, low), high)

  defp signed8(binary, offset),
    do: binary_part(binary, offset, 1) |> then(fn <<value::signed-8>> -> value end)

  defp signed32(binary, offset),
    do: binary_part(binary, offset, 4) |> then(fn <<value::signed-32>> -> value end)

  defp kind(:white, :pawn), do: 0
  defp kind(:white, :knight), do: 1
  defp kind(:white, :bishop), do: 2
  defp kind(:white, :rook), do: 3
  defp kind(:white, :queen), do: 4
  defp kind(:black, :pawn), do: 5
  defp kind(:black, :knight), do: 6
  defp kind(:black, :bishop), do: 7
  defp kind(:black, :rook), do: 8
  defp kind(:black, :queen), do: 9
end
