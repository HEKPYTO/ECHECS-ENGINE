# credo:disable-for-this-file Credo.Check.Warning.RaiseInsideRescue
defmodule EchecsEngine.Trainer do
  @moduledoc """
  Streaming, deterministic SGD for the integer NNUE evaluator.

  Trains `EchecsEngine.Eval` weights from labeled JSONL without ever
  loading the dataset into memory. Each row must contain `fen` and exactly
  one label: `eval_cp` (centipawns, side-to-move), `wdl` (`[win,draw,loss]`
  triplet, `sum==1`, mapped to `(win-loss)*1000`), or `result` (`"1-0"`,
  `"0-1"`, `"1/2-1/2"` as White result, flipped to side-to-move). Rejects
  ambiguous or missing labels.

  ## Determinism

  Partitioning into training/validation uses `:erlang.phash2({seed, FEN}, 10_000)`
  so the split is stable across runs. Epochs average gradients over
  `training_rows` before applying integer-clipped updates. `validation_fraction`
  defaults to `0.2`; set to `0` for no validation.

  Loss is mean squared error over streamed rows; the returned map reports
  `initial_training_loss` / `training_loss` and `initial_validation_loss` /
  `validation_loss` plus `active_updates` and `changed_tensors`.

  ## Scope

  Only active feature rows and the dense tail (`w1/b1/w2` + `psqt`/`b2`) are
  touched; inactive rows stay at their seed. Updates are `±1..64` clipped
  and accumulated per-index then averaged, so training is a bounded integer
  walk, not a float optimizer. Not a strength or Stockfish-parity claim.

  ## Example

      EchecsEngine.Trainer.train!("data/train.jsonl", "priv/echecs.nnue",
        epochs: 1, learning_rate: 0.05, validation_fraction: 0.2, seed: 0)
  """

  alias EchecsEngine.Eval

  @allowed [:epochs, :learning_rate, :validation_fraction, :seed]
  @neurons 64
  @hidden 16

  @doc """
  Trains a `ETNN` v1 evaluator from `input` JSONL to `output` path.

  Options: `:epochs` (pos integer, default `1`), `:learning_rate` (pos float,
  default `0.05`), `:validation_fraction` (`0 <= f < 1`, default `0.2`),
  `:seed` (integer, default `0`). Raises `ArgumentError` on empty inputs,
  empty partitions, or unknown options.

  Returns `%{rows, training_rows, validation_rows, initial_training_loss,
  training_loss, initial_validation_loss, validation_loss, changed_tensors,
  active_updates, batches, output}`.
  """
  @spec train!(Path.t(), Path.t(), keyword()) :: map()
  def train!(input, output, opts \\ []) do
    validate_opts!(opts)
    epochs = Keyword.get(opts, :epochs, 1)
    rate = Keyword.get(opts, :learning_rate, 0.05)
    fraction = Keyword.get(opts, :validation_fraction, 0.2)
    seed = Keyword.get(opts, :seed, 0)
    validate_numbers!(epochs, rate, fraction, seed)

    {rows, training_rows, validation_rows} = count_partitions!(input, fraction, seed)
    if rows == 0, do: raise(ArgumentError, "training data is empty")
    if training_rows == 0, do: raise(ArgumentError, "training partition is empty")

    if fraction > 0 and validation_rows == 0,
      do: raise(ArgumentError, "validation partition is empty")

    before = Eval.seed_weights()
    initial = bootstrap_dense(before)
    initial_training_loss = loss(input, initial, fraction, seed, :training)
    initial_validation_loss = loss(input, initial, fraction, seed, :validation)

    {weights, active_updates} =
      Enum.reduce(1..epochs, {initial, 0}, fn _epoch, {current, updates} ->
        {gradients, epoch_updates} = train_epoch(input, current, rate, fraction, seed)
        {apply_gradients(current, gradients), updates + epoch_updates}
      end)

    training_loss = loss(input, weights, fraction, seed, :training)
    validation_loss = loss(input, weights, fraction, seed, :validation)
    :ok = Eval.dump!(output, weights)

    %{
      rows: rows,
      training_rows: training_rows,
      validation_rows: validation_rows,
      initial_training_loss: initial_training_loss,
      training_loss: training_loss,
      initial_validation_loss: initial_validation_loss,
      validation_loss: validation_loss,
      changed_tensors: changed_tensors(before, weights),
      active_updates: active_updates,
      batches: epochs,
      output: output
    }
  end

  @doc false
  @spec target_for_row!(map(), Echecs.Game.t(), pos_integer()) :: number()
  def target_for_row!(row, game, number) do
    labels = Enum.filter(["eval_cp", "wdl", "result"], &Map.has_key?(row, &1))

    case labels do
      ["eval_cp"] -> eval_target!(row["eval_cp"], number)
      ["wdl"] -> wdl_target!(row["wdl"], number)
      ["result"] -> result_target!(row["result"], game, number)
      [] -> raise ArgumentError, "line #{number}: missing eval_cp, wdl, or result"
      _ -> raise ArgumentError, "line #{number}: ambiguous label"
    end
  end

  defp train_epoch(input, weights, rate, fraction, seed) do
    {gradients, rows} =
      input
      |> File.stream!()
      |> Stream.with_index(1)
      |> Enum.reduce({empty_gradients(), 0}, fn {line, number}, {gradients, rows} ->
        %{game: game, target: target, partition_key: key} = row!(line, number)

        if validation?(key, fraction, seed) do
          {gradients, rows}
        else
          {add_row_gradients(gradients, game, target, weights, rate), rows + 1}
        end
      end)

    {average_gradients(gradients, rows), active_updates(gradients, rows)}
  end

  defp loss(input, weights, fraction, seed, partition) do
    {sum, count} =
      input
      |> File.stream!()
      |> Stream.with_index(1)
      |> Enum.reduce({0.0, 0}, fn {line, number}, {sum, count} ->
        %{game: game, target: target, partition_key: key} = row!(line, number)

        if in_partition?(key, fraction, seed, partition) do
          prediction = Eval.evaluate(game, Eval.refresh(game, weights), weights)
          {sum + :math.pow(target - prediction, 2), count + 1}
        else
          {sum, count}
        end
      end)

    if count == 0, do: 0.0, else: sum / count
  end

  defp add_row_gradients(gradients, game, target, weights, rate) do
    %{raw_score: raw_score, transformed: transformed, hidden: hidden, rows: rows} =
      forward(game, weights)

    raw_target = if game.turn == :white, do: target, else: -target
    step = clip(round((raw_target - raw_score) * rate), -64, 64)

    if step == 0 do
      gradients
    else
      gradients
      |> add(:b2, 0, step * weights.scale)
      |> add_psqt(rows.psqt_entries, step)
      |> add_dense(transformed, hidden, weights, step)
      |> add_features(rows, weights, step)
    end
  end

  defp forward(game, weights) do
    accumulator = Eval.refresh(game, weights)
    features = Eval.training_features(game)

    {own, opponent, own_rows, opponent_rows} =
      if game.turn == :white do
        {accumulator.white, accumulator.black, features.white_rows, features.black_rows}
      else
        {accumulator.black, accumulator.white, features.black_rows, features.white_rows}
      end

    transformed = (Tuple.to_list(own) ++ Tuple.to_list(opponent)) |> Enum.map(&transformed/1)

    hidden =
      for neuron <- 0..(@hidden - 1) do
        total =
          Enum.with_index(transformed)
          |> Enum.reduce(signed32(weights.b1, neuron), fn {value, input}, sum ->
            sum + value * signed8(weights.w1, input * @hidden + neuron)
          end)

        clip(div(total, 64), 0, 127)
      end

    dense =
      signed32(weights.b2, 0) +
        (hidden
         |> Enum.with_index()
         |> Enum.reduce(0, fn {value, neuron}, sum ->
           sum + value * signed8(weights.w2, neuron)
         end))

    %{
      raw_score: accumulator.psqt + div(dense, weights.scale),
      transformed: transformed,
      hidden: hidden,
      rows: %{
        own_rows: own_rows,
        opponent_rows: opponent_rows,
        psqt_entries: features.psqt_entries
      }
    }
  end

  defp add_dense(gradients, transformed, hidden, weights, step) do
    Enum.reduce(0..(@hidden - 1), gradients, fn neuron, current ->
      output_weight = signed8(weights.w2, neuron)
      hidden_step = sign(step * output_weight)

      current =
        current
        |> add(:w2, neuron, sign(step) * max(div(Enum.at(hidden, neuron), 16), 1))
        |> add(:b1, neuron, hidden_step * 64)

      add_w1_gradients(current, transformed, neuron, hidden_step)
    end)
  end

  defp add_w1_gradients(gradients, transformed, neuron, hidden_step) do
    Enum.with_index(transformed)
    |> Enum.reduce(gradients, fn {value, input}, matrix ->
      if value == 0 or hidden_step == 0,
        do: matrix,
        else: add(matrix, :w1, input * @hidden + neuron, hidden_step * sign(value))
    end)
  end

  defp add_features(gradients, rows, weights, step) do
    gradients
    |> add_feature_rows(rows.own_rows, 0, weights, step)
    |> add_feature_rows(rows.opponent_rows, @neurons, weights, step)
  end

  defp add_feature_rows(gradients, rows, input_offset, weights, step) do
    Enum.reduce(rows, gradients, &add_feature_row(&2, &1, input_offset, weights, step))
  end

  defp add_feature_row(gradients, row, input_offset, weights, step) do
    Enum.reduce(0..(@neurons - 1), gradients, fn neuron, current ->
      input = input_offset + neuron
      add(current, :feature, row * @neurons + neuron, sign(step * feature_signal(weights, input)))
    end)
  end

  defp feature_signal(weights, input) do
    Enum.reduce(0..(@hidden - 1), 0, fn hidden, sum ->
      sum + signed8(weights.w1, input * @hidden + hidden) * signed8(weights.w2, hidden)
    end)
  end

  defp add_psqt(gradients, entries, step),
    do: Enum.reduce(entries, gradients, &add(&2, :psqt, &1, step))

  defp add(gradients, _tensor, _index, 0), do: gradients

  defp add(gradients, tensor, index, delta),
    do:
      Map.update!(gradients, tensor, &Map.update(&1, index, delta, fn value -> value + delta end))

  defp empty_gradients,
    do: %{feature: %{}, psqt: %{}, w1: %{}, b1: %{}, w2: %{}, b2: %{}}

  defp active_updates(gradients, _rows),
    do: Enum.reduce(gradients, 0, fn {_tensor, updates}, total -> total + map_size(updates) end)

  defp average_gradients(gradients, 0), do: gradients

  defp average_gradients(gradients, rows) do
    Map.new(gradients, fn {tensor, updates} ->
      {tensor,
       Map.new(updates, fn {index, delta} ->
         {index, average_delta(delta, rows)}
       end)}
    end)
  end

  defp average_delta(0, _rows), do: 0

  defp average_delta(delta, rows) do
    average = div(delta, rows)
    if average == 0, do: sign(delta), else: average
  end

  defp apply_gradients(weights, gradients) do
    %{
      weights
      | feature: apply16(weights.feature, gradients.feature),
        psqt: apply16(weights.psqt, gradients.psqt),
        w1: apply8(weights.w1, gradients.w1),
        b1: apply32(weights.b1, gradients.b1),
        w2: apply8(weights.w2, gradients.w2),
        b2: apply32(weights.b2, gradients.b2)
    }
  end

  defp apply8(binary, updates), do: apply_entries(binary, 1, updates, -128, 127)
  defp apply16(binary, updates), do: apply_entries(binary, 2, updates, -32_768, 32_767)

  defp apply32(binary, updates),
    do: apply_entries(binary, 4, updates, -2_000_000_000, 2_000_000_000)

  defp apply_entries(binary, width, updates, low, high) do
    for index <- 0..(div(byte_size(binary), width) - 1), into: <<>> do
      value = signed(binary, index, width)
      encode(clip(value + Map.get(updates, index, 0), low, high), width)
    end
  end

  defp encode(value, 1), do: <<value::signed-8>>
  defp encode(value, 2), do: <<value::signed-16>>
  defp encode(value, 4), do: <<value::signed-32>>
  defp signed(binary, index, 1), do: signed8(binary, index)
  defp signed(binary, index, 2), do: signed16(binary, index)
  defp signed(binary, index, 4), do: signed32(binary, index)

  defp bootstrap_dense(weights) do
    %{
      weights
      | w1: :binary.copy(<<1::signed-8>>, byte_size(weights.w1)),
        b1: :binary.copy(<<128::signed-32>>, div(byte_size(weights.b1), 4)),
        w2: :binary.copy(<<2::signed-8>>, byte_size(weights.w2))
    }
  end

  defp changed_tensors(before, after_weights) do
    [:feature, :psqt, :w1, :b1, :w2, :b2]
    |> Enum.filter(&(Map.fetch!(before, &1) != Map.fetch!(after_weights, &1)))
  end

  defp count_partitions!(input, fraction, seed) do
    input
    |> File.stream!()
    |> Stream.with_index(1)
    |> Enum.reduce({0, 0, 0}, fn {line, number}, {rows, training, validation} ->
      %{partition_key: key} = row!(line, number)

      if validation?(key, fraction, seed),
        do: {rows + 1, training, validation + 1},
        else: {rows + 1, training + 1, validation}
    end)
  end

  defp row!(line, number) do
    case Jason.decode(String.trim(line)) do
      {:ok, %{"fen" => fen} = row} when is_binary(fen) ->
        game =
          try do
            Echecs.new_game(fen)
          rescue
            _ -> raise ArgumentError, "line #{number}: invalid fen"
          end

        %{
          game: game,
          target: target_for_row!(row, game, number),
          partition_key: Echecs.FEN.to_string(game)
        }

      {:ok, _} ->
        raise ArgumentError, "line #{number}: expected fen and label"

      {:error, _} ->
        raise ArgumentError, "line #{number}: invalid JSON"
    end
  end

  defp eval_target!(value, _) when is_number(value), do: value
  defp eval_target!(_, number), do: raise(ArgumentError, "line #{number}: invalid eval_cp label")

  defp wdl_target!([win, _draw, loss] = wdl, number) do
    if normalized_wdl?(wdl),
      do: (win - loss) * 1_000,
      else: raise(ArgumentError, "line #{number}: invalid wdl label")
  end

  defp wdl_target!(_, number), do: raise(ArgumentError, "line #{number}: invalid wdl label")

  defp normalized_wdl?([win, draw, loss]) do
    Enum.all?([win, draw, loss], &(is_number(&1) and &1 >= 0 and &1 <= 1)) and
      abs(win + draw + loss - 1.0) <= 1.0e-6
  end

  defp result_target!(result, game, number) do
    white_target =
      case result do
        "1-0" -> 1_000
        "0-1" -> -1_000
        "1/2-1/2" -> 0
        _ -> raise ArgumentError, "line #{number}: invalid result label"
      end

    if game.turn == :white, do: white_target, else: -white_target
  end

  defp in_partition?(key, fraction, seed, :validation), do: validation?(key, fraction, seed)

  defp in_partition?(key, fraction, seed, :training), do: not validation?(key, fraction, seed)

  defp validation?(key, fraction, seed),
    do: :erlang.phash2({seed, key}, 10_000) < round(fraction * 10_000)

  defp transformed(value) do
    clipped = clip(div(value, 32), 0, 127)
    clipped + div(clipped * clipped, 127)
  end

  defp signed8(binary, index), do: binary_part(binary, index, 1) |> decode8()
  defp signed16(binary, index), do: binary_part(binary, index * 2, 2) |> decode16()
  defp signed32(binary, index), do: binary_part(binary, index * 4, 4) |> decode32()
  defp decode8(<<value::signed-8>>), do: value
  defp decode16(<<value::signed-16>>), do: value
  defp decode32(<<value::signed-32>>), do: value

  defp sign(number) when number < 0, do: -1
  defp sign(number) when number > 0, do: 1
  defp sign(_), do: 0
  defp clip(value, low, high), do: min(max(value, low), high)

  defp validate_opts!(opts) when is_list(opts) do
    case Enum.find(opts, fn {key, _} -> key not in @allowed end) do
      nil -> :ok
      {key, _} -> raise ArgumentError, "unknown option #{inspect(key)}"
    end
  end

  defp validate_opts!(_), do: raise(ArgumentError, "options must be a keyword list")

  defp validate_numbers!(epochs, rate, fraction, seed)
       when is_integer(epochs) and epochs > 0 and is_number(rate) and rate > 0 and
              is_number(fraction) and fraction >= 0 and fraction < 1 and is_integer(seed),
       do: :ok

  defp validate_numbers!(_, _, _, _), do: raise(ArgumentError, "invalid training options")
end
