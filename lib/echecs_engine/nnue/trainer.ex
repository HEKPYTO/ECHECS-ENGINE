defmodule EchecsEngine.NNUE.Trainer do
  @moduledoc """
  Builds sparse evaluator artifacts from supervised FEN/eval JSONL records.

  This is a lightweight first-stage trainer: it learns a feature table and
  evaluator weights that are compatible with the production sparse evaluator.
  It is intentionally deterministic so artifacts can be benchmarked and
  reproduced before introducing heavier optimizer-based NNUE training.
  """

  @accumulator_size 3072
  @hidden_size 32
  @default_feature_row_width 32

  @type artifact :: %{
          feature_table: %{non_neg_integer() => Nx.Tensor.t() | map()},
          w1: Nx.Tensor.t(),
          b1: Nx.Tensor.t(),
          w2: Nx.Tensor.t(),
          b2: Nx.Tensor.t(),
          training: map()
        }

  @spec fit_jsonl!(String.t() | [String.t()], keyword()) :: artifact()
  def fit_jsonl!(paths, opts \\ []) do
    paths = normalize_paths(paths)
    validation_selector = validation_selector(paths, opts)

    {params, training} = train_sparse_network(paths, validation_selector, opts)

    %{
      feature_table: params.feature_table,
      w1: params.w1,
      b1: params.b1,
      w2: params.w2,
      b2: params.b2,
      training: training
    }
  end

  @spec fit_and_save!(String.t() | [String.t()], String.t(), keyword()) :: :ok
  def fit_and_save!(paths, output_path, opts \\ []) do
    artifact =
      paths
      |> fit_jsonl!(opts)
      |> maybe_quantize(opts)

    training_config = Keyword.get(opts, :training_config, %{})

    EchecsEngine.Checkpoint.save_evaluator_state!(
      output_path,
      artifact,
      Map.merge(%{"source" => "nnue_trainer"}, training_config)
    )
  end

  defp maybe_quantize(artifact, opts) do
    if Keyword.get(opts, :quantize?, false) do
      EchecsEngine.NNUE.Quantization.quantize_artifact(artifact)
    else
      artifact
    end
  end

  defp normalize_paths(paths) when is_list(paths), do: paths
  defp normalize_paths(path) when is_binary(path), do: [path]

  defp example_stream(paths) do
    paths
    |> Stream.flat_map(&record_stream!/1)
    |> Stream.map(&example_from_record!/1)
  end

  defp record_stream!(path) do
    path
    |> File.stream!()
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
    |> Stream.map(&:json.decode/1)
  end

  defp example_from_record!(%{"fen" => fen} = record) do
    game = Echecs.new_game(fen)

    features =
      game
      |> EchecsEngine.NNUE.Features.active_indices()
      |> Map.values()
      |> List.flatten()
      |> Enum.uniq()

    q =
      record
      |> EchecsEngine.Value.target_from_record(game)
      |> EchecsEngine.Value.wdl_to_q()
      |> side_to_white(game)

    %{features: features, target: q}
  end

  defp example_from_record!(_record), do: raise(ArgumentError, "record is missing fen")

  defp side_to_white(q, %{turn: :white}), do: q
  defp side_to_white(q, %{turn: :black}), do: -q

  defp validation_selector(paths, opts) do
    split = Keyword.get(opts, :validation_split, 0)
    seed = Keyword.get(opts, :seed, 0)

    cond do
      is_float(split) and split > 0.0 ->
        {:hash_ratio, max(min(split, 1.0), 0.0), seed}

      is_integer(split) and split > 0 ->
        {:reservoir, reservoir_validation_indices(paths, split, seed)}

      true ->
        :none
    end
  end

  defp validation_member?(:none, _index), do: false

  defp validation_member?({:hash_ratio, split, seed}, index) do
    :erlang.phash2({seed, index}, 1_000_000) / 1_000_000 < split
  end

  defp validation_member?({:reservoir, validation_indices}, index) do
    MapSet.member?(validation_indices, index)
  end

  defp validation_selection_name(:none), do: "none"
  defp validation_selection_name({:hash_ratio, _split, _seed}), do: "hash_ratio"
  defp validation_selection_name({:reservoir, _validation_indices}), do: "reservoir"

  defp reservoir_validation_indices(paths, requested_count, seed) do
    {_seen, sample} =
      paths
      |> indexed_examples()
      |> Enum.reduce({0, []}, fn {index, _example}, {seen, sample} ->
        cond do
          seen < requested_count ->
            {seen + 1, [index | sample]}

          requested_count == 0 ->
            {seen + 1, sample}

          true ->
            replacement_slot = :erlang.phash2({seed, seen, index}, seen + 1)

            if replacement_slot < requested_count do
              {seen + 1, List.replace_at(sample, replacement_slot, index)}
            else
              {seen + 1, sample}
            end
        end
      end)

    sample
    |> MapSet.new()
  end

  defp train_sparse_network(paths, validation_selector, opts) do
    epochs = opts |> Keyword.get(:epochs, 8) |> max(1)
    learning_rate = Keyword.get(opts, :learning_rate, 0.05)
    l2 = Keyword.get(opts, :l2, 0.0)
    feature_row_width = max(Keyword.get(opts, :feature_row_width, @default_feature_row_width), 1)
    feature_init_scale = Keyword.get(opts, :feature_init_scale, 0.05)

    initial_params = %{
      feature_table: %{},
      w1: initialize_w1(),
      b1: Nx.broadcast(0.0, {@hidden_size}),
      w2: initialize_w2(),
      b2: Nx.broadcast(0.0, {1})
    }

    {params, losses, validation_losses, train_count, validation_count} =
      Enum.reduce(1..epochs, {initial_params, [], [], nil, nil}, fn epoch,
                                                                    {params, losses,
                                                                     validation_losses,
                                                                     train_count,
                                                                     validation_count} ->
        {updated_params, total_loss, seen_train_count, seen_validation_count} =
          paths
          |> indexed_examples()
          |> maybe_epoch_order(epoch, opts)
          |> Enum.reduce({params, 0.0, 0, 0}, fn {index, example},
                                                 {params_acc, loss_acc, train_acc, validation_acc} ->
            if validation_member?(validation_selector, index) do
              {params_acc, loss_acc, train_acc, validation_acc + 1}
            else
              {next_params, loss} =
                train_example(
                  params_acc,
                  example,
                  learning_rate,
                  l2,
                  feature_row_width,
                  feature_init_scale
                )

              {next_params, loss_acc + loss, train_acc + 1, validation_acc}
            end
          end)

        mean_loss = total_loss / max(seen_train_count, 1)

        validation_loss =
          if seen_validation_count == 0 do
            nil
          else
            mean_loss(paths, updated_params, validation_selector)
          end

        {
          updated_params,
          [mean_loss | losses],
          [validation_loss | validation_losses],
          train_count || seen_train_count,
          validation_count || seen_validation_count
        }
      end)

    if (train_count || 0) + (validation_count || 0) == 0 do
      raise ArgumentError, "cannot fit sparse evaluator from an empty dataset"
    end

    training = %{
      algorithm: "online_sgd_sparse_nnue",
      epochs: epochs,
      learning_rate: learning_rate,
      l2: l2,
      corpus_mode: "streaming_sparse_features",
      feature_discovery: "lazy",
      validation_selection: validation_selection_name(validation_selector),
      train_examples: train_count,
      validation_examples: validation_count,
      training_loss: Enum.reverse(losses),
      validation_loss:
        validation_losses
        |> Enum.reverse()
        |> Enum.reject(&is_nil/1)
    }

    {params, training}
  end

  defp indexed_examples(paths) do
    paths
    |> example_stream()
    |> Stream.with_index()
    |> Stream.map(fn {example, index} -> {index, example} end)
  end

  defp maybe_epoch_order(stream, epoch, opts) do
    if Keyword.get(opts, :shuffle?, false) do
      shuffle_stream(
        stream,
        max(Keyword.get(opts, :shuffle_buffer_size, 2048), 1),
        Keyword.get(opts, :seed, 0),
        epoch
      )
    else
      stream
    end
  end

  defp shuffle_stream(stream, buffer_size, seed, epoch) do
    stream
    |> Stream.chunk_every(buffer_size)
    |> Stream.with_index()
    |> Stream.flat_map(fn {chunk, chunk_index} ->
      chunk
      |> Enum.with_index()
      |> Enum.sort_by(fn {_item, item_index} ->
        :erlang.phash2({seed, epoch, chunk_index, item_index})
      end)
      |> Enum.map(&elem(&1, 0))
    end)
  end

  defp train_example(
         params,
         %{features: features, target: target},
         learning_rate,
         l2,
         feature_row_width,
         feature_init_scale
       ) do
    {prediction, cache} = forward(params, features)
    error = prediction - target
    dy = Nx.tensor(2.0 * error, type: :f32)

    grad_w2 =
      Nx.reshape(cache.hidden, {@hidden_size, 1})
      |> Nx.multiply(dy)

    grad_b2 = Nx.reshape(dy, {1})
    grad_hidden = Nx.squeeze(params.w2) |> Nx.multiply(dy)

    grad_hidden_pre =
      grad_hidden
      |> Nx.multiply(Nx.multiply(2.0, cache.hidden_clipped))
      |> Nx.multiply(clip_derivative(cache.hidden_pre))

    grad_w1 =
      Nx.dot(
        Nx.reshape(cache.acc_squared, {@accumulator_size, 1}),
        Nx.reshape(grad_hidden_pre, {1, @hidden_size})
      )

    grad_b1 = grad_hidden_pre

    grad_acc =
      Nx.dot(grad_hidden_pre, Nx.transpose(params.w1))
      |> Nx.multiply(Nx.multiply(2.0, cache.acc_clipped))
      |> Nx.multiply(clip_derivative(cache.accumulator))

    updated_params =
      params
      |> Map.update!(:w2, &apply_gradient(&1, grad_w2, learning_rate, l2))
      |> Map.update!(:b2, &apply_gradient(&1, grad_b2, learning_rate, 0.0))
      |> Map.update!(:w1, &apply_gradient(&1, grad_w1, learning_rate, l2))
      |> Map.update!(:b1, &apply_gradient(&1, grad_b1, learning_rate, 0.0))
      |> Map.update!(
        :feature_table,
        &update_feature_rows(
          &1,
          features,
          grad_acc,
          learning_rate,
          l2,
          feature_row_width,
          feature_init_scale
        )
      )

    {updated_params, error * error}
  end

  defp mean_loss(paths, params, validation_selector) do
    {loss, count} =
      paths
      |> indexed_examples()
      |> Stream.filter(fn {index, _example} -> validation_member?(validation_selector, index) end)
      |> Enum.reduce({0.0, 0}, fn {_index, %{features: features, target: target}}, {acc, count} ->
        error = predict(params, features) - target
        {acc + error * error, count + 1}
      end)

    loss / max(count, 1)
  end

  defp predict(params, features) do
    params
    |> forward(features)
    |> elem(0)
  end

  defp forward(params, features) do
    accumulator = accumulator_from_features(params.feature_table, features)
    acc_clipped = Nx.clip(accumulator, 0.0, 1.0)
    acc_squared = Nx.multiply(acc_clipped, acc_clipped)
    hidden_pre = Nx.dot(acc_squared, params.w1) |> Nx.add(params.b1)
    hidden_clipped = Nx.clip(hidden_pre, 0.0, 1.0)
    hidden = Nx.multiply(hidden_clipped, hidden_clipped)

    output =
      hidden
      |> Nx.dot(params.w2)
      |> Nx.add(params.b2)
      |> Nx.squeeze()
      |> Nx.to_number()

    {output,
     %{
       accumulator: accumulator,
       acc_clipped: acc_clipped,
       acc_squared: acc_squared,
       hidden_pre: hidden_pre,
       hidden_clipped: hidden_clipped,
       hidden: hidden
     }}
  end

  defp accumulator_from_features(feature_table, features) do
    Enum.reduce(features, Nx.broadcast(0.0, {@accumulator_size}), fn feature_idx, acc ->
      case Map.get(feature_table, feature_idx) do
        nil -> acc
        %{indices: indices, values: values} -> add_sparse(acc, indices, values)
        %Nx.Tensor{} = row -> Nx.add(acc, row)
      end
    end)
  end

  defp add_sparse(accumulator, indices, values) do
    Enum.zip(indices, values)
    |> Enum.reduce(accumulator, fn {index, value}, acc ->
      current =
        acc
        |> Nx.slice([index], [1])
        |> Nx.add(Nx.tensor([value], type: :f32))

      Nx.put_slice(acc, [index], current)
    end)
  end

  defp update_feature_rows(
         feature_table,
         features,
         grad_acc,
         learning_rate,
         l2,
         feature_row_width,
         feature_init_scale
       ) do
    Enum.reduce(features, feature_table, fn feature_idx, table_acc ->
      updated_row =
        table_acc
        |> Map.get(feature_idx, feature_row(feature_idx, feature_row_width, feature_init_scale))
        |> dense_feature_row()
        |> apply_gradient(grad_acc, learning_rate, l2)
        |> compress_feature_row(feature_row_width)

      Map.put(table_acc, feature_idx, updated_row)
    end)
  end

  defp apply_gradient(param, grad, learning_rate, l2) do
    regularized =
      if l2 == 0.0 do
        grad
      else
        Nx.add(grad, Nx.multiply(param, l2))
      end

    Nx.subtract(param, Nx.multiply(regularized, learning_rate))
  end

  defp clip_derivative(tensor) do
    tensor
    |> Nx.greater(0.0)
    |> Nx.logical_and(Nx.less(tensor, 1.0))
    |> Nx.as_type(:f32)
  end

  defp feature_row(feature_idx, row_width, scale) do
    slots =
      0..(@accumulator_size - 1)
      |> Enum.map(fn slot ->
        score = :erlang.phash2({:feature_slot, feature_idx, slot}, 1_000_000)
        {score, slot}
      end)
      |> Enum.sort(:desc)
      |> Enum.take(row_width)
      |> Enum.map(&elem(&1, 1))
      |> Enum.sort()

    values =
      Enum.map(slots, fn slot ->
        (:erlang.phash2({:feature_value, feature_idx, slot}, 2001) - 1000) / 1000.0 * scale
      end)

    %{indices: slots, values: values}
  end

  defp dense_feature_row(%Nx.Tensor{} = row), do: row

  defp dense_feature_row(%{indices: indices, values: values}) do
    add_sparse(Nx.broadcast(0.0, {@accumulator_size}), indices, values)
  end

  defp compress_feature_row(%Nx.Tensor{} = row, row_width) do
    entries =
      row
      |> Nx.to_flat_list()
      |> Enum.with_index()
      |> Enum.reject(fn {value, _index} -> value == 0.0 end)
      |> Enum.sort_by(fn {value, _index} -> abs(value) end, :desc)
      |> Enum.take(row_width)
      |> Enum.sort_by(&elem(&1, 1))

    %{
      indices: Enum.map(entries, &elem(&1, 1)),
      values: Enum.map(entries, &elem(&1, 0))
    }
  end

  defp initialize_w1 do
    0..(@accumulator_size - 1)
    |> Enum.map(fn row ->
      0..(@hidden_size - 1)
      |> Enum.map(fn col ->
        (:erlang.phash2({:w1, row, col}, 2001) - 1000) / 1000.0 * 0.05
      end)
    end)
    |> Nx.tensor(type: :f32)
  end

  defp initialize_w2 do
    0..(@hidden_size - 1)
    |> Enum.map(fn row ->
      [(:erlang.phash2({:w2, row}, 2001) - 1000) / 1000.0 * 0.05]
    end)
    |> Nx.tensor(type: :f32)
  end
end
