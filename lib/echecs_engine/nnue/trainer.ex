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
  @default_feature_embedding_size 32

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
    feature_row_width = max(Keyword.get(opts, :feature_row_width, @default_feature_row_width), 1)

    %{
      feature_table: export_feature_table(params, feature_row_width),
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
    |> Stream.map(&Jason.decode!/1)
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

    feature_embedding_size =
      max(Keyword.get(opts, :feature_embedding_size, @default_feature_embedding_size), 1)

    feature_init_scale = Keyword.get(opts, :feature_init_scale, 0.05)
    batch_size = opts |> Keyword.get(:batch_size, 64) |> max(1)

    initial_params = %{
      feature_embeddings: %{},
      projection: initialize_projection(feature_embedding_size),
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
        {updated_params, total_loss, seen_train_count} =
          paths
          |> indexed_examples()
          |> maybe_epoch_order(epoch, opts)
          |> Stream.chunk_every(batch_size)
          |> Enum.reduce({params, 0.0, 0}, fn batch, {params_acc, loss_acc, train_acc} ->
            {_validation_batch, training_batch} =
              Enum.split_with(batch, fn {index, _example} ->
                validation_member?(validation_selector, index)
              end)

            case training_batch do
              [] ->
                {params_acc, loss_acc, train_acc}

              training_batch ->
                {next_params, loss} =
                  train_batch(
                    params_acc,
                    Enum.map(training_batch, &elem(&1, 1)),
                    learning_rate,
                    l2,
                    feature_embedding_size,
                    feature_init_scale
                  )

                {
                  next_params,
                  loss_acc + loss * length(training_batch),
                  train_acc + length(training_batch)
                }
            end
          end)

        mean_loss = total_loss / max(seen_train_count, 1)

        {validation_total_loss, seen_validation_count} =
          validation_stats(
            paths,
            updated_params,
            validation_selector,
            batch_size,
            feature_embedding_size,
            feature_init_scale
          )

        validation_loss = validation_mean_loss(validation_total_loss, seen_validation_count)

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
      algorithm: "minibatch_factorized_sparse_nnue",
      epochs: epochs,
      batch_size: batch_size,
      learning_rate: learning_rate,
      l2: l2,
      corpus_mode: "streaming_sparse_features",
      feature_discovery: "lazy",
      feature_transform: "factorized_projection",
      feature_embedding_size: feature_embedding_size,
      export_row_width: feature_row_width,
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

  defp train_batch(
         params,
         examples,
         learning_rate,
         l2,
         feature_embedding_size,
         feature_init_scale
       ) do
    batch_count = length(examples)

    {grads, loss_sum} =
      batch_gradients(params, examples, feature_embedding_size, feature_init_scale)

    updated_params =
      params
      |> Map.update!(:w2, &apply_gradient(&1, grads.w2, learning_rate, l2))
      |> Map.update!(:b2, &apply_gradient(&1, grads.b2, learning_rate, 0.0))
      |> Map.update!(:w1, &apply_gradient(&1, grads.w1, learning_rate, l2))
      |> Map.update!(:b1, &apply_gradient(&1, grads.b1, learning_rate, 0.0))
      |> Map.update!(:projection, &apply_gradient(&1, grads.projection, learning_rate, l2))
      |> Map.update!(
        :feature_embeddings,
        &update_feature_embeddings(
          &1,
          grads.feature_embeddings,
          learning_rate,
          l2,
          feature_embedding_size,
          feature_init_scale
        )
      )

    {updated_params, loss_sum / max(batch_count, 1)}
  end

  defp batch_gradients(params, examples, feature_embedding_size, feature_init_scale) do
    {outputs, cache} = batch_forward(params, examples, feature_embedding_size, feature_init_scale)
    targets = Nx.tensor(Enum.map(examples, & &1.target), type: :f32)
    errors = Nx.subtract(outputs, targets)
    dy = errors |> Nx.multiply(2.0) |> Nx.divide(length(examples))

    grad_w2 =
      Nx.dot(
        Nx.transpose(cache.hidden),
        Nx.reshape(dy, {length(examples), 1})
      )

    grad_b2 = Nx.sum(Nx.reshape(dy, {length(examples), 1}), axes: [0])

    grad_hidden =
      Nx.dot(
        Nx.reshape(dy, {length(examples), 1}),
        Nx.transpose(params.w2)
      )

    grad_hidden_pre =
      grad_hidden
      |> Nx.multiply(Nx.multiply(2.0, cache.hidden_clipped))
      |> Nx.multiply(clip_derivative(cache.hidden_pre))

    grad_w1 = Nx.dot(Nx.transpose(cache.acc_squared), grad_hidden_pre)
    grad_b1 = Nx.sum(grad_hidden_pre, axes: [0])

    grad_acc =
      Nx.dot(grad_hidden_pre, Nx.transpose(params.w1))
      |> Nx.multiply(Nx.multiply(2.0, cache.acc_clipped))
      |> Nx.multiply(clip_derivative(cache.accumulators))

    grad_projection = Nx.dot(Nx.transpose(cache.summed_embeddings), grad_acc)
    grad_feature_embeddings = Nx.dot(grad_acc, Nx.transpose(params.projection))

    feature_embedding_grads =
      Enum.zip(cache.examples, grad_feature_embeddings |> Nx.to_batched(1) |> Enum.to_list())
      |> Enum.reduce(%{}, fn {%{features: features}, grad_batch}, acc ->
        grad_embedding = Nx.squeeze(grad_batch, axes: [0])

        Enum.reduce(features, acc, fn feature_idx, table_acc ->
          Map.update(table_acc, feature_idx, grad_embedding, &Nx.add(&1, grad_embedding))
        end)
      end)

    {%{
       w1: grad_w1,
       b1: grad_b1,
       w2: grad_w2,
       b2: grad_b2,
       projection: grad_projection,
       feature_embeddings: feature_embedding_grads
     }, Nx.sum(Nx.multiply(errors, errors)) |> Nx.to_number()}
  end

  defp validation_mean_loss(_total_loss, 0), do: nil
  defp validation_mean_loss(total_loss, count), do: total_loss / count

  defp validation_stats(
         paths,
         params,
         validation_selector,
         batch_size,
         feature_embedding_size,
         feature_init_scale
       ) do
    paths
    |> indexed_examples()
    |> Stream.filter(fn {index, _example} -> validation_member?(validation_selector, index) end)
    |> Stream.map(&elem(&1, 1))
    |> Stream.chunk_every(batch_size)
    |> Enum.reduce({0.0, 0}, fn examples, {loss_acc, count_acc} ->
      {loss, count} = batch_loss(examples, params, feature_embedding_size, feature_init_scale)
      {loss_acc + loss, count_acc + count}
    end)
  end

  defp batch_loss(examples, params, feature_embedding_size, feature_init_scale) do
    {outputs, _cache} =
      batch_forward(params, examples, feature_embedding_size, feature_init_scale)

    targets = Nx.tensor(Enum.map(examples, & &1.target), type: :f32)
    errors = Nx.subtract(outputs, targets)

    {
      Nx.sum(Nx.multiply(errors, errors)) |> Nx.to_number(),
      length(examples)
    }
  end

  defp batch_forward(params, examples, feature_embedding_size, feature_init_scale) do
    summed_embeddings =
      examples
      |> Enum.map(fn %{features: features} ->
        summed_embedding(
          params.feature_embeddings,
          features,
          feature_embedding_size,
          feature_init_scale
        )
      end)
      |> stack_embeddings(feature_embedding_size)

    accumulators = Nx.dot(summed_embeddings, params.projection)
    acc_clipped = Nx.clip(accumulators, 0.0, 1.0)
    acc_squared = Nx.multiply(acc_clipped, acc_clipped)
    hidden_pre = Nx.dot(acc_squared, params.w1) |> Nx.add(params.b1)
    hidden_clipped = Nx.clip(hidden_pre, 0.0, 1.0)
    hidden = Nx.multiply(hidden_clipped, hidden_clipped)

    outputs =
      hidden
      |> Nx.dot(params.w2)
      |> Nx.add(params.b2)
      |> Nx.reshape({length(examples)})

    {outputs,
     %{
       examples: examples,
       summed_embeddings: summed_embeddings,
       accumulators: accumulators,
       acc_clipped: acc_clipped,
       acc_squared: acc_squared,
       hidden_pre: hidden_pre,
       hidden_clipped: hidden_clipped,
       hidden: hidden
     }}
  end

  defp update_feature_embeddings(
         feature_embeddings,
         embedding_grads,
         learning_rate,
         l2,
         feature_embedding_size,
         feature_init_scale
       ) do
    Enum.reduce(embedding_grads, feature_embeddings, fn {feature_idx, grad_embedding},
                                                        table_acc ->
      updated_embedding =
        table_acc
        |> feature_embedding(feature_idx, feature_embedding_size, feature_init_scale)
        |> apply_gradient(grad_embedding, learning_rate, l2)

      Map.put(table_acc, feature_idx, updated_embedding)
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

  defp feature_embedding(feature_embeddings, feature_idx, embedding_size, scale) do
    Map.get(feature_embeddings, feature_idx) ||
      initialize_feature_embedding(feature_idx, embedding_size, scale)
  end

  defp initialize_feature_embedding(feature_idx, embedding_size, scale) do
    0..(embedding_size - 1)
    |> Enum.map(fn dim ->
      (:erlang.phash2({:feature_embedding, feature_idx, dim}, 2001) - 1000) / 1000.0 * scale
    end)
    |> Nx.tensor(type: :f32)
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

  defp summed_embedding(feature_embeddings, features, embedding_size, scale) do
    features
    |> Enum.map(&feature_embedding(feature_embeddings, &1, embedding_size, scale))
    |> stack_embeddings(embedding_size)
    |> Nx.sum(axes: [0])
  end

  defp stack_embeddings([], embedding_size), do: Nx.broadcast(0.0, {0, embedding_size})
  defp stack_embeddings(embeddings, _embedding_size), do: Nx.stack(embeddings)

  defp export_feature_table(params, row_width) do
    Enum.into(params.feature_embeddings, %{}, fn {feature_idx, embedding} ->
      contribution =
        embedding
        |> Nx.dot(params.projection)
        |> compress_feature_row(row_width)

      {feature_idx, contribution}
    end)
  end

  defp initialize_projection(feature_embedding_size) do
    0..(feature_embedding_size - 1)
    |> Enum.map(fn row ->
      0..(@accumulator_size - 1)
      |> Enum.map(fn col ->
        (:erlang.phash2({:projection, row, col}, 2001) - 1000) / 1000.0 * 0.01
      end)
    end)
    |> Nx.tensor(type: :f32)
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
