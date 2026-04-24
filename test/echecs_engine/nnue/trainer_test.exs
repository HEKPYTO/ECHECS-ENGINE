defmodule EchecsEngine.NNUE.TrainerTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.NNUE.Trainer

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "fits a sparse evaluator artifact from supervised JSONL" do
    path = tmp_path("nnue_train")

    File.write!(
      path,
      IO.iodata_to_binary([
        :json.encode(%{
          "fen" => @start_fen,
          "move" => "e2e4",
          "wdl" => [1.0, 0.0, 0.0]
        }),
        "\n"
      ])
    )

    on_exit(fn -> File.rm(path) end)

    artifact = Trainer.fit_jsonl!(path)

    assert is_map(artifact.feature_table)
    assert map_size(artifact.feature_table) > 0
    assert Nx.shape(artifact.w1) == {3072, 32}
    assert Nx.shape(artifact.b1) == {32}
    assert Nx.shape(artifact.w2) == {32, 1}
    assert Nx.shape(artifact.b2) == {1}

    game = Echecs.new_game(@start_fen)

    accumulator =
      EchecsEngine.NNUE.Accumulator.refresh(game, feature_table: artifact.feature_table)

    score =
      accumulator
      |> EchecsEngine.SparseEvaluator.evaluate(artifact.w1, artifact.b1, artifact.w2, artifact.b2)
      |> Nx.squeeze()
      |> Nx.to_number()

    assert score > 0.0
  end

  test "feature table is not capped by accumulator width" do
    path = tmp_path("nnue_train_uncapped")

    File.write!(
      path,
      IO.iodata_to_binary([
        :json.encode(%{
          "fen" => @start_fen,
          "move" => "e2e4",
          "wdl" => [1.0, 0.0, 0.0]
        }),
        "\n"
      ])
    )

    on_exit(fn -> File.rm(path) end)

    artifact = Trainer.fit_jsonl!(path, max_features: 1)

    assert map_size(artifact.feature_table) > 1
  end

  test "trainer emits compact learned feature rows keyed by feature id" do
    path = tmp_path("nnue_train_compact_rows")

    File.write!(
      path,
      IO.iodata_to_binary([
        :json.encode(%{
          "fen" => @start_fen,
          "move" => "e2e4",
          "wdl" => [1.0, 0.0, 0.0]
        }),
        "\n"
      ])
    )

    on_exit(fn -> File.rm(path) end)

    artifact = Trainer.fit_jsonl!(path)

    {_feature_idx, contribution} = artifact.feature_table |> Map.to_list() |> hd()

    assert %{indices: indices, values: values} = contribution
    assert length(indices) == length(values)
    assert length(indices) > 0
    assert length(indices) <= 64
    assert Enum.all?(indices, &is_integer/1)
    refute Enum.all?(values, &(&1 == 0.0))
    assert artifact.training.corpus_mode == "streaming_sparse_features"
    assert artifact.training.feature_discovery == "lazy"
  end

  test "fits evaluator weights with an optimizer and records training loss" do
    path = tmp_path("nnue_train_optimizer")

    File.write!(
      path,
      IO.iodata_to_binary([
        :json.encode(%{
          "fen" => @start_fen,
          "move" => "e2e4",
          "wdl" => [1.0, 0.0, 0.0]
        }),
        "\n"
      ])
    )

    on_exit(fn -> File.rm(path) end)

    artifact = Trainer.fit_jsonl!(path, epochs: 3, learning_rate: 0.5)

    assert artifact.training.algorithm == "online_sgd_sparse_nnue"
    assert artifact.training.epochs == 3
    assert length(artifact.training.training_loss) == 3

    assert List.last(artifact.training.training_loss) <
             List.first(artifact.training.training_loss)
  end

  test "records deterministic validation loss when validation split is configured" do
    path = tmp_path("nnue_train_validation")

    records = [
      %{"fen" => @start_fen, "move" => "e2e4", "wdl" => [1.0, 0.0, 0.0]},
      %{"fen" => @start_fen, "move" => "d2d4", "wdl" => [0.8, 0.2, 0.0]},
      %{"fen" => @start_fen, "move" => "g1f3", "wdl" => [0.4, 0.4, 0.2]},
      %{"fen" => @start_fen, "move" => "c2c4", "wdl" => [0.2, 0.4, 0.4]}
    ]

    File.write!(
      path,
      records
      |> Enum.map(&[:json.encode(&1), "\n"])
      |> IO.iodata_to_binary()
    )

    on_exit(fn -> File.rm(path) end)

    opts = [epochs: 3, learning_rate: 0.25, validation_split: 0.5, shuffle?: true, seed: 17]
    artifact_a = Trainer.fit_jsonl!(path, opts)
    artifact_b = Trainer.fit_jsonl!(path, opts)

    assert artifact_a.training.train_examples > 0
    assert artifact_a.training.validation_examples > 0
    assert artifact_a.training.train_examples + artifact_a.training.validation_examples == 4
    assert artifact_a.training.validation_selection == "hash_ratio"
    assert length(artifact_a.training.validation_loss) == 3
    assert artifact_a.training.training_loss == artifact_b.training.training_loss
    assert artifact_a.training.validation_loss == artifact_b.training.validation_loss
  end

  test "integer validation split uses bounded deterministic sampling metadata" do
    path = tmp_path("nnue_train_integer_validation")

    records = [
      %{"fen" => @start_fen, "move" => "e2e4", "wdl" => [1.0, 0.0, 0.0]},
      %{"fen" => @start_fen, "move" => "d2d4", "wdl" => [0.8, 0.2, 0.0]},
      %{"fen" => @start_fen, "move" => "g1f3", "wdl" => [0.4, 0.4, 0.2]},
      %{"fen" => @start_fen, "move" => "c2c4", "wdl" => [0.2, 0.4, 0.4]}
    ]

    File.write!(
      path,
      records
      |> Enum.map(&[:json.encode(&1), "\n"])
      |> IO.iodata_to_binary()
    )

    on_exit(fn -> File.rm(path) end)

    artifact = Trainer.fit_jsonl!(path, epochs: 2, validation_split: 2, shuffle?: true, seed: 11)

    assert artifact.training.train_examples == 2
    assert artifact.training.validation_examples == 2
    assert artifact.training.validation_selection == "reservoir"
  end

  test "shuffle buffer size does not change deterministic training metrics" do
    path = tmp_path("nnue_train_shuffle_buffer")

    records = [
      %{"fen" => @start_fen, "move" => "e2e4", "wdl" => [1.0, 0.0, 0.0]},
      %{"fen" => @start_fen, "move" => "d2d4", "wdl" => [0.8, 0.2, 0.0]},
      %{"fen" => @start_fen, "move" => "g1f3", "wdl" => [0.4, 0.4, 0.2]},
      %{"fen" => @start_fen, "move" => "c2c4", "wdl" => [0.2, 0.4, 0.4]}
    ]

    File.write!(
      path,
      records
      |> Enum.map(&[:json.encode(&1), "\n"])
      |> IO.iodata_to_binary()
    )

    on_exit(fn -> File.rm(path) end)

    opts = [
      epochs: 3,
      learning_rate: 0.25,
      validation_split: 0.5,
      shuffle?: true,
      seed: 17,
      shuffle_buffer_size: 2
    ]

    artifact_a = Trainer.fit_jsonl!(path, opts)
    artifact_b = Trainer.fit_jsonl!(path, opts)

    assert artifact_a.training.training_loss == artifact_b.training.training_loss
    assert artifact_a.training.validation_loss == artifact_b.training.validation_loss
  end

  test "fits and saves a loadable evaluator artifact" do
    dataset_path = tmp_path("nnue_train_save")
    output_path = tmp_path("nnue_artifact", ".axon")

    File.write!(
      dataset_path,
      IO.iodata_to_binary([
        :json.encode(%{
          "fen" => @start_fen,
          "move" => "e2e4",
          "eval_cp" => 120
        }),
        "\n"
      ])
    )

    on_exit(fn ->
      File.rm(dataset_path)
      File.rm(output_path)
    end)

    :ok = Trainer.fit_and_save!(dataset_path, output_path, training_config: %{"source" => "unit"})

    assert {:ok, %{artifact: artifact, metadata: metadata}} =
             EchecsEngine.Checkpoint.load_evaluator_state(output_path)

    assert metadata.training_config["source"] == "unit"
    assert map_size(artifact.feature_table) > 0
  end

  defp tmp_path(prefix, suffix \\ ".jsonl") do
    System.tmp_dir!()
    |> Path.join("#{prefix}_#{System.unique_integer([:positive])}#{suffix}")
  end
end
