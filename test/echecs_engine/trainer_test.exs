defmodule EchecsEngine.TrainerTest do
  use ExUnit.Case, async: false

  alias EchecsEngine.{Eval, Trainer}

  test "rejects malformed labels with their line number" do
    data = Path.join(System.tmp_dir!(), "bad-train-#{System.unique_integer([:positive])}.jsonl")
    File.write!(data, ~s({"fen":"8/8/8/8/8/8/8/K6k w - - 0 1}) <> "\n")
    assert_raise ArgumentError, ~r/line 1/, fn -> Trainer.train!(data, temp_artifact(), []) end
  end

  test "normalizes result labels from White outcome to side-to-move targets" do
    row = %{"result" => "1-0"}
    white = Echecs.new_game("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    black = Echecs.new_game("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")

    assert Trainer.target_for_row!(row, white, 1) == 1_000
    assert Trainer.target_for_row!(row, black, 1) == -1_000
  end

  test "rejects ambiguous labels with their line number" do
    data =
      Path.join(System.tmp_dir!(), "ambiguous-train-#{System.unique_integer([:positive])}.jsonl")

    File.write!(
      data,
      ~s({"fen":"4k3/8/8/8/8/8/8/3QK3 w - - 0 1","eval_cp":900,"result":"1-0"}) <> "\n"
    )

    assert_raise ArgumentError, ~r/line 1: ambiguous label/, fn ->
      Trainer.train!(data, temp_artifact(), [])
    end
  end

  test "writes a current evaluator artifact from labeled JSONL" do
    data = Path.join(System.tmp_dir!(), "train-#{System.unique_integer([:positive])}.jsonl")
    File.write!(data, ~s({"fen":"4k3/8/8/8/8/8/8/3QK3 w - - 0 1","eval_cp":700}) <> "\n")
    output = temp_artifact()

    assert %{rows: 1, validation_loss: loss, changed_tensors: tensors} =
             Trainer.train!(data, output, epochs: 1, validation_fraction: 0.0, seed: 1)

    assert is_number(loss)
    assert tensors == [:feature, :psqt, :w1, :b1, :w2, :b2]
    assert is_map(Eval.load!(output))
  end

  test "updates only active sparse rows and reports tensors that actually changed" do
    fen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    game = Echecs.new_game(fen)

    data =
      Path.join(System.tmp_dir!(), "active-train-#{System.unique_integer([:positive])}.jsonl")

    output = temp_artifact()
    File.write!(data, ~s({"fen":"#{fen}","eval_cp":700}) <> "\n")

    before = Eval.seed_weights()

    assert %{changed_tensors: tensors, active_updates: active_updates} =
             Trainer.train!(data, output, epochs: 1, validation_fraction: 0.0, seed: 0)

    after_weights = Eval.load!(output)
    active = Eval.training_features(game)

    assert active_updates > 0
    assert tensors == changed_tensors(before, after_weights)
    assert :feature in tensors and :psqt in tensors

    assert changed_entries(before.feature, after_weights.feature, 2) != []

    assert changed_entries(before.feature, after_weights.feature, 2) -- active.feature_entries ==
             []

    assert changed_entries(before.psqt, after_weights.psqt, 2) != []
    assert changed_entries(before.psqt, after_weights.psqt, 2) -- active.psqt_entries == []
  end

  test "rejects duplicate-only data when its content hash leaves validation empty" do
    data = Path.join(System.tmp_dir!(), "split-train-#{System.unique_integer([:positive])}.jsonl")

    rows =
      Enum.map_join(1..20, "\n", fn n ->
        ~s({"fen":"4k3/8/8/8/8/8/8/3QK3 w - - 0 1","eval_cp":#{n}})
      end)

    File.write!(data, rows <> "\n")

    assert_raise ArgumentError, "validation partition is empty", fn ->
      Trainer.train!(data, temp_artifact(), validation_fraction: 0.5, seed: 1)
    end
  end

  test "accepts only normalized WDL probability triplets" do
    game = Echecs.new_game("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert Trainer.target_for_row!(%{"wdl" => [0.7, 0.2, 0.1]}, game, 1) == 600.0
    assert Trainer.target_for_row!(%{"wdl" => [0.0, 1.0, 0.0]}, game, 1) == 0.0

    for label <- [1.0, [1.1, 0.0, 0.0], [0.5, 0.5, 0.5]] do
      assert_raise ArgumentError, ~r/line 2: invalid wdl label/, fn ->
        Trainer.target_for_row!(%{"wdl" => label}, game, 2)
      end
    end
  end

  test "material corpus improves deterministic holdout loss" do
    data =
      Path.join(System.tmp_dir!(), "material-train-#{System.unique_integer([:positive])}.jsonl")

    rows = [
      ~s({"fen":"4k3/8/8/8/8/8/8/3QK3 w - - 0 1","eval_cp":100}),
      ~s({"fen":"4k3/8/8/8/8/8/8/3QK3 b - - 0 1","result":"0-1"}),
      ~s({"fen":"4k3/8/8/8/8/8/8/3qK3 w - - 0 1","wdl":[0.1,0.2,0.7]}),
      ~s({"fen":"4k3/8/8/8/8/8/8/3qK3 b - - 0 1","wdl":[0.7,0.2,0.1]}),
      ~s({"fen":"4k3/8/8/8/8/8/8/4K2R w - - 0 1","eval_cp":50}),
      ~s({"fen":"4k3/8/8/8/8/8/8/4K2R b - - 0 1","result":"0-1"}),
      ~s({"fen":"4k3/8/8/8/8/8/8/4K1N1 w - - 0 1","wdl":[0.4,0.2,0.4]}),
      ~s({"fen":"4k3/8/8/8/8/8/8/4K1N1 b - - 0 1","eval_cp":-50})
    ]

    File.write!(data, Enum.join(rows, "\n") <> "\n")

    first = temp_artifact()
    second = temp_artifact()

    result =
      Trainer.train!(data, first,
        epochs: 12,
        learning_rate: 0.01,
        validation_fraction: 0.25,
        seed: 0
      )

    second_result =
      Trainer.train!(data, second,
        epochs: 12,
        learning_rate: 0.01,
        validation_fraction: 0.25,
        seed: 0
      )

    assert result.training_rows > 0 and result.validation_rows > 0
    assert result.validation_loss < result.initial_validation_loss
    assert %{second_result | output: first} == result
    assert File.read!(first) == File.read!(second)
  end

  test "mixed side-to-move labels train deterministically across every runtime tensor" do
    data = Path.join(System.tmp_dir!(), "mixed-train-#{System.unique_integer([:positive])}.jsonl")
    first = temp_artifact()
    second = temp_artifact()

    rows = [
      ~s({"fen":"4k3/8/8/8/8/8/8/3QK3 w - - 0 1","eval_cp":700}),
      ~s({"fen":"4k3/8/8/8/8/8/8/3QK3 b - - 0 1","result":"0-1"}),
      ~s({"fen":"4k3/8/8/8/8/8/8/3qK3 w - - 0 1","wdl":[0.0,0.1,0.9]}),
      ~s({"fen":"4k3/8/8/8/8/8/8/3qK3 b - - 0 1","wdl":[1.0,0.0,0.0]})
    ]

    File.write!(data, Enum.join(List.duplicate(rows, 6) |> List.flatten(), "\n") <> "\n")

    opts = [epochs: 4, learning_rate: 0.01, validation_fraction: 0.25, seed: 19]
    first_result = Trainer.train!(data, first, opts)
    second_result = Trainer.train!(data, second, opts)

    assert first_result.training_rows > 0 and first_result.validation_rows > 0
    assert first_result.validation_loss < first_result.initial_validation_loss
    assert first_result.changed_tensors == [:feature, :psqt, :w1, :b1, :w2, :b2]
    assert first_result.active_updates > 0
    assert first_result == %{second_result | output: first}
    assert File.read!(first) == File.read!(second)
  end

  defp temp_artifact,
    do: Path.join(System.tmp_dir!(), "trained-#{System.unique_integer([:positive])}.nnue")

  defp changed_tensors(before, after_weights) do
    [:feature, :psqt, :w1, :b1, :w2, :b2]
    |> Enum.filter(&(Map.fetch!(before, &1) != Map.fetch!(after_weights, &1)))
  end

  defp changed_entries(before, after_weights, width) do
    0..(div(byte_size(before), width) - 1)
    |> Enum.filter(fn index ->
      binary_part(before, index * width, width) !=
        binary_part(after_weights, index * width, width)
    end)
  end
end
