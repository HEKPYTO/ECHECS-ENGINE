defmodule Mix.Tasks.EngineTrainEvaluatorTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "exports a sparse evaluator artifact from JSONL" do
    dataset_path = tmp_path("engine_train_evaluator")
    output_path = tmp_path("engine_train_evaluator_artifact", ".axon")

    File.write!(
      dataset_path,
      IO.iodata_to_binary([
        Jason.encode!(%{
          "fen" => @start_fen,
          "move" => "e2e4",
          "wdl" => [1.0, 0.0, 0.0]
        }),
        "\n"
      ])
    )

    on_exit(fn ->
      File.rm(dataset_path)
      File.rm(output_path)
    end)

    output =
      capture_io(fn ->
        Mix.Tasks.Engine.TrainEvaluator.run([dataset_path, output_path])
      end)

    assert output =~ "saved evaluator:"
    assert File.exists?(output_path)
  end

  test "exports a quantized sparse evaluator artifact" do
    dataset_path = tmp_path("engine_train_evaluator_quantized")
    output_path = tmp_path("engine_train_evaluator_quantized_artifact", ".axon")

    File.write!(
      dataset_path,
      IO.iodata_to_binary([
        Jason.encode!(%{
          "fen" => @start_fen,
          "move" => "e2e4",
          "wdl" => [1.0, 0.0, 0.0]
        }),
        "\n"
      ])
    )

    on_exit(fn ->
      File.rm(dataset_path)
      File.rm(output_path)
    end)

    capture_io(fn ->
      Mix.Tasks.Engine.TrainEvaluator.run(["--quantize", dataset_path, output_path])
    end)

    assert {:ok, %{artifact: artifact}} =
             EchecsEngine.Checkpoint.load_evaluator_state(output_path)

    assert artifact.quantized? == true
  end

  defp tmp_path(prefix, suffix \\ ".jsonl") do
    System.tmp_dir!()
    |> Path.join("#{prefix}_#{System.unique_integer([:positive])}#{suffix}")
  end
end
