defmodule EchecsEngine.EvaluatorArtifactTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Checkpoint

  setup do
    path =
      System.tmp_dir!()
      |> Path.join("echecs_engine_evaluator_#{System.unique_integer([:positive])}.axon")

    on_exit(fn -> File.rm(path) end)

    %{path: path}
  end

  test "saves and loads sparse evaluator artifacts with feature tables", %{path: path} do
    artifact = %{
      feature_table: %{1 => Nx.broadcast(0.0, {3072}) |> Nx.put_slice([1], Nx.tensor([1.0]))},
      w1: Nx.broadcast(0.1, {3072, 32}),
      b1: Nx.broadcast(0.0, {32}),
      w2: Nx.broadcast(0.1, {32, 1}),
      b2: Nx.broadcast(0.0, {1})
    }

    :ok = Checkpoint.save_evaluator_state!(path, artifact, %{"source" => "unit-test"})

    assert {:ok, %{artifact: loaded, metadata: metadata}} = Checkpoint.load_evaluator_state(path)
    assert metadata.training_config["source"] == "unit-test"
    assert is_map(loaded.feature_table)
    assert Nx.shape(loaded.w1) == {3072, 32}
  end
end
