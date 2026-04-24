defmodule EchecsEngine.SimulationTest do
  use ExUnit.Case, async: true

  test "documents supervised JSONL training instead of dummy random simulation" do
    docs = Code.fetch_docs(EchecsEngine.Simulation)

    assert {:docs_v1, _, _, _, %{"en" => module_doc}, _, _} = docs
    assert module_doc =~ "supervised JSONL"
    refute module_doc =~ "dummy self-play"
    refute module_doc =~ "uniform random batches"
    refute module_doc =~ "ResNet"
  end

  test "training epoch count is configurable" do
    assert EchecsEngine.Simulation.training_epochs([]) == 2
    assert EchecsEngine.Simulation.training_epochs(epochs: 7) == 7
    assert EchecsEngine.Simulation.training_epochs(epochs: 0) == 1
  end

  test "loss function accepts Axon trainer target/prediction ordering" do
    loss = EchecsEngine.Simulation.loss_fn()

    targets = %{
      policy: Nx.broadcast(0.0, {2, 4672}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      policy_mask: Nx.broadcast(1.0, {2, 4672}),
      wdl: Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], type: :f32),
      moves_left: Nx.tensor([[0.5], [0.25]], type: :f32)
    }

    predictions = %{
      policy: Nx.broadcast(0.0, {2, 4672}),
      wdl: Nx.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], type: :f32),
      moves_left: Nx.tensor([[0.4], [0.2]], type: :f32)
    }

    value = loss.(targets, predictions)

    assert %Nx.Tensor{} = value
    assert Nx.shape(value) == {}
  end
end
