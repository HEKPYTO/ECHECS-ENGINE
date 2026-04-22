defmodule EchecsEngine.ServingTest do
  use ExUnit.Case, async: false

  test "EchecsEngine.Serving returns policy and value tensors" do
    input_tensor = Nx.broadcast(Nx.tensor(0.0, type: :f32), {119, 8, 8})

    result = Nx.Serving.batched_run(EchecsEngine.Serving, input_tensor)

    assert %{policy: policy_tensor, value: value_tensor} = result
    assert Nx.shape(policy_tensor) == {4672}
    assert Nx.shape(value_tensor) == {1}
  end
end
