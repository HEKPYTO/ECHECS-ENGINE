defmodule EchecsZero.ServingTest do
  use ExUnit.Case, async: false

  test "EchecsZero.Serving returns policy and value tensors" do
    # Create a dummy input tensor for a single state
    input_tensor = Nx.broadcast(Nx.tensor(0.0, type: :f32), {119, 8, 8})

    # Call the serving (which should be registered by the application)
    result = Nx.Serving.batched_run(EchecsZero.Serving, input_tensor)

    # Output should contain policy and value maps/tensors
    assert %{policy: policy_tensor, value: value_tensor} = result
    assert Nx.shape(policy_tensor) == {4672}
    assert Nx.shape(value_tensor) == {1}
  end
end
