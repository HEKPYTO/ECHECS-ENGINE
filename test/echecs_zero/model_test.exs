defmodule EchecsZero.ModelTest do
  use ExUnit.Case

  alias EchecsZero.Model

  test "builds an Axon ResNet model with policy and value heads" do
    model = Model.build()

    # Pass a dummy tensor
    input = Nx.broadcast(0.0, {1, 119, 8, 8})

    {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

    # Initialize parameters
    params = init_fn.(input, %{})

    # Predict
    output = predict_fn.(params, input)

    # Output should be a map/container with policy and value
    assert %{policy: policy_tensor, value: value_tensor} = output

    # Assert shapes
    assert Nx.shape(policy_tensor) == {1, 4672}
    assert Nx.shape(value_tensor) == {1, 1}
  end
end
