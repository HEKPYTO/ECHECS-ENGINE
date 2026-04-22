defmodule EchecsEngine.ModelTest do
  use ExUnit.Case

  alias EchecsEngine.Model

  test "builds an Axon ResNet model with policy and value heads" do
    model = Model.build()

    input = Nx.broadcast(0.0, {1, 119, 8, 8})

    {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

    params = init_fn.(input, Axon.ModelState.empty())

    output = predict_fn.(params, input)

    assert %{policy: policy_tensor, value: value_tensor} = output

    assert Nx.shape(policy_tensor) == {1, 4672}
    assert Nx.shape(value_tensor) == {1, 1}
  end
end
