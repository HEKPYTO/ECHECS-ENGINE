defmodule EchecsEngine.Model.ViTTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Model.ViT

  describe "build/0" do
    test "builds the ViT model that takes {1, 119, 8, 8} and returns policy and value" do
      model = ViT.build()

      # Initialize the model weights
      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

      # Dummy input of batch 1, 119 channels, 8x8
      key = Nx.Random.key(42)
      input = Nx.Random.normal(key, shape: {1, 119, 8, 8}) |> elem(0)

      # Initialize parameters
      params = init_fn.(input, %{})

      # Predict
      result = predict_fn.(params, input)

      assert %{policy: policy, value: value} = result
      assert Nx.shape(policy) == {1, 4672}
      assert Nx.shape(value) == {1, 1}
    end
  end
end
