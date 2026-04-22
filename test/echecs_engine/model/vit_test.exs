defmodule EchecsEngine.Model.ViTTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Model.ViT

  describe "build/0" do
    test "builds the ViT model that takes {1, 119, 8, 8} and returns policy and value" do
      model = ViT.build()

      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

      key = Nx.Random.key(42)
      input = Nx.Random.normal(key, shape: {1, 119, 8, 8}) |> elem(0)

      params = init_fn.(input, Axon.ModelState.empty())

      result = predict_fn.(params, input)

      assert %{policy: policy, value: value} = result
      assert Nx.shape(policy) == {1, 4672}
      assert Nx.shape(value) == {1, 1}
    end
  end
end
