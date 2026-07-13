defmodule EchecsEngine.Model.ViTTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Model.ViT

  describe "build/0" do
    test "builds the ViT model that takes {1, 119, 8, 8} and returns policy, WDL, and moves-left heads" do
      model = ViT.build()

      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

      key = Nx.Random.key(42)
      input = Nx.Random.normal(key, shape: {1, 119, 8, 8}) |> elem(0)

      params = init_fn.(input, Axon.ModelState.empty())

      result = predict_fn.(params, input)

      assert %{policy: policy, wdl: wdl, moves_left: moves_left} = result
      assert Nx.shape(policy) == {1, 4672}
      assert Nx.shape(wdl) == {1, 3}
      assert Nx.shape(moves_left) == {1, 1}
    end
  end

  describe "build/0 parameter isolation" do
    test "each of the four transformer encoder blocks owns independent, non-shared parameters" do
      model = ViT.build()

      {init_fn, _predict_fn} = Axon.build(model, compiler: EXLA)

      key = Nx.Random.key(7)
      input = Nx.Random.normal(key, shape: {1, 119, 8, 8}) |> elem(0)

      params = init_fn.(input, Axon.ModelState.empty())

      param_keys =
        params
        |> Map.get(:parameters)
        |> Map.keys()

      for projection <- ["qkv_proj", "out_proj", "swiglu_gate", "swiglu_up", "swiglu_down"],
          index <- 0..3 do
        expected = "block_#{index}_#{projection}"

        assert expected in param_keys,
               "expected independent parameter #{inspect(expected)} to exist, " <>
                 "got keys: #{inspect(Enum.sort(param_keys))}"
      end
    end
  end

  test "normalizes extreme attention scores without NaN values" do
    probabilities = ViT.attention_probabilities(Nx.tensor([[1_000.0, -1_000.0]], type: :f32))

    assert Nx.to_flat_list(probabilities) == [1.0, 0.0]
  end
end
