defmodule EchecsEngine.SparseEvaluatorTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  test "SparseEvaluator evaluates accumulator correctly using SCReLU" do
    key = Nx.Random.key(42)
    {acc, key} = Nx.Random.uniform(key, shape: {3072}, type: :f32)
    {w1, key} = Nx.Random.uniform(key, shape: {3072, 32}, type: :f32)
    {b1, key} = Nx.Random.uniform(key, shape: {32}, type: :f32)
    {w2, key} = Nx.Random.uniform(key, shape: {32, 1}, type: :f32)
    {b2, _key} = Nx.Random.uniform(key, shape: {1}, type: :f32)

    result = jit(&EchecsEngine.SparseEvaluator.evaluate/5, compiler: EXLA).(acc, w1, b1, w2, b2)

    assert Nx.shape(result) == {1}
  end
end
