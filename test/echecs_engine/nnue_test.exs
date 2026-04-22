defmodule EchecsEngine.NNUETest do
  use ExUnit.Case, async: true

  import Nx.Defn

  test "NNUE evaluates accumulator correctly" do
    # Create dummy tensors for a batch of 1 or shape {256}
    key = Nx.Random.key(42)
    {acc, key} = Nx.Random.uniform(key, shape: {256}, type: :f32)
    {w1, key} = Nx.Random.uniform(key, shape: {256, 32}, type: :f32)
    {b1, key} = Nx.Random.uniform(key, shape: {32}, type: :f32)
    {w2, key} = Nx.Random.uniform(key, shape: {32, 1}, type: :f32)
    {b2, _key} = Nx.Random.uniform(key, shape: {1}, type: :f32)

    # Use EXLA compiler to ensure it compiles
    result = jit(&EchecsEngine.NNUE.evaluate/5, compiler: EXLA).(acc, w1, b1, w2, b2)

    # Validate shape
    assert Nx.shape(result) == {1}
  end
end
