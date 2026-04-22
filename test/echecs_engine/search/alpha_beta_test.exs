defmodule EchecsEngine.Search.AlphaBetaTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  test "evaluates and selects the best leaf accumulator" do
    key = Nx.Random.key(123)

    # 10 candidate leaves
    {accumulators, key} = Nx.Random.uniform(key, shape: {10, 3072}, type: :f32)

    {w1, key} = Nx.Random.uniform(key, shape: {3072, 32}, type: :f32)
    {b1, key} = Nx.Random.uniform(key, shape: {32}, type: :f32)
    {w2, key} = Nx.Random.uniform(key, shape: {32, 1}, type: :f32)
    {b2, _key} = Nx.Random.uniform(key, shape: {1}, type: :f32)

    # Use EXLA
    {best_score, best_idx} =
      jit(&EchecsEngine.Search.AlphaBeta.evaluate_leaves/5, compiler: EXLA).(
        accumulators,
        w1,
        b1,
        w2,
        b2
      )

    assert Nx.shape(best_score) == {}
    assert Nx.shape(best_idx) == {}

    idx = Nx.to_number(best_idx)
    assert idx >= 0 and idx < 10
  end
end
