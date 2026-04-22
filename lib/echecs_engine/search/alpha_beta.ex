defmodule EchecsEngine.Search.AlphaBeta do
  @moduledoc """
  Defines the boundary between Elixir's concurrent search 
  and the JIT-compiled XLA execution.
  """

  import Nx.Defn

  @doc """
  Given a batch of leaf accumulators, evaluates them through the SparseEvaluator network
  and returns the maximum evaluated score and its corresponding index.

  This compiled function acts as the optimized tactical evaluator at the
  frontier of the Elixir Alpha-Beta search tree.
  """
  defn evaluate_leaves(accumulators, w1, b1, w2, b2) do
    scores = EchecsEngine.SparseEvaluator.evaluate(accumulators, w1, b1, w2, b2)

    scores = Nx.squeeze(scores, axes: [-1])

    best_score = Nx.reduce_max(scores)
    best_idx = Nx.argmax(scores)

    {best_score, best_idx}
  end
end
