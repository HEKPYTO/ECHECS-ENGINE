defmodule EchecsEngine.Search.AlphaBeta do
  @moduledoc """
  Defines the boundary between Elixir's concurrent search 
  and the JIT-compiled XLA execution.
  """

  import Nx.Defn

  @doc """
  Given a batch of leaf accumulators, evaluates them through the NNUE network
  and returns the maximum evaluated score and its corresponding index.

  This compiled function acts as the optimized tactical evaluator at the
  frontier of the Elixir Alpha-Beta search tree.
  """
  defn evaluate_leaves(accumulators, w1, b1, w2, b2) do
    # accumulators shape: {num_leaves, 3072}
    # The NNUE evaluate function inherently broadcasts over the batch dimension!
    scores = EchecsEngine.NNUE.evaluate(accumulators, w1, b1, w2, b2)

    # Squeeze the {num_leaves, 1} shape to {num_leaves}
    scores = Nx.squeeze(scores, axes: [-1])

    # Return the maximum score and the index of the best leaf
    best_score = Nx.reduce_max(scores)
    best_idx = Nx.argmax(scores)

    {best_score, best_idx}
  end
end
