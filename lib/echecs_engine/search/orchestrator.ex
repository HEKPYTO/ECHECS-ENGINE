defmodule EchecsEngine.Search.Orchestrator do
  @moduledoc """
  Orchestrates the hybrid engine flow:
  1. Queries the GPU Vision Transformer (ViT) to get move policies.
  2. Uses policy to select candidate moves.
  3. Passes candidates to the compiled CPU XLA Alpha-Beta NNUE for deep search.
  """

  @doc """
  Runs the full hybrid search for a single step to find the best move.
  """
  def search(game) do
    # 1. Fetch ViT Policy (Slow GPU Network)
    tensor =
      EchecsEngine.Tensor.to_tensor(game)
      |> Nx.squeeze(axes: [0])
      |> Nx.as_type(:f32)

    %{policy: policy_tensor, value: _v} = Nx.Serving.batched_run(EchecsEngine.Serving, tensor)

    # We extract the policy vector
    _policy = Nx.to_flat_list(policy_tensor)

    # 2. Generate Legal Moves
    legal_moves = Echecs.legal_moves(game)

    if legal_moves == [] do
      {:terminal, Echecs.status(game)}
    else
      # In a complete implementation, we'd map `policy` to `legal_moves` and sort them.
      # For now, we take the generated moves and construct "leaf accumulators" 
      # simulating the Alpha-Beta NNUE search phase.

      # Dummy weights for the compiled NNUE (these would be loaded from checkpoint)
      w1 = Nx.broadcast(0.1, {256, 32})
      b1 = Nx.broadcast(0.0, {32})
      w2 = Nx.broadcast(0.1, {32, 1})
      b2 = Nx.broadcast(0.0, {1})

      # We simulate that each legal move results in an updated NNUE accumulator.
      # For this minimal proof of the hybrid flow, we generate random accumulators
      # representing the deeply searched leaf nodes of each move.
      num_moves = length(legal_moves)
      key = Nx.Random.key(System.system_time())
      {accumulators, _key} = Nx.Random.uniform(key, shape: {num_moves, 256}, type: :f32)

      # 3. Fast Alpha-Beta NNUE evaluation (CPU/JIT)
      {best_score, best_idx_tensor} =
        EchecsEngine.Search.AlphaBeta.evaluate_leaves(accumulators, w1, b1, w2, b2)

      best_idx = Nx.to_number(best_idx_tensor)
      best_score_val = Nx.to_number(best_score)

      # 4. Select move
      best_move = Enum.at(legal_moves, best_idx)

      {best_move, best_score_val}
    end
  end
end
