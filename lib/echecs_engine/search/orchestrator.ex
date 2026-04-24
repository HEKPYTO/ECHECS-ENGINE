defmodule EchecsEngine.Search.Orchestrator do
  @moduledoc """
  Orchestrates the hybrid engine flow:
  1. Queries the GPU Vision Transformer (ViT) to get move policies.
  2. Uses policy to select candidate moves.
  3. Passes candidates to the compiled CPU XLA Alpha-Beta SparseEvaluator for deep search.
  """

  @default_candidate_limit 8

  @doc """
  Runs the full hybrid search for a single step to find the best move.
  """
  def search(game, opts \\ []) do
    history_games = Keyword.get(opts, :history_games, [])

    tensor =
      EchecsEngine.Tensor.to_tensor(game, history_games)
      |> Nx.squeeze(axes: [0])
      |> Nx.as_type(:f32)

    legal_moves = Echecs.legal_moves(game)

    if legal_moves == [] do
      {:terminal, Echecs.status(game)}
    else
      inference = Keyword.get(opts, :inference, EchecsEngine.Serving)
      output = EchecsEngine.Inference.run(inference, tensor)
      policy_tensor = output.policy
      root_q = EchecsEngine.Value.output_to_q(output)

      candidate_limit =
        opts
        |> Keyword.get(:candidate_limit, @default_candidate_limit)
        |> min(length(legal_moves))
        |> max(1)

      candidates =
        game
        |> EchecsEngine.Policy.legal_move_priors(legal_moves, policy_tensor)
        |> Enum.sort_by(fn {_move, prior} -> prior end, :desc)
        |> Enum.take(candidate_limit)

      candidate_games =
        Enum.map(candidates, fn {move, _prior} ->
          {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)
          {move, next_game}
        end)

      case Keyword.get(opts, :evaluator_weights) do
        %{feature_table: feature_table, w1: w1, b1: b1, w2: w2, b2: b2} ->
          accumulators =
            candidate_games
            |> Enum.map(&elem(&1, 1))
            |> EchecsEngine.Accumulator.batch_from_games(feature_table: feature_table)

          {best_score, best_idx_tensor} =
            EchecsEngine.Search.AlphaBeta.evaluate_leaves(accumulators, w1, b1, w2, b2)

          best_idx = Nx.to_number(best_idx_tensor)
          {best_move, _next_game} = Enum.at(candidate_games, best_idx)

          {best_move, Nx.to_number(best_score)}

        nil ->
          child_qs =
            case Keyword.get(opts, :batched_inference) do
              batch_inference when is_function(batch_inference, 1) ->
                child_tensors =
                  candidate_games
                  |> Enum.map(fn {_move, next_game} ->
                    EchecsEngine.Tensor.to_tensor(
                      next_game,
                      [game | history_games] |> Enum.take(7)
                    )
                    |> Nx.squeeze(axes: [0])
                    |> Nx.as_type(:f32)
                  end)
                  |> Nx.stack()

                batch_output = EchecsEngine.Inference.run_batch(batch_inference, child_tensors)

                0..(length(candidate_games) - 1)
                |> Enum.map(fn idx ->
                  %{wdl: batch_output.wdl[idx]}
                  |> EchecsEngine.Value.output_to_q()
                end)

              _other ->
                Enum.map(candidate_games, fn {_move, next_game} ->
                  next_game
                  |> EchecsEngine.Tensor.to_tensor([game | history_games] |> Enum.take(7))
                  |> Nx.squeeze(axes: [0])
                  |> Nx.as_type(:f32)
                  |> then(&EchecsEngine.Inference.run(inference, &1))
                  |> EchecsEngine.Value.output_to_q()
                end)
            end

          candidate_scores =
            Enum.zip(Enum.zip(candidates, candidate_games), child_qs)
            |> Enum.map(fn {{{move, prior}, {_move, _next_game}}, child_q} ->
              score = child_q + prior * 0.05 + root_q * 0.01
              {move, score}
            end)

          Enum.max_by(candidate_scores, &elem(&1, 1))

        _other ->
          raise ArgumentError,
                "evaluator_weights must be a map containing feature_table, w1, b1, w2, and b2"
      end
    end
  end
end
