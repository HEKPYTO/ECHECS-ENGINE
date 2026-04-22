defmodule EchecsZero.MCTS do
  alias EchecsZero.MCTS.Node

  @c_puct 1.0

  @doc """
  Runs Monte Carlo Tree Search for the given number of iterations.
  Returns the updated root node.
  """
  def search(game, iterations) do
    root = %Node{game: game}

    Enum.reduce(1..iterations, root, fn _, current_root ->
      {new_root, _value} = simulate(current_root)
      new_root
    end)
  end

  defp simulate(%Node{children: children} = node) when children == %{} do
    {node, value} = expand_and_evaluate(node)
    {update_node(node, value), value}
  end

  defp simulate(%Node{children: children} = node) do
    {best_move, best_child} = select_child(node)
    {updated_child, value} = simulate(best_child)

    updated_children = Map.put(children, best_move, updated_child)
    updated_node = %{node | children: updated_children}

    # Backpropagate -value because turn switches
    {update_node(updated_node, -value), -value}
  end

  defp select_child(%Node{children: children, visits: parent_visits}) do
    Enum.max_by(children, fn {_move, child} ->
      puct_score(child, parent_visits)
    end)
  end

  defp puct_score(
         %Node{visits: visits, total_value: total_value, prior_prob: prior},
         parent_visits
       ) do
    q_value = if visits == 0, do: 0.0, else: total_value / visits
    u_value = @c_puct * prior * :math.sqrt(max(parent_visits, 1)) / (1 + visits)
    q_value + u_value
  end

  defp expand_and_evaluate(%Node{game: game} = node) do
    legal_moves = Echecs.legal_moves(game)

    if legal_moves == [] do
      # Determine terminal value based on status
      # Simplistic logic: checkmate is -1 for the current player
      status = Echecs.status(game)

      value =
        case status do
          :checkmate -> -1.0
          # draw, stalemate, etc.
          _ -> 0.0
        end

      {node, value}
    else
      tensor =
        EchecsZero.Tensor.to_tensor(game)
        |> Nx.squeeze(axes: [0])
        |> Nx.as_type(:f32)

      %{policy: _p, value: v} = Nx.Serving.batched_run(EchecsZero.Serving, tensor)

      value = v |> Nx.squeeze() |> Nx.to_number()

      prob = 1.0 / length(legal_moves)

      children =
        Map.new(legal_moves, fn move ->
          {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)
          {move, %Node{game: next_game, prior_prob: prob}}
        end)

      {%{node | children: children}, value}
    end
  end

  defp update_node(%Node{visits: visits, total_value: total} = node, value) do
    %{node | visits: visits + 1, total_value: total + value}
  end
end
