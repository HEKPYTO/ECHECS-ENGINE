defmodule EchecsEngine.MCTS do
  @moduledoc """
  Provides the Monte Carlo Tree Search algorithm for legacy frameworks.
  """

  alias EchecsEngine.MCTS.Node

  @c_puct 1.0
  @fpu_reduction 0.25

  @doc """
  Runs Monte Carlo Tree Search for the given number of iterations.

  Traverses the tree using PUCT, expands leaf nodes using the neural
  network evaluated via `Nx.Serving`, and backpropagates the scalar
  value to update tree statistics. Returns the updated root node.
  """
  def search(game, iterations, opts \\ []) do
    root = %Node{game: game, history_games: Keyword.get(opts, :history_games, []), depth: 0}
    budget = EchecsEngine.Search.TimeManager.new(Keyword.put_new(opts, :iterations, iterations))
    {opts, owned_tt?} = ensure_tt(opts)

    result =
      budget
      |> EchecsEngine.Search.TimeManager.max_iterations()
      |> iteration_sequence()
      |> Enum.reduce_while(root, fn iteration, current_root ->
        if EchecsEngine.Search.TimeManager.expired?(budget, iteration - 1) do
          {:halt, current_root}
        else
          {new_root, _value} = simulate(current_root, opts)
          {:cont, new_root}
        end
      end)

    if owned_tt?, do: :ets.delete(opts[:tt])
    result
  end

  @doc """
  Searches a position and returns the root move with the highest visit count.
  """
  def search_best_move(game, opts \\ []) do
    budget = EchecsEngine.Search.TimeManager.new(opts)
    iterations = EchecsEngine.Search.TimeManager.max_iterations(budget)

    game
    |> search(iterations, opts)
    |> best_move()
  end

  defp iteration_sequence(:infinity), do: Stream.iterate(1, &(&1 + 1))
  defp iteration_sequence(max_iterations), do: 1..max_iterations

  @doc """
  Selects the root move with the highest visit count after search.
  """
  def best_move(%Node{children: children}) when map_size(children) > 0 do
    children
    |> Enum.max_by(fn {_move, child} ->
      {child.visits, Node.value_from_parent_perspective(child), child.prior_prob}
    end)
    |> elem(0)
  end

  @doc """
  Returns the principal variation by following the most-visited child path.
  """
  def principal_variation(node, max_plies \\ 16)

  def principal_variation(%Node{} = node, max_plies) when max_plies > 0 do
    do_principal_variation(node, max_plies, [])
  end

  def principal_variation(_node, _max_plies), do: []

  @doc false
  defp simulate(%Node{children: children} = node, opts) when children == %{} do
    {node, value} = expand_and_evaluate(node, opts)
    {update_node(node, value), value}
  end

  defp simulate(%Node{children: children} = node, opts) do
    {best_move, best_child} = select_child(node, opts)
    {updated_child, value} = simulate(best_child, opts)

    updated_children = Map.replace!(children, best_move, updated_child)
    updated_node = %{node | children: updated_children}

    {update_node(updated_node, -value), -value}
  end

  @doc false
  defp select_child(%Node{children: children, visits: parent_visits} = node, opts) do
    parent_q = node_value(node)
    c_puct = Keyword.get(opts, :c_puct, @c_puct)
    fpu_reduction = Keyword.get(opts, :fpu_reduction, @fpu_reduction)

    Enum.max_by(children, fn {_move, child} ->
      puct_score(child, parent_visits, parent_q, c_puct, fpu_reduction)
    end)
  end

  @doc false
  defp puct_score(
         %Node{visits: visits, total_value: total_value, prior_prob: prior},
         parent_visits,
         parent_q,
         c_puct,
         fpu_reduction
       ) do
    q_value =
      if visits == 0 do
        parent_q - fpu_reduction
      else
        Node.value_from_parent_perspective(%Node{visits: visits, total_value: total_value})
      end

    u_value = c_puct * prior * :math.sqrt(max(parent_visits, 1)) / (1 + visits)
    q_value + u_value
  end

  @doc false
  defp expand_and_evaluate(
         %Node{game: game, history_games: history_games, depth: depth} = node,
         opts
       ) do
    legal_moves = Echecs.legal_moves(game)

    if legal_moves == [] do
      status = Echecs.status(game)

      value =
        case status do
          :checkmate -> -1.0
          _ -> 0.0
        end

      {node, value}
    else
      output = cached_or_infer(game, history_games, opts)

      policy_tensor = output.policy
      value = EchecsEngine.Value.output_to_q(output)

      priors =
        game
        |> EchecsEngine.Policy.legal_move_priors(legal_moves, policy_tensor)
        |> maybe_apply_root_noise(depth, opts)

      children =
        Map.new(priors, fn {move, prior_prob} ->
          {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)
          child_history = [game | history_games] |> Enum.take(7)

          {move,
           %Node{
             game: next_game,
             history_games: child_history,
             depth: depth + 1,
             prior_prob: prior_prob
           }}
        end)

      {%{node | children: children}, value}
    end
  end

  defp ensure_tt(opts) do
    case Keyword.get(opts, :tt) do
      nil -> {Keyword.put(opts, :tt, :ets.new(__MODULE__, [:set, :private])), true}
      _tt -> {opts, false}
    end
  end

  defp cached_or_infer(game, history_games, opts) do
    tt = Keyword.fetch!(opts, :tt)
    key = tt_key(game, history_games, opts)

    case :ets.lookup(tt, key) do
      [{^key, output}] ->
        output

      [] ->
        tensor =
          EchecsEngine.Tensor.to_tensor(game, history_games)
          |> Nx.squeeze(axes: [0])
          |> Nx.as_type(:f32)

        inference = Keyword.get(opts, :inference, EchecsEngine.Serving)
        output = EchecsEngine.Inference.run(inference, tensor)
        true = :ets.insert(tt, {key, output})
        output
    end
  end

  defp tt_key(game, history_games, opts) do
    {inference_identity(opts), game.zobrist_hash,
     Enum.map(Enum.take(history_games, 7), & &1.zobrist_hash)}
  end

  defp inference_identity(opts) do
    case Keyword.get(opts, :inference_id) do
      nil -> Keyword.get(opts, :inference, EchecsEngine.Serving) |> normalize_inference_identity()
      explicit -> explicit
    end
  end

  defp normalize_inference_identity(inference) when is_atom(inference), do: {:serving, inference}

  defp normalize_inference_identity(inference) when is_function(inference, 1) do
    {:function,
     %{
       module: :erlang.fun_info(inference, :module) |> elem(1),
       index: :erlang.fun_info(inference, :index) |> elem(1),
       uniq: :erlang.fun_info(inference, :new_uniq) |> elem(1),
       env_hash: :erlang.fun_info(inference, :env) |> elem(1) |> :erlang.phash2()
     }}
  end

  defp normalize_inference_identity(inference), do: {:term, :erlang.phash2(inference)}

  defp maybe_apply_root_noise(priors, 0, opts) when priors != [] do
    if Keyword.get(opts, :self_play, false) do
      epsilon = Keyword.get(opts, :dirichlet_epsilon, 0.25)
      alpha = Keyword.get(opts, :dirichlet_alpha, 0.3)
      noise = sample_dirichlet(length(priors), alpha, opts)

      Enum.zip(priors, noise)
      |> Enum.map(fn {{move, prior}, noise_weight} ->
        {move, (1.0 - epsilon) * prior + epsilon * noise_weight}
      end)
    else
      priors
    end
  end

  defp maybe_apply_root_noise(priors, _depth, _opts), do: priors

  defp sample_dirichlet(size, alpha, opts) do
    case Keyword.get(opts, :dirichlet_noise_fn) do
      fun when is_function(fun, 2) ->
        fun.(size, alpha)

      _other ->
        1..size
        |> Enum.map(fn _ -> sample_gamma(alpha, opts) end)
        |> normalize_noise()
    end
  end

  defp sample_gamma(alpha, opts) do
    case Keyword.get(opts, :gamma_sampler_fn) do
      fun when is_function(fun, 1) ->
        fun.(alpha)

      _other when alpha < 1.0 ->
        sample_gamma(alpha + 1.0, opts) * :math.pow(max(:rand.uniform(), 1.0e-9), 1.0 / alpha)

      _other ->
        marsaglia_tsang(alpha)
    end
  end

  defp marsaglia_tsang(alpha) do
    d = alpha - 1.0 / 3.0
    c = 1.0 / :math.sqrt(9.0 * d)
    gamma_loop(d, c)
  end

  defp gamma_loop(d, c) do
    x = :rand.normal()
    v = 1.0 + c * x

    cond do
      v <= 0.0 ->
        gamma_loop(d, c)

      true ->
        v3 = v * v * v
        u = :rand.uniform()

        if u < 1.0 - 0.0331 * x * x * x * x or
             :math.log(u) < 0.5 * x * x + d * (1.0 - v3 + :math.log(v3)) do
          d * v3
        else
          gamma_loop(d, c)
        end
    end
  end

  defp normalize_noise(weights) do
    total = Enum.sum(weights)

    if total > 0.0 do
      Enum.map(weights, &(&1 / total))
    else
      uniform = 1.0 / max(length(weights), 1)
      Enum.map(weights, fn _ -> uniform end)
    end
  end

  @doc false
  defp update_node(%Node{visits: visits, total_value: total} = node, value) do
    %{node | visits: visits + 1, total_value: total + value}
  end

  defp node_value(%Node{visits: 0}), do: 0.0
  defp node_value(%Node{visits: visits, total_value: total_value}), do: total_value / visits

  defp do_principal_variation(%Node{children: children}, _remaining, acc)
       when map_size(children) == 0,
       do: Enum.reverse(acc)

  defp do_principal_variation(%Node{children: children}, remaining, acc) do
    {move, child} =
      Enum.max_by(children, fn {_move, child} ->
        {child.visits, Node.value_from_parent_perspective(child), child.prior_prob}
      end)

    if remaining == 1 do
      Enum.reverse([move | acc])
    else
      do_principal_variation(child, remaining - 1, [move | acc])
    end
  end
end
