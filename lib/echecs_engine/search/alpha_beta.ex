defmodule EchecsEngine.Search.AlphaBeta do
  @moduledoc """
  Recursive alpha-beta search with iterative deepening and sparse-evaluator leaves.
  """

  import Nx.Defn

  @neg_inf -1.0e9
  @pos_inf 1.0e9

  @type stats :: %{
          best_move: Echecs.Move.t(),
          score: float(),
          depth: pos_integer(),
          nodes: pos_integer(),
          pv: [String.t()]
        }

  @doc """
  Given a batch of leaf accumulators, evaluates them through the SparseEvaluator network
  and returns the maximum evaluated score and its corresponding index.
  """
  defn evaluate_leaves(accumulators, w1, b1, w2, b2) do
    scores = EchecsEngine.SparseEvaluator.evaluate(accumulators, w1, b1, w2, b2)

    scores = Nx.squeeze(scores, axes: [-1])

    best_score = Nx.reduce_max(scores)
    best_idx = Nx.argmax(scores)

    {best_score, best_idx}
  end

  @spec search(Echecs.Game.t(), keyword()) :: stats()
  def search(game, opts \\ []) do
    node_limit = Keyword.get(opts, :nodes)
    deadline_ms = deadline_ms(opts)
    reporter = Keyword.get(opts, :reporter)
    history_games = Keyword.get(opts, :history_games, [])
    tt = :ets.new(__MODULE__, [:set, :private])
    root_acc = initial_accumulator(game, opts)

    try do
      opts
      |> depth_sequence()
      |> Enum.reduce_while({nil, base_state(tt, deadline_ms, node_limit)}, fn depth,
                                                                              {best, state} ->
        if expired?(state) do
          {:halt, {best || fallback_result(game), state}}
        else
          {result, state} =
            root_search_with_window(game, depth, history_games, root_acc, best, state, opts)

          if is_function(reporter, 1) do
            reporter.(%{
              depth: depth,
              nodes: state.nodes,
              pv: Enum.map(result.pv_moves, &move_to_uci/1)
            })
          end

          {:cont,
           {%{
              best_move: result.best_move,
              score: result.score,
              depth: depth,
              nodes: state.nodes,
              pv: Enum.map(result.pv_moves, &move_to_uci/1)
            }, state}}
        end
      end)
      |> elem(0)
    after
      :ets.delete(tt)
    end
  end

  @spec search_best_move(Echecs.Game.t(), keyword()) :: {Echecs.Move.t(), stats()}
  def search_best_move(game, opts \\ []) do
    result = search(game, opts)
    {result.best_move, result}
  end

  defp depth_sequence(opts) do
    if Keyword.get(opts, :infinite, false) and not is_integer(opts[:depth]) do
      Stream.iterate(1, &(&1 + 1))
    else
      1..max(Keyword.get(opts, :depth, 2), 1)
    end
  end

  @doc false
  @spec transposition_key(Echecs.Game.t(), [Echecs.Game.t()], keyword()) :: term()
  def transposition_key(game, history_games, opts \\ []) do
    {
      game.zobrist_hash,
      Enum.map(Enum.take(history_games || [], 7), & &1.zobrist_hash),
      evaluator_identity(opts),
      inference_identity(opts)
    }
  end

  defp root_search(game, depth, history_games, accumulator, state, opts) do
    root_search(game, depth, history_games, accumulator, state, opts, @neg_inf, @pos_inf)
  end

  defp root_search(game, depth, history_games, accumulator, state, opts, alpha, beta) do
    legal_moves =
      game
      |> ordered_moves(history_games, opts)
      |> limit_root_candidates(opts)

    Enum.reduce_while(legal_moves, {fallback_root_result(game), state, alpha}, fn move,
                                                                                  {best, state,
                                                                                   alpha} ->
      if expired?(state) do
        {:halt, {best, state}}
      else
        {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)
        next_history = [game | history_games] |> Enum.take(7)
        next_acc = next_accumulator(accumulator, game, move, opts)

        {score, child_pv, state} =
          negamax(
            next_game,
            depth - 1,
            -beta,
            -alpha,
            next_history,
            next_acc,
            state,
            opts,
            1,
            true
          )

        score = -score
        pv = [move | child_pv]

        if score > alpha do
          updated = %{best_move: move, score: score, pv_moves: pv}
          {:cont, {updated, state, score}}
        else
          {:cont, {best, state, alpha}}
        end
      end
    end)
    |> then(fn
      {best, state, _alpha} -> {best, state}
      {best, state} -> {best, state}
    end)
  end

  defp root_search_with_window(game, depth, history_games, accumulator, best, state, opts) do
    case aspiration_bounds(depth, best, opts) do
      nil ->
        root_search(game, depth, history_games, accumulator, state, opts)

      {alpha, beta} ->
        {result, state} =
          root_search(game, depth, history_games, accumulator, state, opts, alpha, beta)

        if result.score <= alpha or result.score >= beta do
          root_search(game, depth, history_games, accumulator, state, opts)
        else
          {result, state}
        end
    end
  end

  defp aspiration_bounds(_depth, nil, _opts), do: nil

  defp aspiration_bounds(depth, best, opts) do
    case Keyword.get(opts, :aspiration_window, 0.25) do
      false -> nil
      _ -> do_aspiration_bounds(depth, best, opts)
    end
  end

  defp do_aspiration_bounds(depth, %{score: score}, opts) when depth > 1 do
    case Keyword.get(opts, :aspiration_window, 0.25) do
      window when is_number(window) and window > 0 ->
        {score - window, score + window}

      _ ->
        nil
    end
  end

  defp do_aspiration_bounds(_depth, _best, _opts), do: nil

  defp limit_root_candidates(moves, opts) do
    case Keyword.get(opts, :candidate_limit) do
      limit when is_integer(limit) and limit > 0 -> Enum.take(moves, limit)
      _other -> moves
    end
  end

  defp negamax(
         game,
         depth,
         alpha,
         beta,
         history_games,
         accumulator,
         state,
         opts,
         ply,
         allow_null?
       ) do
    state = bump_nodes(state)

    cond do
      expired?(state) ->
        {evaluate_position(game, history_games, accumulator, opts), [], state}

      Echecs.legal_moves(game) == [] ->
        {terminal_score(game, depth), [], state}

      depth <= 0 ->
        quiescence(game, alpha, beta, history_games, accumulator, state, opts, ply)

      tt_entry = lookup_tt(state.tt, game, history_games, depth, alpha, beta, opts) ->
        {tt_entry.score, tt_entry.pv_moves, state}

      null_cutoff =
          maybe_null_move_cutoff(
            game,
            depth,
            beta,
            history_games,
            accumulator,
            state,
            opts,
            ply,
            allow_null?
          ) ->
        null_cutoff

      true ->
        search_children(game, depth, alpha, beta, history_games, accumulator, state, opts, ply)
    end
  end

  defp search_children(game, depth, alpha, beta, history_games, accumulator, state, opts, ply) do
    legal_moves = ordered_moves(game, history_games, opts, ply)

    original_alpha = alpha

    {best_score, best_pv, state, _final_alpha, cutoff?} =
      Enum.reduce_while(legal_moves, {@neg_inf, [], state, alpha, false}, fn move,
                                                                             {best_score, best_pv,
                                                                              state, alpha,
                                                                              cutoff?} ->
        if expired?(state) do
          {:halt, {best_score, best_pv, state, alpha, false}}
        else
          {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)
          next_history = [game | history_games] |> Enum.take(7)
          next_acc = next_accumulator(accumulator, game, move, opts)

          {score, child_pv, state} =
            negamax(
              next_game,
              depth - 1,
              -beta,
              -alpha,
              next_history,
              next_acc,
              state,
              opts,
              ply + 1,
              true
            )

          score = -score

          {best_score, best_pv, alpha} =
            if score > best_score do
              {score, [move | child_pv], max(alpha, score)}
            else
              {best_score, best_pv, alpha}
            end

          if alpha >= beta do
            {:halt, {best_score, best_pv, state, alpha, true}}
          else
            {:cont, {best_score, best_pv, state, alpha, cutoff?}}
          end
        end
      end)

    store_tt(
      state.tt,
      game,
      history_games,
      depth,
      best_score,
      best_pv,
      opts,
      tt_flag(best_score, original_alpha, beta, cutoff?)
    )

    {best_score, best_pv, state}
  end

  defp quiescence(game, alpha, beta, history_games, accumulator, state, opts, ply) do
    stand_pat = evaluate_position(game, history_games, accumulator, opts)

    cond do
      stand_pat >= beta ->
        {stand_pat, [], state}

      true ->
        alpha = max(alpha, stand_pat)

        captures =
          Echecs.legal_moves(game)
          |> Enum.filter(&(capture_bonus(game, &1) > 0))
          |> order_capture_moves(game)

        Enum.reduce_while(captures, {stand_pat, [], state, alpha}, fn move,
                                                                      {best_score, best_pv, state,
                                                                       alpha} ->
          if expired?(state) do
            {:halt, {best_score, best_pv, state, alpha}}
          else
            {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)
            next_history = [game | history_games] |> Enum.take(7)
            next_acc = next_accumulator(accumulator, game, move, opts)

            {score, child_pv, state} =
              quiescence(
                next_game,
                -beta,
                -alpha,
                next_history,
                next_acc,
                bump_nodes(state),
                opts,
                ply + 1
              )

            score = -score

            {best_score, best_pv, alpha} =
              if score > best_score do
                {score, [move | child_pv], max(alpha, score)}
              else
                {best_score, best_pv, alpha}
              end

            if alpha >= beta do
              {:halt, {best_score, best_pv, state, alpha}}
            else
              {:cont, {best_score, best_pv, state, alpha}}
            end
          end
        end)
        |> then(fn {best_score, best_pv, state, _alpha} -> {best_score, best_pv, state} end)
    end
  end

  defp maybe_null_move_cutoff(
         game,
         depth,
         beta,
         history_games,
         accumulator,
         state,
         opts,
         ply,
         allow_null?
       ) do
    cond do
      not allow_null? ->
        nil

      Keyword.get(opts, :null_move_pruning, true) != true ->
        nil

      depth < 3 ->
        nil

      Echecs.Game.in_check?(game) ->
        nil

      true ->
        reduction = if depth >= 6, do: 3, else: 2
        null_game = make_null_move(game)
        next_history = [game | history_games] |> Enum.take(7)

        {score, _pv, state} =
          negamax(
            null_game,
            depth - 1 - reduction,
            -beta,
            -beta + 1,
            next_history,
            accumulator,
            state,
            opts,
            ply + 1,
            false
          )

        score = -score

        if score >= beta do
          {score, [], state}
        else
          nil
        end
    end
  end

  defp ordered_moves(game, history_games, opts, ply \\ 0) do
    moves = Echecs.legal_moves(game)

    policy_ordering_plies = Keyword.get(opts, :policy_ordering_plies, 1)

    case {Keyword.get(opts, :inference), ply < policy_ordering_plies} do
      {nil, _policy_allowed?} ->
        order_capture_moves(moves, game)

      {_inference, false} ->
        order_capture_moves(moves, game)

      {inference, true} ->
        tensor =
          EchecsEngine.Tensor.to_tensor(game, history_games)
          |> Nx.squeeze(axes: [0])
          |> Nx.as_type(:f32)

        policy = EchecsEngine.Inference.run(inference, tensor).policy

        game
        |> EchecsEngine.Policy.legal_move_priors(moves, policy)
        |> Enum.sort_by(
          fn {move, prior} ->
            {capture_bonus(game, move) + promotion_bonus(move), prior}
          end,
          :desc
        )
        |> Enum.map(&elem(&1, 0))
    end
  end

  defp order_capture_moves(moves, game) do
    Enum.sort_by(moves, &(capture_bonus(game, &1) + promotion_bonus(&1)), :desc)
  end

  defp capture_bonus(game, %Echecs.Move{} = move) do
    board =
      case game.board do
        board when is_tuple(board) -> board
        board -> Echecs.Board.from_struct(board)
      end

    cond do
      move.special == :en_passant -> 10
      Echecs.Board.at_tuple(board, move.to) != nil -> 10
      true -> 0
    end
  end

  defp promotion_bonus(%Echecs.Move{promotion: nil}), do: 0
  defp promotion_bonus(%Echecs.Move{promotion: :queen}), do: 4
  defp promotion_bonus(%Echecs.Move{}), do: 2

  defp evaluate_position(game, history_games, nil, opts),
    do: evaluate_without_accumulator(game, history_games, opts)

  defp evaluate_position(game, history_games, accumulator, opts) do
    case Keyword.get(opts, :evaluator_weights) do
      %{w1: w1, b1: b1, w2: w2, b2: b2} ->
        accumulator
        |> EchecsEngine.SparseEvaluator.evaluate(w1, b1, w2, b2)
        |> Nx.squeeze()
        |> Nx.to_number()
        |> white_score_to_side_to_move(game)

      _other ->
        evaluate_without_accumulator(game, history_games, opts)
    end
  end

  defp evaluate_without_accumulator(game, history_games, opts) do
    case Keyword.get(opts, :evaluator_fn) do
      evaluator when is_function(evaluator, 1) ->
        evaluator.(game)

      _other ->
        case Keyword.get(opts, :inference) do
          nil ->
            0.0

          inference ->
            tensor =
              EchecsEngine.Tensor.to_tensor(game, history_games)
              |> Nx.squeeze(axes: [0])
              |> Nx.as_type(:f32)

            inference
            |> EchecsEngine.Inference.run(tensor)
            |> EchecsEngine.Value.output_to_q()
        end
    end
  end

  defp initial_accumulator(game, opts) do
    case Keyword.get(opts, :evaluator_weights) do
      %{feature_table: feature_table} ->
        EchecsEngine.Accumulator.from_game(game, feature_table: feature_table)

      _other ->
        nil
    end
  end

  defp next_accumulator(nil, _game, _move, _opts), do: nil

  defp next_accumulator(accumulator, game, move, opts) do
    case Keyword.get(opts, :evaluator_weights) do
      %{feature_table: feature_table} ->
        EchecsEngine.NNUE.Accumulator.update_after_move(
          accumulator,
          game,
          move,
          feature_table: feature_table
        )

      _other ->
        nil
    end
  end

  defp terminal_score(game, depth) do
    case Echecs.status(game) do
      :checkmate -> -100_000.0 + depth
      _other -> 0.0
    end
  end

  defp base_state(tt, deadline_ms, node_limit) do
    %{tt: tt, nodes: 0, deadline_ms: deadline_ms, node_limit: node_limit}
  end

  defp bump_nodes(state), do: %{state | nodes: state.nodes + 1}

  defp expired?(%{deadline_ms: deadline_ms, node_limit: node_limit, nodes: nodes}) do
    (is_integer(node_limit) and nodes >= node_limit) or
      (is_integer(deadline_ms) and System.monotonic_time(:millisecond) >= deadline_ms)
  end

  defp deadline_ms(opts) do
    case Keyword.get(opts, :movetime) do
      ms when is_integer(ms) -> System.monotonic_time(:millisecond) + max(ms - 2, 1)
      _other -> nil
    end
  end

  defp make_null_move(%Echecs.Game{} = game) do
    next_turn = Echecs.Piece.opponent(game.turn)
    next_en_passant = nil
    next_fullmove = if game.turn == :black, do: game.fullmove + 1, else: game.fullmove
    next_halfmove = game.halfmove + 1
    next_hash = Echecs.Zobrist.hash(game.board, next_turn, game.castling, next_en_passant)

    %Echecs.Game{
      game
      | turn: next_turn,
        en_passant: next_en_passant,
        halfmove: next_halfmove,
        fullmove: next_fullmove,
        history: [next_hash | game.history],
        zobrist_hash: next_hash
    }
  end

  defp lookup_tt(tt, game, history_games, depth, alpha, beta, opts) do
    key = transposition_key(game, history_games, opts)

    case :ets.lookup(tt, key) do
      [{_key, %{depth: cached_depth} = entry}] when cached_depth >= depth ->
        usable_tt_entry(entry, alpha, beta)

      _other ->
        nil
    end
  end

  defp usable_tt_entry(%{flag: :exact} = entry, _alpha, _beta), do: entry

  defp usable_tt_entry(%{flag: :lower, score: score} = entry, _alpha, beta) when score >= beta,
    do: entry

  defp usable_tt_entry(%{flag: :upper, score: score} = entry, alpha, _beta) when score <= alpha,
    do: entry

  defp usable_tt_entry(_entry, _alpha, _beta), do: nil

  defp store_tt(_tt, _game, _history_games, _depth, _score, [], _opts, _flag), do: :ok

  defp store_tt(tt, game, history_games, depth, score, pv_moves, opts, flag) do
    :ets.insert(
      tt,
      {transposition_key(game, history_games, opts),
       %{depth: depth, score: score, pv_moves: pv_moves, flag: flag}}
    )

    :ok
  end

  defp tt_flag(_score, _alpha, _beta, true), do: :lower
  defp tt_flag(score, original_alpha, _beta, _cutoff?) when score <= original_alpha, do: :upper
  defp tt_flag(_score, _alpha, _beta, _cutoff?), do: :exact

  defp fallback_result(game) do
    move = List.first(Echecs.legal_moves(game))
    %{best_move: move, score: 0.0, depth: 1, nodes: 1, pv: [move_to_uci(move)]}
  end

  defp fallback_root_result(game) do
    move = List.first(Echecs.legal_moves(game))
    %{best_move: move, score: @neg_inf, pv_moves: [move]}
  end

  defp move_to_uci(%Echecs.Move{} = move) do
    Echecs.Board.to_algebraic(move.from) <>
      Echecs.Board.to_algebraic(move.to) <>
      promotion_suffix(move.promotion)
  end

  defp promotion_suffix(nil), do: ""
  defp promotion_suffix(:queen), do: "q"
  defp promotion_suffix(:rook), do: "r"
  defp promotion_suffix(:bishop), do: "b"
  defp promotion_suffix(:knight), do: "n"

  defp white_score_to_side_to_move(score, %{turn: :white}), do: score
  defp white_score_to_side_to_move(score, %{turn: :black}), do: -score

  defp evaluator_identity(opts) do
    cond do
      Keyword.has_key?(opts, :evaluator_id) ->
        {:evaluator_id, Keyword.fetch!(opts, :evaluator_id)}

      Keyword.has_key?(opts, :evaluator_fn) ->
        {:evaluator_fn, function_identity(Keyword.fetch!(opts, :evaluator_fn))}

      Keyword.has_key?(opts, :evaluator_weights) ->
        {:evaluator_weights, :erlang.phash2(Keyword.fetch!(opts, :evaluator_weights))}

      true ->
        :none
    end
  end

  defp inference_identity(opts) do
    cond do
      Keyword.has_key?(opts, :inference_id) ->
        {:inference_id, Keyword.fetch!(opts, :inference_id)}

      Keyword.has_key?(opts, :inference) ->
        {:inference, function_or_term_identity(Keyword.fetch!(opts, :inference))}

      true ->
        :none
    end
  end

  defp function_or_term_identity(fun) when is_function(fun, 1), do: function_identity(fun)
  defp function_or_term_identity(term), do: :erlang.phash2(term)

  defp function_identity(fun) do
    %{
      module: :erlang.fun_info(fun, :module) |> elem(1),
      index: :erlang.fun_info(fun, :index) |> elem(1),
      uniq: :erlang.fun_info(fun, :new_uniq) |> elem(1),
      env_hash: :erlang.fun_info(fun, :env) |> elem(1) |> :erlang.phash2()
    }
  end
end
