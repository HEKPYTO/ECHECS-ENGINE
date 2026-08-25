# credo:disable-for-this-file Credo.Check.Refactor.CyclomaticComplexity
# credo:disable-for-this-file Credo.Check.Refactor.Nesting
# credo:disable-for-this-file Credo.Check.Refactor.FunctionArity
defmodule EchecsEngine.Search do
  @moduledoc false

  require Echecs.Move
  import Bitwise
  alias Echecs.Bitboard.{Magic, Precomputed}
  alias EchecsEngine.Eval

  @mate 32_000
  @inf 32_001
  @tt_slots 262_144
  @null_min_depth 4
  @reverse_futility_depth 1
  @reverse_futility_margin 300
  @forward_futility_margin 300
  @see_pruning_depth 2
  @see_pruning_margin 200
  @allowed [
    :depth,
    :nodes,
    :movetime,
    :reporter,
    :stop_ref,
    :eval,
    :tt_slots,
    :trace,
    :unordered,
    :exact_depth,
    :disable_lmr,
    :disable_null,
    :disable_futility,
    :disable_see_pruning,
    :root_window
  ]

  @spec best_move(Echecs.Game.t(), keyword()) ::
          {:ok, integer(), map()} | {:terminal, atom()} | {:error, term()}
  def best_move(game, opts \\ []) do
    with :ok <- validate_opts(opts), {:ok, limits} <- limits(opts) do
      moves = Echecs.MoveGen.legal_moves_int(game)

      cond do
        moves == [] -> {:terminal, terminal(game)}
        Echecs.Game.draw?(game) -> {:terminal, :draw}
        opts[:exact_depth] -> exact_best_move(game, opts, limits)
        true -> iterative(game, opts, limits)
      end
    end
  rescue
    error -> {:error, error}
  end

  defp exact_best_move(game, opts, limits) do
    accumulator = if is_function(opts[:eval], 1), do: nil, else: Eval.refresh(game, evaluator())
    {score, move, nodes} = exact_root(game, accumulator, limits.max_depth, opts, 0)

    {:ok, move,
     %{
       score: score,
       depth: limits.max_depth,
       seldepth: limits.max_depth,
       nodes: nodes,
       time_ms: 0,
       pv: [EchecsEngine.Move.to_uci(move)],
       tt_hits: 0,
       tt_slots: 0,
       tt_entries: 0
     }}
  end

  defp exact_root(game, accumulator, depth, opts, ply) do
    moves = Echecs.MoveGen.legal_moves_int(game)

    {score, move, nodes} =
      Enum.reduce(moves, {-@inf, nil, 0}, fn move, {best, best_move, nodes} ->
        child = Echecs.Game.make_move_int(game, move)

        {child_score, child_nodes} =
          exact_negamax(
            child,
            child_accumulator(accumulator, game, move, child, opts),
            depth - 1,
            opts,
            ply + 1
          )

        score = -child_score

        if score > best or (score == best and (is_nil(best_move) or move < best_move)),
          do: {score, move, nodes + child_nodes},
          else: {best, best_move, nodes + child_nodes}
      end)

    {score, move, nodes}
  end

  defp exact_negamax(game, accumulator, depth, opts, ply) do
    moves = Echecs.MoveGen.legal_moves_int(game)

    cond do
      moves == [] ->
        {terminal_score(game, ply), 1}

      Echecs.Game.draw?(game) ->
        {0, 1}

      depth == 0 ->
        {evaluate(game, accumulator, opts), 1}

      true ->
        Enum.reduce(moves, {-@inf, 1}, fn move, {best, nodes} ->
          child_game = Echecs.Game.make_move_int(game, move)

          {child_score, child_nodes} =
            exact_negamax(
              child_game,
              child_accumulator(accumulator, game, move, child_game, opts),
              depth - 1,
              opts,
              ply + 1
            )

          {max(best, -child_score), nodes + child_nodes}
        end)
    end
  end

  defp iterative(game, opts, limits) do
    moves = order_moves(game, Echecs.MoveGen.legal_moves_int(game))
    started = System.monotonic_time(:millisecond)
    slots = opts[:tt_slots] || @tt_slots
    tt = :ets.new(__MODULE__, [:set, :private, read_concurrency: true])

    state = %{
      nodes: 0,
      seldepth: 0,
      started: started,
      limits: limits,
      opts: opts,
      tt: tt,
      tt_slots: slots,
      tt_hits: 0,
      tt_stores: 0,
      age: 0,
      history: %{},
      killers: %{},
      null?: false,
      rep_context: repetition_context(game)
    }

    accumulator = if is_function(opts[:eval], 1), do: nil, else: Eval.refresh(game, evaluator())

    {fallback_move, _stage, _see} = hd(moves)

    fallback = %{
      best: fallback_move,
      score: 0,
      depth: 0,
      seldepth: 0,
      pv: [fallback_move]
    }

    try do
      {last, state} =
        Enum.reduce_while(1..limits.max_depth, {fallback, state}, fn depth, {last, current} ->
          case root(game, accumulator, depth, current) do
            {:stop, stopped} ->
              {:halt, {last, stopped}}

            {next, next_state} ->
              report(next, next_state, opts)
              {:cont, {next, next_state}}
          end
        end)

      {:ok, last.best,
       %{
         score: last.score,
         depth: last.depth,
         seldepth: last.seldepth,
         nodes: state.nodes,
         time_ms: max(System.monotonic_time(:millisecond) - started, 0),
         pv: Enum.map(last.pv, &EchecsEngine.Move.to_uci/1),
         tt_hits: state.tt_hits,
         tt_slots: slots,
         tt_entries: :ets.info(tt, :size)
       }}
    after
      :ets.delete(tt)
    end
  end

  defp root(game, accumulator, depth, state) do
    {root_alpha, _root_beta} = root_window(state.opts)

    Enum.reduce_while(
      order_moves(game, Echecs.MoveGen.legal_moves_int(game), nil, state, 0),
      {nil, root_alpha, state},
      fn {move, _stage, _see}, {best, alpha, current} ->
        if stopped?(current),
          do: {:halt, {:stop, current}},
          else: root_move(game, accumulator, move, depth, best, alpha, current)
      end
    )
    |> case do
      {:stop, state} -> {:stop, state}
      {best, _alpha, state} -> {best, state}
    end
  end

  defp root_move(game, accumulator, move, depth, best, alpha, state) do
    child = Echecs.Game.make_move_int(game, move)
    child_accumulator = child_accumulator(accumulator, game, move, child, state.opts)
    pv? = is_nil(best)

    reduced? =
      not is_nil(best) and state.opts[:disable_lmr] != true and depth >= 4 and
        not tactical?(game, move) and not Echecs.Game.in_check?(game) and
        not Echecs.Game.in_check?(child)

    child_depth = if reduced?, do: depth - 2, else: depth - 1
    if reduced?, do: trace(state, {:lmr_reduce, 0, move, depth - 1, child_depth})

    {child_alpha, child_beta} = if pv?, do: {-@inf, -alpha}, else: {-alpha - 1, -alpha}

    case negamax_child(child, child_accumulator, child_depth, child_alpha, child_beta, 1, state) do
      {:stop, state} ->
        {:halt, {:stop, state}}

      {raw_score, pv, state} ->
        score = -raw_score

        tie_research? = not pv? and move < best.best

        result =
          if (score > alpha and (reduced? or not pv?)) or tie_research? do
            if reduced?, do: trace(state, {:lmr_research, 0, move, depth - 1, child_depth})

            {full_alpha, full_beta} =
              if tie_research?, do: {-@inf, @inf}, else: {-@inf, -alpha}

            case negamax_child(
                   child,
                   child_accumulator,
                   depth - 1,
                   full_alpha,
                   full_beta,
                   1,
                   state
                 ) do
              {full_score, full_pv, full_state} -> {-full_score, full_pv, full_state}
              {:stop, stopped_state} -> {:stop, stopped_state}
            end
          else
            {score, pv, state}
          end

        case result do
          {:stop, stopped_state} ->
            {:halt, {:stop, stopped_state}}

          {score, pv, state} ->
            candidate = %{
              best: move,
              score: score,
              depth: depth,
              seldepth: state.seldepth,
              pv: [move | pv]
            }

            {best, alpha} =
              if is_nil(best) or score > best.score or (score == best.score and move < best.best),
                do: {candidate, max(alpha, score)},
                else: {best, alpha}

            {:cont, {best, alpha, state}}
        end
    end
  end

  defp negamax(game, accumulator, depth, alpha, beta, ply, state) do
    state = bump(state, ply)
    moves = Echecs.MoveGen.legal_moves_int(game)
    {alpha, beta} = mate_window(alpha, beta, ply)

    cond do
      stopped?(state) ->
        {:stop, state}

      moves == [] ->
        {terminal_score(game, ply), [], state}

      Echecs.Game.draw?(game) ->
        {0, [], state}

      alpha >= beta ->
        {alpha, [], state}

      depth <= 0 and state.opts[:exact_depth] ->
        exact_leaf(game, accumulator, ply, state)

      depth <= 0 ->
        qsearch(game, accumulator, alpha, beta, ply, state)

      true ->
        static_eval = static_eval_for_node(game, accumulator, depth, alpha, beta, state)

        case reverse_futility(game, depth, alpha, beta, ply, static_eval, state) do
          {:cut, score} ->
            {score, [], state}

          :search ->
            null_or_search(
              game,
              accumulator,
              moves,
              depth,
              alpha,
              beta,
              ply,
              static_eval,
              state
            )
        end
    end
  end

  defp null_or_search(game, accumulator, moves, depth, alpha, beta, ply, static_eval, state) do
    allowed? = null_allowed?(game, depth, alpha, beta, static_eval, state)

    if is_pid(state.opts[:trace]) and depth >= @null_min_depth do
      trace(state, {
        :null_guard,
        ply,
        depth,
        alpha,
        beta,
        static_eval,
        state.null?,
        Echecs.Game.in_check?(game),
        non_pawn_material?(game, game.turn),
        allowed?
      })
    end

    if allowed? do
      trace(state, {:null_try, ply, depth})
      null = make_null_move(game)

      trace(state, {
        :null_position,
        game.turn,
        null.turn,
        game.en_passant,
        null.en_passant,
        game.halfmove,
        null.halfmove,
        game.fullmove,
        null.fullmove,
        null.zobrist_hash,
        length(null.history)
      })

      case negamax(null, accumulator, depth - 3, -beta, -beta + 1, ply + 1, %{state | null?: true}) do
        {child_score, _pv, null_state} when -child_score >= beta ->
          trace(state, {:null_fail_high, ply, depth})
          trace(state, {:null_verify, ply, depth})

          case negamax(game, accumulator, depth - 2, alpha, beta, ply, %{null_state | null?: true}) do
            {score, pv, verified} when score >= beta ->
              trace(state, {:null_cutoff, ply, depth})
              {score, pv, %{verified | null?: false}}

            {_score, _pv, verified} ->
              search_tt_node(
                game,
                accumulator,
                moves,
                depth,
                alpha,
                beta,
                ply,
                static_eval,
                %{verified | null?: false}
              )

            {:stop, verified} ->
              {:stop, %{verified | null?: false}}
          end

        {:stop, stopped} ->
          {:stop, %{stopped | null?: false}}

        {_score, _pv, continued} ->
          search_tt_node(
            game,
            accumulator,
            moves,
            depth,
            alpha,
            beta,
            ply,
            static_eval,
            %{continued | null?: false}
          )
      end
    else
      search_tt_node(game, accumulator, moves, depth, alpha, beta, ply, static_eval, state)
    end
  end

  defp null_allowed?(game, depth, alpha, beta, static_eval, state) do
    depth >= @null_min_depth and state.null? != true and state.opts[:disable_null] != true and
      state.opts[:exact_depth] != true and
      beta - alpha <= 1 and not Echecs.Game.in_check?(game) and beta > -@mate + 1_000 and
      beta < @mate - 1_000 and static_eval >= beta and non_pawn_material?(game, game.turn)
  end

  defp static_eval_for_node(game, accumulator, depth, alpha, beta, state) do
    needed? =
      state.opts[:exact_depth] != true and beta - alpha <= 1 and
        ((state.opts[:disable_futility] != true and depth <= @reverse_futility_depth) or
           (state.opts[:disable_null] != true and depth >= @null_min_depth and state.null? != true))

    if needed?, do: evaluate(game, accumulator, state.opts), else: nil
  end

  defp reverse_futility(game, depth, alpha, beta, ply, static_eval, state) do
    margin = depth * @reverse_futility_margin

    allowed? =
      state.opts[:disable_futility] != true and state.opts[:exact_depth] != true and
        depth <= @reverse_futility_depth and beta - alpha <= 1 and
        not Echecs.Game.in_check?(game) and beta > -@mate + 1_000 and beta < @mate - 1_000 and
        non_pawn_material?(game, game.turn) and static_eval - margin >= beta and
        Echecs.MoveGen.has_legal_move?(game)

    if allowed? do
      score = static_eval - margin
      trace(state, {:reverse_futility, ply, depth, static_eval, beta})
      {:cut, score}
    else
      :search
    end
  end

  defp non_pawn_material?(game, color) do
    Enum.any?(0..63, fn square ->
      case Echecs.Board.at_tuple(game.board, square) do
        {^color, type} when type in [:knight, :bishop, :rook, :queen] -> true
        _ -> false
      end
    end)
  end

  defp make_null_move(game) do
    turn = opponent(game.turn)
    hash = Echecs.Zobrist.hash(game.board, turn, game.castling, nil)

    %{
      game
      | turn: turn,
        en_passant: nil,
        halfmove: game.halfmove + 1,
        fullmove: if(game.turn == :black, do: game.fullmove + 1, else: game.fullmove),
        zobrist_hash: hash,
        history: [hash]
    }
  end

  defp exact_leaf(game, accumulator, ply, state) do
    if Echecs.MoveGen.legal_moves_int(game) == [],
      do: {terminal_score(game, ply), [], state},
      else: {evaluate(game, accumulator, state.opts), [], state}
  end

  defp search_tt_node(game, accumulator, moves, depth, alpha, beta, ply, static_eval, state) do
    if state.null? do
      pvs(
        game,
        accumulator,
        order_moves(game, moves, nil, state, ply),
        depth,
        alpha,
        beta,
        ply,
        static_eval,
        state,
        [],
        initial_best_score(alpha, state),
        false
      )
    else
      search_tt_node_with_table(
        game,
        accumulator,
        moves,
        depth,
        alpha,
        beta,
        ply,
        static_eval,
        state
      )
    end
  end

  defp search_tt_node_with_table(
         game,
         accumulator,
         moves,
         depth,
         alpha,
         beta,
         ply,
         static_eval,
         state
       ) do
    hash = game.zobrist_hash
    key = tt_key(game, state.rep_context)
    trace(state, {:tt_context, ply, hash, key})

    case probe_tt(state, hash, key, depth, alpha, beta, ply) do
      {:cut, score, move, state} ->
        {score, if(move, do: [move], else: []), state}

      {:move, move, state} ->
        store_tt_result(
          pvs(
            game,
            accumulator,
            order_moves(game, moves, move, state, ply),
            depth,
            alpha,
            beta,
            ply,
            static_eval,
            state,
            [],
            initial_best_score(alpha, state),
            false
          ),
          state,
          hash,
          key,
          depth,
          alpha,
          beta,
          ply
        )

      {:miss, state} ->
        store_tt_result(
          pvs(
            game,
            accumulator,
            order_moves(game, moves, nil, state, ply),
            depth,
            alpha,
            beta,
            ply,
            static_eval,
            state,
            [],
            initial_best_score(alpha, state),
            false
          ),
          state,
          hash,
          key,
          depth,
          alpha,
          beta,
          ply
        )
    end
  end

  defp store_tt_result({:stop, state}, _old, _hash, _key, _depth, _alpha, _beta, _ply),
    do: {:stop, state}

  defp store_tt_result({score, pv, state}, _old, hash, key, depth, alpha, beta, ply) do
    bound = if score <= alpha, do: :upper, else: if(score >= beta, do: :lower, else: :exact)
    {score, pv, store_tt(state, hash, key, depth, bound, score, List.first(pv), ply)}
  end

  defp pvs(
         _game,
         _accumulator,
         [],
         _depth,
         _alpha,
         _beta,
         _ply,
         _static_eval,
         state,
         pv,
         best_score,
         _searched
       ),
       do: {best_score, pv, state}

  defp pvs(
         game,
         accumulator,
         [{move, stage, see_score} | rest],
         depth,
         alpha,
         beta,
         ply,
         static_eval,
         state,
         best_pv,
         best_score,
         searched
       ) do
    child = Echecs.Game.make_move_int(game, move)
    child_accumulator = child_accumulator(accumulator, game, move, child, state.opts)

    cond do
      see_prunable?(game, child, move, stage, see_score, depth, alpha, beta, ply, state, searched) ->
        trace(state, {:see_prune, ply, depth, move, see_score})

        pvs(
          game,
          accumulator,
          rest,
          depth,
          alpha,
          beta,
          ply,
          static_eval,
          state,
          best_pv,
          best_score,
          true
        )

      forward_futility?(game, child, move, depth, alpha, beta, static_eval, state, searched) ->
        trace(state, {:forward_futility, ply, depth, move, static_eval, alpha})

        pvs(
          game,
          accumulator,
          rest,
          depth,
          alpha,
          beta,
          ply,
          static_eval,
          state,
          best_pv,
          best_score,
          true
        )

      true ->
        {child_alpha, child_beta} =
          if searched,
            do: {-alpha - 1, -alpha},
            else: {-beta, -alpha}

        reduced? =
          searched and state.opts[:disable_lmr] != true and state.opts[:exact_depth] != true and
            depth >= 4 and
            not tactical?(game, move) and not Echecs.Game.in_check?(game) and
            not Echecs.Game.in_check?(child)

        child_depth = if reduced?, do: depth - 2, else: depth - 1
        if reduced?, do: trace(state, {:lmr_reduce, ply, move, depth - 1, child_depth})

        case negamax_child(
               child,
               child_accumulator,
               child_depth,
               child_alpha,
               child_beta,
               ply + 1,
               state
             ) do
          {:stop, state} ->
            {:stop, state}

          {raw_score, child_pv, state} ->
            score = -raw_score

            tie_research? =
              searched and score == alpha and
                match?([best_move | _] when move < best_move, best_pv)

            re_search? =
              (searched and score > alpha and score < beta) or (reduced? and score > alpha) or
                tie_research?

            if reduced? and score > alpha,
              do: trace(state, {:lmr_research, ply, move, depth - 1, child_depth})

            result =
              if re_search?,
                do:
                  negamax_child(
                    child,
                    child_accumulator,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    state
                  ),
                else: {raw_score, child_pv, state}

            case result do
              {:stop, state} ->
                {:stop, state}

              {rescore, repv, state} ->
                score = if re_search?, do: -rescore, else: score

                {best_score, pv} =
                  if score > best_score or
                       (score == best_score and
                          (best_pv == [] or move < List.first(best_pv))),
                     do: {score, [move | repv]},
                     else: {best_score, best_pv}

                alpha = max(alpha, score)

                if alpha >= beta,
                  do: {best_score, pv, quiet_cutoff(state, game, move, ply, depth)},
                  else:
                    pvs(
                      game,
                      accumulator,
                      rest,
                      depth,
                      alpha,
                      beta,
                      ply,
                      static_eval,
                      state,
                      pv,
                      best_score,
                      true
                    )
            end
        end
    end
  end

  defp see_prunable?(
         game,
         child,
         move,
         stage,
         see_score,
         depth,
         alpha,
         beta,
         ply,
         state,
         searched
       ) do
    allowed? =
      state.opts[:disable_see_pruning] != true and state.opts[:exact_depth] != true and
        depth in 1..@see_pruning_depth and searched and beta - alpha <= 1 and
        not Echecs.Game.in_check?(game) and not Echecs.Game.in_check?(child) and
        beta > -@mate + 1_000 and beta < @mate - 1_000 and non_pawn_material?(game, game.turn) and
        stage == :bad_capture and is_integer(see_score) and
        see_score < -depth * @see_pruning_margin and Echecs.Move.unpack_promotion(move) == nil and
        Echecs.Move.unpack_special(move) != :en_passant

    if stage == :tt, do: trace(state, {:see_prune_guard, ply, move, :tt, allowed?})
    allowed?
  end

  defp forward_futility?(game, child, move, depth, alpha, beta, static_eval, state, searched) do
    state.opts[:disable_futility] != true and state.opts[:exact_depth] != true and depth == 1 and
      searched and beta - alpha <= 1 and not Echecs.Game.in_check?(game) and
      not Echecs.Game.in_check?(child) and beta > -@mate + 1_000 and beta < @mate - 1_000 and
      non_pawn_material?(game, game.turn) and not tactical?(game, move) and
      static_eval + @forward_futility_margin <= alpha
  end

  defp qsearch(game, accumulator, alpha, beta, ply, state) do
    moves = Echecs.MoveGen.legal_moves_int(game)

    cond do
      moves == [] ->
        {terminal_score(game, ply), [], state}

      Echecs.Game.in_check?(game) ->
        pvs(
          game,
          accumulator,
          order_moves(game, moves, nil, state, ply),
          0,
          alpha,
          beta,
          ply,
          nil,
          state,
          [],
          initial_best_score(alpha, state),
          false
        )

      true ->
        stand_pat = evaluate(game, accumulator, state.opts)

        if stand_pat >= beta,
          do: {stand_pat, [], state},
          else:
            qsearch_captures(
              game,
              accumulator,
              Enum.filter(moves, &tactical?(game, &1)),
              alpha,
              beta,
              ply,
              state,
              stand_pat
            )
    end
  end

  defp qsearch_captures(_game, _accumulator, [], _alpha, _beta, _ply, state, stand_pat),
    do: {stand_pat, [], state}

  defp qsearch_captures(game, accumulator, captures, alpha, beta, ply, state, stand_pat) do
    pvs(
      game,
      accumulator,
      order_moves(game, captures, nil, state, ply),
      0,
      max(alpha, stand_pat),
      beta,
      ply,
      stand_pat,
      state,
      [],
      stand_pat,
      false
    )
  end

  defp terminal_score(game, ply), do: if(Echecs.Game.in_check?(game), do: -@mate + ply, else: 0)

  defp mate_window(alpha, beta, ply),
    do: {max(alpha, -@mate + ply), min(beta, @mate - ply)}

  defp root_window(opts) do
    case opts[:root_window] do
      {alpha, beta} when is_integer(alpha) and is_integer(beta) and alpha < beta -> {alpha, beta}
      nil -> {-@inf, @inf}
    end
  end

  defp initial_best_score(_alpha, _state), do: -@inf

  defp evaluate(game, accumulator, opts) when is_list(opts) do
    case opts[:eval] do
      eval when is_function(eval, 1) -> eval.(game)
      _ -> evaluate(game, accumulator, %{})
    end
  end

  defp evaluate(game, _accumulator, %{eval: eval}) when is_function(eval, 1), do: eval.(game)

  defp evaluate(game, accumulator, _opts), do: Eval.evaluate(game, accumulator, evaluator())

  defp child_accumulator(nil, _game, _move, _child, _opts), do: nil

  defp child_accumulator(accumulator, game, move, child, _opts),
    do: Eval.update(accumulator, game, move, child, evaluator())

  defp negamax_child(child, accumulator, depth, alpha, beta, ply, state) do
    parent_context = state.rep_context
    child_context = next_repetition_context(parent_context, child)

    trace(
      state,
      {:tt_context_roll, ply, parent_context, child_context, child.halfmove == 0,
       child.zobrist_hash}
    )

    case negamax(child, accumulator, depth, alpha, beta, ply, %{
           state
           | rep_context: child_context
         }) do
      {:stop, child_state} -> {:stop, %{child_state | rep_context: parent_context}}
      {score, pv, child_state} -> {score, pv, %{child_state | rep_context: parent_context}}
    end
  end

  defp evaluator do
    key = {__MODULE__, :weights}

    case :persistent_term.get(key, nil) do
      nil ->
        weights = Eval.load!(Application.app_dir(:echecs_engine, "priv/echecs.nnue"))
        :persistent_term.put(key, weights)
        weights

      weights ->
        weights
    end
  end

  defp order_moves(game, moves, tt_move, state, ply) do
    ordered =
      moves
      |> Enum.map(&move_entry(game, &1, tt_move))
      |> order_entries(game, state, ply)

    trace(
      state,
      {:order, ply,
       Enum.map(ordered, fn {move, stage, see_score} -> {stage, move, see_score} end)}
    )

    ordered
  end

  defp order_moves(game, moves),
    do: order_moves(game, moves, nil, %{opts: %{}, history: %{}, killers: %{}}, 0)

  defp move_entry(game, move, tt_move) do
    see_score = if tactical?(game, move), do: see(game, move), else: nil

    stage =
      cond do
        move == tt_move -> :tt
        is_nil(see_score) -> :quiet
        see_score >= 0 -> :good_capture
        true -> :bad_capture
      end

    {move, stage, see_score}
  end

  defp order_entries(entries, game, state, ply) do
    if state.opts[:unordered] do
      entries
    else
      {tt, rest} = Enum.split_with(entries, &(elem(&1, 1) == :tt))
      {captures, quiets} = Enum.split_with(rest, &(elem(&1, 1) != :quiet))
      {good, bad} = Enum.split_with(captures, &(elem(&1, 2) >= 0))

      good =
        Enum.sort_by(good, fn {move, _stage, see_score} -> {-see_score, -mvv(game, move)} end)

      bad = Enum.sort_by(bad, &elem(&1, 2))
      quiets = Enum.sort_by(quiets, fn {move, _stage, _see} -> quiet_rank(state, ply, move) end)
      tt ++ good ++ quiets ++ bad
    end
  end

  defp quiet_rank(state, ply, move),
    do:
      {if(move in Map.get(state.killers, ply, []), do: 0, else: 1),
       -Map.get(state.history, history_key(move), 0)}

  defp tactical?(game, move),
    do:
      not is_nil(Echecs.Board.at_tuple(game.board, Echecs.Move.unpack_to(move))) or
        Echecs.Move.unpack_promotion(move) != nil or
        Echecs.Move.unpack_special(move) == :en_passant

  defp see(game, move) do
    from = Echecs.Move.unpack_from(move)
    to = Echecs.Move.unpack_to(move)
    moved = Echecs.Board.at_tuple(game.board, from)
    captured = captured_piece(game, move)
    promotion = Echecs.Move.unpack_promotion(move)

    if is_nil(moved) or (is_nil(captured) and is_nil(promotion)) do
      0
    else
      board = Echecs.Board.make_move_on_board_tuple(game.board, move, game.turn)
      moved_value = value(promotion || elem(moved, 1))

      value(captured && elem(captured, 1)) + moved_value - value(elem(moved, 1)) -
        exchange(board, opponent(game.turn), to, moved_value)
    end
  end

  defp exchange(board, side, target, victim_value) do
    case least_attacker(board, side, target) do
      nil ->
        0

      {from, type} ->
        move = Echecs.Move.pack(from, target, nil, nil)
        next = Echecs.Board.make_move_on_board_tuple(board, move, side)
        max(0, victim_value - exchange(next, opponent(side), target, value(type)))
    end
  end

  defp least_attacker(board, side, target) do
    0..63
    |> Enum.reduce([], fn square, attackers ->
      case Echecs.Board.at_tuple(board, square) do
        {^side, type} ->
          if attacks?(board, square, type, side, target) and
               legal_exchange_attacker?(board, side, square, target),
             do: [{square, type} | attackers],
             else: attackers

        _ ->
          attackers
      end
    end)
    |> Enum.min_by(fn {_square, type} -> value(type) end, fn -> nil end)
  end

  defp legal_exchange_attacker?(board, side, from, target) do
    next =
      Echecs.Board.make_move_on_board_tuple(board, Echecs.Move.pack(from, target, nil, nil), side)

    {white_king, black_king} = Echecs.Game.king_positions(next)
    king = if(side == :white, do: white_king, else: black_king)

    is_integer(king) and not Echecs.Board.attacked?(next, king, opponent(side))
  end

  defp attacks?(_board, from, :pawn, color, target),
    do: (Precomputed.get_pawn_attacks(from, color) &&& 1 <<< target) != 0

  defp attacks?(_board, from, :knight, _color, target),
    do: (Precomputed.get_knight_attacks(from) &&& 1 <<< target) != 0

  defp attacks?(_board, from, :king, _color, target),
    do: (Precomputed.get_king_attacks(from) &&& 1 <<< target) != 0

  defp attacks?(board, from, :bishop, _color, target),
    do: (Magic.get_bishop_attacks(from, elem(board, 14)) &&& 1 <<< target) != 0

  defp attacks?(board, from, :rook, _color, target),
    do: (Magic.get_rook_attacks(from, elem(board, 14)) &&& 1 <<< target) != 0

  defp attacks?(board, from, :queen, _color, target),
    do:
      attacks?(board, from, :bishop, :white, target) or
        attacks?(board, from, :rook, :white, target)

  defp captured_piece(game, move) do
    if Echecs.Move.unpack_special(move) == :en_passant do
      square = Echecs.Move.unpack_to(move) + if(game.turn == :white, do: 8, else: -8)
      Echecs.Board.at_tuple(game.board, square)
    else
      Echecs.Board.at_tuple(game.board, Echecs.Move.unpack_to(move))
    end
  end

  defp mvv(game, move),
    do: value(captured_piece(game, move) && elem(captured_piece(game, move), 1))

  defp value(nil), do: 0
  defp value(:pawn), do: 100
  defp value(:knight), do: 320
  defp value(:bishop), do: 330
  defp value(:rook), do: 500
  defp value(:queen), do: 900
  defp value(:king), do: 20_000
  defp opponent(:white), do: :black
  defp opponent(:black), do: :white
  defp history_key(move), do: {Echecs.Move.unpack_from(move), Echecs.Move.unpack_to(move)}

  defp quiet_cutoff(state, game, move, ply, depth) do
    if tactical?(game, move) do
      state
    else
      killers =
        [move | Enum.reject(Map.get(state.killers, ply, []), &(&1 == move))] |> Enum.take(2)

      history =
        Map.update(
          state.history,
          history_key(move),
          depth * depth,
          &min(&1 + depth * depth, 1_000_000)
        )

      trace(state, {:quiet_cutoff, ply, move, killers})
      %{state | killers: Map.put(state.killers, ply, killers), history: history}
    end
  end

  defp trace(%{opts: opts}, event) do
    cond do
      is_pid(opts[:trace]) -> send(opts[:trace], {:search_trace, event})
      is_function(opts[:trace], 1) -> opts[:trace].(event)
      true -> :ok
    end
  end

  defp trace(_, _), do: :ok

  defp bump(state, ply), do: %{state | nodes: state.nodes + 1, seldepth: max(state.seldepth, ply)}

  defp stopped?(state),
    do:
      state.nodes >= state.limits.nodes or
        System.monotonic_time(:millisecond) >= state.limits.deadline or
        atomics_stopped?(state.opts[:stop_ref])

  defp atomics_stopped?(nil), do: false
  defp atomics_stopped?(ref), do: :atomics.get(ref, 1) != 0

  defp probe_tt(state, hash, key, depth, alpha, beta, ply) do
    index = rem(hash, state.tt_slots)

    case :ets.lookup(state.tt, index) do
      [{^index, ^key, stored_depth, _age, bound, stored_score, move}]
      when stored_depth >= depth ->
        score = from_tt(stored_score, ply)
        state = %{state | tt_hits: state.tt_hits + 1}

        if bound == :exact or (bound == :lower and score >= beta) or
             (bound == :upper and score <= alpha),
           do: {:cut, score, move, state},
           else: {:move, move, state}

      [{^index, ^key, _stored_depth, _age, _bound, _score, move}] ->
        {:move, move, %{state | tt_hits: state.tt_hits + 1}}

      _ ->
        {:miss, state}
    end
  end

  defp store_tt(state, hash, key, depth, bound, score, move, ply) do
    index = rem(hash, state.tt_slots)
    age = state.age + 1
    entry = {index, key, depth, age, bound, to_tt(score, ply), move}
    trace(state, {:tt_store, ply, depth, bound, score})

    case :ets.lookup(state.tt, index) do
      [] ->
        :ets.insert(state.tt, entry)

      [{^index, _old_key, old_depth, old_age, _bound, _score, _move}]
      when depth > old_depth or age - old_age > 8 ->
        :ets.insert(state.tt, entry)

      _ ->
        :ok
    end

    %{state | age: age, tt_stores: state.tt_stores + 1}
  end

  defp tt_key(game, rep_context), do: {game.zobrist_hash, game.halfmove, rep_context}

  defp repetition_context(game) do
    game.history
    |> Enum.take(game.halfmove + 1)
    |> Enum.reverse()
    |> Enum.reduce(0, &mix_repetition_context(&2, &1))
  end

  defp next_repetition_context(_parent, game) when game.halfmove == 0,
    do: mix_repetition_context(0, game.zobrist_hash)

  defp next_repetition_context(parent, game),
    do: mix_repetition_context(parent, game.zobrist_hash)

  defp mix_repetition_context(context, zobrist),
    do: :erlang.phash2({context, zobrist}, 2_147_483_647)

  defp to_tt(score, ply) when score > @mate - 1_000, do: score + ply
  defp to_tt(score, ply) when score < -@mate + 1_000, do: score - ply
  defp to_tt(score, _ply), do: score
  defp from_tt(score, ply) when score > @mate - 1_000, do: score - ply
  defp from_tt(score, ply) when score < -@mate + 1_000, do: score + ply
  defp from_tt(score, _ply), do: score

  defp report(last, state, opts) do
    if is_function(opts[:reporter], 1) do
      opts[:reporter].(%{
        depth: last.depth,
        seldepth: last.seldepth,
        score: last.score,
        nodes: state.nodes,
        time_ms: max(System.monotonic_time(:millisecond) - state.started, 0),
        pv: Enum.map(last.pv, &EchecsEngine.Move.to_uci/1)
      })
    end
  end

  defp limits(opts) do
    keys = Enum.filter([:depth, :nodes, :movetime], &Keyword.has_key?(opts, &1))
    now = System.monotonic_time(:millisecond)

    cond do
      length(keys) > 1 ->
        {:error, :ambiguous_limit}

      opts[:depth] && (not is_integer(opts[:depth]) or opts[:depth] < 1) ->
        {:error, :invalid_depth}

      opts[:nodes] && (not is_integer(opts[:nodes]) or opts[:nodes] < 1) ->
        {:error, :invalid_nodes}

      opts[:movetime] && (not is_integer(opts[:movetime]) or opts[:movetime] < 1) ->
        {:error, :invalid_movetime}

      opts[:tt_slots] && (not is_integer(opts[:tt_slots]) or opts[:tt_slots] < 1) ->
        {:error, :invalid_tt_slots}

      Keyword.has_key?(opts, :root_window) and
          not match?(
            {alpha, beta} when is_integer(alpha) and is_integer(beta) and alpha < beta,
            opts[:root_window]
          ) ->
        {:error, :invalid_root_window}

      true ->
        {:ok,
         %{
           max_depth: opts[:depth] || if(opts[:nodes] || opts[:movetime], do: 64, else: 4),
           nodes: opts[:nodes] || 9_223_372_036_854_775_807,
           deadline: now + (opts[:movetime] || 9_223_372_036_854_775_807)
         }}
    end
  end

  defp validate_opts(opts) when is_list(opts) do
    case Enum.find(opts, fn {key, _} -> key not in @allowed end) do
      nil -> :ok
      {key, _} -> {:error, {:unknown_option, key}}
    end
  end

  defp validate_opts(_), do: {:error, :invalid_options}
  defp terminal(game), do: if(Echecs.Game.in_check?(game), do: :checkmate, else: :stalemate)
end
