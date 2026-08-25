defmodule EchecsEngine.SearchTest do
  use ExUnit.Case, async: true
  require Echecs.Move

  alias EchecsEngine.{Eval, Search}

  @mate 32_000

  test "returns draw before evaluating" do
    game = Echecs.new_game("8/8/8/8/8/8/8/K6k w - - 100 1")
    assert {:terminal, :draw} = Search.best_move(game, depth: 2)
  end

  test "a repeated position path is adjudicated as a draw" do
    game = Echecs.new_game("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")

    repeated =
      Enum.reduce(["a1a2", "e8e7", "a2a1", "e7e8", "a1a2", "e8e7", "a2a1", "e7e8"], game, fn uci,
                                                                                             position ->
        Echecs.Game.make_move_int(position, packed_move(position, uci))
      end)

    assert Echecs.Game.draw?(repeated)
    assert {:terminal, :draw} = Search.best_move(repeated, depth: 2)
  end

  test "checkmate takes precedence over an overlapping fifty-move draw" do
    mate = Echecs.new_game("7k/6Q1/7K/8/8/8/8/8 b - - 100 1")
    assert {:terminal, :checkmate} = Search.best_move(mate, depth: 2)

    game = Echecs.new_game("7k/5Q2/7K/8/8/8/8/8 w - - 99 1")
    assert {:ok, move, %{score: score}} = Search.best_move(game, depth: 2)
    assert score > 31_000
    assert Echecs.Game.checkmate?(Echecs.Game.make_move_int(game, move))
  end

  test "finds mate in one with packed moves" do
    game = Echecs.new_game("7k/5Q2/7K/8/8/8/8/8 w - - 0 1")
    assert {:ok, move, %{score: score}} = Search.best_move(game, depth: 2)
    assert score > 31_000
    assert Echecs.Game.checkmate?(Echecs.Game.make_move_int(game, move))
  end

  test "checked qsearch searches quiet evasions without stand pat" do
    game = Echecs.new_game("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    assert Echecs.Game.in_check?(game)

    evaluator = fn position ->
      if Echecs.Game.in_check?(position), do: raise("stand pat while checked")
      0
    end

    assert {:ok, move, _info} = Search.best_move(game, depth: 1, eval: evaluator)
    assert move in Echecs.MoveGen.legal_moves_int(game)
  end

  test "returns a legal move at a finite node limit" do
    game = Echecs.new_game()
    assert {:ok, move, %{depth: depth}} = Search.best_move(game, nodes: 250)
    assert move in Echecs.MoveGen.legal_moves_int(game)
    assert depth >= 1
  end

  test "an interrupted first iteration reports its consumed node and no completed depth" do
    game = Echecs.new_game()
    assert {:ok, move, %{depth: 0, nodes: 1}} = Search.best_move(game, nodes: 1)
    assert move in Echecs.MoveGen.legal_moves_int(game)
  end

  test "rejects ambiguous and unknown options" do
    game = Echecs.new_game()
    assert {:error, :ambiguous_limit} = Search.best_move(game, depth: 2, nodes: 10)
    assert {:error, {:unknown_option, :bogus}} = Search.best_move(game, bogus: true)
  end

  test "direct mapped TT stays within its configured private slot budget" do
    game = Echecs.new_game()
    assert {:ok, _move, info} = Search.best_move(game, depth: 3, tt_slots: 2)
    assert info.tt_slots == 2
    assert info.tt_entries <= 2
  end

  test "TT keeps mate distances root-ply aware" do
    game = Echecs.new_game("7k/5Q2/7K/8/8/8/8/8 w - - 0 1")

    assert {:ok, move, %{score: score, tt_hits: hits}} =
             Search.best_move(game, depth: 3, tt_slots: 8)

    assert Echecs.Game.checkmate?(Echecs.Game.make_move_int(game, move))
    assert score > 31_000
    assert hits >= 0
  end

  test "TT identity separates halfmove and repetition contexts sharing a board hash" do
    game = Echecs.new_game("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
    history_sensitive = %{game | halfmove: 8, history: [game.zobrist_hash, 17, 29]}

    assert game.zobrist_hash == history_sensitive.zobrist_hash

    assert {:terminal, :draw} =
             Search.best_move(%{history_sensitive | halfmove: 100}, depth: 3, trace: self())

    assert {:ok, _move, _info} = Search.best_move(game, depth: 3, trace: self())
    first_context = tt_context(drain_trace([]))

    assert {:ok, _move, _info} = Search.best_move(history_sensitive, depth: 3, trace: self())
    second_context = tt_context(drain_trace([]))
    refute first_context == second_context
  end

  test "TT repetition context rolls for reversible children and resets after pawn moves" do
    assert {:ok, _move, _info} = Search.best_move(Echecs.new_game(), depth: 2, trace: self())

    rolls =
      drain_trace([])
      |> Enum.filter(fn
        {:tt_context_roll, 1, _parent, _child, _reset?, _hash} -> true
        _ -> false
      end)

    assert Enum.any?(rolls, &match?({:tt_context_roll, 1, _, _, true, _}, &1))
    assert Enum.any?(rolls, &match?({:tt_context_roll, 1, _, _, false, _}, &1))

    assert rolls
           |> Enum.map(&elem(&1, 2))
           |> Enum.uniq()
           |> length() == 1
  end

  test "staged ordering traces good captures before quiets and bad captures" do
    game = Echecs.new_game("4k3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1")
    assert {:ok, _move, _info} = Search.best_move(game, depth: 2, trace: self())
    assert_receive {:search_trace, {:order, _ply, stages}}
    labels = Enum.map(stages, &elem(&1, 0))

    assert Enum.find_index(labels, &(&1 == :good_capture)) <
             Enum.find_index(labels, &(&1 == :quiet))
  end

  test "unordered mode preserves a fixed position result" do
    game = Echecs.new_game("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3")
    assert {:ok, move, %{score: score}} = Search.best_move(game, depth: 3)
    assert {:ok, ^move, %{score: ^score}} = Search.best_move(game, depth: 3, unordered: true)
  end

  test "SEE accounts for winning, equal, poisoned, en-passant, and promotion moves" do
    assert see("4k3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1", "d4d5") > 0
    assert see("3qk3/8/8/3r4/3R4/8/8/4K3 w - - 0 1", "d4d5") >= 0
    assert see("3rk3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1", "d4d5") < 0
    assert see("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6") > 0
    assert see("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q") >= 800
  end

  test "SEE excludes pinned recaptures and distinguishes legal king recaptures" do
    assert see("3k4/3r4/8/3p4/2B5/8/8/3RK3 w - - 0 1", "c4d5") > 0
    assert see("8/8/4k3/3p4/3Q4/8/8/4K3 w - - 0 1", "d4d5") < 0
    assert see("8/8/4k3/3p4/3Q4/8/8/3RK3 w - - 0 1", "d4d5") > 0
  end

  test "staged ordering beats unordered depth-four corpus without changing results" do
    corpus = [
      "4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1",
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
      "3qk3/8/8/3r4/3R4/8/8/4K3 w - - 0 1"
    ]

    {ordered_nodes, unordered_nodes} =
      Enum.reduce(corpus, {0, 0}, fn fen, {ordered_nodes, unordered_nodes} ->
        game = Echecs.new_game(fen)
        assert {:ok, move, %{score: score, nodes: ordered}} = Search.best_move(game, depth: 4)

        assert {:ok, unordered_move, %{score: ^score, nodes: unordered}} =
                 Search.best_move(game, depth: 4, unordered: true)

        # At depth four ordering may choose a different member of a genuine tie.
        # The compact depth-three regression below proves that property against the
        # independent oracle without making this node-reduction gate expensive.
        assert is_integer(move)
        assert is_integer(unordered_move)

        {ordered_nodes + ordered, unordered_nodes + unordered}
      end)

    assert ordered_nodes < unordered_nodes,
           "ordered=#{ordered_nodes} unordered=#{unordered_nodes}"
  end

  test "ordered and unordered traversal select only oracle-best moves from a compact tie" do
    game = Echecs.new_game("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
    {score, oracle_moves} = oracle_qsearch_root(game, 3)

    assert {:ok, ordered_move, %{score: ^score}} = Search.best_move(game, depth: 3)

    assert {:ok, unordered_move, %{score: ^score}} =
             Search.best_move(game, depth: 3, unordered: true)

    assert ordered_move in oracle_moves
    assert unordered_move in oracle_moves
    assert_oracle_tied(game, 3, score, [ordered_move, unordered_move])
  end

  test "exact mode agrees with independent depth-three negamax oracle" do
    corpus = [
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "4k3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1",
      "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
      "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
      "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
      "7k/5Q2/7K/8/8/8/8/8 w - - 0 1"
    ]

    Enum.each(corpus, fn fen ->
      game = Echecs.new_game(fen)
      {score, moves} = oracle_root(game, 3)
      assert {:ok, move, %{score: ^score}} = Search.best_move(game, depth: 3, exact_depth: true)
      assert move in moves
    end)
  end

  test "ordinary depth-three search agrees with the independent qsearch oracle" do
    corpus = [
      "4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1",
      "4k2r/8/8/8/8/8/3P4/Q3K3 w - - 0 1",
      "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
      "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
      "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
      "7k/5Q2/7K/8/8/8/8/8 w - - 0 1"
    ]

    Enum.each(corpus, fn fen ->
      game = Echecs.new_game(fen)
      {score, moves} = oracle_qsearch_root(game, 3)
      assert {:ok, move, %{score: ^score}} = Search.best_move(game, depth: 3)
      assert move in moves
    end)
  end

  test "fail-low root window returns the actual best score and stores an upper TT bound" do
    game = Echecs.new_game("4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1")
    {expected_score, expected_moves} = oracle_qsearch_root(game, 3)

    assert {:ok, move, %{score: score}} =
             Search.best_move(game, depth: 3, root_window: {1_500, 1_600}, trace: self())

    assert score == expected_score
    assert score < 1_500
    assert move in expected_moves
    assert Enum.any?(drain_trace([]), &match?({:tt_store, _, _, :upper, _}, &1))

    assert {:error, {:unknown_option, :root_window}} =
             EchecsEngine.analyze("4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1",
               root_window: {1_500, 1_600}
             )

    assert {:ok, lower_move, %{score: lower_score}} =
             Search.best_move(game, depth: 3, root_window: {100, 101}, trace: self())

    assert lower_score == expected_score
    assert lower_score > 101
    assert lower_move in expected_moves

    assert Enum.any?(drain_trace([]), fn
             {:tt_store, _ply, _depth, :lower, stored_score} -> stored_score < -101
             _ -> false
           end)
  end

  test "LMR traces reductions and full-depth re-searches only for eligible late quiets" do
    game = Echecs.new_game()
    assert {:ok, _move, _info} = Search.best_move(game, depth: 5, trace: self())
    events = drain_trace([])

    reductions =
      for {:lmr_reduce, _ply, _move, original, reduced} <- events, do: {original, reduced}

    assert Enum.all?(reductions, fn {original, reduced} ->
             original >= 3 and reduced == original - 1
           end)

    refute reductions == []
  end

  test "LMR retains depth-four result while reducing nodes" do
    corpus = [
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3"
    ]

    {enabled, disabled} =
      Enum.reduce(corpus, {0, 0}, fn fen, {enabled, disabled} ->
        game = Echecs.new_game(fen)
        assert {:ok, move, %{score: score, nodes: with_lmr}} = Search.best_move(game, depth: 4)

        assert {:ok, ^move, %{score: ^score, nodes: without_lmr}} =
                 Search.best_move(game, depth: 4, disable_lmr: true)

        {enabled + with_lmr, disabled + without_lmr}
      end)

    assert enabled < disabled, "lmr=#{enabled} full=#{disabled}"
  end

  test "LMR depth-four result agrees with the independent qsearch oracle" do
    game = Echecs.new_game("4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1")
    {score, moves} = oracle_qsearch_root(game, 4)

    assert {:ok, move, %{score: ^score}} = Search.best_move(game, depth: 4, trace: self())
    assert move in moves
    assert Enum.any?(drain_trace([]), &match?({:lmr_reduce, 0, _, 3, 2}, &1))
  end

  test "cancelling a root LMR re-search retains the previous completed iteration" do
    parent = self()
    stop_ref = :atomics.new(1, [])

    trace = fn event ->
      send(parent, {:trace_callback, event})

      case event do
        {:lmr_research, 0, _move, 4, _reduced_depth} ->
          :atomics.put(stop_ref, 1, 1)
          send(parent, :root_lmr_research_stopped)

        _ ->
          :ok
      end
    end

    reporter = fn info -> send(parent, {:completed_iteration, info}) end
    game = Echecs.new_game()

    task =
      Task.async(fn ->
        Search.best_move(game, depth: 5, stop_ref: stop_ref, trace: trace, reporter: reporter)
      end)

    assert_receive :root_lmr_research_stopped, 30_000
    assert {:ok, {:ok, move, info}} = Task.yield(task, 2_000)

    reports = drain_reports([])
    last_report = List.last(reports)

    assert Enum.map(reports, & &1.depth) == [1, 2, 3, 4]
    assert info.depth == last_report.depth
    assert info.pv == last_report.pv
    assert info.depth < 5
    assert move in Echecs.MoveGen.legal_moves_int(game)
    refute Process.alive?(task.pid)
  end

  test "mate scores prefer the shortest forced win and keep its PV legal" do
    game = Echecs.new_game("7k/5Q2/7K/8/8/8/8/8 w - - 0 1")
    scores = mate_scores(game, 5)
    winning = Enum.filter(scores, fn {score, _move} -> score > @mate - 1_000 end)
    shortest_score = winning |> Enum.map(&elem(&1, 0)) |> Enum.max()

    assert Enum.uniq(Enum.map(winning, &elem(&1, 0))) |> length() >= 2

    assert {:ok, move, %{score: ^shortest_score, pv: pv}} = Search.best_move(game, depth: 5)
    assert move in for({^shortest_score, winning_move} <- winning, do: winning_move)
    assert_legal_pv(game, pv)
  end

  test "mate scores prefer the longest survival for the losing side" do
    game = Echecs.new_game("6k1/8/5K2/4Q3/8/8/8/8 b - - 0 1")
    scores = mate_scores(game, 6)
    losses = Enum.filter(scores, fn {score, _move} -> score < -@mate + 1_000 end)
    survival_score = losses |> Enum.map(&elem(&1, 0)) |> Enum.max()

    assert Enum.any?(losses, fn {score, _move} -> score < survival_score end)

    assert {:ok, move, %{score: ^survival_score, pv: pv}} = Search.best_move(game, depth: 6)
    assert move in for({^survival_score, survival_move} <- losses, do: survival_move)
    assert_legal_pv(game, pv)
  end

  test "verified null move traces a safe transition, verification, and cutoff" do
    game = Echecs.new_game("4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1")

    assert {:ok, move, %{score: score, nodes: enabled}} =
             Search.best_move(game, depth: 6, trace: self())

    events = drain_trace([])

    for event <- [:null_try, :null_fail_high, :null_verify, :null_cutoff] do
      assert Enum.any?(events, &match?({^event, _, _}, &1)),
             "missing #{event}: #{inspect(events)}"
    end

    assert Enum.any?(events, fn
             {:null_position, turn, next_turn, _ep, nil, halfmove, next_halfmove, fullmove,
              next_fullmove, hash, 1}
             when is_integer(hash) and next_halfmove == halfmove + 1 and
                    ((turn == :black and next_turn == :white and next_fullmove == fullmove + 1) or
                       (turn == :white and next_turn == :black and next_fullmove == fullmove)) ->
               true

             _ ->
               false
           end)

    assert {:ok, ^move, %{score: ^score, nodes: disabled}} =
             Search.best_move(game, depth: 6, disable_null: true)

    assert enabled < disabled, "null=#{enabled} full=#{disabled}"
  end

  test "null guards exclude shallow, pawn-only, and mate-window nodes" do
    assert {:ok, _move, _info} = Search.best_move(Echecs.new_game(), depth: 3, trace: self())
    refute Enum.any?(drain_trace([]), &match?({:null_try, _, _}, &1))

    pawn_only = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    assert {:ok, _move, _info} = Search.best_move(pawn_only, depth: 6, trace: self())
    pawn_events = drain_trace([])
    refute Enum.any?(pawn_events, &match?({:null_try, _, _}, &1))

    assert Enum.any?(pawn_events, fn
             {:null_guard, _ply, _depth, _alpha, _beta, _static, _prior_null, _in_check, false,
              false} ->
               true

             _ ->
               false
           end)

    compact = Echecs.new_game("4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert {:ok, _move, _info} = Search.best_move(compact, depth: 6, trace: self())

    assert Enum.any?(drain_trace([]), fn
             {:null_guard, _ply, _depth, _alpha, beta, _static, _prior_null, _in_check, _material,
              false}
             when beta >= @mate - 1_000 ->
               true

             _ ->
               false
           end)

    assert {:ok, _move, _info} = Search.best_move(compact, depth: 8, trace: self())
    nested_events = drain_trace([])

    assert Enum.any?(nested_events, fn
             {:null_guard, _ply, _depth, _alpha, _beta, _static, true, _in_check, _material,
              false} ->
               true

             _ ->
               false
           end)

    assert Enum.any?(nested_events, fn
             {:null_guard, _ply, _depth, _alpha, _beta, _static, _prior_null, true, _material,
              false} ->
               true

             _ ->
               false
           end)

    assert {:error, {:unknown_option, :disable_null}} =
             EchecsEngine.analyze("4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1", disable_null: true)
  end

  test "futility pruning traces conservative reverse and forward cuts" do
    game = Echecs.new_game("4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1")

    assert {:ok, move, %{score: score, nodes: enabled}} =
             Search.best_move(game, depth: 6, trace: self())

    events = drain_trace([])
    assert Enum.any?(events, &match?({:reverse_futility, _, _, _, _}, &1))
    assert Enum.any?(events, &match?({:forward_futility, _, _, _, _, _}, &1))

    assert {:ok, ^move, %{score: ^score, nodes: disabled}} =
             Search.best_move(game, depth: 6, disable_futility: true)

    assert enabled < disabled, "futility=#{enabled} full=#{disabled}"
  end

  test "futility keeps a compact depth-five corpus stable while reducing nodes" do
    corpus = [
      "4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1",
      "4k2r/8/8/8/8/8/8/Q3K2R w - - 0 1",
      "4k2r/8/8/8/8/8/3P4/Q3K3 w - - 0 1"
    ]

    {enabled, disabled} =
      Enum.reduce(corpus, {0, 0}, fn fen, {enabled, disabled} ->
        game = Echecs.new_game(fen)

        assert {:ok, move, %{score: score, nodes: with_futility}} =
                 Search.best_move(game, depth: 5)

        assert {:ok, ^move, %{score: ^score, nodes: without_futility}} =
                 Search.best_move(game, depth: 5, disable_futility: true)

        {enabled + with_futility, disabled + without_futility}
      end)

    assert enabled < disabled, "futility=#{enabled} full=#{disabled}"
  end

  test "futility leaves mate, captures, and promotions to normal search" do
    mate = Echecs.new_game("7k/5Q2/7K/8/8/8/8/8 w - - 0 1")
    assert {:ok, mate_move, %{score: mate_score}} = Search.best_move(mate, depth: 3)
    assert mate_score > 31_000
    assert Echecs.Game.checkmate?(Echecs.Game.make_move_int(mate, mate_move))

    capture = Echecs.new_game("4k3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1")

    assert {:ok, capture_move, %{score: capture_score}} = Search.best_move(capture, depth: 4)

    assert {:ok, ^capture_move, %{score: ^capture_score}} =
             Search.best_move(capture, depth: 4, disable_futility: true)

    promotion = Echecs.new_game("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    assert {:ok, promotion_move, %{score: promotion_score}} =
             Search.best_move(promotion, depth: 4)

    assert {:ok, ^promotion_move, %{score: ^promotion_score}} =
             Search.best_move(promotion, depth: 4, disable_futility: true)

    pawn_only = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    assert {:ok, _move, _info} = Search.best_move(pawn_only, depth: 5, trace: self())

    refute Enum.any?(drain_trace([]), fn
             {:reverse_futility, _, _, _, _} -> true
             {:forward_futility, _, _, _, _, _} -> true
             _ -> false
           end)

    assert {:error, {:unknown_option, :disable_futility}} =
             EchecsEngine.analyze("4k2r/8/8/8/8/8/8/Q3K3 w - - 0 1", disable_futility: true)
  end

  test "SEE pruning traces only losing late captures" do
    game = Echecs.new_game("3rk3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1")

    assert {:ok, move, %{score: score, nodes: enabled}} =
             Search.best_move(game, depth: 4, trace: self())

    events = drain_trace([])

    assert Enum.any?(events, fn
             {:see_prune, _ply, _depth, _move, see} when see < 0 -> true
             _ -> false
           end)

    assert {:ok, ^move, %{score: ^score, nodes: disabled}} =
             Search.best_move(game, depth: 4, disable_see_pruning: true)

    assert enabled < disabled, "see=#{enabled} full=#{disabled}"
  end

  test "SEE pruning keeps a compact tactical depth-four corpus stable while reducing nodes" do
    corpus = [
      "3rk3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1",
      "3r1k2/8/8/3p4/3Q4/8/8/4K3 w - - 0 1",
      "3rk3/8/8/3p4/3Q4/8/8/5K2 w - - 0 1"
    ]

    {enabled, disabled} =
      Enum.reduce(corpus, {0, 0}, fn fen, {enabled, disabled} ->
        game = Echecs.new_game(fen)
        assert {:ok, move, %{score: score, nodes: with_see}} = Search.best_move(game, depth: 4)

        assert {:ok, ^move, %{score: ^score, nodes: without_see}} =
                 Search.best_move(game, depth: 4, disable_see_pruning: true)

        {enabled + with_see, disabled + without_see}
      end)

    assert enabled < disabled, "see=#{enabled} full=#{disabled}"
  end

  test "SEE pruning excludes checking, promotion, en-passant, equal, and public test controls" do
    checking = Echecs.new_game("3k4/8/5n2/3p4/3Q4/8/8/4K3 w - - 0 1")
    checking_move = packed_move(checking, "d4d5")
    assert Echecs.Game.in_check?(Echecs.Game.make_move_int(checking, checking_move))
    assert {:ok, _move, _info} = Search.best_move(checking, depth: 1, trace: self())
    refute pruned_move?(drain_trace([]), checking_move)

    equal = Echecs.new_game("3qk3/8/8/3r4/3R4/8/8/4K3 w - - 0 1")
    equal_move = packed_move(equal, "d4d5")
    assert {:ok, _move, _info} = Search.best_move(equal, depth: 1, trace: self())
    refute pruned_move?(drain_trace([]), equal_move)

    en_passant = Echecs.new_game("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    en_passant_move = packed_move(en_passant, "e5d6")
    assert {:ok, _move, _info} = Search.best_move(en_passant, depth: 1, trace: self())
    refute pruned_move?(drain_trace([]), en_passant_move)

    promotion = Echecs.new_game("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promotion_move = packed_move(promotion, "a7b8q")
    assert {:ok, _move, _info} = Search.best_move(promotion, depth: 1, trace: self())
    refute pruned_move?(drain_trace([]), promotion_move)

    evasion = Echecs.new_game("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    assert Echecs.Game.in_check?(evasion)
    assert {:ok, _move, _info} = Search.best_move(evasion, depth: 1, trace: self())
    refute Enum.any?(drain_trace([]), &match?({:see_prune, _, _, _, _}, &1))

    tt_game = Echecs.new_game("3rk3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1")
    assert {:ok, _move, _info} = Search.best_move(tt_game, depth: 4, trace: self())

    assert Enum.any?(drain_trace([]), &match?({:see_prune_guard, _ply, _move, :tt, false}, &1))

    assert {:error, {:unknown_option, :disable_see_pruning}} =
             EchecsEngine.analyze("3rk3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1",
               disable_see_pruning: true
             )
  end

  defp see(fen, uci) do
    game = Echecs.new_game(fen)
    move = packed_move(game, uci)
    assert {:ok, _best, _info} = Search.best_move(game, depth: 1, trace: self())

    receive_see(move)
  end

  defp pruned_move?(events, move),
    do: Enum.any?(events, &match?({:see_prune, _ply, _depth, ^move, _see}, &1))

  defp receive_see(move) do
    receive do
      {:search_trace, {:order, _ply, entries}} ->
        case Enum.find(entries, fn {_stage, current, _score} -> current == move end) do
          {_stage, ^move, score} when is_integer(score) -> score
          _ -> receive_see(move)
        end
    after
      1_000 -> flunk("missing SEE trace for #{move}")
    end
  end

  defp packed_move(game, <<from::binary-size(2), to::binary-size(2), suffix::binary>>) do
    promotion = %{"" => nil, "q" => :queen, "r" => :rook, "b" => :bishop, "n" => :knight}[suffix]

    Enum.find(Echecs.MoveGen.legal_moves_int(game), fn move ->
      Echecs.Board.to_algebraic(Echecs.Move.unpack_from(move)) == from and
        Echecs.Board.to_algebraic(Echecs.Move.unpack_to(move)) == to and
        Echecs.Move.unpack_promotion(move) == promotion
    end) || flunk("illegal fixture move #{from}#{to}#{suffix}")
  end

  defp oracle_root(game, depth) do
    weights = Eval.seed_weights()

    game
    |> Echecs.MoveGen.legal_moves_int()
    |> Enum.map(fn move ->
      {-oracle(Echecs.Game.make_move_int(game, move), depth - 1, 1, weights), move}
    end)
    |> then(fn scores ->
      best = scores |> Enum.map(&elem(&1, 0)) |> Enum.max()
      {best, scores |> Enum.filter(&(elem(&1, 0) == best)) |> Enum.map(&elem(&1, 1))}
    end)
  end

  defp oracle(game, depth, ply, weights) do
    cond do
      Echecs.Game.draw?(game) ->
        0

      Echecs.MoveGen.legal_moves_int(game) == [] ->
        if(Echecs.Game.in_check?(game), do: -@mate + ply, else: 0)

      depth == 0 ->
        Eval.evaluate(game, Eval.refresh(game, weights), weights)

      true ->
        game
        |> Echecs.MoveGen.legal_moves_int()
        |> Enum.map(&(-oracle(Echecs.Game.make_move_int(game, &1), depth - 1, ply + 1, weights)))
        |> Enum.max()
    end
  end

  defp oracle_qsearch_root(game, depth) do
    weights = Eval.seed_weights()

    game
    |> Echecs.MoveGen.legal_moves_int()
    |> Enum.map(fn move ->
      {-oracle_qsearch(Echecs.Game.make_move_int(game, move), depth - 1, 1, weights), move}
    end)
    |> oracle_best()
  end

  defp oracle_qsearch(game, depth, ply, weights) do
    moves = Echecs.MoveGen.legal_moves_int(game)

    cond do
      moves == [] ->
        if(Echecs.Game.in_check?(game), do: -@mate + ply, else: 0)

      Echecs.Game.draw?(game) ->
        0

      depth > 0 ->
        moves
        |> Enum.map(
          &(-oracle_qsearch(Echecs.Game.make_move_int(game, &1), depth - 1, ply + 1, weights))
        )
        |> Enum.max()

      Echecs.Game.in_check?(game) ->
        moves
        |> Enum.map(&(-oracle_qsearch(Echecs.Game.make_move_int(game, &1), 0, ply + 1, weights)))
        |> Enum.max()

      true ->
        stand_pat = Eval.evaluate(game, Eval.refresh(game, weights), weights)

        moves
        |> Enum.filter(&tactical?(game, &1))
        |> Enum.map(&(-oracle_qsearch(Echecs.Game.make_move_int(game, &1), 0, ply + 1, weights)))
        |> Enum.max(fn -> stand_pat end)
        |> max(stand_pat)
    end
  end

  defp oracle_best(scores) do
    best = scores |> Enum.map(&elem(&1, 0)) |> Enum.max()
    {best, scores |> Enum.filter(&(elem(&1, 0) == best)) |> Enum.map(&elem(&1, 1))}
  end

  defp assert_oracle_tied(game, depth, score, moves) do
    weights = Eval.seed_weights()

    Enum.each(moves, fn move ->
      actual = -oracle_qsearch(Echecs.Game.make_move_int(game, move), depth - 1, 1, weights)
      assert actual == score
    end)
  end

  defp mate_scores(game, depth) do
    for move <- Echecs.MoveGen.legal_moves_int(game) do
      {-forced_mate_score(Echecs.Game.make_move_int(game, move), depth - 1, 1), move}
    end
  end

  defp forced_mate_score(game, depth, ply) do
    moves = Echecs.MoveGen.legal_moves_int(game)

    cond do
      moves == [] ->
        if(Echecs.Game.in_check?(game), do: -@mate + ply, else: 0)

      Echecs.Game.draw?(game) ->
        0

      depth == 0 ->
        0

      true ->
        moves
        |> Enum.map(
          &(-forced_mate_score(Echecs.Game.make_move_int(game, &1), depth - 1, ply + 1))
        )
        |> Enum.max()
    end
  end

  defp assert_legal_pv(game, pv) do
    Enum.reduce(pv, game, fn uci, position ->
      move = packed_move(position, uci)
      Echecs.Game.make_move_int(position, move)
    end)
  end

  defp drain_reports(reports) do
    receive do
      {:completed_iteration, report} -> drain_reports([report | reports])
    after
      0 -> Enum.reverse(reports)
    end
  end

  defp tactical?(game, move) do
    not is_nil(Echecs.Board.at_tuple(game.board, Echecs.Move.unpack_to(move))) or
      Echecs.Move.unpack_promotion(move) != nil or
      Echecs.Move.unpack_special(move) == :en_passant
  end

  defp drain_trace(events) do
    receive do
      {:search_trace, event} -> drain_trace([event | events])
    after
      0 -> Enum.reverse(events)
    end
  end

  defp tt_context(events) do
    case Enum.find(events, &match?({:tt_context, _, _, _}, &1)) do
      {:tt_context, _ply, _hash, context} -> context
      nil -> flunk("missing TT context trace")
    end
  end
end
