defmodule EchecsEngine.MCTSTest do
  use ExUnit.Case

  alias EchecsEngine.MCTS
  alias EchecsEngine.MCTS.Node

  @tag timeout: 120_000
  test "search/2 performs MCTS and returns an updated root node" do
    game = Echecs.new_game()
    iterations = 5

    root_node = MCTS.search(game, iterations)

    assert %Node{} = root_node
    assert root_node.visits == iterations
    assert root_node.game == game
    assert map_size(root_node.children) > 0

    children_visits =
      Enum.map(root_node.children, fn {_, child} -> child.visits end) |> Enum.sum()

    assert children_visits == iterations - 1
  end

  test "search/3 assigns priors from the policy head instead of uniform values" do
    game = Echecs.new_game()
    legal_moves = Echecs.legal_moves(game)
    target_move = Enum.find(legal_moves, &(&1.from == 52 and &1.to == 36))
    comparison_move = Enum.find(legal_moves, &(&1.from == 51 and &1.to == 35))

    inference = fn _tensor ->
      policy = Nx.broadcast(-10.0, {4672})

      policy =
        Nx.put_slice(
          policy,
          [EchecsEngine.Policy.move_index(game, target_move)],
          Nx.tensor([10.0])
        )

      %{policy: policy, wdl: Nx.tensor([0.6, 0.2, 0.2]), moves_left: Nx.tensor([30.0])}
    end

    root_node = MCTS.search(game, 1, inference: inference)

    assert %Node{} = root_node

    assert root_node.children[target_move].prior_prob >
             root_node.children[comparison_move].prior_prob

    total_prior =
      root_node.children
      |> Map.values()
      |> Enum.reduce(0.0, fn child, acc -> acc + child.prior_prob end)

    assert_in_delta total_prior, 1.0, 1.0e-6
  end

  test "best_move/1 selects the root move with the highest visit count" do
    game = Echecs.new_game()
    legal_moves = Echecs.legal_moves(game)
    target_move = Enum.find(legal_moves, &(&1.from == 52 and &1.to == 36))
    other_move = Enum.find(legal_moves, &(&1.from == 51 and &1.to == 35))

    root = %Node{
      game: game,
      children: %{
        target_move => %Node{game: game, visits: 8, total_value: 1.0},
        other_move => %Node{game: game, visits: 3, total_value: 3.0}
      }
    }

    assert MCTS.best_move(root) == target_move
  end

  test "search_best_move/3 returns the visit-count selected move" do
    game = Echecs.new_game()
    legal_moves = Echecs.legal_moves(game)
    target_move = Enum.find(legal_moves, &(&1.from == 52 and &1.to == 36))

    inference = fn _tensor ->
      policy = Nx.broadcast(-10.0, {4672})

      policy =
        Nx.put_slice(
          policy,
          [EchecsEngine.Policy.move_index(game, target_move)],
          Nx.tensor([10.0])
        )

      %{policy: policy, wdl: Nx.tensor([0.7, 0.2, 0.1]), moves_left: Nx.tensor([30.0])}
    end

    assert ^target_move = MCTS.search_best_move(game, iterations: 3, inference: inference)
  end

  test "search/3 accepts an infinite iteration budget when a movetime deadline is provided" do
    game = Echecs.new_game()

    root_node =
      MCTS.search(game, :infinity,
        movetime: 5,
        inference: fn _tensor ->
          %{
            policy: Nx.broadcast(0.0, {4672}),
            wdl: Nx.tensor([0.6, 0.2, 0.2]),
            moves_left: Nx.tensor([30.0])
          }
        end
      )

    assert %Node{} = root_node
    assert root_node.visits > 0
  end

  test "search/3 reuses cached evaluations through a transposition table" do
    game = Echecs.new_game()
    parent = self()
    tt = :ets.new(:mcts_test_tt, [:set, :private])

    inference = fn _tensor ->
      send(parent, :inference_call)

      %{
        policy: Nx.broadcast(0.0, {4672}),
        wdl: Nx.tensor([0.6, 0.2, 0.2]),
        moves_left: Nx.tensor([30.0])
      }
    end

    _ = MCTS.search(game, 1, inference: inference, tt: tt)
    _ = MCTS.search(game, 1, inference: inference, tt: tt)

    assert_receive :inference_call

    refute_receive :inference_call, 20

    :ets.delete(tt)
  end

  test "search/3 scopes the transposition cache by inference identity" do
    game = Echecs.new_game()
    parent = self()
    tt = :ets.new(:mcts_test_tt_identity, [:set, :private])

    inference_a = fn _tensor ->
      send(parent, :inference_a)

      %{
        policy: Nx.broadcast(0.0, {4672}),
        wdl: Nx.tensor([0.6, 0.2, 0.2]),
        moves_left: Nx.tensor([30.0])
      }
    end

    inference_b = fn _tensor ->
      send(parent, :inference_b)

      %{
        policy: Nx.broadcast(0.0, {4672}),
        wdl: Nx.tensor([0.2, 0.2, 0.6]),
        moves_left: Nx.tensor([30.0])
      }
    end

    _ = MCTS.search(game, 1, inference: inference_a, tt: tt)
    _ = MCTS.search(game, 1, inference: inference_b, tt: tt)

    assert_receive :inference_a
    assert_receive :inference_b

    :ets.delete(tt)
  end

  test "search/3 scopes function inference identity by closure environment" do
    game = Echecs.new_game()
    parent = self()
    tt = :ets.new(:mcts_test_tt_closure_identity, [:set, :private])

    inference_a = closure_inference(parent, :inference_a, [0.6, 0.2, 0.2])
    inference_b = closure_inference(parent, :inference_b, [0.2, 0.2, 0.6])

    _ = MCTS.search(game, 1, inference: inference_a, tt: tt)
    _ = MCTS.search(game, 1, inference: inference_b, tt: tt)

    assert_receive :inference_a
    assert_receive :inference_b

    :ets.delete(tt)
  end

  test "search/3 supports self-play root noise" do
    game = Echecs.new_game()
    legal_moves = Echecs.legal_moves(game)
    target_move = Enum.find(legal_moves, &(&1.from == 52 and &1.to == 36))
    comparison_move = List.last(legal_moves)

    inference = fn _tensor ->
      policy = Nx.broadcast(-10.0, {4672})

      policy =
        Nx.put_slice(
          policy,
          [EchecsEngine.Policy.move_index(game, target_move)],
          Nx.tensor([10.0])
        )

      %{policy: policy, wdl: Nx.tensor([0.7, 0.2, 0.1]), moves_left: Nx.tensor([30.0])}
    end

    root_node =
      MCTS.search(game, 1,
        inference: inference,
        self_play: true,
        dirichlet_epsilon: 1.0,
        dirichlet_noise_fn: fn size, _alpha ->
          List.duplicate(0.0, size - 1) ++ [1.0]
        end
      )

    assert root_node.children[comparison_move].prior_prob == 1.0
    assert root_node.children[target_move].prior_prob == 0.0
  end

  test "search/3 uses gamma variates for built-in Dirichlet sampling" do
    game = Echecs.new_game()
    parent = self()

    inference = fn _tensor ->
      %{
        policy: Nx.broadcast(0.0, {4672}),
        wdl: Nx.tensor([0.7, 0.2, 0.1]),
        moves_left: Nx.tensor([30.0])
      }
    end

    _root =
      MCTS.search(game, 1,
        inference: inference,
        self_play: true,
        gamma_sampler_fn: fn alpha ->
          send(parent, {:gamma_alpha, alpha})
          1.0
        end
      )

    assert_receive {:gamma_alpha, _alpha}
  end

  defp closure_inference(parent, tag, wdl) do
    fn _tensor ->
      send(parent, tag)

      %{
        policy: Nx.broadcast(0.0, {4672}),
        wdl: Nx.tensor(wdl),
        moves_left: Nx.tensor([30.0])
      }
    end
  end
end
