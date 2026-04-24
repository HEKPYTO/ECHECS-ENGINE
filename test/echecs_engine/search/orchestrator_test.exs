defmodule EchecsEngine.Search.OrchestratorTest do
  use ExUnit.Case

  test "hybrid search successfully evaluates a board state and picks a move" do
    game = Echecs.new_game()

    result = EchecsEngine.Search.Orchestrator.search(game)

    case result do
      {move, score} ->
        assert %Echecs.Move{} = move
        assert is_float(score)

      other ->
        flunk("Expected {move, score}, got #{inspect(other)}")
    end
  end

  test "hybrid search uses the model policy to choose the candidate set" do
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

      %{policy: policy, wdl: Nx.tensor([0.7, 0.2, 0.1]), moves_left: Nx.tensor([35.0])}
    end

    assert {move, score} =
             EchecsEngine.Search.Orchestrator.search(game,
               inference: inference,
               candidate_limit: 1
             )

    assert move == target_move
    assert is_float(score)
  end

  test "without evaluator weights, search uses policy and WDL instead of placeholder sparse weights" do
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

      %{policy: policy, wdl: Nx.tensor([0.8, 0.1, 0.1]), moves_left: Nx.tensor([30.0])}
    end

    assert {^target_move, score} =
             EchecsEngine.Search.Orchestrator.search(game,
               inference: inference,
               candidate_limit: 4
             )

    assert score > 0.7
  end

  test "without sparse evaluator weights, orchestrator evaluates candidate child positions" do
    game = Echecs.new_game()
    legal_moves = Echecs.legal_moves(game)
    high_prior_move = Enum.find(legal_moves, &(&1.from == 52 and &1.to == 36))
    better_child_move = Enum.find(legal_moves, &(&1.from == 51 and &1.to == 35))

    {:ok, counter} = Agent.start_link(fn -> 0 end)

    inference = fn _tensor ->
      call_idx =
        Agent.get_and_update(counter, fn idx ->
          {idx, idx + 1}
        end)

      case call_idx do
        0 ->
          policy = Nx.broadcast(-10.0, {4672})

          policy =
            policy
            |> Nx.put_slice(
              [EchecsEngine.Policy.move_index(game, high_prior_move)],
              Nx.tensor([10.0])
            )
            |> Nx.put_slice(
              [EchecsEngine.Policy.move_index(game, better_child_move)],
              Nx.tensor([9.0])
            )

          %{policy: policy, wdl: Nx.tensor([0.5, 0.4, 0.1]), moves_left: Nx.tensor([24.0])}

        1 ->
          %{
            policy: Nx.broadcast(0.0, {4672}),
            wdl: Nx.tensor([0.3, 0.3, 0.4]),
            moves_left: Nx.tensor([24.0])
          }

        _ ->
          %{
            policy: Nx.broadcast(0.0, {4672}),
            wdl: Nx.tensor([0.8, 0.1, 0.1]),
            moves_left: Nx.tensor([24.0])
          }
      end
    end

    assert {move, score} =
             EchecsEngine.Search.Orchestrator.search(game,
               inference: inference,
               candidate_limit: 2
             )

    assert move == better_child_move
    assert score > 0.0
  end

  test "with batched child inference configured, orchestrator evaluates candidate positions in one batch" do
    game = Echecs.new_game()
    legal_moves = Echecs.legal_moves(game)
    high_prior_move = Enum.find(legal_moves, &(&1.from == 52 and &1.to == 36))
    better_child_move = Enum.find(legal_moves, &(&1.from == 51 and &1.to == 35))
    parent = self()

    inference = fn _tensor ->
      policy = Nx.broadcast(-10.0, {4672})

      policy =
        policy
        |> Nx.put_slice(
          [EchecsEngine.Policy.move_index(game, high_prior_move)],
          Nx.tensor([10.0])
        )
        |> Nx.put_slice(
          [EchecsEngine.Policy.move_index(game, better_child_move)],
          Nx.tensor([9.0])
        )

      %{policy: policy, wdl: Nx.tensor([0.5, 0.4, 0.1]), moves_left: Nx.tensor([24.0])}
    end

    batched_inference = fn tensor ->
      send(parent, {:batch_shape, Nx.shape(tensor)})

      %{
        policy: Nx.broadcast(0.0, {2, 4672}),
        wdl: Nx.tensor([[0.3, 0.3, 0.4], [0.8, 0.1, 0.1]]),
        moves_left: Nx.tensor([[24.0], [24.0]])
      }
    end

    assert {move, score} =
             EchecsEngine.Search.Orchestrator.search(game,
               inference: inference,
               batched_inference: batched_inference,
               candidate_limit: 2
             )

    assert_receive {:batch_shape, {2, 119, 8, 8}}
    assert move == better_child_move
    assert score > 0.0
  end
end
