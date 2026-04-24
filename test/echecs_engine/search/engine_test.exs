defmodule EchecsEngine.Search.EngineTest do
  use ExUnit.Case, async: false

  alias EchecsEngine.Search.Engine

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "best_move/2 refuses alpha-beta without an evaluator or inference fallback" do
    game = Echecs.new_game(@start_fen)

    assert {:error, :missing_evaluator} =
             Engine.best_move(game,
               depth: 1,
               evaluator_paths: []
             )
  end

  test "best_move/2 defaults to alpha-beta and honors evaluator scores over policy bias" do
    game = Echecs.new_game(@start_fen)
    target_move = Enum.find(Echecs.legal_moves(game), &(&1.from == 52 and &1.to == 36))
    policy_move = Enum.find(Echecs.legal_moves(game), &(&1.from == 51 and &1.to == 35))

    {:ok, target_game} =
      Echecs.make_move(game, target_move.from, target_move.to, target_move.promotion)

    {:ok, policy_game} =
      Echecs.make_move(game, policy_move.from, policy_move.to, policy_move.promotion)

    target_fen = Echecs.FEN.to_string(target_game)
    policy_fen = Echecs.FEN.to_string(policy_game)

    inference = fn _tensor ->
      policy = Nx.broadcast(-10.0, {4672})

      policy =
        Nx.put_slice(
          policy,
          [EchecsEngine.Policy.move_index(game, policy_move)],
          Nx.tensor([10.0])
        )

      %{policy: policy, wdl: Nx.tensor([0.7, 0.2, 0.1]), moves_left: Nx.tensor([30.0])}
    end

    evaluator_fn = fn position ->
      case Echecs.FEN.to_string(position) do
        ^target_fen -> -1.0
        ^policy_fen -> 1.0
        _other -> 0.0
      end
    end

    assert {:ok, ^target_move} =
             Engine.best_move(game, depth: 1, inference: inference, evaluator_fn: evaluator_fn)
  end

  test "public best_move/2 forwards go-style search options" do
    game = Echecs.new_game(@start_fen)
    target_move = Enum.find(Echecs.legal_moves(game), &(&1.from == 52 and &1.to == 36))

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

    assert {:ok, "e2e4"} =
             EchecsEngine.best_move(@start_fen,
               depth: 1,
               inference: inference,
               evaluator_fn: fn _position -> 0.0 end
             )
  end

  test "best_move/2 still supports the experimental mcts backend explicitly" do
    game = Echecs.new_game(@start_fen)
    parent = self()

    inference = fn _tensor ->
      send(parent, :inference_call)

      %{
        policy: Nx.broadcast(0.0, {4672}),
        wdl: Nx.tensor([0.4, 0.4, 0.2]),
        moves_left: Nx.tensor([30.0])
      }
    end

    assert {:ok, %Echecs.Move{}} = Engine.best_move(game, backend: :mcts, inference: inference)

    calls =
      receive_all_calls(0)

    assert calls > 1
  end

  test "best_move/2 loads sparse evaluator artifacts for alpha-beta when paths are provided" do
    game = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    legal_moves = Echecs.legal_moves(game)
    target_move = Enum.find(legal_moves, &(&1.from == 52 and &1.to == 36))

    next_games =
      Enum.map(legal_moves, fn move ->
        {:ok, next_game} = Echecs.make_move(game, move.from, move.to, move.promotion)
        {move, next_game}
      end)

    target_game = next_games |> Enum.find(fn {move, _game} -> move == target_move end) |> elem(1)

    target_features =
      target_game
      |> EchecsEngine.NNUE.Features.active_indices()
      |> Map.values()
      |> List.flatten()
      |> MapSet.new()

    other_features =
      next_games
      |> Enum.reject(fn {move, _game} -> move == target_move end)
      |> Enum.reduce(MapSet.new(), fn {_move, next_game}, acc ->
        features =
          next_game
          |> EchecsEngine.NNUE.Features.active_indices()
          |> Map.values()
          |> List.flatten()
          |> MapSet.new()

        MapSet.union(acc, features)
      end)

    feature_table =
      [target_features, other_features]
      |> Enum.reduce(MapSet.new(), &MapSet.union/2)
      |> Enum.map(fn feature_idx ->
        contribution =
          cond do
            feature_idx in target_features and feature_idx not in other_features ->
              Nx.broadcast(0.0, {3072}) |> Nx.put_slice([0], Nx.tensor([1.0]))

            true ->
              Nx.broadcast(0.0, {3072})
          end

        {feature_idx, contribution}
      end)
      |> Map.new()

    artifact = %{
      feature_table: feature_table,
      w1: Nx.broadcast(0.0, {3072, 32}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      b1: Nx.broadcast(0.0, {32}),
      w2: Nx.broadcast(0.0, {32, 1}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      b2: Nx.broadcast(0.0, {1})
    }

    path =
      System.tmp_dir!()
      |> Path.join("echecs_engine_eval_#{System.unique_integer([:positive])}.axon")

    on_exit(fn -> File.rm(path) end)

    :ok =
      EchecsEngine.Checkpoint.save_evaluator_state!(path, artifact, %{"source" => "engine-test"})

    assert {:ok, ^target_move} =
             Engine.best_move(game,
               depth: 1,
               evaluator_paths: [path]
             )
  end

  test "default evaluator misses are not cached for process lifetime" do
    cache_key = {Engine, :default_evaluator}
    old_cache = :persistent_term.get(cache_key, :__missing__)
    old_path = Application.get_env(:echecs_engine, :evaluator_path, :__missing__)
    path = temp_evaluator_path()

    on_exit(fn ->
      case old_cache do
        :__missing__ -> :persistent_term.erase(cache_key)
        value -> :persistent_term.put(cache_key, value)
      end

      restore_evaluator_path(old_path)
      File.rm(path)
    end)

    Application.put_env(:echecs_engine, :evaluator_path, path)
    :persistent_term.put(cache_key, :error)

    game = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    feature_table = feature_table_for(game)

    artifact = %{
      feature_table: feature_table,
      w1: Nx.broadcast(0.0, {3072, 32}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      b1: Nx.broadcast(0.0, {32}),
      w2: Nx.broadcast(0.0, {32, 1}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      b2: Nx.broadcast(0.0, {1})
    }

    :ok =
      EchecsEngine.Checkpoint.save_evaluator_state!(path, artifact, %{"source" => "cache-test"})

    assert {:ok, %Echecs.Move{}} = Engine.best_move(game, depth: 1)
  end

  test "default evaluator cache reloads when the artifact signature changes" do
    cache_key = {Engine, :default_evaluator}
    old_cache = :persistent_term.get(cache_key, :__missing__)
    old_path = Application.get_env(:echecs_engine, :evaluator_path, :__missing__)
    path = temp_evaluator_path()

    on_exit(fn ->
      case old_cache do
        :__missing__ -> :persistent_term.erase(cache_key)
        value -> :persistent_term.put(cache_key, value)
      end

      restore_evaluator_path(old_path)
      File.rm(path)
    end)

    Application.put_env(:echecs_engine, :evaluator_path, path)
    :persistent_term.put(cache_key, {:ok, :stale_weights, [{path, :enoent}]})

    game = Echecs.new_game("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    feature_table = feature_table_for(game)

    artifact = %{
      feature_table: feature_table,
      w1: Nx.broadcast(0.0, {3072, 32}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      b1: Nx.broadcast(0.0, {32}),
      w2: Nx.broadcast(0.0, {32, 1}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]])),
      b2: Nx.broadcast(0.0, {1})
    }

    :ok =
      EchecsEngine.Checkpoint.save_evaluator_state!(path, artifact, %{
        "source" => "cache-signature-test"
      })

    assert {:ok, %Echecs.Move{}} = Engine.best_move(game, depth: 1)
  end

  test "best_move/2 emits iterative info with principal variation when a reporter is provided" do
    game = Echecs.new_game(@start_fen)
    target_move = Enum.find(Echecs.legal_moves(game), &(&1.from == 52 and &1.to == 36))
    parent = self()

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

    reporter = fn info -> send(parent, {:info, info}) end

    assert {:ok, ^target_move} =
             Engine.best_move(
               game,
               depth: 2,
               inference: inference,
               evaluator_fn: fn _position -> 0.0 end,
               reporter: reporter
             )

    assert_receive {:info, %{depth: 1, nodes: nodes_1, pv: ["e2e4" | _rest]}}
    assert is_integer(nodes_1)
    assert nodes_1 > 1

    assert_receive {:info, %{depth: 2, nodes: nodes_2, pv: ["e2e4" | _rest]}}
    assert is_integer(nodes_2)
    assert nodes_2 > nodes_1
  end

  defp receive_all_calls(count) do
    receive do
      :inference_call -> receive_all_calls(count + 1)
    after
      0 -> count
    end
  end

  defp feature_table_for(game) do
    game
    |> EchecsEngine.NNUE.Features.active_indices()
    |> Map.values()
    |> List.flatten()
    |> Enum.uniq()
    |> Map.new(fn feature_idx ->
      {feature_idx, Nx.broadcast(0.0, {3072}) |> Nx.put_slice([0], Nx.tensor([1.0]))}
    end)
  end

  defp temp_evaluator_path do
    System.tmp_dir!()
    |> Path.join("echecs_engine_default_eval_#{System.unique_integer([:positive])}.axon")
  end

  defp restore_evaluator_path(:__missing__),
    do: Application.delete_env(:echecs_engine, :evaluator_path)

  defp restore_evaluator_path(path),
    do: Application.put_env(:echecs_engine, :evaluator_path, path)
end
