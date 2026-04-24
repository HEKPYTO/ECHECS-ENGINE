defmodule EchecsEngine.DatasetTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Dataset

  @start_fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

  test "builds supervised tensors from FEN, move, result, and engine eval" do
    record = %{
      "fen" => @start_fen,
      "move" => "e2e4",
      "result" => "1-0",
      "eval_cp" => 120,
      "moves_left" => 42
    }

    assert {:ok, {input, targets}} = Dataset.example_from_record(record)

    assert Nx.shape(input) == {119, 8, 8}
    assert Nx.shape(targets.policy) == {4672}
    assert Nx.shape(targets.policy_mask) == {4672}
    assert Nx.shape(targets.wdl) == {3}
    assert Nx.shape(targets.moves_left) == {1}

    game = Echecs.new_game(@start_fen)
    move = Enum.find(Echecs.legal_moves(game), &(&1.from == 52 and &1.to == 36))
    move_idx = EchecsEngine.Policy.move_index(game, move)

    assert targets.policy[move_idx] |> Nx.to_number() == 1.0
    assert targets.policy_mask[move_idx] |> Nx.to_number() == 1.0
    assert Nx.sum(targets.policy) |> Nx.to_number() == 1.0
    assert Nx.sum(targets.wdl) |> Nx.to_number() == 1.0
  end

  test "accepts optional history_fens for stacked-position encoding" do
    record = %{
      "fen" => "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
      "history_fens" => [@start_fen],
      "move" => "g1f3",
      "result" => "1-0"
    }

    assert {:ok, {input, _targets}} = Dataset.example_from_record(record)
    assert input[14 + 0][6][4] |> Nx.to_number() == 1
  end

  test "rejects records whose target move is illegal for the FEN" do
    record = %{
      "fen" => @start_fen,
      "move" => "e2e5",
      "result" => "1-0"
    }

    assert {:error, {:illegal_move, "e2e5"}} = Dataset.example_from_record(record)
  end

  test "lazily streams JSONL records into batches" do
    path =
      Path.join(System.tmp_dir!(), "echecs_dataset_#{System.unique_integer([:positive])}.jsonl")

    json =
      :json.encode(%{
        "fen" => @start_fen,
        "move" => "e2e4",
        "result" => "1-0",
        "eval_cp" => 80
      })

    File.write!(path, [json, "\n", json, "\n"])

    on_exit(fn -> File.rm(path) end)

    batches = Dataset.batches_from_jsonl!(path, batch_size: 2)

    refute is_list(batches)

    assert [{inputs, targets}] = Enum.to_list(batches)
    assert Nx.shape(inputs) == {2, 119, 8, 8}
    assert Nx.shape(targets.policy) == {2, 4672}
    assert Nx.shape(targets.policy_mask) == {2, 4672}
    assert Nx.shape(targets.wdl) == {2, 3}
    assert Nx.shape(targets.moves_left) == {2, 1}
  end

  test "streams JSONL shards in order" do
    paths =
      for idx <- 1..2 do
        path =
          Path.join(
            System.tmp_dir!(),
            "echecs_dataset_shard_#{idx}_#{System.unique_integer([:positive])}.jsonl"
          )

        json =
          :json.encode(%{
            "fen" => @start_fen,
            "move" => if(idx == 1, do: "e2e4", else: "d2d4"),
            "result" => "1-0"
          })

        File.write!(path, [json, "\n"])
        path
      end

    on_exit(fn -> Enum.each(paths, &File.rm/1) end)

    assert batches = Dataset.batches_from_jsonl!(paths, batch_size: 1)
    assert length(Enum.to_list(batches)) == 2
  end

  test "drops incomplete trailing batches for compiled training loops" do
    path =
      Path.join(
        System.tmp_dir!(),
        "echecs_dataset_drop_partial_#{System.unique_integer([:positive])}.jsonl"
      )

    json =
      :json.encode(%{
        "fen" => @start_fen,
        "move" => "e2e4",
        "result" => "1-0"
      })

    File.write!(path, List.duplicate([json, "\n"], 3))

    on_exit(fn -> File.rm(path) end)

    [{inputs, targets}] =
      path
      |> Dataset.batches_from_jsonl!(batch_size: 2)
      |> Enum.to_list()

    assert Nx.shape(inputs) == {2, 119, 8, 8}
    assert Nx.shape(targets.policy) == {2, 4672}
  end
end
