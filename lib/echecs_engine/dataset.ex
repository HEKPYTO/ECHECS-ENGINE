defmodule EchecsEngine.Dataset do
  @moduledoc """
  Supervised FEN/eval dataset pipeline for policy, WDL, and moves-left targets.
  """

  @spec batches_from_jsonl!(String.t() | [String.t()], keyword()) :: Enumerable.t()
  def batches_from_jsonl!(paths, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 64)

    paths
    |> normalize_paths()
    |> Stream.flat_map(&record_lines!/1)
    |> Stream.map(fn record ->
      case example_from_record(record) do
        {:ok, example} -> example
        {:error, reason} -> raise ArgumentError, "invalid dataset record: #{inspect(reason)}"
      end
    end)
    |> Stream.chunk_every(batch_size, batch_size, :discard)
    |> Stream.map(&batch_examples/1)
  end

  @spec example_from_record(map()) :: {:ok, {Nx.Tensor.t(), map()}} | {:error, term()}
  def example_from_record(%{"fen" => fen, "move" => uci_move} = record) do
    with {:ok, game} <- parse_game(fen),
         {:ok, history_games} <- parse_history_games(Map.get(record, "history_fens", [])),
         {:ok, move} <- legal_move_from_uci(game, uci_move) do
      input =
        game
        |> EchecsEngine.Tensor.to_tensor(history_games)
        |> Nx.squeeze(axes: [0])
        |> Nx.as_type(:f32)

      policy = policy_target(game, move)
      policy_mask = policy_mask(game)
      wdl = EchecsEngine.Value.target_from_record(record, game)
      moves_left = Nx.tensor([Map.get(record, "moves_left", 0) / 128.0], type: :f32)

      {:ok,
       {input,
        %{
          policy: policy,
          policy_mask: policy_mask,
          wdl: wdl,
          moves_left: moves_left
        }}}
    end
  end

  def example_from_record(_record), do: {:error, :missing_required_fields}

  defp decode_record!(line), do: Jason.decode!(line)

  defp normalize_paths(paths) when is_list(paths), do: paths
  defp normalize_paths(path) when is_binary(path), do: [path]

  defp record_lines!(path) do
    path
    |> File.stream!()
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
    |> Stream.map(&decode_record!/1)
  end

  defp parse_game(fen) do
    {:ok, Echecs.new_game(fen)}
  rescue
    error -> {:error, {:invalid_fen, error}}
  end

  defp parse_history_games(history_fens) when is_list(history_fens) do
    Enum.reduce_while(history_fens, {:ok, []}, fn fen, {:ok, games} ->
      case parse_game(fen) do
        {:ok, game} -> {:cont, {:ok, games ++ [game]}}
        {:error, reason} -> {:halt, {:error, {:invalid_history_fen, reason}}}
      end
    end)
  end

  defp parse_history_games(_other), do: {:error, :invalid_history_fens}

  defp legal_move_from_uci(game, <<from::binary-size(2), to::binary-size(2), rest::binary>>) do
    promotion = parse_promotion(rest)
    from_idx = Echecs.Board.to_index(from)
    to_idx = Echecs.Board.to_index(to)

    case Enum.find(Echecs.legal_moves(game), fn move ->
           move.from == from_idx and move.to == to_idx and move.promotion == promotion
         end) do
      nil -> {:error, {:illegal_move, from <> to <> rest}}
      move -> {:ok, move}
    end
  rescue
    _error -> {:error, {:invalid_move, from <> to <> rest}}
  end

  defp legal_move_from_uci(_game, uci), do: {:error, {:invalid_move, uci}}

  defp parse_promotion(""), do: nil
  defp parse_promotion("q"), do: :queen
  defp parse_promotion("r"), do: :rook
  defp parse_promotion("b"), do: :bishop
  defp parse_promotion("n"), do: :knight

  defp policy_target(game, move) do
    Nx.broadcast(0.0, {4672})
    |> Nx.put_slice([EchecsEngine.Policy.move_index(game, move)], Nx.tensor([1.0], type: :f32))
  end

  defp policy_mask(game) do
    Enum.reduce(Echecs.legal_moves(game), Nx.broadcast(0.0, {4672}), fn move, mask ->
      Nx.put_slice(
        mask,
        [EchecsEngine.Policy.move_index(game, move)],
        Nx.tensor([1.0], type: :f32)
      )
    end)
  end

  defp batch_examples(examples) do
    {inputs, targets} = Enum.unzip(examples)

    batched_targets = %{
      policy: Nx.stack(Enum.map(targets, & &1.policy)),
      policy_mask: Nx.stack(Enum.map(targets, & &1.policy_mask)),
      wdl: Nx.stack(Enum.map(targets, & &1.wdl)),
      moves_left: Nx.stack(Enum.map(targets, & &1.moves_left))
    }

    {Nx.stack(inputs), batched_targets}
  end
end
