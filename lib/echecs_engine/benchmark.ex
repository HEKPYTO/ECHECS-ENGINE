defmodule EchecsEngine.Benchmark do
  @moduledoc """
  Runs deterministic FEN best-move benchmark suites.
  """

  @type position_result :: %{
          id: String.t(),
          fen: String.t(),
          expected: [String.t()],
          got: String.t() | nil,
          correct?: boolean(),
          time_us: non_neg_integer()
        }

  @type result :: %{
          total: non_neg_integer(),
          correct: non_neg_integer(),
          accuracy: float(),
          positions: [position_result()]
        }

  @type match_result :: %{
          games: non_neg_integer(),
          results: [map()]
        }

  @spec run_jsonl!(String.t(), keyword()) :: result()
  def run_jsonl!(path, opts \\ []) do
    best_move = Keyword.get(opts, :best_move, &EchecsEngine.best_move/2)
    search_opts = Keyword.get(opts, :search_opts, [])

    positions =
      path
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Stream.map(&Jason.decode!/1)
      |> Enum.map(&run_position(&1, best_move, search_opts))

    total = length(positions)
    correct = Enum.count(positions, & &1.correct?)

    %{
      total: total,
      correct: correct,
      accuracy: if(total == 0, do: 0.0, else: correct / total),
      positions: positions
    }
  end

  @spec run_match_jsonl!(String.t(), keyword()) :: match_result()
  def run_match_jsonl!(path, opts \\ []) do
    white = Keyword.get(opts, :white, &EchecsEngine.best_move/2)
    black = Keyword.get(opts, :black, &EchecsEngine.best_move/2)
    plies = Keyword.get(opts, :plies, 80)
    search_opts = Keyword.get(opts, :search_opts, [])

    results =
      path
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Stream.map(&Jason.decode!/1)
      |> Enum.map(fn record -> play_match(record, white, black, plies, search_opts) end)

    %{games: length(results), results: results}
  end

  defp run_position(record, best_move, search_opts) do
    fen = Map.fetch!(record, "fen")
    expected = expected_moves(record)

    {time_us, result} = :timer.tc(fn -> best_move.(fen, search_opts) end)
    got = move_from_result(result)

    %{
      id: Map.get(record, "id", fen),
      fen: fen,
      expected: expected,
      got: got,
      correct?: got in expected,
      time_us: time_us
    }
  end

  defp expected_moves(%{"best" => moves}) when is_list(moves), do: moves
  defp expected_moves(%{"best" => move}) when is_binary(move), do: [move]
  defp expected_moves(%{"bm" => move}) when is_binary(move), do: [move]
  defp expected_moves(_record), do: []

  defp move_from_result({:ok, move}), do: move
  defp move_from_result({:terminal, _status}), do: "0000"
  defp move_from_result({:error, _reason}), do: nil

  defp play_match(record, white, black, plies, search_opts) do
    fen = Map.fetch!(record, "fen")
    game = Echecs.new_game(fen)

    {final_game, moves, status} =
      Enum.reduce_while(1..plies, {game, [], :ok}, fn _ply, {current_game, moves, _status} ->
        if Echecs.legal_moves(current_game) == [] do
          {:halt, {current_game, moves, Echecs.status(current_game)}}
        else
          history_games =
            moves
            |> Enum.reduce({Echecs.new_game(fen), []}, fn uci, {replay_game, history} ->
              {:ok, next_game} = apply_uci_move(replay_game, uci)
              {next_game, [replay_game | history] |> Enum.take(7)}
            end)
            |> elem(1)

          player = if(current_game.turn == :white, do: white, else: black)

          case player.(
                 Echecs.FEN.to_string(current_game),
                 Keyword.put(search_opts, :history_games, history_games)
               ) do
            {:ok, move} ->
              case apply_uci_move(current_game, move) do
                {:ok, next_game} -> {:cont, {next_game, moves ++ [move], :ok}}
                {:error, reason} -> {:halt, {current_game, moves, {:illegal_move, move, reason}}}
              end

            {:terminal, status} ->
              {:halt, {current_game, moves, status}}

            {:error, reason} ->
              {:halt, {current_game, moves, {:error, reason}}}
          end
        end
      end)

    %{
      fen: fen,
      moves: moves,
      final_fen: Echecs.FEN.to_string(final_game),
      status: status
    }
  end

  @doc """
  Returns a simple SPRT-style decision from aggregate game outcomes.

  This uses a log-likelihood ratio over win/loss decisive games. Draws count
  toward sample size but do not move the ratio.
  """
  @spec sprt(map(), keyword()) :: map()
  def sprt(%{wins: wins, losses: losses, draws: draws}, opts \\ []) do
    elo0 = Keyword.get(opts, :elo0, 0.0)
    elo1 = Keyword.get(opts, :elo1, 5.0)
    alpha = Keyword.get(opts, :alpha, 0.05)
    beta = Keyword.get(opts, :beta, 0.05)

    p0 = elo_to_score(elo0)
    p1 = elo_to_score(elo1)

    llr =
      wins * :math.log(p1 / p0) +
        losses * :math.log((1.0 - p1) / (1.0 - p0))

    lower = :math.log(beta / (1.0 - alpha))
    upper = :math.log((1.0 - beta) / alpha)

    %{
      wins: wins,
      losses: losses,
      draws: draws,
      llr: llr,
      lower_bound: lower,
      upper_bound: upper,
      decision: sprt_decision(llr, lower, upper)
    }
  end

  defp elo_to_score(elo), do: 1.0 / (1.0 + :math.pow(10.0, -elo / 400.0))

  defp sprt_decision(llr, lower, upper) do
    cond do
      llr >= upper -> :accept
      llr <= lower -> :reject
      true -> :continue
    end
  end

  defp apply_uci_move(game, <<from::binary-size(2), to::binary-size(2), rest::binary>>) do
    Echecs.make_move(
      game,
      Echecs.Board.to_index(from),
      Echecs.Board.to_index(to),
      parse_promotion(rest)
    )
  rescue
    error -> {:error, error}
  end

  defp apply_uci_move(_game, _move), do: {:error, :invalid_move}

  defp parse_promotion(""), do: nil
  defp parse_promotion("q"), do: :queen
  defp parse_promotion("r"), do: :rook
  defp parse_promotion("b"), do: :bishop
  defp parse_promotion("n"), do: :knight
  defp parse_promotion(_), do: nil
end
