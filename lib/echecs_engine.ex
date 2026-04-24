defmodule EchecsEngine do
  @moduledoc """
  Public API for ECHECS-ENGINE.
  """

  @doc """
  Hello world.

  ## Examples

      iex> EchecsEngine.hello()
      :world

  """
  def hello do
    :world
  end

  @doc """
  Returns the best move for a FEN position as a UCI coordinate string.

  ## Examples

      iex> {:ok, move} = EchecsEngine.best_move("8/8/8/8/8/8/4P3/4K2k w - - 0 1", allow_zero_evaluator: true)
      iex> String.match?(move, ~r/^[a-h][1-8][a-h][1-8][qrbn]?$/)
      true

  """
  @spec best_move(String.t(), keyword()) ::
          {:ok, String.t()} | {:terminal, atom()} | {:error, term()}
  def best_move(fen, opts \\ []) when is_binary(fen) do
    with {:ok, game} <- parse_fen(fen),
         {:ok, search_opts} <- normalize_search_opts(opts) do
      case EchecsEngine.Search.Engine.best_move(game, search_opts) do
        {:ok, %Echecs.Move{} = move} -> {:ok, move_to_uci(move)}
        {:terminal, status} -> {:terminal, status}
        {:error, reason} -> {:error, reason}
      end
    end
  end

  defp parse_fen(fen) do
    {:ok, Echecs.new_game(fen)}
  rescue
    error -> {:error, error}
  end

  defp normalize_search_opts(opts) do
    with {:ok, history_games} <- parse_history_games(opts) do
      {:ok, Keyword.put(opts, :history_games, history_games)}
    end
  end

  defp parse_history_games(opts) do
    cond do
      Keyword.has_key?(opts, :history_games) and is_list(opts[:history_games]) ->
        {:ok, opts[:history_games]}

      Keyword.has_key?(opts, :history_fens) and is_list(opts[:history_fens]) ->
        Enum.reduce_while(opts[:history_fens], {:ok, []}, fn history_fen, {:ok, games} ->
          case parse_fen(history_fen) do
            {:ok, game} -> {:cont, {:ok, games ++ [game]}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)

      true ->
        {:ok, []}
    end
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
end
