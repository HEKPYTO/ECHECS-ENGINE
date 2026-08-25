defmodule EchecsEngine do
  @moduledoc "Public pure-Elixir chess-engine API."

  @allowed [:depth, :nodes, :movetime, :reporter, :stop_ref]

  @spec analyze(String.t(), keyword()) ::
          {:ok, map()} | {:terminal, :checkmate | :stalemate | :draw} | {:error, term()}
  def analyze(fen, opts \\ [])

  def analyze(fen, opts) when is_binary(fen) do
    with :ok <- public_opts(opts), {:ok, game} <- parse(fen) do
      case EchecsEngine.Search.best_move(game, opts) do
        {:ok, move, info} -> {:ok, Map.put(info, :bestmove, EchecsEngine.Move.to_uci(move))}
        other -> other
      end
    end
  end

  def analyze(_, _), do: {:error, :invalid_fen}

  @spec best_move(String.t(), keyword()) ::
          {:ok, String.t()} | {:terminal, atom()} | {:error, term()}
  def best_move(fen, opts \\ []) do
    case analyze(fen, opts) do
      {:ok, %{bestmove: move}} -> {:ok, move}
      other -> other
    end
  end

  defp public_opts(opts) when is_list(opts) do
    case Enum.find(opts, fn {key, _} -> key not in @allowed end) do
      nil -> :ok
      {key, _} -> {:error, {:unknown_option, key}}
    end
  end

  defp public_opts(_), do: {:error, :invalid_options}

  defp parse(fen) do
    {:ok, Echecs.new_game(fen)}
  rescue
    error -> {:error, error}
  end
end
