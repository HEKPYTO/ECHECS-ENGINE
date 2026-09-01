defmodule EchecsEngine do
  @moduledoc """
  Public pure-Elixir chess-engine API.

  Thin facade over `EchecsEngine.Search`. `Echecs` owns all chess rules and
  packed move encoding; this module owns the engine contract: FEN parsing,
  option validation, and the stable `{:ok, map()}` / `{:terminal, atom()}` /
  `{:error, term()}` return shape consumed by `EchecsEngine.UCI` and
  `Mix.Tasks.Engine.Best`.

  ## Search limits

  Exactly one of `depth`, `nodes`, or `movetime` may be supplied. When none
  is given, depth `4` is used. All limits are delegated to
  `EchecsEngine.Search.best_move/2`.

  ## Examples

      iex> {:ok, info} = EchecsEngine.analyze("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth: 2)
      iex> is_binary(info.bestmove)
      true

      iex> EchecsEngine.best_move("8/8/8/8/8/8/8/8 w - - 0 1", depth: 1)
      {:terminal, :stalemate}
  """

  @allowed [:depth, :nodes, :movetime, :reporter, :stop_ref]

  @typedoc "Supported public search options."
  @type analyze_opt ::
          {:depth, pos_integer()}
          | {:nodes, pos_integer()}
          | {:movetime, pos_integer()}
          | {:reporter, (map() -> any())}
          | {:stop_ref, :atomics.atomics_ref()}

  @typedoc "Successful search info returned by `analyze/2`."
  @type analyze_info :: %{
          required(:bestmove) => String.t(),
          required(:score) => integer(),
          required(:depth) => non_neg_integer(),
          required(:seldepth) => non_neg_integer(),
          required(:nodes) => non_neg_integer(),
          required(:time_ms) => non_neg_integer(),
          required(:pv) => [String.t()],
          required(:tt_hits) => non_neg_integer(),
          required(:tt_slots) => pos_integer(),
          required(:tt_entries) => non_neg_integer()
        }

  @doc """
  Analyses `fen` and returns the best move with full search info.

  Returns `{:error, :invalid_fen}` for non-binary input, `{:error, {:unknown_option, key}}`
  for unsupported options, and `{:terminal, reason}` for positions with no legal moves
  or drawn positions.

  ## Options

    * `:depth` - search depth in plies (default `4`)
    * `:nodes` - hard node limit
    * `:movetime` - soft time limit in milliseconds
    * `:reporter` - `fn info -> any()` called after each completed iteration
    * `:stop_ref` - `:atomics` ref; search stops cooperatively when index `1` is non-zero
  """
  @spec analyze(String.t(), keyword()) ::
          {:ok, analyze_info()} | {:terminal, :checkmate | :stalemate | :draw} | {:error, term()}
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

  @doc """
  Returns only the best UCI move string for `fen`.

  Convenience wrapper around `analyze/2` that discards the info map.

  ## Examples

      iex> {:ok, "e2e4"} = EchecsEngine.best_move("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", depth: 1)
      ...> {:ok, move} = EchecsEngine.best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth: 1)
      ...> String.length(move) >= 4
      true
  """
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
