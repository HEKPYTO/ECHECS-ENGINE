defmodule EchecsEngine.Search.Engine do
  @moduledoc """
  High-level search facade for public APIs and UCI.
  """

  alias EchecsEngine.Search.TimeManager

  @default_depth 2
  @evaluator_cache_key {__MODULE__, :default_evaluator}

  @spec best_move(Echecs.Game.t(), keyword()) ::
          {:ok, %Echecs.Move{}} | {:terminal, atom()} | {:error, term()}
  def best_move(game, opts \\ []) do
    if Echecs.legal_moves(game) == [] do
      {:terminal, Echecs.status(game)}
    else
      search_opts =
        opts
        |> default_search_opts()
        |> maybe_attach_evaluator()

      case Keyword.get(search_opts, :backend, :alpha_beta) do
        :mcts ->
          {:ok, EchecsEngine.MCTS.search_best_move(game, search_opts)}

        _alpha_beta ->
          with :ok <- ensure_alpha_beta_evaluator(search_opts) do
            {move, _stats} = EchecsEngine.Search.AlphaBeta.search_best_move(game, search_opts)
            {:ok, move}
          end
      end
    end
  end

  defp default_search_opts(opts) do
    if TimeManager.budgeted?(opts) do
      opts
    else
      Keyword.put_new(opts, :depth, @default_depth)
    end
  end

  defp maybe_attach_evaluator(opts) do
    cond do
      Keyword.has_key?(opts, :evaluator_weights) ->
        opts

      Keyword.has_key?(opts, :evaluator_fn) ->
        opts

      true ->
        case default_evaluator_weights(Keyword.get(opts, :evaluator_paths)) do
          {:ok, weights} -> Keyword.put(opts, :evaluator_weights, weights)
          _other -> opts
        end
    end
  end

  defp default_evaluator_weights(nil) do
    paths = EchecsEngine.Checkpoint.default_evaluator_paths()
    signature = evaluator_paths_signature(paths)

    case :persistent_term.get(@evaluator_cache_key, :unset) do
      {:ok, weights, ^signature} ->
        {:ok, weights}

      _miss_or_unset ->
        case load_evaluator_weights(paths) do
          {:ok, weights} = value ->
            :persistent_term.put(@evaluator_cache_key, {:ok, weights, signature})
            value

          :error ->
            :error
        end
    end
  end

  defp default_evaluator_weights(paths) when is_list(paths), do: load_evaluator_weights(paths)

  defp evaluator_paths_signature(paths) do
    Enum.map(paths, fn path ->
      case File.stat(path, time: :posix) do
        {:ok, stat} -> {path, stat.mtime, stat.size}
        {:error, reason} -> {path, reason}
      end
    end)
  end

  defp load_evaluator_weights(paths) do
    case EchecsEngine.Checkpoint.load_evaluator_state(paths) do
      {:ok, %{artifact: artifact}} ->
        {:ok, EchecsEngine.NNUE.Quantization.dequantize_artifact(artifact)}

      {:error, _reason} ->
        :error
    end
  end

  defp ensure_alpha_beta_evaluator(opts) do
    cond do
      Keyword.get(opts, :allow_zero_evaluator, false) ->
        :ok

      Keyword.has_key?(opts, :evaluator_weights) ->
        :ok

      Keyword.has_key?(opts, :evaluator_fn) ->
        :ok

      Keyword.has_key?(opts, :inference) ->
        :ok

      true ->
        {:error, :missing_evaluator}
    end
  end
end
