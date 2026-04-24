defmodule EchecsEngine.NNUE.Accumulator do
  @moduledoc """
  Refresh/update helpers for the sparse evaluator accumulator.

  The sparse evaluator consumes a compact 3072-wide vector produced by summing
  learned feature-table contributions for active HalfKP-like features.
  """

  @size 3072

  @spec refresh(Echecs.Game.t(), keyword()) :: Nx.Tensor.t()
  def refresh(game, opts \\ []) do
    active_features =
      game
      |> EchecsEngine.NNUE.Features.active_indices()
      |> Map.values()
      |> List.flatten()

    feature_table =
      Keyword.get(opts, :feature_table) ||
        raise ArgumentError,
              "feature_table is required for NNUE accumulator refresh; hashed fallback has been removed"

    learned_accumulator(active_features, feature_table)
  end

  @spec update_after_move(Nx.Tensor.t(), Echecs.Game.t(), Echecs.Move.t(), keyword()) ::
          Nx.Tensor.t()
  def update_after_move(accumulator, game, move, opts \\ []) do
    feature_table = feature_table!(opts)
    delta = EchecsEngine.NNUE.Features.delta_indices(game, move)

    accumulator
    |> remove_contributions(delta.removed, feature_table)
    |> add_contributions(delta.added, feature_table)
  end

  defp learned_accumulator(indices, feature_table) do
    zero = Nx.broadcast(0.0, {@size})

    add_contributions(zero, indices, feature_table)
  end

  defp add_contributions(accumulator, indices, feature_table) do
    Enum.reduce(indices, accumulator, fn feature_idx, acc ->
      case Map.get(feature_table, feature_idx) do
        nil -> acc
        contribution -> add_contribution(acc, contribution)
      end
    end)
  end

  defp remove_contributions(accumulator, indices, feature_table) do
    Enum.reduce(indices, accumulator, fn feature_idx, acc ->
      case Map.get(feature_table, feature_idx) do
        nil -> acc
        contribution -> remove_contribution(acc, contribution)
      end
    end)
  end

  defp add_contribution(accumulator, %Nx.Tensor{} = contribution),
    do: Nx.add(accumulator, contribution)

  defp add_contribution(accumulator, %{indices: indices, values: values}),
    do: add_sparse(accumulator, indices, values, 1.0)

  defp remove_contribution(accumulator, %Nx.Tensor{} = contribution),
    do: Nx.subtract(accumulator, contribution)

  defp remove_contribution(accumulator, %{indices: indices, values: values}),
    do: add_sparse(accumulator, indices, values, -1.0)

  defp add_sparse(accumulator, indices, values, sign) do
    Enum.zip(indices, values)
    |> Enum.reduce(accumulator, fn {index, value}, acc ->
      current =
        acc
        |> Nx.slice([index], [1])
        |> Nx.add(Nx.tensor([value * sign], type: :f32))

      Nx.put_slice(acc, [index], current)
    end)
  end

  defp feature_table!(opts) do
    Keyword.get(opts, :feature_table) ||
      raise ArgumentError,
            "feature_table is required for NNUE accumulator refresh; hashed fallback has been removed"
  end
end
