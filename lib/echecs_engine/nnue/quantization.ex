defmodule EchecsEngine.NNUE.Quantization do
  @moduledoc """
  Quantizes sparse evaluator artifacts for compact checkpoint storage.

  The current runtime evaluator still consumes floats, so loading dequantizes the
  tensors. This gives deterministic artifact compression now and keeps the door
  open for an integer evaluator kernel later.
  """

  @qmax 32_767.0

  @spec quantize_artifact(map(), keyword()) :: map()
  def quantize_artifact(artifact, _opts \\ []) do
    artifact
    |> Map.put(:quantized?, true)
    |> Map.update!(:feature_table, fn table ->
      Map.new(table, fn {feature_idx, contribution} ->
        {feature_idx, quantize_contribution(contribution)}
      end)
    end)
    |> Map.update!(:w1, &quantize_tensor/1)
    |> Map.update!(:b1, &quantize_tensor/1)
    |> Map.update!(:w2, &quantize_tensor/1)
    |> Map.update!(:b2, &quantize_tensor/1)
  end

  @spec dequantize_artifact(map()) :: map()
  def dequantize_artifact(%{quantized?: true} = artifact) do
    artifact
    |> Map.delete(:quantized?)
    |> Map.update!(:feature_table, fn table ->
      Map.new(table, fn {feature_idx, payload} ->
        {feature_idx, dequantize_contribution(payload)}
      end)
    end)
    |> Map.update!(:w1, &dequantize_tensor/1)
    |> Map.update!(:b1, &dequantize_tensor/1)
    |> Map.update!(:w2, &dequantize_tensor/1)
    |> Map.update!(:b2, &dequantize_tensor/1)
  end

  def dequantize_artifact(artifact), do: artifact

  defp quantize_contribution(%Nx.Tensor{} = tensor), do: quantize_tensor(tensor)

  defp quantize_contribution(%{indices: indices, values: values}) do
    max_abs =
      values
      |> Enum.map(&abs/1)
      |> Enum.max(fn -> 0.0 end)

    scale = if max_abs > 0.0, do: max_abs / @qmax, else: 1.0

    quantized_values =
      Enum.map(values, fn value ->
        value
        |> Kernel./(scale)
        |> round()
        |> max(round(-@qmax))
        |> min(round(@qmax))
      end)

    %{sparse?: true, indices: indices, values: quantized_values, scale: scale}
  end

  defp dequantize_contribution(%{sparse?: true, indices: indices, values: values, scale: scale}) do
    %{indices: indices, values: Enum.map(values, &(&1 * scale))}
  end

  defp dequantize_contribution(payload), do: dequantize_tensor(payload)

  defp quantize_tensor(%Nx.Tensor{} = tensor) do
    max_abs =
      tensor
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    scale = if max_abs > 0.0, do: max_abs / @qmax, else: 1.0

    values =
      tensor
      |> Nx.divide(scale)
      |> Nx.round()
      |> Nx.clip(-@qmax, @qmax)
      |> Nx.as_type({:s, 16})

    %{values: values, scale: scale}
  end

  defp dequantize_tensor(%{values: values, scale: scale}) do
    values
    |> Nx.as_type(:f32)
    |> Nx.multiply(scale)
  end
end
