defmodule EchecsEngine.NNUE.QuantizationTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.NNUE.Quantization

  test "quantizes and dequantizes sparse evaluator tensors" do
    artifact = %{
      feature_table: %{7 => Nx.broadcast(0.25, {3072})},
      w1: Nx.broadcast(0.125, {3072, 32}),
      b1: Nx.broadcast(-0.25, {32}),
      w2: Nx.broadcast(0.5, {32, 1}),
      b2: Nx.tensor([0.03125], type: :f32)
    }

    quantized = Quantization.quantize_artifact(artifact)

    assert quantized.quantized? == true
    assert Nx.type(quantized.w1.values) == {:s, 16}
    assert Nx.type(quantized.feature_table[7].values) == {:s, 16}

    dequantized = Quantization.dequantize_artifact(quantized)

    assert close?(dequantized.w1, artifact.w1)
    assert close?(dequantized.b1, artifact.b1)
    assert close?(dequantized.w2, artifact.w2)
    assert close?(dequantized.b2, artifact.b2)
    assert close?(dequantized.feature_table[7], artifact.feature_table[7])
  end

  test "passes unquantized artifacts through unchanged" do
    artifact = %{w1: Nx.tensor([1.0])}

    assert Quantization.dequantize_artifact(artifact) == artifact
  end

  test "quantizes and dequantizes compact sparse feature rows" do
    artifact = %{
      feature_table: %{7 => %{indices: [1, 9], values: [0.25, -0.5]}},
      w1: Nx.broadcast(0.125, {3072, 32}),
      b1: Nx.broadcast(0.0, {32}),
      w2: Nx.broadcast(0.5, {32, 1}),
      b2: Nx.tensor([0.03125], type: :f32)
    }

    quantized = Quantization.quantize_artifact(artifact)

    assert quantized.feature_table[7].sparse? == true
    assert quantized.feature_table[7].indices == [1, 9]

    dequantized = Quantization.dequantize_artifact(quantized)

    assert dequantized.feature_table[7].indices == [1, 9]
    assert_in_delta Enum.at(dequantized.feature_table[7].values, 0), 0.25, 1.0e-3
    assert_in_delta Enum.at(dequantized.feature_table[7].values, 1), -0.5, 1.0e-3
  end

  defp close?(left, right) do
    left
    |> Nx.subtract(right)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
    |> Kernel.<(1.0e-3)
  end
end
