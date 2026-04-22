defmodule EchecsEngine.NNUE do
  @moduledoc """
  Implements the Fast Efficiently Updatable Neural Network (NNUE).
  This module defines the shallow XLA-compiled `defn` network used
  at the leaves of the Alpha-Beta search tree.
  """

  import Nx.Defn

  @doc """
  Evaluates the accumulator vector into a scalar board score.

  ## Parameters
  - `acc`: The accumulator tensor of shape `{256}`.
  - `w1`: Dense layer 1 weights `{256, 32}`.
  - `b1`: Dense layer 1 bias `{32}`.
  - `w2`: Dense layer 2 weights `{32, 1}`.
  - `b2`: Dense layer 2 bias `{1}`.
  """
  defn evaluate(acc, w1, b1, w2, b2) do
    # Clipped ReLU on the accumulator (standard NNUE practice)
    # We clip between 0 and 127 (or 1.0 depending on quantization).
    # Using standard ReLU for floating point implementation.
    h1 = Nx.max(0.0, acc)

    # Dense layer -> ReLU
    h2 = Nx.dot(h1, w1) |> Nx.add(b1) |> Nx.max(0.0)

    # Final dense layer (Output)
    Nx.dot(h2, w2) |> Nx.add(b2)
  end
end
