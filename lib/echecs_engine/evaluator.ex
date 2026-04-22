defmodule EchecsEngine.SparseEvaluator do
  @moduledoc """
  Implements an updatable neural network architecture.

  This module defines the evaluation structure:
  - Eliminates classical evaluation.
  - Utilizes an expanded accumulator (3072 dimensions).
  - Employs SCReLU (Squared Clipped ReLU) for feature extraction
    without increasing network depth, allowing for efficient CPU evaluation.
  """

  import Nx.Defn

  @doc """
  Evaluates the accumulator vector into a scalar board score using 
  SCReLU activations.

  ## Parameters
  - `acc`: The accumulator tensor of shape `{3072}` (or batch mapped).
  - `w1`: Dense layer 1 weights `{3072, 32}`.
  - `b1`: Dense layer 1 bias `{32}`.
  - `w2`: Dense layer 2 weights `{32, 1}`.
  - `b2`: Dense layer 2 bias `{1}`.
  """
  defn evaluate(acc, w1, b1, w2, b2) do
    clipped = Nx.clip(acc, 0.0, 1.0)

    h1 = Nx.multiply(clipped, clipped)

    h2_pre = Nx.dot(h1, w1) |> Nx.add(b1)
    h2_clipped = Nx.clip(h2_pre, 0.0, 1.0)
    h2 = Nx.multiply(h2_clipped, h2_clipped)

    Nx.dot(h2, w2) |> Nx.add(b2)
  end
end
