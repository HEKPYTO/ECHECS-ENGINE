defmodule EchecsEngine.NNUE do
  @moduledoc """
  Implements the Fast Efficiently Updatable Neural Network (NNUE).

  Updated to reflect the latest Stockfish 17+ architecture:
  - Eliminates classical evaluation completely.
  - Utilizes a massively widened Accumulator (e.g., 3072 dimensions, simulated here).
  - Employs SCReLU (Squared Clipped ReLU) for extreme non-linear feature extraction
    without adding depth, allowing the CPU to evaluate rapidly.
  """

  import Nx.Defn

  @doc """
  Evaluates the accumulator vector into a scalar board score using 
  Stockfish 17+ style SCReLU activations.

  ## Parameters
  - `acc`: The accumulator tensor of shape `{3072}` (or batch mapped).
  - `w1`: Dense layer 1 weights `{3072, 32}`.
  - `b1`: Dense layer 1 bias `{32}`.
  - `w2`: Dense layer 2 weights `{32, 1}`.
  - `b2`: Dense layer 2 bias `{1}`.
  """
  defn evaluate(acc, w1, b1, w2, b2) do
    # SCReLU (Squared Clipped ReLU) is the hallmark of modern Stockfish NNUE.
    # Clip between 0.0 and 1.0 (standardized, SF uses 0..127 integer math)
    clipped = Nx.clip(acc, 0.0, 1.0)

    # Square the activation: x^2
    # This allows a shallow 2-layer network to capture highly non-linear piece interactions
    h1 = Nx.multiply(clipped, clipped)

    # Dense layer -> SCReLU
    h2_pre = Nx.dot(h1, w1) |> Nx.add(b1)
    h2_clipped = Nx.clip(h2_pre, 0.0, 1.0)
    h2 = Nx.multiply(h2_clipped, h2_clipped)

    # Final dense layer (Output)
    Nx.dot(h2, w2) |> Nx.add(b2)
  end
end
