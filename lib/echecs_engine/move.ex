defmodule EchecsEngine.Move do
  @moduledoc """
  Packed-move to UCI string conversion.

  `Echecs` encodes moves as integers packing `from`, `to`, promotion and
  special flags. This module is the single translation point between that
  internal representation and the UCI strings consumed by `EchecsEngine`,
  `EchecsEngine.Search`, and `EchecsEngine.UCI`.

  `nil` maps to `"0000"` (the UCI null-move sentinel) so search and UCI
  layers can preserve the same sentinel without branching.

  ## Examples

      iex> move = Echecs.Move.pack(12, 28, nil, nil)
      iex> EchecsEngine.Move.to_uci(move)
      "e2e4"
  """

  require Echecs.Move

  @doc """
  Converts a packed move integer (or `nil`) to a UCI string.

  Promotion suffixes are `q`, `r`, `b`, `n`. Returns `"0000"` for `nil`.
  """
  @spec to_uci(integer() | nil) :: String.t()
  def to_uci(nil), do: "0000"

  def to_uci(move),
    do:
      Echecs.Board.to_algebraic(Echecs.Move.unpack_from(move)) <>
        Echecs.Board.to_algebraic(Echecs.Move.unpack_to(move)) <>
        promotion(Echecs.Move.unpack_promotion(move))

  defp promotion(nil), do: ""
  defp promotion(:queen), do: "q"
  defp promotion(:rook), do: "r"
  defp promotion(:bishop), do: "b"
  defp promotion(:knight), do: "n"
end
