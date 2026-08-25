defmodule EchecsEngine.Move do
  @moduledoc false

  require Echecs.Move

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
