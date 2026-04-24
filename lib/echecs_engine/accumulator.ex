defmodule EchecsEngine.Accumulator do
  @moduledoc """
  Builds deterministic sparse-evaluator inputs from game states.
  """

  @spec from_game(Echecs.Game.t(), keyword()) :: Nx.Tensor.t()
  def from_game(game, opts \\ []) do
    EchecsEngine.NNUE.Accumulator.refresh(game, opts)
  end

  @spec batch_from_games([Echecs.Game.t()], keyword()) :: Nx.Tensor.t()
  def batch_from_games(games, opts \\ []) do
    games
    |> Enum.map(&from_game(&1, opts))
    |> Nx.stack()
  end
end
