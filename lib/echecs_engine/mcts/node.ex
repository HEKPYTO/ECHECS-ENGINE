defmodule EchecsEngine.MCTS.Node do
  @moduledoc """
  Represents a node in the Monte Carlo Tree Search.
  """

  defstruct [
    :game,
    history_games: [],
    depth: 0,
    visits: 0,
    total_value: 0.0,
    prior_prob: 0.0,
    children: %{}
  ]

  @type t :: %__MODULE__{
          game: any(),
          history_games: [Echecs.Game.t()],
          depth: non_neg_integer(),
          visits: integer(),
          total_value: float(),
          prior_prob: float(),
          children: %{any() => t()}
        }

  @spec value_from_parent_perspective(t()) :: float()
  def value_from_parent_perspective(%__MODULE__{visits: 0}), do: 0.0

  def value_from_parent_perspective(%__MODULE__{visits: visits, total_value: total_value}) do
    -total_value / visits
  end
end
