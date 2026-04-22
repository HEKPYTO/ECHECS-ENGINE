defmodule EchecsZero.MCTS.Node do
  @moduledoc """
  Represents a node in the Monte Carlo Tree Search.
  """

  defstruct [
    :game,
    visits: 0,
    total_value: 0.0,
    prior_prob: 0.0,
    children: %{}
  ]

  @type t :: %__MODULE__{
          game: any(),
          visits: integer(),
          total_value: float(),
          prior_prob: float(),
          children: %{any() => t()}
        }
end
