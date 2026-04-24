defmodule EchecsEngine.Search.TimeManager do
  @moduledoc """
  Converts UCI/search options into bounded search budgets.
  """

  defstruct max_iterations: 128, deadline_ms: nil

  @type t :: %__MODULE__{max_iterations: pos_integer() | :infinity, deadline_ms: integer() | nil}

  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    now = System.monotonic_time(:millisecond)
    movetime = Keyword.get(opts, :movetime)

    %__MODULE__{
      max_iterations: iteration_cap(opts),
      deadline_ms: if(is_integer(movetime), do: now + max(movetime - 2, 1), else: nil)
    }
  end

  @spec max_iterations(t()) :: pos_integer() | :infinity
  def max_iterations(%__MODULE__{max_iterations: max_iterations}), do: max_iterations

  @spec expired?(t(), non_neg_integer()) :: boolean()
  def expired?(%__MODULE__{max_iterations: :infinity} = budget, _completed_iterations) do
    deadline_expired?(budget)
  end

  def expired?(%__MODULE__{} = budget, completed_iterations) do
    completed_iterations >= budget.max_iterations or deadline_expired?(budget)
  end

  @spec budgeted?(keyword()) :: boolean()
  def budgeted?(opts) do
    Keyword.has_key?(opts, :nodes) or Keyword.has_key?(opts, :depth) or
      Keyword.has_key?(opts, :movetime) or Keyword.has_key?(opts, :iterations) or
      Keyword.has_key?(opts, :wtime) or Keyword.has_key?(opts, :btime) or
      Keyword.get(opts, :infinite, false)
  end

  defp iteration_cap(opts) do
    cond do
      opts[:iterations] == :infinity -> :infinity
      is_integer(opts[:iterations]) -> max(opts[:iterations], 1)
      is_integer(opts[:nodes]) -> max(opts[:nodes], 1)
      is_integer(opts[:depth]) -> max(opts[:depth] * 32, 1)
      is_integer(opts[:movetime]) -> max(min(opts[:movetime] * 4, 1024), 1)
      opts[:infinite] == true -> :infinity
      true -> 128
    end
  end

  defp deadline_expired?(%__MODULE__{deadline_ms: nil}), do: false

  defp deadline_expired?(%__MODULE__{deadline_ms: deadline_ms}) do
    System.monotonic_time(:millisecond) >= deadline_ms
  end
end
