defmodule EchecsEngine.Search.TimeManagerTest do
  use ExUnit.Case, async: true

  alias EchecsEngine.Search.TimeManager

  test "builds a node-limited budget from go options" do
    budget = TimeManager.new(nodes: 16)

    assert TimeManager.max_iterations(budget) == 16
    refute TimeManager.expired?(budget, 0)
    assert TimeManager.expired?(budget, 16)
  end

  test "builds a movetime budget with default node cap" do
    budget = TimeManager.new(movetime: 5)

    assert TimeManager.max_iterations(budget) > 0
    assert is_integer(budget.deadline_ms)
  end

  test "treats infinite analysis as uncapped" do
    budget = TimeManager.new(infinite: true)

    assert TimeManager.max_iterations(budget) == :infinity
    refute TimeManager.expired?(budget, 10_000)
  end
end
