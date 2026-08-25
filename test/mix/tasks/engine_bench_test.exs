defmodule Mix.Tasks.Engine.BenchTest do
  use ExUnit.Case, async: false

  alias Mix.Tasks.Engine.Bench, as: BenchTask

  test "match mode requires both engine commands, arguments, and an opening book" do
    assert_raise Mix.Error, ~r/--base-cmd CMD/, fn ->
      BenchTask.run(["match", "--base-cmd", "mix"])
    end
  end
end
