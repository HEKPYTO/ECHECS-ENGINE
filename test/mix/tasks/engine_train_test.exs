defmodule Mix.Tasks.EngineTrainTest do
  use ExUnit.Case, async: true

  test "requires a dataset path argument" do
    assert_raise Mix.Error, ~r/usage: mix engine.train path\/to\/train.jsonl/, fn ->
      Mix.Tasks.Engine.Train.run([])
    end
  end
end
