defmodule EchecsEngineTest do
  use ExUnit.Case
  doctest EchecsEngine

  test "greets the world" do
    assert EchecsEngine.hello() == :world
  end
end
