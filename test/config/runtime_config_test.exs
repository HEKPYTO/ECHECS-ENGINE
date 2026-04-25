defmodule EchecsEngine.RuntimeConfigTest do
  use ExUnit.Case, async: false

  @runtime Path.expand("../../config/runtime.exs", __DIR__)

  setup do
    previous_target = System.get_env("XLA_TARGET")
    previous_preallocate = System.get_env("EXLA_PREALLOCATE")
    previous_fraction = System.get_env("EXLA_MEMORY_FRACTION")
    previous_clients = Application.get_env(:exla, :clients)
    previous_preferred = Application.get_env(:exla, :preferred_clients)

    on_exit(fn ->
      restore_env("XLA_TARGET", previous_target)
      restore_env("EXLA_PREALLOCATE", previous_preallocate)
      restore_env("EXLA_MEMORY_FRACTION", previous_fraction)
      Application.put_env(:exla, :clients, previous_clients)
      Application.put_env(:exla, :preferred_clients, previous_preferred)
    end)

    :ok
  end

  test "uses host as the default preferred client" do
    {runtime, _imports} = Config.Reader.read_imports!(@runtime)

    exla_config = runtime[:exla]
    assert exla_config[:preferred_clients] == [:host]
    clients = exla_config[:clients]
    assert clients[:host][:platform] == :host
    assert clients[:cuda][:platform] == :cuda
  end

  test "prefers cuda when XLA_TARGET is a cuda target" do
    System.put_env("XLA_TARGET", "cuda12")
    {runtime, _imports} = Config.Reader.read_imports!(@runtime)

    assert runtime[:exla][:preferred_clients] == [:cuda, :host]
  end

  defp restore_env(key, nil), do: System.delete_env(key)
  defp restore_env(key, value), do: System.put_env(key, value)
end
