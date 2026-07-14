defmodule EchecsEngine.RuntimeConfigTest do
  use ExUnit.Case, async: false

  @runtime Path.expand("../../config/runtime.exs", __DIR__)

  setup do
    previous_target = System.get_env("XLA_TARGET")
    previous_preallocate = System.get_env("EXLA_PREALLOCATE")
    previous_fraction = System.get_env("EXLA_MEMORY_FRACTION")
    previous_xla_flags = System.get_env("XLA_FLAGS")
    previous_miopen_find = System.get_env("MIOPEN_FIND_MODE")
    previous_miopen_db = System.get_env("MIOPEN_USER_DB_PATH")
    previous_clients = Application.get_env(:exla, :clients)
    previous_preferred = Application.get_env(:exla, :preferred_clients)

    on_exit(fn ->
      restore_env("XLA_TARGET", previous_target)
      restore_env("EXLA_PREALLOCATE", previous_preallocate)
      restore_env("EXLA_MEMORY_FRACTION", previous_fraction)
      restore_env("XLA_FLAGS", previous_xla_flags)
      restore_env("MIOPEN_FIND_MODE", previous_miopen_find)
      restore_env("MIOPEN_USER_DB_PATH", previous_miopen_db)
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

  test "prefers rocm when XLA_TARGET is rocm" do
    System.put_env("XLA_TARGET", "rocm")
    System.delete_env("XLA_FLAGS")
    System.delete_env("MIOPEN_FIND_MODE")
    System.delete_env("MIOPEN_USER_DB_PATH")

    {runtime, _imports} = Config.Reader.read_imports!(@runtime)

    assert runtime[:exla][:preferred_clients] == [:rocm, :host]

    clients = runtime[:exla][:clients]
    assert clients[:rocm][:platform] == :rocm
    assert clients[:rocm][:preallocate] == false
    assert clients[:rocm][:memory_fraction] == 0.25
  end

  test "disables the xla gpu autotuner and configures miopen cache for rocm" do
    System.put_env("XLA_TARGET", "rocm")
    System.delete_env("XLA_FLAGS")
    System.delete_env("MIOPEN_FIND_MODE")
    System.delete_env("MIOPEN_USER_DB_PATH")

    Config.Reader.read_imports!(@runtime)

    assert System.get_env("XLA_FLAGS") =~ "--xla_gpu_autotune_level=0"
    assert System.get_env("MIOPEN_FIND_MODE") == "2"
    assert System.get_env("MIOPEN_USER_DB_PATH") == "/tmp/miopen_user_db"
  end

  test "respects an operator-provided xla autotune level for rocm" do
    System.put_env("XLA_TARGET", "rocm")
    System.put_env("XLA_FLAGS", "--xla_gpu_autotune_level=1")

    Config.Reader.read_imports!(@runtime)

    assert System.get_env("XLA_FLAGS") == "--xla_gpu_autotune_level=1"
  end

  defp restore_env(key, nil), do: System.delete_env(key)
  defp restore_env(key, value), do: System.put_env(key, value)
end
