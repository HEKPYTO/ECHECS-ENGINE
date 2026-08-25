defmodule ProjectSurfaceTest do
  use ExUnit.Case, async: true

  test "runtime surface is pure Elixir" do
    for path <- ["mix.exs", "mix.lock", ".gitignore"] do
      source = File.read!(path)

      for legacy <- [":nx", ":axon", ":exla", "model", "gpu", "miopen"] do
        refute String.downcase(source) =~ legacy, "#{path} retains #{legacy}"
      end
    end

    refute File.exists?("lib/echecs_engine/mcts.ex")

    for obsolete <- ["Dockerfile.match", "Dockerfile.nvidia", "Dockerfile.rocm"] do
      refute File.exists?(obsolete)
    end
  end

  test "container surface runs the production UCI release" do
    dockerfile = File.read!("Dockerfile")
    compose = File.read!("docker-compose.yml")

    assert File.exists?(".dockerignore")
    assert dockerfile =~ "mix release"
    assert dockerfile =~ "EchecsEngine.UCI.run()"
    assert dockerfile =~ "USER 65532:65532"
    refute String.downcase(dockerfile) =~ ~r/cuda|gpu|nvidia|rocm|xla/
    assert compose =~ "stdin_open: true"
    assert compose =~ "tty: true"
  end
end
