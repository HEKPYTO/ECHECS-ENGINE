defmodule EchecsEngine.DockerConfigTest do
  use ExUnit.Case, async: true

  @compose Path.expand("../docker-compose.yml", __DIR__)
  @dockerfile Path.expand("../Dockerfile", __DIR__)
  @match_dockerfile Path.expand("../Dockerfile.match", __DIR__)

  test "engine-match is configured as a dockerized fastchess gate" do
    compose = File.read!(@compose)

    assert compose =~ "services:"
    assert compose =~ "\n  engine:\n"
    assert compose =~ "\n  engine-nvidia:\n"
    assert compose =~ "engine-match:"
    assert compose =~ ~r/engine-match:.*dockerfile: Dockerfile\.match/s
    assert compose =~ ~r/engine-match:.*image: echecs_engine:match/s
    assert compose =~ ~r/engine-match:.*XLA_TARGET=cpu/s
    assert compose =~ ~r/engine-match:.*\.\/models:\/app\/models:z/s
  end

  test "backend images use distinct tags and ABI-matched bases" do
    compose = File.read!(@compose)

    assert compose =~
             ~r/engine:\n.*BUILDER_BASE: elixir:1\.19-slim.*RUNTIME_BASE: debian:trixie-slim/s

    assert compose =~ ~r/engine:\n.*image: echecs_engine:cpu/s

    assert compose =~
             ~r/engine-nvidia:.*BUILDER_BASE: hexpm\/elixir:1\.19\.5-erlang-28\.3\.1-ubuntu-noble-20260410.*RUNTIME_BASE: nvidia\/cuda:12\.9\.1-devel-ubuntu24\.04/s

    assert compose =~ ~r/engine-nvidia:.*image: echecs_engine:nvidia/s
    assert compose =~ ~r/engine-nvidia:.*platform: linux\/amd64/s
    assert compose =~ ~r/engine-match:.*image: echecs_engine:match/s
  end

  test "Dockerfile supports CPU and CUDA build configuration only" do
    dockerfile = File.read!(@dockerfile)

    assert dockerfile =~ "ARG BUILDER_BASE=elixir:1.19-slim"
    assert dockerfile =~ "ARG RUNTIME_BASE=debian:trixie-slim"
    assert dockerfile =~ "ARG CUDA_RUNTIME_PACKAGES="
    assert dockerfile =~ "if [ \"${XLA_TARGET}\" = \"cuda12\" ]; then"
    assert dockerfile =~ "RUN mix deps.get --only $MIX_ENV"
    assert dockerfile =~ "RUN cd deps/echecs && elixir scripts/generate_magic_cache.exs"
  end

  test "Dockerfile.match builds fastchess and installs stockfish" do
    dockerfile = File.read!(@match_dockerfile)

    assert dockerfile =~ "ARG FASTCHESS_REF=v1.8.0-alpha"
    assert dockerfile =~ "https://github.com/Disservin/fastchess.git"
    assert dockerfile =~ "stockfish"
    assert dockerfile =~ "make -j"
    assert dockerfile =~ "PATH=/usr/games:${PATH}"
    assert dockerfile =~ "/usr/local/bin/fastchess"
  end
end
