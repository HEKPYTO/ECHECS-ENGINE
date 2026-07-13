defmodule EchecsEngine.DockerConfigTest do
  use ExUnit.Case, async: true

  @compose Path.expand("../docker-compose.yml", __DIR__)
  @dockerfile Path.expand("../Dockerfile", __DIR__)
  @rocm_dockerfile Path.expand("../Dockerfile.rocm", __DIR__)
  @match_dockerfile Path.expand("../Dockerfile.match", __DIR__)

  test "engine-match is configured as a dockerized fastchess gate" do
    compose = File.read!(@compose)

    assert compose =~ "services:"
    assert compose =~ "\n  engine:\n"
    assert compose =~ "\n  engine-nvidia:\n"
    assert compose =~ "\n  engine-amd:\n"
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

    assert compose =~
             ~r/engine-amd:.*dockerfile: Dockerfile\.rocm.*BUILDER_BASE: hexpm\/elixir:1\.19\.5-erlang-28\.3\.1-ubuntu-noble-20260410.*XLA_TARGET: rocm.*XLA_BUILD: "true".*RUNTIME_BASE: rocm\/dev-ubuntu-24\.04:7\.2\.4-complete/s

    assert compose =~ ~r/engine-amd:.*image: echecs_engine:amd/s
    assert compose =~ ~r/engine-amd:.*platform: linux\/amd64/s
    assert compose =~ ~r/engine-amd:.*- \/dev\/kfd.*- \/dev\/dri/s
    assert compose =~ ~r/engine-amd:.*ROCM_PATH=\/opt\/rocm/s
    assert compose =~ ~r/engine-match:.*image: echecs_engine:match/s
  end

  test "Dockerfile keeps CPU and CUDA build configuration isolated" do
    dockerfile = File.read!(@dockerfile)

    assert dockerfile =~ "ARG BUILDER_BASE=elixir:1.19-slim"
    assert dockerfile =~ "ARG RUNTIME_BASE=debian:trixie-slim"
    assert dockerfile =~ "ARG CUDA_RUNTIME_PACKAGES="
    assert dockerfile =~ "if [ \"${XLA_TARGET}\" = \"cuda12\" ]; then"
    assert dockerfile =~ "RUN mix deps.get --only $MIX_ENV"
    assert dockerfile =~ "RUN cd deps/echecs && elixir scripts/generate_magic_cache.exs"
  end

  test "Dockerfile.rocm builds XLA from source against the ROCm runtime" do
    dockerfile = File.read!(@rocm_dockerfile)

    assert dockerfile =~
             "ARG BUILDER_BASE=hexpm/elixir:1.19.5-erlang-28.3.1-ubuntu-noble-20260410"

    assert dockerfile =~ "ARG RUNTIME_BASE=rocm/dev-ubuntu-24.04:7.2.4-complete"
    assert dockerfile =~ "ARG BAZEL_VERSION=7.7.0"

    assert dockerfile =~
             "ARG BAZEL_SHA256=fe7e799cbc9140f986b063e06800a3d4c790525075c877d00a7112669824acbf"

    assert dockerfile =~ "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}"
    assert dockerfile =~ "sha256sum --check --strict"
    refute dockerfile =~ "FROM bazelbuild/bazel"
    assert dockerfile =~ "FROM ${RUNTIME_BASE} AS builder"
    assert dockerfile =~ "COPY --from=elixir /usr/local/bin /usr/local/bin"
    assert dockerfile =~ "COPY --from=elixir /usr/local/lib/elixir /usr/local/lib/elixir"
    assert dockerfile =~ "COPY --from=elixir /usr/local/lib/erlang /usr/local/lib/erlang"
    refute dockerfile =~ "COPY --from=rocm /opt/rocm /opt/rocm"
    assert dockerfile =~ "XLA_TARGET=${XLA_TARGET}"
    assert dockerfile =~ "XLA_BUILD=${XLA_BUILD}"
    assert dockerfile =~ "ROCM_PATH=/opt/rocm"
    assert dockerfile =~ "PYTHON_BIN_PATH=/usr/bin/python3"
    assert dockerfile =~ "xxd"
    assert dockerfile =~ "ln -sf /usr/bin/clang-18 /usr/local/bin/clang"
    assert dockerfile =~ "ln -sf /usr/bin/clang++-18 /usr/local/bin/clang++"
    assert dockerfile =~ "miopen-hip-dev"

    assert dockerfile =~
             "ARG ROCM_AMDGPU_TARGETS=gfx90a,gfx942,gfx1030,gfx1100,gfx1200,gfx1201"

    assert dockerfile =~ ~r/mix deps.get.*TF_ROCM_AMDGPU_TARGETS/s
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
