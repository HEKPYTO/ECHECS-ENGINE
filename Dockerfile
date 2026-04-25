ARG BUILDER_BASE=elixir:1.19-slim
ARG RUNTIME_BASE=debian:trixie-slim
ARG CUDA_RUNTIME_PACKAGES="libcudnn9-cuda-12 libnvshmem3-cuda-12"

FROM ${BUILDER_BASE} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG XLA_TARGET=cpu

ENV MIX_ENV=prod \
    LANG=C.UTF-8 \
    XLA_TARGET=${XLA_TARGET}

RUN apt-get update -y && \
    apt-get install -y build-essential git curl python3 bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mix local.hex --force && \
    mix local.rebar --force

COPY mix.exs mix.lock ./
RUN mix deps.get --only $MIX_ENV
RUN cd deps/echecs && elixir scripts/generate_magic_cache.exs
RUN mix deps.compile

COPY . .

RUN mix compile
RUN mix release

FROM ${RUNTIME_BASE} AS runner

ARG DEBIAN_FRONTEND=noninteractive
ARG XLA_TARGET=cpu
ARG CUDA_RUNTIME_PACKAGES

USER root

ENV MIX_ENV=prod \
    LANG=C.UTF-8 \
    XLA_TARGET=${XLA_TARGET} \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib

RUN apt-get update -y && \
    apt-get install -y libstdc++6 openssl ca-certificates bash && \
    if [ "${XLA_TARGET}" = "cuda12" ]; then \
      apt-get install -y --allow-change-held-packages ${CUDA_RUNTIME_PACKAGES}; \
      if [ -e /usr/lib/x86_64-linux-gnu/nvshmem/12/nvshmem_transport_ibrc.so.5 ] && \
         [ ! -e /usr/lib/x86_64-linux-gnu/nvshmem/12/nvshmem_transport_ibrc.so.3 ]; then \
        ln -sf /usr/lib/x86_64-linux-gnu/nvshmem/12/nvshmem_transport_ibrc.so.5 \
          /usr/lib/x86_64-linux-gnu/nvshmem/12/nvshmem_transport_ibrc.so.3; \
      fi; \
      latest_nvrtc="$(find /usr/local -path '*libnvrtc-builtins.so.12.*' | sort | tail -n 1)"; \
      if [ -n "${latest_nvrtc}" ] && [ -d /usr/local/cuda/lib64 ]; then \
        ln -sf "${latest_nvrtc}" /usr/local/cuda/lib64/libnvrtc-builtins.so.12.9; \
      fi; \
      ldconfig; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/_build/prod/rel/echecs_engine ./

ENTRYPOINT ["/app/bin/echecs_engine"]
CMD ["start"]
