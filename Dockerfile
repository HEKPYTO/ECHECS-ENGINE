ARG BUILDER_IMAGE=hexpm/elixir:1.19.5-erlang-28.3.1-debian-bookworm-20260518-slim
ARG RUNTIME_IMAGE=debian:bookworm-slim

FROM ${BUILDER_IMAGE} AS builder

ENV MIX_ENV=prod \
    LANG=C.UTF-8

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mix local.hex --force && mix local.rebar --force

COPY mix.exs mix.lock ./
RUN mix deps.get --only prod
RUN cd deps/echecs && elixir scripts/generate_magic_cache.exs
RUN mix deps.compile

COPY lib lib
COPY priv priv
RUN mix compile --warnings-as-errors && mix release

FROM ${RUNTIME_IMAGE} AS runner

ENV LANG=C.UTF-8

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libncurses6 libstdc++6 openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder --chown=65532:65532 /app/_build/prod/rel/echecs_engine ./

USER 65532:65532
ENTRYPOINT ["/app/bin/echecs_engine"]
CMD ["eval", "EchecsEngine.UCI.run()"]
