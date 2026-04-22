FROM elixir:1.19-slim AS builder

ENV MIX_ENV=prod \
    LANG=C.UTF-8

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

FROM debian:trixie-slim AS runner

ENV MIX_ENV=prod \
    LANG=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y libstdc++6 openssl ca-certificates bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/_build/prod/rel/echecs_engine ./

ENTRYPOINT ["/app/bin/echecs_engine"]
CMD ["start"]
