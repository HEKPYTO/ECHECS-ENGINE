# Builder stage
FROM elixir:1.19-slim AS builder

# Set build environment
ENV MIX_ENV=prod \
    LANG=C.UTF-8

# Install OS dependencies required for compilation (especially EXLA)
RUN apt-get update -y && \
    apt-get install -y build-essential git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Hex and Rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Copy dependency files
COPY mix.exs mix.lock ./

# Fetch and compile dependencies
RUN mix deps.get --only $MIX_ENV
RUN cd deps/echecs && elixir scripts/generate_magic_cache.exs
RUN mix deps.compile

# Copy application source code
COPY . .

# Compile application
RUN mix compile

# Build the release
RUN mix release

# Runner stage
FROM debian:trixie-slim AS runner

# Set runtime environment
ENV MIX_ENV=prod \
    LANG=C.UTF-8

# Install standard runtime dependencies (libstdc++ needed by EXLA usually)
RUN apt-get update -y && \
    apt-get install -y libstdc++6 openssl ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the release from the builder stage
COPY --from=builder /app/_build/prod/rel/echecs_engine ./

# Set the entrypoint to the compiled release binary
ENTRYPOINT ["/app/bin/echecs_engine"]
CMD ["start"]
