# Builder stage
FROM elixir:1.19-alpine AS builder

# Set build environment
ENV MIX_ENV=prod \
    LANG=C.UTF-8

# Install OS dependencies required for compilation (especially EXLA)
# Alpine uses apk instead of apt-get
RUN apk update && \
    apk add --no-cache build-base git curl python3 bash && \
    rm -rf /var/cache/apk/*

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
FROM alpine:3.20 AS runner

# Set runtime environment
ENV MIX_ENV=prod \
    LANG=C.UTF-8

# Install standard runtime dependencies
# Alpine needs libstdc++ for XLA/C++ native extensions, and ncurses-libs/openssl for Erlang runtime
RUN apk update && \
    apk add --no-cache libstdc++ openssl ncurses-libs bash libgcc && \
    rm -rf /var/cache/apk/*

WORKDIR /app

# Copy the release from the builder stage
COPY --from=builder /app/_build/prod/rel/echecs_engine ./

# Set the entrypoint to the compiled release binary
ENTRYPOINT ["/app/bin/echecs_engine"]
CMD ["start"]
