apk add --no-cache build-base git curl python3 bash gcompat libstdc++ ncurses-libs
export XLA_TARGET=x86_64-linux-gnu-cpu
export EXLA_TARGET=x86_64-linux-gnu-cpu
mix deps.compile exla --force
