# ECHECS-ENGINE

ECHECS-ENGINE is a compact pure-Elixir chess engine. `Echecs` supplies all chess rules and packed move generation; this project supplies an integer evaluator, fail-soft PVS search, UCI, benchmarks, and offline evaluator training.

```sh
mix deps.get
mix engine.seed
mix engine.best --depth 4 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
mix engine.uci
```

The tracked `priv/echecs.nnue` file is a deterministic version-one (`ETNN`) signed-integer artifact. It has 16 king buckets, 64-neuron accumulators, and a 128→16→1 tail. The seed forwards material-aware PSQT values, so it is usable without a separate model download.

`EchecsEngine.analyze/2` and `best_move/2` accept exactly one of `depth`, `nodes`, or `movetime` (default depth 4). The UCI loop supports standard position commands and cooperative `stop`.

For a deterministic local smoke measurement, run `mix engine.bench`; its recorded baseline is intentionally a pure-Elixir correctness baseline, not a Stockfish-parity claim. For a paired strength match, install `fastchess` separately and pass both UCI commands plus an opening book; fastchess is the authority for normalized-Elo SPRT:

```sh
mix engine.bench match \
  --base-cmd mix --base-args "engine.uci" \
  --candidate-cmd mix --candidate-args "engine.uci" \
  --book openings.pgn --rounds 100 --tc 10+0.1 --concurrency 2 --dir "$PWD"
```

Quote each `--*-args` value as one command line; it is forwarded to fastchess as its
single engine `args=` field. Opening books ending in `.epd` use fastchess EPD mode;
other books use PGN mode.

To produce a new runtime artifact from labeled JSONL, run:

```sh
mix engine.train_evaluator data.jsonl output.nnue
```

Each JSONL row needs a `fen` and exactly one label. `eval_cp` is a numeric
centipawn target from the FEN side-to-move perspective. `wdl` is only a normalized
three-number `[win, draw, loss]` probability triplet (each 0..1, sum 1), mapped to
`(win - loss) * 1000` from that same perspective. A `result` (`"1-0"`, `"0-1"`,
or `"1/2-1/2"`) is a White outcome and is converted to the FEN side-to-move
perspective. Training is
deterministic streamed offline integer SGD over active NNUE/PSQT features and the
dense tail; it is not a shipped-strength or Stockfish-parity claim.
