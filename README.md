# ECHECS-ENGINE

ECHECS-ENGINE is a chess engine and training stack written in Elixir. It uses Nx, Axon, and EXLA for neural evaluation, alpha-beta search, supervised training, and UCI integration.

## Features

- Alpha-beta search with iterative deepening, quiescence, transposition tables, and UCI time controls.
- An optional MCTS backend and a sparse NNUE-style evaluator for checkpointed search.
- Supervised policy/WDL/moves-left training from JSONL data, plus PGN-to-JSONL conversion.
- UCI, benchmark, SPRT, and external engine-match commands.
- CPU, NVIDIA CUDA, and AMD ROCm container paths. ROCm XLA builds from source and takes longer than the other paths.

## Requirements

- Elixir and Erlang compatible with [mix.exs](mix.exs)
- Native dependencies for the selected Nx backend
- Docker and Docker Compose for container-based CPU or GPU workflows

## Quick Start

```bash
mix deps.get
elixir deps/echecs/scripts/generate_magic_cache.exs
mix engine.best --allow-zero-evaluator "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

`--allow-zero-evaluator` is a smoke-test mode only. Meaningful alpha-beta results require a sparse evaluator artifact, an `evaluator_fn`, or an inference callback. See [Finding a Move](#finding-a-move) and [Setup & Training](#setup--training) for the full workflow.

## Architectural Design

This engine utilizes a dual-network approach, bridging functional concurrency on the Erlang VM with compiled XLA numerical graphs.

### 1. Spatial Self-Attention Policy Network (GPU/CPU)
The policy network interprets the 8x8 chess board as a sequence of 64 discrete tokens.
- Employs **Multi-Head Self-Attention** to process interactions across the entire board simultaneously.
- Incorporates **SwiGLU (Swish-Gated Linear Units)** inside the feed-forward blocks.
- The network runs batched via `Nx.Serving`, enabling optimized utilization to generate move probability distributions.

### 2. Sparse Evaluator Path (CPU)
The production search path is alpha-beta first and can use a checkpointed sparse NNUE-style evaluator.
- Utilizes a `3072`-wide accumulator populated from HalfKP-like two-perspective features.
- Requires an explicit learned feature table for sparse evaluator use; the old hashed fallback path has been removed.
- Executes via `Nx.Defn` for compiled XLA evaluation.

### 3. Search
The public `EchecsEngine.best_move/2` API defaults to recursive alpha-beta search with iterative deepening, quiescence, principal variation reporting, and optional sparse-evaluator artifacts. MCTS is available when selected explicitly with `backend: :mcts`.

## Setup & Training

The environment is securely containerized using an optimized, lightweight Debian Slim base to ensure seamless native compatibility with XLA and varying hardware accelerators.

## Finding a Move

The public engine boundary accepts a FEN and returns a UCI-style best move:

```elixir
EchecsEngine.best_move(
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  allow_zero_evaluator: true
)
#=> {:ok, "<uci-move>"}
```

The default alpha-beta path expects either a sparse evaluator artifact, an explicit
`evaluator_fn`, or a neural `inference` callback. For smoke tests only, pass
`allow_zero_evaluator: true` to permit the deliberately weak zero-evaluator fallback.

From the terminal:

```bash
mix engine.best --allow-zero-evaluator "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

For GUI and benchmark integration, run the UCI protocol loop:

```bash
mix engine.uci
```

Best-move benchmark suites are JSONL files with FENs and accepted moves:

```json
{"id":"start","fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","best":["e2e4","d2d4"]}
```

Run a suite with:

```bash
mix engine.bench path/to/suite.jsonl
```

For aggregate match results, use the SPRT helper:

```bash
mix engine.bench --sprt 120 95 185 --elo1 5 --alpha 0.05 --beta 0.05
```

For external UCI engine-vs-engine gates on the host, use `cutechess-cli` or `fastchess` through:

```bash
mix engine.match --dry-run --engine-a "mix engine.uci" --engine-b stockfish --games 200 --sprt 0 5
mix engine.match --runner fastchess --engine-a "mix engine.uci" --engine-b stockfish --games 200
```

For a Dockerized match/SPRT path, build the dedicated fastchess runner and invoke the
same task with `--docker`:

```bash
docker compose build engine-match
mix engine.match --docker --engine-a "mix engine.uci" --engine-b stockfish --games 200 --sprt 0 5
```

UCI `go` budgets such as `movetime`, `depth`, `nodes`, `wtime`, `btime`, `winc`, `binc`, `movestogo`, and `infinite` are parsed and forwarded into the alpha-beta search path. Infinite searches run asynchronously and can be interrupted with `stop`.

**Default CPU Training (No GPU Required):**
```bash
docker compose build engine
docker compose up -d engine
```

The containerized training entrypoints expect a mounted dataset at `./data/train.jsonl`, which is exposed in the container as `/app/data/train.jsonl` through the `ENGINE_DATASET` environment variable.

For local runs without Docker:

```bash
mix engine.train path/to/train.jsonl
```

To train the sparse evaluator artifact used by the alpha-beta path:

```bash
mix engine.train_evaluator path/to/train.jsonl models/echecs_engine_evaluator.axon
mix engine.train_evaluator --quantize path/to/train.jsonl models/echecs_engine_evaluator.axon
```

Training expects supervised JSONL records with FEN positions, played moves, results, and optional calibrated labels:

```json
{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","move":"e2e4","result":"1-0","eval_cp":80,"moves_left":42}
{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","move":"d2d4","wdl":[0.35,0.50,0.15],"moves_left":38}
```

Each record is converted into a 119-plane board tensor, a legal-move-masked policy target, a WDL target, and a moves-left target. The tensor now uses an `8 x 14 + 7` history-stack schema: eight positions of piece-plus-repetition planes and seven current auxiliary planes. Explicit `wdl` or `eval_wdl` labels are preferred when available; `eval_cp` remains a fallback conversion.

**NVIDIA GPU Training:**
```bash
docker compose build engine-nvidia
docker compose up -d engine-nvidia
```

**AMD GPU Training (ROCm):**

This path requires a supported AMD GPU, a Linux host with ROCm installed, and Docker access to
`/dev/kfd` and `/dev/dri`. ROCm AMD support is currently source-build-only: XLA is compiled from
source during the image build, so a cold build can take substantial time and disk space. It is
intentionally a separate image: it does not change the CPU or NVIDIA CUDA build paths.

```bash
docker compose build engine-amd
docker compose up -d engine-amd
```

Use `docker compose logs -f engine-amd` to confirm that EXLA selected the ROCm client before
starting a long training run.

### Checkpointing & Model Exporting

The system features robust continuous checkpointing to ensure long-term training runs never lose progress.

**Continuous Recovery:**
During the simulation run, the `Axon.Loop` automatically saves the full training state, including `model_state`, optimizer momentum, and loop progress, at the end of every epoch to `models/echecs_engine_latest.axon`.
If your Docker container is preempted or stopped, starting it again will locate this file, re-hydrate the optimizer, and resume training from the stored loop state.

**Exporting for Production Inference:**
Training checkpoints are mathematically heavy because they contain historical gradient momentum. To export purely the network weights for fast, read-only inference inside the `Nx.Serving` cluster:

```bash
# This strips the optimizer bounds and extracts a production .axon model
mix engine.export
```
This generates `models/echecs_engine_production.axon`. Production model artifacts now include metadata describing the tensor schema and output heads so incompatible checkpoints fail earlier and more clearly.

## License

ECHECS-ENGINE is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).
