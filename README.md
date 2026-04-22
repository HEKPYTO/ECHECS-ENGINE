# ECHECS-ENGINE

**ECHECS-ENGINE** is a hybrid chess engine written in Elixir, leveraging the Axon and Nx libraries for neural network evaluation.

## Architectural Design

This engine utilizes a dual-network approach, bridging functional concurrency on the Erlang VM with compiled XLA numerical graphs.

### 1. Spatial Self-Attention Policy Network (GPU/CPU)
The policy network interprets the 8x8 chess board as a sequence of 64 discrete tokens.
- Employs **Multi-Head Self-Attention** to process interactions across the entire board simultaneously.
- Incorporates **SwiGLU (Swish-Gated Linear Units)** inside the feed-forward blocks.
- The network runs batched via `Nx.Serving`, enabling optimized utilization to generate move probability distributions.

### 2. Sparse Vector Evaluator (CPU)
To evaluate terminal leaf nodes efficiently during the search phase, the engine implements a **JIT-compiled Sparse Evaluator**.
- Utilizes an **Accumulator** array (3072 dimensions) that updates incrementally via vector addition and subtraction during piece movement.
- Employs **SCReLU (Squared Clipped ReLU)** activations. Squaring the clipped input allows a shallow network design to capture complex piece interactions.
- Written within an `Nx.Defn` macro, allowing the evaluation loop to execute as compiled XLA machine code.

### 3. Hybrid Orchestration
The orchestrator coordinates the search phase. It evaluates the initial position using the Spatial Attention network to prune candidate moves. The remaining candidate moves are then unrolled and evaluated by the compiled CPU execution loop using the Sparse Evaluator.

## Setup & Training

The environment is securely containerized using an optimized, lightweight Debian Slim base to ensure seamless native compatibility with XLA and varying hardware accelerators.

**Default CPU Training (No GPU Required):**
```bash
docker compose build engine
docker compose up -d engine
```

**NVIDIA GPU Training:**
```bash
docker compose build engine-nvidia
docker compose up -d engine-nvidia
```

**AMD GPU Training (ROCm):**
```bash
docker compose build engine-amd
docker compose up -d engine-amd
```

### Checkpointing & Model Exporting

The system features robust continuous checkpointing to ensure long-term training runs never lose progress.

**Continuous Recovery:**
During the simulation run, the `Axon.Loop` automatically auto-saves the entire execution graph (including `model_state` and `optimizer_state` momentum) at the end of every epoch to `models/echecs_engine_latest.ckpt`. 
If your Docker container is preempted or stopped, starting it again will automatically locate this file, re-hydrate the optimizer, and resume training seamlessly from the exact epoch it left off.

**Exporting for Production Inference:**
Training checkpoints are mathematically heavy because they contain historical gradient momentum. To export purely the network weights for fast, read-only inference inside the `Nx.Serving` cluster:

```bash
# This strips the optimizer bounds and extracts a production .axon model
mix echecs.export
```
This generates `models/echecs_engine_production.axon` which can be securely loaded for rapid engine play.
