# ECHECS-ENGINE

**ECHECS-ENGINE** is a hybrid chess engine written in Elixir, leveraging the Axon and Nx libraries for neural network evaluation.

## Architectural Design

This engine utilizes a dual-network approach, bridging functional concurrency on the Erlang VM with compiled XLA numerical graphs.

### 1. Spatial Self-Attention Policy Network (GPU)
The policy network interprets the 8x8 chess board as a sequence of 64 discrete tokens.
- Employs **Multi-Head Self-Attention** to process interactions across the entire board simultaneously.
- Incorporates **SwiGLU (Swish-Gated Linear Units)** inside the feed-forward blocks.
- The network runs batched via `Nx.Serving`, enabling optimized GPU utilization to generate move probability distributions.

### 2. Sparse Vector Evaluator (CPU)
To evaluate terminal leaf nodes efficiently during the search phase, the engine implements a **JIT-compiled Sparse Evaluator**.
- Utilizes an **Accumulator** array (3072 dimensions) that updates incrementally via vector addition and subtraction during piece movement.
- Employs **SCReLU (Squared Clipped ReLU)** activations. Squaring the clipped input allows a shallow network design to capture complex piece interactions.
- Written within an `Nx.Defn` macro, allowing the evaluation loop to execute as compiled XLA machine code.

### 3. Hybrid Orchestration
The orchestrator coordinates the search phase. It evaluates the initial position using the Spatial Attention network to prune candidate moves. The remaining candidate moves are then unrolled and evaluated by the compiled CPU execution loop using the Sparse Evaluator.

## Setup & Training

The environment is containerized using Alpine Linux.

```bash
docker compose build engine-amd
docker compose up engine-amd
```

The system initializes the build and invokes `EchecsEngine.Simulation` to begin the training loop. Checkpoints are automatically persisted to the local `/models` directory.
