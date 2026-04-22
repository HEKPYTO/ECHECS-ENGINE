# ECHECS-ENGINE

**ECHECS-ENGINE** is a state-of-the-art (SOTA), one-of-a-kind chess engine written purely in Elixir.

## Unique Architectural Design

This engine discards conventional paradigms (such as legacy CNN heuristics or purely rigid alphabeta constructs) in favor of a novel, hybrid architecture built directly on top of the Erlang VM and XLA compilation. 

By conducting deep research into the bleeding edge of modern machine learning and chess programming (circa 2026), we forged a hybrid model that utilizes the most groundbreaking concepts in the industry:

### 1. SwiGLU Spatial Self-Attention (The "Slow" GPU Network)
Rather than relying on standard convolutional layers that suffer from strict local receptive fields, our policy network treats the 8x8 chess board as a sequence of 64 discrete tokens. 
- We employ **Multi-Head Self-Attention** to allow pieces on opposite ends of the board to interact in a single layer.
- We utilize **SwiGLU (Swish-Gated Linear Units)** inside the Feed-Forward blocks—a SOTA activation function borrowed from the most advanced modern LLMs, proving vastly superior to standard ReLU or Mish networks. 
- This deep network runs batched via `Nx.Serving` on the GPU, yielding extremely accurate move probabilities.

### 2. Sparse Vector Evaluator (The "Fast" CPU Graph)
To rapidly evaluate millions of leaf nodes without bottlenecking on the GPU, the engine utilizes a **JIT-compiled Sparse Evaluator**.
- Instead of re-evaluating the entire board for every move, it utilizes an **Accumulator** (3072 dimensions). When a piece moves, the engine simply executes a fast vector subtraction (old piece) and addition (new piece).
- This is paired with **SCReLU (Squared Clipped ReLU)** activations. Squaring the clamped input allows a phenomenally shallow network (just two layers) to capture highly complex, non-linear piece relationships without adding depth.
- Entirely written inside an `Nx.Defn` macro, the entire shallow search tree evaluates directly in raw, natively compiled machine code, completely bypassing standard functional overhead.

### 3. Hybrid Orchestration & Gumbel Search Principles
The engine bridges the functional Erlang VM and the XLA execution graphs. It queries the massive Spatial Attention Network to radically prune the candidate moves. The surviving candidates are unrolled into a compiled CPU execution loop where the Sparse Evaluator resolves deep tactical lines.

## Setup & Training

The environment is strictly Dockerized on an Alpine Linux base to maintain lightweight runner boundaries.

```bash
docker compose build engine
docker compose up engine
```

The system automatically initializes the multi-stage build, generating the needed tensor execution boundaries, and begins the `EchecsEngine.Simulation` to continuously train the internal evaluators.
