# AutoKernel

**Autoresearch for GPU kernels.** Give it any PyTorch model, go to sleep, wake up to optimized Triton kernels.

![AutoKernel Progress](progress.png)

Inspired by [@karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- which demonstrated autonomous AI agents for research. AutoKernel applies the same philosophy to GPU kernel optimization: agent modifies one file, runs a fixed evaluation, keeps or reverts, repeats forever.

## How It Works

Give AutoKernel any PyTorch model. It will:

1. **Profile** the model to find which GPU kernels are bottlenecks
2. **Extract** each bottleneck as a standalone Triton kernel
3. **Optimize** each kernel autonomously (edit, benchmark, keep/revert -- forever)
4. **Verify** end-to-end correctness and report the total speedup

The agent reads `program.md` -- the "research org code" -- which contains comprehensive instructions for autonomous operation. It edits `kernel.py` one kernel at a time, runs `bench.py` (fixed benchmark with 5-stage correctness checks + roofline analysis), and either keeps or reverts the change. The orchestrator decides when to move to the next kernel using Amdahl's law.

Each experiment takes ~90 seconds. That's ~40 experiments/hour, ~320 overnight, across all kernels.

## Quick Start

**Requirements:** NVIDIA GPU (tested on H100/A100/RTX 4090), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/RightNow-AI/autokernel.git
cd autokernel
uv sync

# One-time setup: test data + baselines
uv run prepare.py

# Profile the DAC-VAE model (ships with the repo -- no external dependencies)
uv run profile.py --model models/dacvae.py --class-name DACVAE \
 --input-shape 1,1,44100 --dtype float32 --audio vae_test.wav

# Extract top bottleneck kernels
uv run extract.py --top 5

# Verify benchmark works
uv run bench.py
```

## Running the Agent

Spin up Claude, Codex, or any coding agent in this directory:

```
Read program.md and let's kick off a new experiment. Start with setup.
```

The agent will:
1. Profile your model and present the optimization plan
2. Create a branch (e.g., `autokernel/mar11-dacvae`)
3. Optimize each bottleneck kernel in priority order
4. Verify end-to-end correctness and report total speedup

`program.md` is intentionally comprehensive so the agent can run 10+ hours without getting stuck. It includes a 6-tier optimization playbook, decision framework, crash handling, and Amdahl's law reasoning.

## The Pipeline

```
                 profile.py              extract.py           bench.py (loop)         verify.py
Any PyTorch  -->  Rank kernels  -->  Generate baseline  -->  Optimize each  -->  End-to-end
   model          by GPU time       Triton kernels          kernel (agent)       verification
```

| Tool | What it does |
|------|-------------|
| `profile.py` | Profiles any PyTorch model with `torch.profiler`, ranks kernels by GPU time, classifies as compute/memory-bound |
| `extract.py` | Extracts top-N bottleneck kernels from profiling results into standalone Triton kernel files |
| `orchestrate.py` | Multi-kernel scheduler: decides which kernel to optimize next using Amdahl's law, tracks aggregate progress |
| `bench.py` | Fixed benchmark: 5-stage correctness (smoke, shape sweep, numerical stability, determinism, edge cases) + performance + roofline |
| `verify.py` | Plugs optimized kernels back into the model, checks end-to-end correctness, reports total speedup |

## Supported Kernels

4 kernel types covering the core operations of the DAC-VAE audio codec:

| Kernel | Description | Key Metric |
|--------|-------------|------------|
| **matmul** | Dense matrix multiplication (VAE bottleneck 1x1 projections) | TFLOPS |
| **conv1d** | 1D convolution (encoder downsampling, residual units) | TFLOPS |
| **conv_transpose1d** | 1D transposed convolution (decoder upsampling) | TFLOPS |
| **snake_activation** | Snake activation: x + (1/alpha) * sin(alpha*x)^2 | GB/s |

Each has a PyTorch reference in `reference.py` and a starter Triton kernel in `kernels/`.

## Example Model

Self-contained DAC-VAE model definition ships with AutoKernel (no external dependencies):

| Model | File | Params | Usage |
|-------|------|--------|-------|
| DAC-VAE (full) | `models/dacvae.py` | 76.6M | `--class-name DACVAE --input-shape 1,1,44100` |
| DAC-VAE (small) | `models/dacvae.py` | 3.7M | `--class-name DACVAESmall --input-shape 1,1,44100` |
| Custom | `models/custom.py` | -- | Template for your own model |

## Project Structure

```
autokernel/
  kernel.py             the file the agent modifies (one kernel at a time)
  program.md            agent instructions -- the "research org code"

  bench.py              fixed benchmark + 5-stage correctness harness
  reference.py          PyTorch reference implementations (ground truth)
  prepare.py            one-time setup: test data, baselines

  profile.py            profile any PyTorch model, rank kernels by GPU time
  extract.py            extract bottleneck kernels into workspace/
  orchestrate.py        multi-kernel scheduler (Amdahl's law)
  verify.py             end-to-end model verification + speedup report
  analysis.py           experiment visualization (generates progress.png)

  kernels/              starter Triton kernels (4 types: conv1d, conv_transpose1d, snake_activation, matmul)
  models/               self-contained model definitions (DAC-VAE)
  workspace/            runtime artifacts (gitignored)
```

## Design Choices

**Why Triton.** Readable Python-like syntax the agent can understand and modify without mastering inline PTX or SASS. Well-tuned Triton regularly reaches 80-95% of cuBLAS/cuDNN. The agent needs to iterate fast -- Triton compiles in seconds, not minutes.

**Correctness first.** The benchmark checks kernel output against PyTorch before measuring performance. A fast but wrong kernel is immediately reverted. This prevents the agent from "optimizing" by producing garbage.

**Amdahl's law orchestration.** The orchestrator prioritizes by impact. A 1.5x speedup on a 30% kernel (1.18x end-to-end) beats a 3x speedup on a 5% kernel (1.03x end-to-end). It moves on when diminishing returns set in.

**Single file to modify.** The agent only touches `kernel.py`. Scope stays manageable, diffs reviewable, reverts clean.

**TSV logging.** Results go to a plain `results.tsv` file. Human-readable, git-friendly, trivially parseable, no infrastructure.

## Results Format

Every experiment is logged to `results.tsv` (tab-separated):

| Column | Description |
|--------|-------------|
| `experiment` | Sequential experiment number (0 = baseline) |
| `tag` | Short identifier |
| `kernel_type` | Which kernel (e.g., `conv1d`) |
| `throughput_tflops` | Measured throughput (higher is better) |
| `latency_us` | Execution time in microseconds |
| `pct_peak` | Percentage of GPU theoretical peak |
| `speedup_vs_pytorch` | Speedup vs PyTorch/cuDNN |
| `correctness` | PASS, FAIL, TIMEOUT, or CRASH |
| `peak_vram_mb` | Peak GPU memory usage |
| `description` | What was tried |

## Credits

This project is **autoresearch for GPU kernels** -- directly inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), the original experiment in autonomous AI research agents. Karpathy showed that an AI agent can run hundreds of experiments overnight, methodically exploring a search space and logging every result. AutoKernel applies that same loop -- agent edits one file, runs a fixed evaluation, keeps or reverts -- to the domain of GPU kernel optimization with Triton.

Built by the team behind [Forge](https://www.rightnowai.co/forge).

## License

MIT
