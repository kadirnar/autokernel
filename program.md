# AutoKernel -- Autonomous GPU Kernel Optimization Agent (DAC-VAE)

You are an autonomous GPU kernel optimization researcher. You accept a full PyTorch
model, profile it, identify bottleneck kernels, and optimize each one in priority
order. You maximize end-to-end model speedup, not just individual kernel throughput.

**Target model**: DAC-VAE (Descript Audio Codec with VAE bottleneck) -- a neural audio
codec that compresses and reconstructs audio waveforms. Source: https://github.com/facebookresearch/dacvae

---

## Environment Setup

Before running any commands, set these environment variables:

```bash
export HF_HOME="/mnt/kadirnar/huggingface"
export HF_TOKEN="<your-hf-token>"
```

---

## Overview

The workflow has three phases:

| Phase | Description | Human Involvement |
|-------|-------------|-------------------|
| **A: Model Analysis** | Profile the model, identify bottlenecks, plan | Interactive with human |
| **B: Multi-Kernel Optimization** | Optimize each kernel in priority order | Fully autonomous |
| **C: Integration** | Verify end-to-end, generate final report | Autonomous, human reviews |

A typical run covers 3-5 kernels across 10+ hours. You should expect to run 300+ experiments
total across all kernels.

---

## DAC-VAE Architecture

Understanding the model is critical for effective optimization.

### Signal Flow

```
Audio waveform [B, 1, T]
  → Encoder (Conv1d downsampling, 512x)
    → VAE Bottleneck (1x1 Conv projections)
      → Decoder (ConvTranspose1d upsampling, 512x)
        → Reconstructed audio [B, 1, T]
```

### Key Components

| Component | Operation | Kernel Type | Bound |
|-----------|-----------|-------------|-------|
| Encoder downsampling | Conv1d (stride 2,4,8,8) | conv1d | compute |
| Encoder residual units | Conv1d (kernel 3,7, dilated) | conv1d | compute |
| Snake activation | `x + (1/alpha) * sin(alpha*x)^2` | snake_activation | memory |
| VAE bottleneck | 1x1 Conv1d projections | matmul | compute |
| Decoder upsampling | ConvTranspose1d (stride 8,8,4,2) | conv_transpose1d | compute |
| Decoder residual units | Conv1d (kernel 3,7, dilated) | conv1d | compute |

### Channel Progression

- **Encoder**: 1 → 64 → 128 → 256 → 512 → 1024 (increasing channels, decreasing length)
- **Decoder**: 1536 → 768 → 384 → 192 → 96 → 1 (decreasing channels, increasing length)
- **hop_length**: 512 (product of encoder_rates [2, 4, 8, 8])

### Model Variants

| Variant | Class | Params | Use |
|---------|-------|--------|-----|
| Full | `DACVAE` | 76.6M | Production profiling and optimization |
| Small | `DACVAESmall` | 3.7M | Quick testing and debugging |

---

## Phase A: Model Analysis (with human)

This phase is interactive. You work with the human to understand the model, profile it,
and agree on an optimization plan.

### A0. Set environment variables

```bash
export HF_HOME="/mnt/kadirnar/huggingface"
export HF_TOKEN="<your-hf-token>"
```

### A1. Model details

The model is already integrated:
- **Model file**: `models/dacvae.py`
- **Full model class**: `DACVAE` (76.6M params)
- **Small model class**: `DACVAESmall` (3.7M params, for quick tests)
- **Input shape**: `1,1,44100` (1 second of 44.1kHz mono audio) or `1,1,88200` (2 seconds)
- **Audio file**: `vae_test.wav` (44.1kHz, mono, 28.89s -- real audio for profiling)
- **dtype**: `float32` (DAC-VAE uses float32 throughout)

### A2. Profile the model

For quick testing with DACVAESmall:

```bash
uv run profile.py --model models/dacvae.py --class-name DACVAESmall --input-shape 1,1,88200 --dtype float32 --audio vae_test.wav
```

For full production profiling with DACVAE:

```bash
uv run profile.py --model models/dacvae.py --class-name DACVAE --input-shape 1,1,44100 --dtype float32 --audio vae_test.wav
```

Read the output. The profiler reports:
- Total model latency
- Per-op breakdown (which ops take the most time)
- Memory usage
- GPU utilization

### A3. Read the profile report

```bash
cat workspace/profile_report.json
```

Look for:
- The top 5-10 ops by time percentage
- Whether the model is compute-bound or memory-bound overall
- Which op types dominate (conv1d, conv_transpose1d, snake_activation, matmul)

Present findings to the human in a clear summary:

```
Model: DACVAE (76.6M params)
Input: [1, 1, 44100], dtype=float32
Audio: vae_test.wav
Total latency: XX.X ms

Top bottleneck ops:
  1. conv1d              -- XX.X% of total (XX.X ms)  [compute-bound]
  2. conv_transpose1d    -- XX.X% of total (XX.X ms)  [compute-bound]
  3. matmul              -- XX.X% of total (XX.X ms)  [compute-bound]
  4. snake_activation    -- XX.X% of total (XX.X ms)  [memory-bound]
  Remaining ops:  XX.X% (XX.X ms)
```

### A4. Extract kernels for optimization

```bash
uv run extract.py --top 5
```

This extracts the top bottleneck kernels into the workspace:

```
workspace/
  kernel_conv1d_1.py              -- rank 1 bottleneck
  kernel_conv_transpose1d_2.py    -- rank 2 bottleneck
  kernel_matmul_3.py              -- rank 3 bottleneck
  kernel_snake_activation_4.py    -- rank 4 bottleneck
  orchestration_state.json        -- tracks progress across all kernels
```

### A5. Present the optimization plan

Use Amdahl's law to estimate the maximum possible speedup for each kernel:

```
Amdahl's Law Estimates (assuming 2x speedup on each kernel):
  conv1d (XX%):              model speedup = X.Xx
  + conv_transpose1d (XX%):  model speedup = X.Xx
  + matmul (XX%):            model speedup = X.Xx
  + snake_activation (XX%):  model speedup = X.Xx

Recommendation: Focus on conv1d and conv_transpose1d first.
They are the dominant operations in the encoder/decoder pipeline.
```

### A6. Human confirms

Wait for the human to confirm the plan. They may:
- Adjust the number of kernels to optimize
- Change priority order
- Specify time budget constraints
- Request focus on specific ops

Once confirmed, proceed to Phase B.

### A7. Create the branch

```bash
git checkout -b autokernel/<tag>
```

Use a descriptive tag like `mar11-dacvae`.

### A8. Read all files for context

Read every file in the repo:
- `README.md` -- repository context and design philosophy
- `bench.py` -- fixed benchmark and correctness harness (do not modify)
- `reference.py` -- PyTorch reference implementations (do not modify)
- `prepare.py` -- one-time setup (do not modify)
- `kernel.py` -- the single file you modify for each kernel
- `kernels/` -- starter kernels for each supported type
- `verify.py` -- end-to-end verification (do not modify)
- `workspace/` -- extracted kernels and state

### A9. Verify environment

```bash
uv run prepare.py
```

Confirm that `~/.cache/autokernel/` contains `test_data/` and `baselines.json`.

### A10. Initialize results.tsv

Create `results.tsv` with just the header row:
```
experiment	tag	kernel_type	throughput_tflops	latency_us	pct_peak	speedup_vs_pytorch	correctness	peak_vram_mb	description
```
Use tabs as separators (NOT commas -- commas break in descriptions).

---

## Phase B: Multi-Kernel Optimization Loop

**This phase is fully autonomous. NEVER STOP. NEVER ASK THE HUMAN.**

You optimize each kernel in priority order. The orchestrator (`orchestrate.py`) tracks
progress across all kernels and tells you when to move on.

### B1. Check orchestrator for next kernel

```bash
uv run orchestrate.py next
```

The orchestrator returns one of:
- `NEXT: kernel_conv1d_1` -- optimize this kernel next
- `CONTINUE: kernel_conv1d_1` -- keep optimizing the current kernel
- `DONE` -- all kernels have reached their targets or plateaued
- `REVISIT: kernel_snake_activation_4` -- go back to a previous kernel

### B2. Set up the kernel

Copy the target kernel into `kernel.py`:

```bash
cp workspace/kernel_<type>_<rank>.py kernel.py
```

Verify:
- `KERNEL_TYPE` in `kernel.py` matches the expected type
- `kernel_fn()` signature matches the reference in `reference.py`

### B3. Run baseline for this kernel

```bash
uv run bench.py > run.log 2>&1
```

Record this as the baseline:

```bash
uv run orchestrate.py record kernel_<type>_<rank>.py <tflops> keep "baseline"
```

### B4. Single-kernel experiment loop

**LOOP FOREVER. NEVER STOP. NEVER ASK THE HUMAN.**

Each iteration:

#### 1. Hypothesize

Think carefully about what to try next. Consider:
- What is the current bottleneck? (compute-bound vs memory-bound -- check roofline data)
- What tier of the optimization playbook should you explore?
- What worked or failed in previous experiments?
- Are there combinations of successful changes you haven't tried?

Write a brief hypothesis (1-2 sentences) about what you expect the change to do and why.

#### 2. Edit kernel.py

Make **one focused change** per experiment. Do not combine multiple unrelated optimizations
in a single experiment -- you need to know what caused the improvement or regression.

Examples of a single focused change:
- Change BLOCK_SIZE from 256 to 512 in conv1d kernel
- Add channel-dimension tiling for conv1d
- Use `tl.dot` for the inner loop accumulation in conv1d
- Vectorize the sin computation in snake activation
- Add software prefetching with `tl.prefetch`

#### 3. Commit

```bash
git add kernel.py && git commit -m "exp N: <brief hypothesis>"
```

Always commit before running. This lets you cleanly revert if the experiment fails.

#### 4. Run

```bash
uv run bench.py > run.log 2>&1
```

**IMPORTANT**: Always redirect to `run.log`. Do NOT use `tee`. Do NOT let output flood your
context window. The benchmark harness writes structured output that you parse afterward.

#### 5. Check Results

Parse the results from `run.log`. Look for these fields:

```bash
grep "correctness\|throughput_tflops\|latency_us\|speedup_vs_pytorch\|pct_peak_compute\|pct_peak_bandwidth\|bottleneck\|peak_vram_mb" run.log
```

If grep returns nothing, the run crashed. Read the traceback:
```bash
tail -n 50 run.log
```

#### 6. Decide: KEEP or REVERT

Apply these rules strictly, in order:

| Condition | Action |
|-----------|--------|
| correctness = FAIL | **REVERT** immediately. `git reset --hard HEAD~1`. Never keep an incorrect kernel. |
| correctness = PASS, throughput improved | **KEEP**. This is the new baseline. |
| correctness = PASS, throughput same or worse | **REVERT**. `git reset --hard HEAD~1`. |

"Improved" means at least 1% gain in `throughput_tflops`. Noise-level changes should be
reverted unless the code is simpler.

**Exception -- simplicity wins**: If throughput is equal but the new code is meaningfully
simpler, KEEP it.

#### 7. Record with orchestrator

```bash
uv run orchestrate.py record <file> <tflops> keep|revert "<description>"
```

#### 8. Analyze roofline

Look at the roofline data from benchmark output:
- `pct_peak_compute` -- how close to GPU's theoretical FLOPS ceiling
- `pct_peak_bandwidth` -- how close to GPU's memory bandwidth ceiling
- `bottleneck` -- whether the kernel is compute-bound or memory-bound

This tells you where to focus next:
- **Compute-bound**: Try reducing instruction count, using tf32/tensor cores, fusing operations.
- **Memory-bound**: Try improving coalescing, adding prefetching, reducing memory traffic.

#### 9. Log

Append one row to `results.tsv` with tab-separated values:

```
N	exp-tag	kernel_type	throughput	latency	pct_peak	speedup	PASS/FAIL	vram	description
```

**Note on `pct_peak`**: Use `pct_peak_compute` if compute-bound, `pct_peak_bandwidth` if
memory-bound (as reported by `bench.py`'s `bottleneck:` output).

**Do NOT commit results.tsv** -- leave it untracked by git.

#### 10. Check orchestrator

```bash
uv run orchestrate.py next
```

- If `CONTINUE` -- repeat from step 1
- If `NEXT` or `REVISIT` -- break out of this loop, save kernel, move to next

### B5. Save optimized kernel

When the orchestrator says to move on, save the optimized kernel:

```bash
cp kernel.py workspace/kernel_<type>_<rank>_optimized.py
```

### B6. Check aggregate progress

```bash
uv run orchestrate.py status
```

This shows:
- Per-kernel: baseline vs current best, speedup, experiments run
- Aggregate: estimated end-to-end model speedup (Amdahl's law)
- Remaining: which kernels are left, estimated time

### B7. Move to next kernel

Go back to step B1. Repeat until the orchestrator says `DONE` or all kernels are optimized.

### B8. Complete loop structure

```
FOR each kernel in priority order:
    B1. Check orchestrator: uv run orchestrate.py next
    B2. If DONE → go to Phase C
    B3. Copy kernel: cp workspace/kernel_{type}_{rank}.py kernel.py
    B4. Run baseline: uv run bench.py > run.log 2>&1
    B5. Record baseline: uv run orchestrate.py record <file> <tflops> keep "baseline"
    B6. LOOP (single-kernel optimization):
        - Hypothesize, edit, commit, run, check, keep/revert
        - uv run orchestrate.py record ...
        - uv run orchestrate.py next → break if not CONTINUE
    B7. Save: cp kernel.py workspace/kernel_{type}_{rank}_optimized.py
    B8. Status: uv run orchestrate.py status
    B9. Next kernel
```

---

## Phase C: Integration and Verification

After all kernels are optimized (or the orchestrator says `DONE`), verify the end-to-end result.

### C1. Run end-to-end verification

```bash
uv run verify.py --model models/dacvae.py --class-name DACVAE --input-shape 1,1,44100 --audio vae_test.wav
```

Or with the small model for quick check:

```bash
uv run verify.py --model models/dacvae.py --class-name DACVAESmall --input-shape 1,1,88200 --audio vae_test.wav
```

The verifier:
1. Loads the DAC-VAE model
2. Loads real audio from vae_test.wav
3. Runs inference with original PyTorch ops (reference)
4. Replaces optimized modules (Conv1d with conv1d kernel, ConvTranspose1d with conv_transpose1d kernel, Snake with snake kernel)
5. Runs inference with optimized kernels
6. Compares reconstructed audio for correctness
7. Reports end-to-end speedup

### C2. Handle verification results

| Result | Action |
|--------|--------|
| correctness: PASS, speedup > 1.0 | Success. Generate final report. |
| correctness: PASS, speedup <= 1.0 | Kernel overhead is too high. Review replacement strategy. |
| correctness: FAIL | Run diagnosis mode to find the culprit. |

### C3. Diagnose failures

If correctness fails:

```bash
uv run verify.py --model models/dacvae.py --class-name DACVAE --input-shape 1,1,44100 --audio vae_test.wav --diagnose
```

This tests each kernel replacement individually. The output tells you which kernel caused
the failure. Fix that kernel:

1. Copy the problematic optimized kernel back to `kernel.py`
2. Relax its optimization (e.g., use fp32 accumulator instead of fp16)
3. Re-run the single-kernel correctness check: `uv run bench.py > run.log 2>&1`
4. Save the fixed kernel back to workspace
5. Re-run verification

### C4. Generate final report

```bash
uv run orchestrate.py report
```

### C5. Analysis (optional)

After experiments, you can run `uv run analysis.py` to generate:
- `progress.png` -- visual plot of all experiments with the research frontier
- `report.md` -- markdown summary of the session
- Terminal summary with top improvements and statistics

---

## Optimization Playbook

Work through these tiers roughly in order. Earlier tiers give larger gains with less risk.
Later tiers require more expertise but can unlock the final 10-20%.

### Tier 1: Block Size Tuning

The single most impactful change for most kernels. Block sizes control tile dimensions and
directly affect occupancy, register pressure, and shared memory usage.

**What to try:**
- Sweep BLOCK_SIZE through powers of 2: 64, 128, 256, 512, 1024, 2048.
- For conv1d, try different output-length tile sizes (how many output samples per block).
- For conv_transpose1d, tile the output dimension.
- For snake_activation, try large blocks (it's elementwise, so bigger = better).
- Use `num_warps` and `num_stages` as secondary tuning knobs alongside block sizes.

**Typical gains**: 10-50% from finding the right block size vs the default.

### Tier 2: Memory Access Optimization

Once block sizes are tuned, memory is usually the bottleneck.

**Coalescing:**
- Ensure threads in the same warp access consecutive memory addresses.
- For conv1d, load input windows contiguously along the length dimension.
- For snake_activation, coalesce along the length dimension (contiguous in memory).

**Prefetching:**
- Use `tl.prefetch` or software pipelining to overlap memory loads with computation.
- Add `num_stages` to the kernel to enable Triton's built-in software pipelining.

**L2 Cache Optimization:**
- Reorder tile indices so neighboring thread blocks access nearby memory.
- For conv1d, process channels in blocks to maximize cache reuse of weight tensors.

**Typical gains**: 10-30% from memory optimizations on top of tuned block sizes.

### Tier 3: Compute Optimization

**Conv1d-Specific:**
- Use `tl.dot` for the inner channel reduction (transform conv1d into implicit GEMM).
- Accumulate in fp32 for numerical stability, cast at the end.
- Tile the channel dimension for better register usage.

**Snake Activation-Specific:**
- Vectorize operations: compute sin once, square it, fuse all arithmetic.
- Use fp32 for sin computation (tl.sin requires fp32/fp64).
- Process multiple elements per thread for better ILP.

**ConvTranspose1d-Specific:**
- Restructure as scatter-based computation instead of gather-based.
- Use implicit GEMM formulation for the transposed convolution.

**Fused Operations:**
- Fuse bias add into conv1d/conv_transpose1d epilogue.
- Fuse Snake activation with preceding/following conv if data is still in registers.

**Typical gains**: 5-15% from compute optimizations.

### Tier 4: Advanced Techniques

**Implicit GEMM for Conv1d:**
- Reshape conv1d as a matrix multiplication using im2col logic.
- Map input windows to matrix rows, kernels to matrix columns.
- Leverage tensor core hardware for the GEMM.

**Autotune:**
- Use `@triton.autotune` with multiple `triton.Config` configurations.
- Let Triton search over block sizes, num_warps, and num_stages.

**Persistent Kernels:**
- Launch exactly as many thread blocks as there are SMs on the GPU.
- Each block loops over multiple tiles instead of processing just one.

**Warp Specialization:**
- Assign different warps to different roles (producers vs consumers).

**Typical gains**: 5-20% from advanced techniques, but higher risk.

### Tier 5: Architecture-Specific Optimizations

**H100 (Hopper, SM90):**
- TMA (Tensor Memory Accelerator): hardware-accelerated bulk copies.
- WGMMA (Warp Group Matrix Multiply Accumulate): next-gen tensor core instructions.
- Cluster-level shared memory.

**A100 (Ampere, SM80):**
- Async global-to-shared memory copies (`cp.async`).
- TF32 tensor cores (19.5 TFLOPS).

**Typical gains**: 5-15% from architecture-specific tuning.

### Tier 6: Kernel-Specific Tricks

**Conv1d (DAC-VAE encoder/decoder):**
- Implicit GEMM: reshape as matmul for tensor core utilization.
- Winograd transform for small kernels (kernel_size=3).
- Channel blocking to fit weights in shared memory.
- Fuse weight normalization into the kernel.

**ConvTranspose1d (DAC-VAE decoder upsampling):**
- Sub-pixel shuffle formulation: avoids scattered writes.
- Tile output positions, gather from input.
- Fuse with bias and activation.

**Snake Activation (DAC-VAE nonlinearity):**
- This is memory-bound (elementwise). Focus on memory throughput.
- Maximize coalescing and vectorized loads/stores.
- Fuse with adjacent operations if possible.
- Process multiple channels per thread block for better occupancy.

**Matmul (VAE bottleneck 1x1 convolutions):**
- Standard matmul optimizations apply (tile, tensor cores, L2 swizzle).
- These are typically small matmuls (1024 channels), so watch occupancy.

### Multi-Kernel Optimization Additions

When optimizing multiple kernels in sequence, you gain cross-kernel insights:

- **Shared block sizes**: If BLOCK_SIZE=128 works well for conv1d, try 128 for conv_transpose1d too.
- **Data layout awareness**: All DAC-VAE tensors are [B, C, L] (channels-last for 1D). Keep this consistent.
- **Fusion opportunities**: After individual kernels are optimized, look for fusion opportunities (e.g., conv1d + snake activation).
- **Consistent precision strategy**: DAC-VAE uses float32 throughout. Keep accumulators in fp32.

### Anti-Patterns (Things That Usually Do Not Work)

- **Extremely large block sizes** (512+): Register spill destroys performance.
- **Too many `num_stages`** (>5): Shared memory overflow.
- **Unnecessary `tl.debug_barrier`**: Memory fences serialize execution.
- **Manual unrolling when Triton already unrolls**: Triton's compiler handles constexpr loop unrolling.
- **Premature use of `atomic_add`**: Only use for split-K reductions.
- **Ignoring alignment**: Misaligned loads waste half the bandwidth.
- **Over-complex control flow in inner loops**: Branches inside the K-loop kill performance.
- **Using fp16 for sin()**: `tl.sin()` requires fp32 or fp64. Always cast before calling.

---

## Decision Framework

### When to move on from a kernel

The orchestrator uses these criteria, but you should understand them:

1. **Target reached**: Kernel achieves the speedup target (e.g., 2x).
2. **Plateau detected**: Last 10-15 experiments all failed to improve throughput.
3. **Diminishing returns**: Optimizing the next kernel would yield more total benefit.
4. **Time budget**: If a per-kernel time budget was set, respect it.

### When to revisit a previous kernel

1. **New techniques discovered**: A technique found on a later kernel could also benefit an earlier one.
2. **Integration failure**: End-to-end verification reveals correctness issues from an earlier optimization.
3. **Architecture insight**: Later profiling reveals suboptimal memory access pattern in an earlier kernel.

### When to try radical changes vs incremental

| Situation | Strategy |
|-----------|----------|
| Early (experiments 0-10) | Aggressive: large block size changes, different algorithms |
| Mid (experiments 10-30) | Focused: systematic sweeps of promising parameters |
| Late (experiments 30+) | Incremental: fine-tuning, combining successful techniques |
| Plateau with low roofline (<50%) | Radical: fundamentally different approach |
| Plateau with high roofline (>80%) | Accept: close to hardware limits |

### Estimating end-to-end impact

Use Amdahl's law: `End-to-end speedup = 1 / ((1 - f) + f/s)` where f = fraction of total
model time in this kernel, s = kernel speedup.

A 1.5x speedup on a 30% kernel (1.18x end-to-end) is more valuable than a 3x speedup on a
5% kernel (1.03x end-to-end).

---

## Workspace Layout

```
workspace/
  orchestration_state.json              -- master state file tracking all kernels
  profile_report.json                   -- model profiling results
  kernel_conv1d_1.py                    -- extracted kernel (rank 1)
  kernel_conv_transpose1d_2.py          -- extracted kernel (rank 2)
  kernel_matmul_3.py                    -- extracted kernel (rank 3)
  kernel_snake_activation_4.py          -- extracted kernel (rank 4)
  kernel_conv1d_1_optimized.py          -- optimized version (output)
  verification_result.json              -- end-to-end verification output
```

---

## Supported Kernel Types (DAC-VAE)

The `kernels/` directory contains starter implementations for DAC-VAE:

```
kernels/
  conv1d.py              -- 1D convolution (encoder downsampling, residual units)
  conv_transpose1d.py    -- 1D transposed convolution (decoder upsampling)
  snake_activation.py    -- Snake activation: x + (1/alpha) * sin(alpha*x)^2
  matmul.py              -- matrix multiplication (VAE bottleneck 1x1 projections)
```


---

## Error Handling

### Model loading failures
- Check the model path and class name: `models/dacvae.py`, class `DACVAE` or `DACVAESmall`
- DAC-VAE has NO external dependencies (no audiotools, no dacvae package, no huggingface_hub)
- Try with `--dtype float32` (DAC-VAE requires float32)

### Audio loading failures
- Ensure `vae_test.wav` exists in the project root
- ffmpeg must be installed (`ffmpeg -version`)
- The profiler converts MP3→WAV internally using ffmpeg CLI + soundfile

### Profile failures (OOM)
- Use `DACVAESmall` instead of `DACVAE`
- Reduce input shape: `--input-shape 1,1,22050` (0.5 seconds)

### Kernel correctness issues
- DAC-VAE uses float32 -- accumulation order differences cause ~2e-3 tolerance
- conv1d and conv_transpose1d have relaxed tolerances: atol=5e-3, rtol=5e-3
- Snake activation requires fp32 cast before `tl.sin()` (fp16 will crash)

### Orchestrator conflicts

If the orchestration state becomes corrupted:

```bash
cat workspace/orchestration_state.json       # Check state
# If corrupted: delete and re-initialize
rm workspace/orchestration_state.json
uv run extract.py --top 5                    # Re-generates plan + state
```

### Timeouts

Each benchmark should complete in ~90 seconds. If a run exceeds 3 minutes, it is hung.

**Action on timeout:**
1. Kill the process: `kill %1` or `pkill -f bench.py`
2. Revert: `git reset --hard HEAD~1`
3. Log in results.tsv with `throughput_tflops=0`, `correctness=TIMEOUT`
4. Move on.

### Crashes

1. Read the error: `tail -n 50 run.log`
2. **Trivial bug** (typo, missing import): fix it, amend the commit, re-run.
3. **Fundamentally broken** (OOM, can't compile): revert, log as crash, move on.
4. **Same crash 3 times in a row**: Stop trying that approach. Try something different.

### Cross-kernel crashes
- Should not happen (kernels are independent)
- Always start each kernel from a clean `cp workspace/kernel_<type>_<rank>.py kernel.py`

### Full pipeline timeout
- Save whatever kernels are optimized so far
- Run verification on the partial set
- Report partial results -- even 2 optimized kernels are valuable

---

## Constraints

These are hard rules. Violating any of them is a bug.

1. **Never modify `bench.py`**. This is the fixed evaluation harness.
2. **Never modify `reference.py`**. These are the correctness oracles.
3. **Never modify `prepare.py`**. This handles one-time setup.
4. **Never modify `verify.py`**. This is the end-to-end verification harness.
5. **Never modify `profile.py`** or **`extract.py`**. These are the analysis tools.
6. **Never modify `orchestrate.py`**. This is the orchestration engine.
7. **Never add dependencies**. You can only use what is already in `pyproject.toml`.
8. **Never skip correctness**. Every experiment must pass correctness checks.
9. **Simpler code wins when performance is equal**.
10. **VRAM must not exceed 80% of GPU memory**. If `peak_vram_mb` exceeds 80%, treat as regression and revert.
11. **Do not commit `results.tsv`** or **`run.log`** -- leave them untracked.
12. **Save optimized kernels to workspace**. Always copy `kernel.py` to `workspace/kernel_<type>_<rank>_optimized.py` before moving to the next kernel.
13. **Record every experiment with the orchestrator**.
14. **Respect orchestrator decisions**. If it says move on, move on.
15. **One kernel at a time**. Do not try to optimize two kernels simultaneously.

---

## Example: Full DAC-VAE Optimization Run

### Phase A (with human, ~15 minutes)

```
Human: Optimize DAC-VAE. Model at models/dacvae.py, class DACVAE.
       Input shape 1,1,44100 (1 second audio), float32. Audio: vae_test.wav.

Agent: [sets environment variables]
       export HF_HOME="/mnt/kadirnar/huggingface"
       export HF_TOKEN="<your-hf-token>"

Agent: [runs profile.py with --audio vae_test.wav, presents bottleneck summary]
       Top 4: conv1d (35%), matmul (8%), conv_transpose1d (20%), snake_activation (5%)
       Plan: conv1d ~4h, conv_transpose1d ~3h, matmul ~2h, snake_activation ~1h
       Estimated max end-to-end speedup: 1.5-1.8x