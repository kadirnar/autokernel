"""
AutoKernel -- One-time setup and baseline benchmarking.

Verifies environment (CUDA, Triton, PyTorch), generates deterministic test data,
runs a smoke test on the current kernel, and benchmarks PyTorch reference
implementations so that future experiments have a cached baseline to compare
against.

Usage:
    uv run prepare.py
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (shared with bench.py -- keep in sync)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autokernel")
TEST_DATA_DIR = os.path.join(CACHE_DIR, "test_data")
BASELINES_PATH = os.path.join(CACHE_DIR, "baselines.json")

# Matmul test sizes (must match bench.py)
MATMUL_SIZES = [
    ("tiny",    {"M": 128,  "N": 128,  "K": 128}),
    ("small",   {"M": 512,  "N": 512,  "K": 512}),
    ("medium",  {"M": 1024, "N": 1024, "K": 1024}),
    ("large",   {"M": 2048, "N": 2048, "K": 2048}),
    ("xlarge",  {"M": 4096, "N": 4096, "K": 4096}),
]

# Conv1d test sizes for DAC-VAE baselines
CONV1D_SIZES = [
    ("tiny",   {"batch": 1, "in_channels": 1,   "length": 4096,  "out_channels": 64,  "kernel_size": 7, "stride": 1}),
    ("small",  {"batch": 1, "in_channels": 64,  "length": 4096,  "out_channels": 128, "kernel_size": 4, "stride": 2}),
    ("medium", {"batch": 1, "in_channels": 128, "length": 2048,  "out_channels": 256, "kernel_size": 8, "stride": 4}),
    ("large",  {"batch": 1, "in_channels": 256, "length": 512,   "out_channels": 512, "kernel_size": 16, "stride": 8}),
]

# ConvTranspose1d test sizes for DAC-VAE baselines
CONV_TRANSPOSE1D_SIZES = [
    ("tiny",   {"batch": 1, "in_channels": 1024, "length": 8,   "out_channels": 768, "kernel_size": 16, "stride": 8}),
    ("small",  {"batch": 1, "in_channels": 768,  "length": 64,  "out_channels": 384, "kernel_size": 16, "stride": 8}),
    ("medium", {"batch": 1, "in_channels": 384,  "length": 512, "out_channels": 192, "kernel_size": 8,  "stride": 4}),
    ("large",  {"batch": 1, "in_channels": 192,  "length": 2048, "out_channels": 96, "kernel_size": 4,  "stride": 2}),
]

# Snake activation test sizes for DAC-VAE baselines
SNAKE_SIZES = [
    ("tiny",   {"batch": 1, "channels": 64,   "length": 1024}),
    ("small",  {"batch": 1, "channels": 128,  "length": 4096}),
    ("medium", {"batch": 1, "channels": 512,  "length": 1024}),
    ("large",  {"batch": 1, "channels": 1024, "length": 512}),
]

TEST_DTYPES = [torch.float16, torch.bfloat16]

# Number of warmup and benchmark iterations for baseline timing
_WARMUP_ITERS = 25
_BENCH_ITERS = 100

# Deterministic seed for reproducibility
_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dtype_tag(dtype: torch.dtype) -> str:
    """Short string tag for a dtype, e.g. 'fp16', 'bf16'."""
    return {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]


def _matmul_flops(M: int, N: int, K: int) -> int:
    """FLOPs for a single matmul C[M,N] = A[M,K] @ B[K,N]."""
    return 2 * M * N * K


def _conv1d_flops(s: dict) -> int:
    """FLOPs for conv1d."""
    padding = (s["kernel_size"] - 1) // 2
    L_out = (s["length"] + 2 * padding - s["kernel_size"]) // s["stride"] + 1
    return 2 * s["batch"] * s["out_channels"] * L_out * s["in_channels"] * s["kernel_size"]


def _conv_transpose1d_flops(s: dict) -> int:
    """FLOPs for conv_transpose1d."""
    return 2 * s["batch"] * s["in_channels"] * s["length"] * s["out_channels"] * s["kernel_size"]


def _snake_flops(s: dict) -> int:
    """FLOPs for snake activation (~6 ops per element)."""
    return 6 * s["batch"] * s["channels"] * s["length"]


def _benchmark_fn(fn, *args, warmup: int = _WARMUP_ITERS, iters: int = _BENCH_ITERS):
    """
    Benchmark *fn* using CUDA events. Returns median latency in microseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize()
    for i in range(iters):
        start_events[i].record()
        fn(*args)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    return median_ms * 1000.0  # convert to microseconds


# ---------------------------------------------------------------------------
# Step 1-4: Environment verification
# ---------------------------------------------------------------------------

def verify_environment() -> None:
    """Print GPU specs, PyTorch version, Triton version. Exit on failure."""

    print("=== AutoKernel Setup ===\n")

    # -- CUDA & GPU --
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. A CUDA-capable GPU is required.")
        sys.exit(1)

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)
    mem_gb = props.total_mem / (1024 ** 3)
    sm_count = props.multi_processor_count
    cc_major = props.major
    cc_minor = props.minor

    # Driver and CUDA runtime versions
    # torch.version.cuda gives the CUDA toolkit version PyTorch was compiled with
    cuda_version = torch.version.cuda or "unknown"

    # nvidia-smi driver version -- fall back gracefully
    driver_str = "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            driver_str = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    print(f"GPU: {gpu_name}")
    print(f"  Memory: {mem_gb:.1f} GB")
    print(f"  SM Count: {sm_count}")
    print(f"  Compute Capability: {cc_major}.{cc_minor}")
    print(f"  Driver: {driver_str}")
    print(f"  CUDA: {cuda_version}")
    print()

    # -- PyTorch --
    print(f"PyTorch: {torch.__version__}")

    # -- Triton --
    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("ERROR: Triton is not installed. Install with: pip install triton")
        sys.exit(1)

    print()


# ---------------------------------------------------------------------------
# Step 5-6: Generate & cache test data
# ---------------------------------------------------------------------------

def generate_test_data() -> None:
    """Generate deterministic test tensors for all sizes and dtypes."""

    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    print("Generating test data...")

    gen = torch.Generator(device="cpu")

    # --- Matmul test data ---
    for size_name, dims in MATMUL_SIZES:
        M, N, K = dims["M"], dims["N"], dims["K"]
        for dtype in TEST_DTYPES:
            tag = _dtype_tag(dtype)
            label = f"  matmul/{size_name}/{tag}"

            save_dir = os.path.join(TEST_DATA_DIR, "matmul", size_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{tag}.pt")

            if os.path.exists(save_path):
                print(f"{label} ... cached")
                continue

            # Deterministic generation -- seed is fixed per (size, dtype) pair
            gen.manual_seed(_SEED)
            A = torch.randn(M, K, generator=gen, dtype=dtype)
            B = torch.randn(K, N, generator=gen, dtype=dtype)

            torch.save({"A": A, "B": B}, save_path)
            print(f"{label} ... ok")

    # --- Conv1d test data ---
    for size_name, dims in CONV1D_SIZES:
        for dtype in [torch.float32]:
            tag = _dtype_tag(dtype)
            label = f"  conv1d/{size_name}/{tag}"

            save_dir = os.path.join(TEST_DATA_DIR, "conv1d", size_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{tag}.pt")

            if os.path.exists(save_path):
                print(f"{label} ... cached")
                continue

            gen.manual_seed(_SEED)
            x = torch.randn(dims["batch"], dims["in_channels"], dims["length"], generator=gen, dtype=dtype)
            weight = torch.randn(dims["out_channels"], dims["in_channels"], dims["kernel_size"], generator=gen, dtype=dtype) * 0.02
            bias = torch.randn(dims["out_channels"], generator=gen, dtype=dtype) * 0.01

            torch.save({"x": x, "weight": weight, "bias": bias, "stride": dims["stride"]}, save_path)
            print(f"{label} ... ok")

    # --- ConvTranspose1d test data ---
    for size_name, dims in CONV_TRANSPOSE1D_SIZES:
        for dtype in [torch.float32]:
            tag = _dtype_tag(dtype)
            label = f"  conv_transpose1d/{size_name}/{tag}"

            save_dir = os.path.join(TEST_DATA_DIR, "conv_transpose1d", size_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{tag}.pt")

            if os.path.exists(save_path):
                print(f"{label} ... cached")
                continue

            gen.manual_seed(_SEED)
            x = torch.randn(dims["batch"], dims["in_channels"], dims["length"], generator=gen, dtype=dtype)
            weight = torch.randn(dims["in_channels"], dims["out_channels"], dims["kernel_size"], generator=gen, dtype=dtype) * 0.02
            bias = torch.randn(dims["out_channels"], generator=gen, dtype=dtype) * 0.01

            torch.save({"x": x, "weight": weight, "bias": bias, "stride": dims["stride"]}, save_path)
            print(f"{label} ... ok")

    # --- Snake activation test data ---
    for size_name, dims in SNAKE_SIZES:
        for dtype in [torch.float32]:
            tag = _dtype_tag(dtype)
            label = f"  snake_activation/{size_name}/{tag}"

            save_dir = os.path.join(TEST_DATA_DIR, "snake_activation", size_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{tag}.pt")

            if os.path.exists(save_path):
                print(f"{label} ... cached")
                continue

            gen.manual_seed(_SEED)
            x = torch.randn(dims["batch"], dims["channels"], dims["length"], generator=gen, dtype=dtype)
            alpha = torch.ones(1, dims["channels"], 1, dtype=dtype)

            torch.save({"x": x, "alpha": alpha}, save_path)
            print(f"{label} ... ok")

    print()


# ---------------------------------------------------------------------------
# Step 7: Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> None:
    """Import kernel.py, run on tiny input, check correctness."""

    print("Smoke test...")

    # Import kernel
    try:
        import kernel  # noqa: F401
        print("  Import kernel.py: ok")
    except Exception as e:
        print(f"  Import kernel.py: FAIL ({e})")
        sys.exit(1)

    kernel_type = getattr(kernel, "KERNEL_TYPE", "matmul")

    if kernel_type == "matmul":
        from reference import matmul_ref
        dtype = torch.float16
        dims = dict(MATMUL_SIZES)["tiny"]
        M, N, K = dims["M"], dims["N"], dims["K"]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_SEED)
        A = torch.randn(M, K, generator=gen, dtype=dtype).cuda()
        B = torch.randn(K, N, generator=gen, dtype=dtype).cuda()
        try:
            C_kernel = kernel.kernel_fn(A, B)
            torch.cuda.synchronize()
            print("  Run kernel (tiny, fp16): ok")
        except Exception as e:
            print(f"  Run kernel (tiny, fp16): FAIL ({e})")
            sys.exit(1)
        C_ref = matmul_ref(A, B)
        torch.cuda.synchronize()
        atol, rtol = 1e-2, 1e-2
        if torch.allclose(C_kernel, C_ref, atol=atol, rtol=rtol):
            print("  Correctness check: PASS")
        else:
            max_diff = (C_kernel - C_ref).abs().max().item()
            print(f"  Correctness check: FAIL (max diff = {max_diff:.6f})")

    elif kernel_type == "conv1d":
        from reference import conv1d_ref
        dtype = torch.float32
        dims = CONV1D_SIZES[0][1]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_SEED)
        x = torch.randn(dims["batch"], dims["in_channels"], dims["length"], generator=gen, dtype=dtype).cuda()
        weight = torch.randn(dims["out_channels"], dims["in_channels"], dims["kernel_size"], generator=gen, dtype=dtype).cuda() * 0.02
        bias = torch.randn(dims["out_channels"], generator=gen, dtype=dtype).cuda() * 0.01
        padding = (dims["kernel_size"] - 1) // 2
        try:
            y_kernel = kernel.kernel_fn(x, weight, bias, stride=dims["stride"], padding=padding)
            torch.cuda.synchronize()
            print("  Run kernel (tiny, fp32): ok")
        except Exception as e:
            print(f"  Run kernel (tiny, fp32): FAIL ({e})")
            sys.exit(1)
        y_ref = conv1d_ref(x, weight, bias, stride=dims["stride"], padding=padding)
        torch.cuda.synchronize()
        atol, rtol = 1e-4, 1e-4
        if torch.allclose(y_kernel, y_ref, atol=atol, rtol=rtol):
            print("  Correctness check: PASS")
        else:
            max_diff = (y_kernel - y_ref).abs().max().item()
            print(f"  Correctness check: FAIL (max diff = {max_diff:.6f})")

    elif kernel_type == "conv_transpose1d":
        from reference import conv_transpose1d_ref
        dtype = torch.float32
        dims = CONV_TRANSPOSE1D_SIZES[0][1]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_SEED)
        x = torch.randn(dims["batch"], dims["in_channels"], dims["length"], generator=gen, dtype=dtype).cuda()
        weight = torch.randn(dims["in_channels"], dims["out_channels"], dims["kernel_size"], generator=gen, dtype=dtype).cuda() * 0.02
        bias = torch.randn(dims["out_channels"], generator=gen, dtype=dtype).cuda() * 0.01
        padding = (dims["kernel_size"] - dims["stride"]) // 2
        output_padding = 1 if (dims["kernel_size"] - dims["stride"]) % 2 != 0 else 0
        try:
            y_kernel = kernel.kernel_fn(x, weight, bias, stride=dims["stride"], padding=padding, output_padding=output_padding)
            torch.cuda.synchronize()
            print("  Run kernel (tiny, fp32): ok")
        except Exception as e:
            print(f"  Run kernel (tiny, fp32): FAIL ({e})")
            sys.exit(1)
        y_ref = conv_transpose1d_ref(x, weight, bias, stride=dims["stride"], padding=padding, output_padding=output_padding)
        torch.cuda.synchronize()
        atol, rtol = 1e-4, 1e-4
        if torch.allclose(y_kernel, y_ref, atol=atol, rtol=rtol):
            print("  Correctness check: PASS")
        else:
            max_diff = (y_kernel - y_ref).abs().max().item()
            print(f"  Correctness check: FAIL (max diff = {max_diff:.6f})")

    elif kernel_type == "snake_activation":
        from reference import snake_activation_ref
        dtype = torch.float32
        dims = SNAKE_SIZES[0][1]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_SEED)
        x = torch.randn(dims["batch"], dims["channels"], dims["length"], generator=gen, dtype=dtype).cuda()
        alpha = torch.ones(1, dims["channels"], 1, dtype=dtype).cuda()
        try:
            y_kernel = kernel.kernel_fn(x, alpha)
            torch.cuda.synchronize()
            print("  Run kernel (tiny, fp32): ok")
        except Exception as e:
            print(f"  Run kernel (tiny, fp32): FAIL ({e})")
            sys.exit(1)
        y_ref = snake_activation_ref(x, alpha)
        torch.cuda.synchronize()
        atol, rtol = 1e-5, 1e-5
        if torch.allclose(y_kernel, y_ref, atol=atol, rtol=rtol):
            print("  Correctness check: PASS")
        else:
            max_diff = (y_kernel - y_ref).abs().max().item()
            print(f"  Correctness check: FAIL (max diff = {max_diff:.6f})")

    else:
        print(f"  Smoke test for kernel type '{kernel_type}' -- skipped (no specific test)")

    print()


# ---------------------------------------------------------------------------
# Step 8: Benchmark PyTorch baselines
# ---------------------------------------------------------------------------

def benchmark_baselines() -> dict:
    """Benchmark PyTorch reference ops at all sizes and dtypes. Returns results dict."""

    print("Benchmarking PyTorch baselines...")
    results = {}

    # --- Matmul baselines ---
    for size_name, dims in MATMUL_SIZES:
        M, N, K = dims["M"], dims["N"], dims["K"]
        flops = _matmul_flops(M, N, K)

        for dtype in TEST_DTYPES:
            tag = _dtype_tag(dtype)

            # Load cached test data if available, else generate on the fly
            save_path = os.path.join(TEST_DATA_DIR, "matmul", size_name, f"{tag}.pt")
            if os.path.exists(save_path):
                data = torch.load(save_path, weights_only=True)
                A = data["A"].cuda()
                B = data["B"].cuda()
            else:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(_SEED)
                A = torch.randn(M, K, generator=gen, dtype=dtype).cuda()
                B = torch.randn(K, N, generator=gen, dtype=dtype).cuda()

            latency_us = _benchmark_fn(torch.matmul, A, B)
            tflops = flops / (latency_us * 1e-6) / 1e12

            key = f"matmul_{size_name}_{tag}"
            results[key] = {
                "kernel_type": "matmul",
                "size": size_name,
                "dtype": tag,
                "M": M, "N": N, "K": K,
                "latency_us": round(latency_us, 2),
                "throughput_tflops": round(tflops, 3),
            }

            print(f"  matmul {size_name} {tag}: {tflops:.1f} TFLOPS ({latency_us:.2f} us)")

            del A, B
            torch.cuda.empty_cache()

    # --- Conv1d baselines ---
    for size_name, dims in CONV1D_SIZES:
        flops = _conv1d_flops(dims)
        for dtype in [torch.float32]:
            tag = _dtype_tag(dtype)

            gen = torch.Generator(device="cpu")
            gen.manual_seed(_SEED)
            x = torch.randn(dims["batch"], dims["in_channels"], dims["length"], generator=gen, dtype=dtype).cuda()
            weight = torch.randn(dims["out_channels"], dims["in_channels"], dims["kernel_size"], generator=gen, dtype=dtype).cuda() * 0.02
            bias = torch.randn(dims["out_channels"], generator=gen, dtype=dtype).cuda() * 0.01
            padding = (dims["kernel_size"] - 1) // 2

            latency_us = _benchmark_fn(F.conv1d, x, weight, bias, dims["stride"], padding)
            tflops = flops / (latency_us * 1e-6) / 1e12

            key = f"conv1d_{size_name}_{tag}"
            results[key] = {
                "kernel_type": "conv1d",
                "size": size_name,
                "dtype": tag,
                "latency_us": round(latency_us, 2),
                "throughput_tflops": round(tflops, 3),
            }

            print(f"  conv1d {size_name} {tag}: {tflops:.3f} TFLOPS ({latency_us:.2f} us)")

            del x, weight, bias
            torch.cuda.empty_cache()

    # --- ConvTranspose1d baselines ---
    for size_name, dims in CONV_TRANSPOSE1D_SIZES:
        flops = _conv_transpose1d_flops(dims)
        for dtype in [torch.float32]:
            tag = _dtype_tag(dtype)

            gen = torch.Generator(device="cpu")
            gen.manual_seed(_SEED)
            x = torch.randn(dims["batch"], dims["in_channels"], dims["length"], generator=gen, dtype=dtype).cuda()
            weight = torch.randn(dims["in_channels"], dims["out_channels"], dims["kernel_size"], generator=gen, dtype=dtype).cuda() * 0.02
            bias = torch.randn(dims["out_channels"], generator=gen, dtype=dtype).cuda() * 0.01
            padding = (dims["kernel_size"] - dims["stride"]) // 2
            output_padding = 1 if (dims["kernel_size"] - dims["stride"]) % 2 != 0 else 0

            latency_us = _benchmark_fn(F.conv_transpose1d, x, weight, bias, dims["stride"], padding, output_padding)
            tflops = flops / (latency_us * 1e-6) / 1e12

            key = f"conv_transpose1d_{size_name}_{tag}"
            results[key] = {
                "kernel_type": "conv_transpose1d",
                "size": size_name,
                "dtype": tag,
                "latency_us": round(latency_us, 2),
                "throughput_tflops": round(tflops, 3),
            }

            print(f"  conv_transpose1d {size_name} {tag}: {tflops:.3f} TFLOPS ({latency_us:.2f} us)")

            del x, weight, bias
            torch.cuda.empty_cache()

    # --- Snake activation baselines ---
    def _snake_ref(x, alpha):
        return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)

    for size_name, dims in SNAKE_SIZES:
        flops = _snake_flops(dims)
        for dtype in [torch.float32]:
            tag = _dtype_tag(dtype)

            gen = torch.Generator(device="cpu")
            gen.manual_seed(_SEED)
            x = torch.randn(dims["batch"], dims["channels"], dims["length"], generator=gen, dtype=dtype).cuda()
            alpha = torch.ones(1, dims["channels"], 1, dtype=dtype).cuda()

            latency_us = _benchmark_fn(_snake_ref, x, alpha)
            # Report as GB/s for memory-bound ops
            bytes_total = (2 * dims["batch"] * dims["channels"] * dims["length"] + dims["channels"]) * 4  # fp32
            gb_s = bytes_total / (latency_us * 1e-6) / 1e9

            key = f"snake_activation_{size_name}_{tag}"
            results[key] = {
                "kernel_type": "snake_activation",
                "size": size_name,
                "dtype": tag,
                "latency_us": round(latency_us, 2),
                "throughput_gb_s": round(gb_s, 3),
            }

            print(f"  snake_activation {size_name} {tag}: {gb_s:.1f} GB/s ({latency_us:.2f} us)")

            del x, alpha
            torch.cuda.empty_cache()

    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1-4: Verify environment
    verify_environment()

    # Step 5: Create cache directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Step 6: Generate test data
    generate_test_data()

    # Step 7: Smoke test
    smoke_test()

    # Step 8: Benchmark baselines
    baselines = benchmark_baselines()

    # Save baselines
    with open(BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"Baselines saved to {BASELINES_PATH}")

    # Step 9: Summary
    print()
    print("Ready to run experiments!")


if __name__ == "__main__":
    main()
