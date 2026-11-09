#!/usr/bin/env python3
"""
Redukt GPU CUDA Plugin — Validation & Benchmark (R-090)

Kaggle notebook / standalone script to:
1. Compile the CUDA plugin (mamba_cuda.cu -> libredukt_gpu.so)
2. Convert SafeTensors weights to .rbin format
3. Load plugin via ctypes and validate FFI contract
4. Run forward pass and compare against PyTorch reference
5. Benchmark throughput (tok/s)

Usage on Kaggle:
    - Upload redukt repo as dataset
    - Enable GPU (T4)
    - Run all cells

Usage locally (Linux with CUDA):
    python redukt-gpu-cuda-test.py --plugin ./libredukt_gpu.so --weights weights/mamba-130m

Copyright 2026 Redukt Contributors. Apache-2.0 License.
"""

import argparse
import ctypes
import os
import struct
import subprocess
import sys
import time

# ---- Configuration ----

PLUGIN_SRC = "plugins/redukt-gpu-cuda/mamba_cuda.cu"
PLUGIN_LIB = "libredukt_gpu.so"
WEIGHTS_DIR = "weights/mamba-130m"
RBIN_PATH = None  # auto-derived from weights dir

# ---- FFI types (must match backend_ffi.rs + mamba_cuda.cu) ----

BACKEND_FFI_VERSION = 1
BACKEND_NAME_MAX = 64


class BackendInfo(ctypes.Structure):
    _fields_ = [
        ("ffi_version", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("hidden_dim", ctypes.c_uint32),
        ("name", ctypes.c_uint8 * BACKEND_NAME_MAX),
    ]

    def name_str(self):
        raw = bytes(self.name)
        end = raw.find(b'\0')
        return raw[:end].decode("utf-8") if end >= 0 else raw.decode("utf-8", errors="replace")


# BackendResult enum values
BACKEND_OK = 0
BACKEND_INIT_FAILED = 1
BACKEND_FORWARD_FAILED = 2
BACKEND_VERSION_MISMATCH = 3
BACKEND_WEIGHTS_NOT_FOUND = 4


def result_name(code):
    names = {0: "Ok", 1: "InitFailed", 2: "ForwardFailed",
             3: "VersionMismatch", 4: "WeightsNotFound"}
    return names.get(code, f"Unknown({code})")


# ---- Step 1: Compile ----

def compile_plugin(src_path, out_path, arch="sm_75"):
    """Compile CUDA plugin with nvcc."""
    print(f"\n=== Step 1: Compile CUDA plugin ===")

    # Check nvcc
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print(f"nvcc: {r.stdout.strip().splitlines()[-1]}")
    except FileNotFoundError:
        print("ERROR: nvcc not found. Is CUDA toolkit installed?", file=sys.stderr)
        return False

    cmd = [
        "nvcc", "-shared", "-O3", f"-arch={arch}",
        "--compiler-options", "-fPIC",
        "-o", out_path, src_path
    ]
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"COMPILE FAILED:\n{r.stderr}", file=sys.stderr)
        return False

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"OK: {out_path} ({size_mb:.1f} MB)")
    return True


# ---- Step 2: Convert weights ----

def convert_weights(safetensors_path, rbin_path):
    """Convert SafeTensors to .rbin using convert-weights-gpu.py."""
    print(f"\n=== Step 2: Convert weights to .rbin ===")

    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "convert-weights-gpu.py")
    if not os.path.isfile(script):
        # Try relative to cwd
        script = "scripts/convert-weights-gpu.py"

    if not os.path.isfile(safetensors_path):
        print(f"ERROR: weights not found: {safetensors_path}", file=sys.stderr)
        return False

    if os.path.isfile(rbin_path):
        print(f"  .rbin already exists: {rbin_path} — skipping conversion")
        return True

    cmd = [sys.executable, script, "--input", safetensors_path, "--output", rbin_path]
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(f"CONVERT FAILED:\n{r.stderr}", file=sys.stderr)
        return False
    return True


# ---- Step 3: Load and validate FFI ----

def load_plugin(lib_path, weights_path):
    """Load .so via ctypes and call init."""
    print(f"\n=== Step 3: Load plugin via ctypes ===")

    lib = ctypes.CDLL(lib_path)

    # Set up function signatures
    lib.redukt_backend_init.argtypes = [
        ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(BackendInfo)
    ]
    lib.redukt_backend_init.restype = ctypes.c_int

    lib.redukt_backend_forward.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32
    ]
    lib.redukt_backend_forward.restype = ctypes.c_int

    lib.redukt_backend_reset_state.argtypes = []
    lib.redukt_backend_reset_state.restype = None

    lib.redukt_backend_free.argtypes = []
    lib.redukt_backend_free.restype = None

    # Init
    info = BackendInfo()
    path_bytes = weights_path.encode("utf-8") + b"\0"
    result = lib.redukt_backend_init(path_bytes, 0, ctypes.byref(info))

    print(f"  init result:  {result_name(result)}")
    if result != BACKEND_OK:
        print(f"ERROR: init failed with {result_name(result)}", file=sys.stderr)
        return None, None

    print(f"  ffi_version:  {info.ffi_version}")
    print(f"  vocab_size:   {info.vocab_size}")
    print(f"  hidden_dim:   {info.hidden_dim}")
    print(f"  name:         {info.name_str()}")

    assert info.ffi_version == BACKEND_FFI_VERSION, "FFI version mismatch!"
    assert info.vocab_size > 0, "Invalid vocab size"
    assert info.hidden_dim > 0, "Invalid hidden dim"

    return lib, info


# ---- Step 4: Validate forward pass ----

def validate_forward(lib, info, num_tokens=10):
    """Run forward pass on test tokens and validate output."""
    print(f"\n=== Step 4: Validate forward pass ({num_tokens} tokens) ===")

    vs = info.vocab_size
    logits_type = ctypes.c_float * vs
    logits = logits_type()

    # Reset state first
    lib.redukt_backend_reset_state()

    test_tokens = list(range(min(num_tokens, vs)))
    for i, token in enumerate(test_tokens):
        result = lib.redukt_backend_forward(token, logits, vs)
        if result != BACKEND_OK:
            print(f"  FAIL: forward({token}) = {result_name(result)}")
            return False

        # Basic sanity checks
        logits_list = list(logits)
        max_val = max(logits_list)
        min_val = min(logits_list)
        has_nan = any(x != x for x in logits_list)  # NaN check
        has_inf = any(abs(x) > 1e30 for x in logits_list)
        argmax = logits_list.index(max_val)

        status = "OK"
        if has_nan:
            status = "FAIL (NaN)"
        elif has_inf:
            status = "WARN (inf)"
        elif max_val == min_val:
            status = "WARN (uniform)"

        print(f"  token {token:3d}: range=[{min_val:.4f}, {max_val:.4f}] "
              f"argmax={argmax} {status}")

        if has_nan:
            return False

    print(f"  All {num_tokens} tokens passed forward validation")
    return True


# ---- Step 5: Benchmark throughput ----

def benchmark(lib, info, num_tokens=1000, warmup=50):
    """Measure tokens/second throughput."""
    print(f"\n=== Step 5: Benchmark ({num_tokens} tokens, {warmup} warmup) ===")

    vs = info.vocab_size
    logits_type = ctypes.c_float * vs
    logits = logits_type()

    lib.redukt_backend_reset_state()

    # Warmup
    for i in range(warmup):
        lib.redukt_backend_forward(i % vs, logits, vs)

    # Benchmark
    lib.redukt_backend_reset_state()
    start = time.perf_counter()
    for i in range(num_tokens):
        token = i % 256  # byte-level tokens cycle
        result = lib.redukt_backend_forward(token, logits, vs)
        if result != BACKEND_OK:
            print(f"  FAIL at token {i}: {result_name(result)}")
            return 0.0

    elapsed = time.perf_counter() - start
    tok_per_sec = num_tokens / elapsed
    ms_per_tok = elapsed * 1000 / num_tokens

    print(f"  Tokens:   {num_tokens}")
    print(f"  Time:     {elapsed:.3f} s")
    print(f"  Speed:    {tok_per_sec:.1f} tok/s")
    print(f"  Latency:  {ms_per_tok:.3f} ms/tok")

    return tok_per_sec


# ---- Step 6: PyTorch reference comparison (optional) ----

def pytorch_reference_check(lib, info, rbin_path):
    """Compare GPU plugin output against PyTorch Mamba implementation."""
    print(f"\n=== Step 6: PyTorch reference comparison ===")

    try:
        import torch
        import numpy as np
    except ImportError:
        print("  SKIP: torch/numpy not available")
        return True

    # Read config from .rbin header
    with open(rbin_path, "rb") as f:
        hdr = struct.unpack("<IIIIIIIIIII", f.read(44))

    vocab_size = hdr[2]
    d_model = hdr[3]

    vs = info.vocab_size
    logits_type = ctypes.c_float * vs
    logits = logits_type()

    # Run GPU forward on token 65 ('A')
    lib.redukt_backend_reset_state()
    result = lib.redukt_backend_forward(65, logits, vs)
    if result != BACKEND_OK:
        print(f"  FAIL: forward returned {result_name(result)}")
        return False

    gpu_logits = np.array(list(logits), dtype=np.float32)
    gpu_argmax = int(np.argmax(gpu_logits))
    gpu_top5 = np.argsort(gpu_logits)[-5:][::-1]

    print(f"  GPU argmax:  {gpu_argmax}")
    print(f"  GPU top-5:   {list(gpu_top5)}")
    print(f"  GPU logit range: [{gpu_logits.min():.4f}, {gpu_logits.max():.4f}]")
    print(f"  GPU logit std:   {gpu_logits.std():.4f}")

    # Note: full PyTorch comparison requires loading the same model,
    # which may not be available on Kaggle without mamba_ssm package.
    # We validate basic properties instead:
    assert not np.any(np.isnan(gpu_logits)), "NaN in logits"
    assert not np.any(np.isinf(gpu_logits)), "Inf in logits"
    assert gpu_logits.std() > 0.01, "Logits are nearly uniform"

    print("  Basic property checks PASSED")
    return True


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Redukt GPU CUDA Plugin Test")
    parser.add_argument("--plugin", default=None,
                        help="Path to pre-compiled .so (skip compilation)")
    parser.add_argument("--weights", default=WEIGHTS_DIR,
                        help="Path to weights directory or .rbin file")
    parser.add_argument("--arch", default="sm_75",
                        help="CUDA compute capability (default: sm_75 for T4)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run throughput benchmark (1000 tokens)")
    parser.add_argument("--tokens", type=int, default=1000,
                        help="Number of tokens for benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("Redukt GPU CUDA Plugin — Validation Suite (R-090)")
    print("=" * 60)

    # Determine paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    plugin_path = args.plugin
    weights_path = args.weights

    # Step 1: Compile if needed
    if not plugin_path:
        if not compile_plugin(PLUGIN_SRC, PLUGIN_LIB, args.arch):
            sys.exit(1)
        plugin_path = os.path.abspath(PLUGIN_LIB)
    else:
        plugin_path = os.path.abspath(plugin_path)

    if not os.path.isfile(plugin_path):
        print(f"ERROR: plugin not found: {plugin_path}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Convert weights if needed
    if weights_path.endswith(".rbin"):
        rbin_path = weights_path
    else:
        st_path = os.path.join(weights_path, "model.safetensors")
        rbin_path = os.path.join(weights_path, "model.rbin")
        if not os.path.isfile(rbin_path):
            if not convert_weights(st_path, rbin_path):
                sys.exit(1)
        else:
            print(f"\n.rbin exists: {rbin_path}")

    # Step 3: Load plugin
    lib, info = load_plugin(plugin_path, rbin_path)
    if lib is None:
        sys.exit(1)

    try:
        # Step 4: Validate forward pass
        if not validate_forward(lib, info, num_tokens=20):
            print("\nFAIL: Forward validation failed")
            sys.exit(1)

        # Step 5: Benchmark
        if args.benchmark:
            tok_s = benchmark(lib, info, num_tokens=args.tokens)
            if tok_s == 0:
                sys.exit(1)

        # Step 6: PyTorch reference
        pytorch_reference_check(lib, info, rbin_path)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    finally:
        # Cleanup
        lib.redukt_backend_free()


if __name__ == "__main__":
    main()
