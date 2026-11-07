"""
MambaByte Cross-Framework Validation -- Kaggle Notebook
=======================================================
Validates that MambaByte weights produce consistent results between
PyTorch (training) and Rust (Redukt inference).

Outputs:
- validation-report.txt: numerical agreement + compression tests
- logits-comparison.csv: per-position BPB comparison

Prerequisites:
- Trained MambaByte weights (model.safetensors) in /kaggle/working/
  or downloadable from HuggingFace
- Kaggle GPU enabled (T4 or P100)

Tests:
  1. PyTorch cross-entropy BPB on enwik8 10KB
  2. Redukt AZTH compression BPB on enwik8 10KB
  3. BPB agreement (PyTorch CE vs Rust AZTH, threshold 5%)
  4. Redukt roundtrip (compress -> decompress -> diff)
  5. Redukt determinism (2 runs, SHA-256 identical)
  6. Multi-file BPB validation (range check)

License: Apache-2.0
"""

import os
import sys
import math
import time
import csv
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# Section 1: Configuration
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 768
D_INNER = 1536       # 2 * d_model
D_STATE = 16
D_CONV = 4
DT_RANK = 48         # ceil(768 / 16)
N_LAYER = 24

WORK_DIR = "/kaggle/working"
REPO_URL = "https://github.com/DavidCaastro/redukt.git"
REPO_DIR = f"{WORK_DIR}/redukt"
DATA_DIR = f"{WORK_DIR}/validation-data"
RESULTS_DIR = f"{WORK_DIR}/validation-results"
REPORT_TXT = f"{RESULTS_DIR}/validation-report.txt"
LOGITS_CSV = f"{RESULTS_DIR}/logits-comparison.csv"

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_10K = f"{DATA_DIR}/enwik8_10k.bin"

# Thresholds
BPB_AGREE_WARN = 0.01   # 1% difference -> WARNING
BPB_AGREE_FAIL = 0.05   # 5% difference -> FAIL
BPB_RANGE_LOW = 0.85    # expected BPB range for trained model
BPB_RANGE_HIGH = 1.40

# Weights path: check multiple locations
WEIGHTS_CANDIDATES = [
    f"{WORK_DIR}/mambabyte-96m.safetensors",
    f"{WORK_DIR}/checkpoints/mambabyte-96m.safetensors",
    f"{WORK_DIR}/model.safetensors",
]

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[DEVICE] CPU only")


# ---------------------------------------------------------------------------
# Section 2: Model Definition (copied from mambabyte-train.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MambaBlock(nn.Module):
    """Single Mamba block with sequential SSM scan."""

    def __init__(self, d_model: int, d_inner: int, d_state: int,
                 d_conv: int, dt_rank: int):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_inner, bias=True,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)
        x_branch = x_branch[:, :, :residual.shape[1]]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        y = self._ssm_scan(x_branch)
        y = y * F.silu(z)
        out = self.out_proj(y) + residual
        return out

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        B_batch, L, D = x.shape
        x_dbl = self.x_proj(x)
        dt_input = x_dbl[:, :, :self.dt_rank]
        B_ssm = x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_ssm = x_dbl[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_input))
        A = -torch.exp(self.A_log)

        h = torch.zeros(B_batch, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            dt_t = dt[:, t, :]
            B_t = B_ssm[:, t, :]
            C_t = C_ssm[:, t, :]
            x_t = x[:, t, :]

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = dA * h + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y


class MambaByte(nn.Module):
    """MambaByte: byte-level Mamba language model (96M parameters)."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL,
                 d_inner: int = D_INNER, d_state: int = D_STATE,
                 d_conv: int = D_CONV, dt_rank: int = DT_RANK,
                 n_layer: int = N_LAYER):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Section 3: Utilities
# ---------------------------------------------------------------------------

def run(cmd: str, cwd: Optional[str] = None, timeout: int = 600
        ) -> Tuple[int, str]:
    """Run shell command. Returns (returncode, stdout)."""
    print(f"$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, timeout=timeout,
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        print(result.stdout[-2000:])
    if result.stderr.strip():
        print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode, result.stdout


def section(title: str):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def sha256_file(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_weights() -> Optional[str]:
    """Find MambaByte SafeTensors weights in known locations."""
    for candidate in WEIGHTS_CANDIDATES:
        if os.path.exists(candidate):
            print(f"[WEIGHTS] Found: {candidate}")
            return candidate
    print("[WEIGHTS] No trained weights found in any candidate location")
    return None


# ---------------------------------------------------------------------------
# Section 4: Environment Setup
# ---------------------------------------------------------------------------

def setup_environment():
    """Clone Redukt and build from source. Download test data."""
    section("Phase 1: Environment Setup")

    # Install Rust if needed
    rc, _ = run("rustc --version")
    if rc != 0:
        print("Installing Rust toolchain...")
        run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y")
        os.environ["PATH"] = (
            f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]
        )
        run("rustc --version")

    # Clone repo
    if not os.path.exists(REPO_DIR):
        run(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")

    # Build release binary
    print("Building Redukt (release)...")
    rc, _ = run("cargo build --release", cwd=REPO_DIR, timeout=1200)
    if rc != 0:
        print("ERROR: Redukt build failed")
        return False

    print("Redukt built successfully")
    return True


def setup_test_data():
    """Download enwik8 and prepare test files."""
    section("Phase 2: Prepare Test Data")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Download enwik8 10KB
    if not os.path.exists(ENWIK8_10K):
        print("Downloading enwik8...")
        try:
            import urllib.request
            import zipfile
            import io

            zip_path = f"{DATA_DIR}/enwik8.zip"
            urllib.request.urlretrieve(ENWIK8_URL, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open("enwik8") as f:
                    data = f.read(10240)

            with open(ENWIK8_10K, "wb") as f:
                f.write(data)

            os.remove(zip_path)
            print(f"[DATA] enwik8 10KB saved ({len(data)} bytes)")
        except Exception as e:
            print(f"[DATA] enwik8 download failed: {e}")
            # Fallback: deterministic synthetic data
            import random
            rng = random.Random(42)
            data = bytes(rng.randint(0, 255) for _ in range(10240))
            with open(ENWIK8_10K, "wb") as f:
                f.write(data)
            print("[DATA] Using deterministic synthetic fallback")
    else:
        print(f"[DATA] enwik8 10KB already present")

    # Create additional test files (synthetic, deterministic)
    test_files = {
        "text-like.bin": _make_text_like(10240),
        "binary-random.bin": _make_binary_random(10240),
        "structured.bin": _make_structured(10240),
    }
    for name, data in test_files.items():
        path = f"{DATA_DIR}/{name}"
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
            print(f"[DATA] Created {name} ({len(data)} bytes)")


def _make_text_like(size: int) -> bytes:
    """Create text-like data with English character distribution."""
    import random
    rng = random.Random(1337)
    chars = (
        b"etaoinsrhld "
        b"ETAOINSRHLD "
        b"cumwfgypbvkjxqz\n"
    )
    return bytes(rng.choice(chars) for _ in range(size))


def _make_binary_random(size: int) -> bytes:
    """Create high-entropy binary data (deterministic)."""
    import random
    rng = random.Random(7777)
    return bytes(rng.randint(0, 255) for _ in range(size))


def _make_structured(size: int) -> bytes:
    """Create structured data with repeating records."""
    import struct as st
    record = st.pack("<IHH8s", 0xDEADBEEF, 256, 42, b"redukt\x00\x00")
    reps = size // len(record) + 1
    return (record * reps)[:size]


# ---------------------------------------------------------------------------
# Section 5: PyTorch BPB Computation
# ---------------------------------------------------------------------------

def load_pytorch_model(weights_path: str) -> Optional[MambaByte]:
    """Load MambaByte model from SafeTensors into PyTorch."""
    section("Phase 3: Load PyTorch Model")

    try:
        from safetensors.torch import load_file
    except ImportError:
        print("[PYTORCH] Installing safetensors...")
        run("pip install -q safetensors")
        from safetensors.torch import load_file

    model = MambaByte()
    print(f"[PYTORCH] Model parameters: {model.num_params():,}")

    # Load weights from SafeTensors
    print(f"[PYTORCH] Loading weights from: {weights_path}")
    state = load_file(weights_path)

    # Map SafeTensors keys back to model state dict
    model_state = {}

    # Embedding
    if "backbone.embeddings.weight" in state:
        model_state["embedding.weight"] = state["backbone.embeddings.weight"]
    elif "embedding.weight" in state:
        model_state["embedding.weight"] = state["embedding.weight"]

    # Layers
    for i in range(N_LAYER):
        pfx_st = f"backbone.layers.{i}"
        pfx_m = f"layers.{i}"

        key_map = {
            f"{pfx_st}.norm.weight": f"{pfx_m}.norm.weight",
            f"{pfx_st}.mixer.in_proj.weight": f"{pfx_m}.in_proj.weight",
            f"{pfx_st}.mixer.conv1d.weight": f"{pfx_m}.conv1d.weight",
            f"{pfx_st}.mixer.conv1d.bias": f"{pfx_m}.conv1d.bias",
            f"{pfx_st}.mixer.x_proj.weight": f"{pfx_m}.x_proj.weight",
            f"{pfx_st}.mixer.dt_proj.weight": f"{pfx_m}.dt_proj.weight",
            f"{pfx_st}.mixer.dt_proj.bias": f"{pfx_m}.dt_proj.bias",
            f"{pfx_st}.mixer.A_log": f"{pfx_m}.A_log",
            f"{pfx_st}.mixer.D": f"{pfx_m}.D",
            f"{pfx_st}.mixer.out_proj.weight": f"{pfx_m}.out_proj.weight",
        }
        for st_key, m_key in key_map.items():
            if st_key in state:
                model_state[m_key] = state[st_key]

    # Final norm
    if "backbone.norm_f.weight" in state:
        model_state["norm_f.weight"] = state["backbone.norm_f.weight"]
    elif "norm_f.weight" in state:
        model_state["norm_f.weight"] = state["norm_f.weight"]

    # lm_head is tied to embedding -- set from embedding
    if "embedding.weight" in model_state:
        model_state["lm_head.weight"] = model_state["embedding.weight"]

    # Try loading; report missing/unexpected keys
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[PYTORCH] WARNING: missing keys: {missing}")
    if unexpected:
        print(f"[PYTORCH] WARNING: unexpected keys: {unexpected}")

    model = model.to(DEVICE)
    model.eval()
    print("[PYTORCH] Model loaded and set to eval mode")
    return model


@torch.no_grad()
def compute_pytorch_bpb(model: MambaByte, file_path: str,
                        seq_len: int = 8192) -> Tuple[float, List[float]]:
    """
    Compute cross-entropy BPB on a file using the PyTorch model.

    Returns:
        (overall_bpb, per_chunk_bpb_list)
    """
    with open(file_path, "rb") as f:
        data = f.read()

    tokens = torch.tensor(list(data), dtype=torch.long).unsqueeze(0)
    tokens = tokens.to(DEVICE)

    total_loss = 0.0
    total_tokens = 0
    chunk_bpbs = []

    for start in range(0, tokens.shape[1] - 1, seq_len):
        end = min(start + seq_len, tokens.shape[1] - 1)
        input_ids = tokens[:, start:end]
        labels = tokens[:, start + 1:end + 1]

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            labels.reshape(-1),
            reduction="sum",
        )

        chunk_tokens = labels.numel()
        chunk_loss = loss.item()
        total_loss += chunk_loss
        total_tokens += chunk_tokens

        chunk_bpb = (chunk_loss / chunk_tokens) / math.log(2)
        chunk_bpbs.append(chunk_bpb)

    overall_bpb = (total_loss / total_tokens) / math.log(2)
    return overall_bpb, chunk_bpbs


# ---------------------------------------------------------------------------
# Section 6: Rust Compression
# ---------------------------------------------------------------------------

REDUKT_BIN = f"{REPO_DIR}/target/release/redukt"


def copy_weights_for_rust(weights_path: str):
    """Copy MambaByte weights to Redukt weights directory for the Rust loader."""
    dst_dir = f"{REPO_DIR}/weights/mambabyte-96m"
    os.makedirs(dst_dir, exist_ok=True)
    dst = f"{dst_dir}/model.safetensors"
    if not os.path.exists(dst):
        run(f"cp {weights_path} {dst}")
        print(f"[RUST] Weights copied to {dst}")
    else:
        print(f"[RUST] Weights already at {dst}")


def rust_compress(input_path: str, output_path: str,
                  bytes_limit: int = 10240) -> Tuple[bool, float]:
    """
    Compress a file using Redukt with mambabyte backbone.

    Returns:
        (success, bpb)
    """
    rc, stdout = run(
        f"{REDUKT_BIN} compress --input {input_path} --output {output_path} "
        f"--backbone mambabyte --bytes {bytes_limit}",
        cwd=REPO_DIR,
    )
    if rc != 0:
        return False, 0.0

    if not os.path.exists(output_path):
        return False, 0.0

    in_size = min(os.path.getsize(input_path), bytes_limit)
    out_size = os.path.getsize(output_path)
    bpb = (out_size * 8) / in_size if in_size > 0 else 0.0
    return True, bpb


def rust_decompress(input_path: str, output_path: str) -> bool:
    """Decompress an AZTH file using Redukt. Returns success."""
    rc, _ = run(
        f"{REDUKT_BIN} decompress --input {input_path} --output {output_path}",
        cwd=REPO_DIR,
    )
    return rc == 0


# ---------------------------------------------------------------------------
# Section 7: Validation Tests
# ---------------------------------------------------------------------------

class ValidationResult:
    """Single test result."""

    def __init__(self, name: str, passed: bool, detail: str):
        self.name = name
        self.passed = passed
        self.detail = detail

    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def __str__(self) -> str:
        return f"[{self.status()}] {self.name}: {self.detail}"


def test_pytorch_bpb(model: MambaByte) -> ValidationResult:
    """Test 1: PyTorch cross-entropy BPB on enwik8 10KB."""
    section("Test 1: PyTorch BPB")

    bpb, chunks = compute_pytorch_bpb(model, ENWIK8_10K)
    print(f"  PyTorch BPB on enwik8 10KB: {bpb:.4f}")
    print(f"  Chunks: {len(chunks)}, range [{min(chunks):.4f}, {max(chunks):.4f}]")

    in_range = BPB_RANGE_LOW <= bpb <= BPB_RANGE_HIGH
    detail = f"BPB={bpb:.4f} (expected {BPB_RANGE_LOW}-{BPB_RANGE_HIGH})"
    if not in_range:
        detail += " -- OUT OF EXPECTED RANGE"
        # Not necessarily a failure: model may still be training or have
        # different characteristics. Warn but do not block.
        print(f"  WARNING: BPB {bpb:.4f} outside expected range")

    return ValidationResult("PyTorch BPB range", in_range, detail)


def test_rust_compression() -> Tuple[ValidationResult, float]:
    """Test 2: Redukt AZTH compression BPB on enwik8 10KB."""
    section("Test 2: Rust Compression BPB")

    azth_path = f"{DATA_DIR}/enwik8_10k.mambabyte.azth"
    ok, bpb = rust_compress(ENWIK8_10K, azth_path)

    if not ok:
        return ValidationResult(
            "Rust compression", False,
            "Redukt compress --backbone mambabyte failed"
        ), 0.0

    print(f"  Rust AZTH BPB on enwik8 10KB: {bpb:.4f}")
    in_range = BPB_RANGE_LOW <= bpb <= BPB_RANGE_HIGH
    detail = f"BPB={bpb:.4f}"
    if not in_range:
        detail += f" (outside {BPB_RANGE_LOW}-{BPB_RANGE_HIGH})"

    return ValidationResult("Rust compression BPB", in_range, detail), bpb


def test_bpb_agreement(pytorch_bpb: float, rust_bpb: float
                       ) -> ValidationResult:
    """Test 3: BPB agreement between PyTorch CE and Rust AZTH."""
    section("Test 3: BPB Agreement (PyTorch vs Rust)")

    if rust_bpb == 0.0 or pytorch_bpb == 0.0:
        return ValidationResult(
            "BPB agreement", False,
            "Cannot compare: one or both BPB values are zero"
        )

    # AZTH BPB includes header overhead and coder inefficiency, so Rust BPB
    # is expected to be slightly higher than PyTorch CE BPB.
    # We compare relative difference.
    diff = abs(rust_bpb - pytorch_bpb)
    rel_diff = diff / pytorch_bpb

    print(f"  PyTorch CE BPB:  {pytorch_bpb:.4f}")
    print(f"  Rust AZTH BPB:   {rust_bpb:.4f}")
    print(f"  Absolute diff:   {diff:.4f}")
    print(f"  Relative diff:   {rel_diff:.2%}")

    if rel_diff > BPB_AGREE_FAIL:
        status = "FAIL"
        passed = False
    elif rel_diff > BPB_AGREE_WARN:
        status = "WARNING"
        passed = True
    else:
        status = "PASS"
        passed = True

    detail = (
        f"PyTorch={pytorch_bpb:.4f}, Rust={rust_bpb:.4f}, "
        f"diff={rel_diff:.2%} ({status})"
    )
    if not passed:
        detail += " -- NUMERICAL DIVERGENCE DETECTED"

    return ValidationResult("BPB agreement", passed, detail)


def test_roundtrip() -> ValidationResult:
    """Test 4: Compress -> decompress -> byte-identical."""
    section("Test 4: Roundtrip Integrity")

    azth_path = f"{DATA_DIR}/roundtrip_test.azth"
    dec_path = f"{DATA_DIR}/roundtrip_test.dec"

    # Compress
    ok, _ = rust_compress(ENWIK8_10K, azth_path)
    if not ok:
        return ValidationResult(
            "Roundtrip", False, "Compression failed"
        )

    # Decompress
    ok = rust_decompress(azth_path, dec_path)
    if not ok:
        return ValidationResult(
            "Roundtrip", False, "Decompression failed"
        )

    # Compare SHA-256
    original_hash = sha256_file(ENWIK8_10K)
    decoded_hash = sha256_file(dec_path)

    passed = original_hash == decoded_hash
    detail = f"SHA-256 match={passed}"
    if not passed:
        detail += f" (orig={original_hash[:16]}..., dec={decoded_hash[:16]}...)"
    else:
        detail += f" ({original_hash[:16]}...)"

    print(f"  Original: {original_hash}")
    print(f"  Decoded:  {decoded_hash}")
    print(f"  Match:    {passed}")

    return ValidationResult("Roundtrip integrity", passed, detail)


def test_determinism() -> ValidationResult:
    """Test 5: Two compression runs produce identical AZTH output."""
    section("Test 5: Determinism")

    azth_1 = f"{DATA_DIR}/det_run1.azth"
    azth_2 = f"{DATA_DIR}/det_run2.azth"

    ok1, bpb1 = rust_compress(ENWIK8_10K, azth_1)
    ok2, bpb2 = rust_compress(ENWIK8_10K, azth_2)

    if not (ok1 and ok2):
        return ValidationResult(
            "Determinism", False,
            f"Compression failed (run1={ok1}, run2={ok2})"
        )

    hash1 = sha256_file(azth_1)
    hash2 = sha256_file(azth_2)

    passed = hash1 == hash2
    detail = f"SHA-256 identical={passed}, BPB run1={bpb1:.4f} run2={bpb2:.4f}"
    if not passed:
        detail += f" (h1={hash1[:16]}..., h2={hash2[:16]}...)"

    print(f"  Run 1: {hash1}")
    print(f"  Run 2: {hash2}")
    print(f"  Match: {passed}")

    return ValidationResult("Determinism", passed, detail)


def test_multifile_bpb() -> ValidationResult:
    """Test 6: BPB on multiple file types within expected ranges."""
    section("Test 6: Multi-File BPB")

    test_files = {
        "enwik8_10k.bin": (0.5, 2.0),
        "text-like.bin": (0.5, 4.0),
        "binary-random.bin": (4.0, 9.0),
        "structured.bin": (0.1, 4.0),
    }

    all_ok = True
    details = []

    for name, (low, high) in test_files.items():
        fpath = f"{DATA_DIR}/{name}"
        if not os.path.exists(fpath):
            details.append(f"{name}: SKIP (not found)")
            continue

        azth = f"{DATA_DIR}/{name}.multifile.azth"
        ok, bpb = rust_compress(fpath, azth)

        if not ok:
            details.append(f"{name}: FAIL (compression error)")
            all_ok = False
            continue

        in_range = low <= bpb <= high
        status = "OK" if in_range else "OUT-OF-RANGE"
        details.append(f"{name}: BPB={bpb:.4f} [{low}-{high}] {status}")
        print(f"  {name}: BPB={bpb:.4f} (expected {low}-{high}) {status}")

        if not in_range:
            all_ok = False

    return ValidationResult(
        "Multi-file BPB", all_ok, "; ".join(details)
    )


# ---------------------------------------------------------------------------
# Section 8: Report Generation
# ---------------------------------------------------------------------------

def generate_report(results: List[ValidationResult],
                    pytorch_bpb: float, rust_bpb: float):
    """Write validation-report.txt with all test results."""
    section("Generating Report")

    lines = [
        "MambaByte Cross-Framework Validation Report",
        "=" * 50,
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform: Kaggle ({DEVICE})",
        f"Architecture: MambaByte-96M "
        f"(d={D_MODEL}, d_inner={D_INNER}, n_layer={N_LAYER})",
        "",
        "Results:",
        "-" * 50,
    ]

    pass_count = 0
    fail_count = 0
    for r in results:
        lines.append(str(r))
        if r.passed:
            pass_count += 1
        else:
            fail_count += 1

    lines.append("")
    lines.append("-" * 50)
    lines.append(f"Total: {pass_count} PASS, {fail_count} FAIL "
                 f"out of {len(results)} tests")

    if pytorch_bpb > 0 and rust_bpb > 0:
        lines.append("")
        lines.append("Numerical Summary:")
        lines.append(f"  PyTorch CE BPB:  {pytorch_bpb:.6f}")
        lines.append(f"  Rust AZTH BPB:   {rust_bpb:.6f}")
        rel = abs(rust_bpb - pytorch_bpb) / pytorch_bpb
        lines.append(f"  Relative diff:   {rel:.4%}")

    overall = "PASS" if fail_count == 0 else "FAIL"
    lines.append("")
    lines.append(f"OVERALL: {overall}")

    report = "\n".join(lines)

    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")
    print(f"Report saved: {REPORT_TXT}")
    print()
    print(report)


def generate_logits_csv(model: MambaByte):
    """
    Export per-position cross-entropy loss to CSV for detailed comparison.

    Since Redukt does not have a logit-export mode, we export PyTorch
    per-token CE loss which can be compared with future Rust instrumentation.
    """
    with open(ENWIK8_10K, "rb") as f:
        data = f.read()

    tokens = torch.tensor(list(data), dtype=torch.long).unsqueeze(0)
    tokens = tokens.to(DEVICE)

    model.eval()
    with torch.no_grad():
        # Process all at once (10KB fits easily)
        logits = model(tokens[:, :-1])  # (1, N-1, 256)
        targets = tokens[:, 1:]         # (1, N-1)

        # Per-token cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)  # (1, N-1, 256)
        per_token_ce = -log_probs[0, range(targets.shape[1]),
                                  targets[0]].cpu().numpy()

    # Write CSV: position, byte_value, cross_entropy_nats, bpb
    rows = []
    for i in range(len(per_token_ce)):
        ce = float(per_token_ce[i])
        bpb = ce / math.log(2)
        rows.append({
            "position": i + 1,
            "byte_value": int(data[i + 1]),
            "ce_nats": f"{ce:.6f}",
            "ce_bpb": f"{bpb:.6f}",
        })

    with open(LOGITS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["position", "byte_value", "ce_nats", "ce_bpb"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[CSV] Per-token logits saved: {LOGITS_CSV} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Section 9: Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  MambaByte Cross-Framework Validation")
    print("=" * 70)

    results: List[ValidationResult] = []
    pytorch_bpb = 0.0
    rust_bpb = 0.0

    # --- Phase 1: Find weights ---
    weights_path = find_weights()
    if weights_path is None:
        print("\nNo MambaByte weights found. Checking if training just completed...")
        # On Kaggle, training output lands in /kaggle/working/
        # Also check for .pt checkpoint files
        ckpts = sorted(
            Path(WORK_DIR).glob("checkpoints/mambabyte_step*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if ckpts:
            print(f"Found checkpoint: {ckpts[-1]}")
            print("Converting checkpoint to SafeTensors...")
            # Load checkpoint, export SafeTensors, then proceed
            ckpt = torch.load(str(ckpts[-1]), map_location="cpu",
                              weights_only=False)
            model = MambaByte()
            model.load_state_dict(ckpt["model_state_dict"])
            st_path = f"{WORK_DIR}/mambabyte-96m.safetensors"
            try:
                from safetensors.torch import save_file
                # Use the export convention from mambabyte-train.py
                state = {}
                state["backbone.embeddings.weight"] = (
                    model.embedding.weight.data.cpu()
                )
                for i, layer in enumerate(model.layers):
                    pfx = f"backbone.layers.{i}"
                    state[f"{pfx}.norm.weight"] = layer.norm.weight.data.cpu()
                    state[f"{pfx}.mixer.in_proj.weight"] = (
                        layer.in_proj.weight.data.cpu()
                    )
                    state[f"{pfx}.mixer.conv1d.weight"] = (
                        layer.conv1d.weight.data.cpu()
                    )
                    state[f"{pfx}.mixer.conv1d.bias"] = (
                        layer.conv1d.bias.data.cpu()
                    )
                    state[f"{pfx}.mixer.x_proj.weight"] = (
                        layer.x_proj.weight.data.cpu()
                    )
                    state[f"{pfx}.mixer.dt_proj.weight"] = (
                        layer.dt_proj.weight.data.cpu()
                    )
                    state[f"{pfx}.mixer.dt_proj.bias"] = (
                        layer.dt_proj.bias.data.cpu()
                    )
                    state[f"{pfx}.mixer.A_log"] = layer.A_log.data.cpu()
                    state[f"{pfx}.mixer.D"] = layer.D.data.cpu()
                    state[f"{pfx}.mixer.out_proj.weight"] = (
                        layer.out_proj.weight.data.cpu()
                    )
                state["backbone.norm_f.weight"] = model.norm_f.weight.data.cpu()
                state["lm_head.weight"] = model.embedding.weight.data.cpu()
                save_file(state, st_path)
                print(f"Exported SafeTensors: {st_path}")
                weights_path = st_path
            except Exception as e:
                print(f"SafeTensors export failed: {e}")
        else:
            print("\nERROR: No weights or checkpoints found.")
            print("Expected locations:")
            for c in WEIGHTS_CANDIDATES:
                print(f"  - {c}")
            print("\nRun mambabyte-train.py first, then re-run this script.")
            print("Skipping PyTorch tests; running Rust-only tests if possible.\n")

    # --- Phase 2: PyTorch BPB ---
    model = None
    if weights_path is not None:
        model = load_pytorch_model(weights_path)
        if model is not None:
            r = test_pytorch_bpb(model)
            results.append(r)
            # Extract BPB from detail
            pytorch_bpb = float(r.detail.split("=")[1].split(" ")[0])
            print(f"  Stored PyTorch BPB: {pytorch_bpb:.4f}")

            # Export per-token logits CSV
            generate_logits_csv(model)

    # --- Phase 3: Build Redukt ---
    build_ok = setup_environment()
    if not build_ok:
        results.append(ValidationResult(
            "Redukt build", False, "cargo build --release failed"
        ))
        generate_report(results, pytorch_bpb, rust_bpb)
        return

    # --- Phase 4: Prepare test data ---
    setup_test_data()

    # Copy weights for Rust loader
    if weights_path is not None:
        copy_weights_for_rust(weights_path)

    # --- Phase 5: Rust compression BPB ---
    r_rust, rust_bpb = test_rust_compression()
    results.append(r_rust)

    # --- Phase 6: BPB agreement ---
    if pytorch_bpb > 0 and rust_bpb > 0:
        r_agree = test_bpb_agreement(pytorch_bpb, rust_bpb)
        results.append(r_agree)
    else:
        results.append(ValidationResult(
            "BPB agreement", False,
            f"Skipped (PyTorch={pytorch_bpb:.4f}, Rust={rust_bpb:.4f})"
        ))

    # --- Phase 7: Roundtrip ---
    r_rt = test_roundtrip()
    results.append(r_rt)

    # --- Phase 8: Determinism ---
    r_det = test_determinism()
    results.append(r_det)

    # --- Phase 9: Multi-file BPB ---
    r_multi = test_multifile_bpb()
    results.append(r_multi)

    # --- Phase 10: Report ---
    generate_report(results, pytorch_bpb, rust_bpb)

    # Summary exit
    failed = sum(1 for r in results if not r.passed)
    if failed > 0:
        print(f"\n{failed} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} tests PASSED")


if __name__ == "__main__":
    main()
