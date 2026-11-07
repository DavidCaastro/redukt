"""
MambaByte Training -- Kaggle Notebook
=====================================
Train byte-level Mamba SSM (96M params) from scratch for neural compression.
Supports Kaggle TPU v5e-8 and GPU T4.

Architecture: MambaByte-96M
  - vocab_size=256 (byte-level, no BPE)
  - d_model=768, d_inner=1536, d_state=16, d_conv=4, dt_rank=48
  - n_layer=24, tied embeddings
  - Based on Mamba (Gu & Dao 2023) + MambaByte (Wang et al. 2024)

Usage:
  - On Kaggle: enable TPU v5e-8 or GPU T4 accelerator
  - Outputs: checkpoints to /kaggle/working/checkpoints/
  - Resume: automatically resumes from latest checkpoint
  - Final export: SafeTensors format compatible with Redukt Rust loader

License: Apache-2.0
"""

import os
import sys
import math
import time
import glob
import json
import struct
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

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

SEQ_LEN = 8192       # bytes per sequence
BATCH_SIZE = 48       # total across all devices
LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 76000   # ~30B bytes / (48 * 8192)
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1
SAVE_EVERY = 5000
EVAL_EVERY = 1000
LOG_EVERY = 100

WORK_DIR = "/kaggle/working"
CKPT_DIR = f"{WORK_DIR}/checkpoints"
DATA_DIR = f"{WORK_DIR}/data"

# Derived
DTYPE = torch.float32  # TPU v5e supports bf16 but float32 is safer for correctness

# ---------------------------------------------------------------------------
# Section 2: Device Setup
# ---------------------------------------------------------------------------

USE_TPU = False
USE_GPU = False
DEVICE = None
WORLD_SIZE = 1

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    USE_TPU = True
    DEVICE = xm.xla_device()
    WORLD_SIZE = xm.xrt_world_size()
    print(f"[DEVICE] TPU detected: {DEVICE}, world_size={WORLD_SIZE}")
except ImportError:
    if torch.cuda.is_available():
        USE_GPU = True
        DEVICE = torch.device("cuda:0")
        print(f"[DEVICE] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        DEVICE = torch.device("cpu")
        print("[DEVICE] CPU only -- training will be very slow")

PER_DEVICE_BATCH = BATCH_SIZE // WORLD_SIZE

# ---------------------------------------------------------------------------
# Section 3: Model Definition (Pure PyTorch -- no mamba_ssm dependency)
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
    """
    Single Mamba block with parallel (convolutional) SSM computation.

    Structure:
        input -> RMSNorm -> in_proj (d_model -> 2*d_inner) -> split [x, z]
        x -> conv1d(d_conv=4, causal) -> SiLU -> SSM(d_state=16) -> y
        output = y * SiLU(z) -> out_proj (d_inner -> d_model) -> residual add

    The SSM scan is implemented as a sequential loop over the sequence length.
    TODO: Replace with fused CUDA kernel (selective_scan_fn) for 3-5x speedup.
    For TPU, the sequential scan is acceptable since XLA can optimize it.
    """

    def __init__(self, d_model: int, d_inner: int, d_state: int,
                 d_conv: int, dt_rank: int):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        # Pre-norm
        self.norm = RMSNorm(d_model)

        # Input projection: d_model -> 2 * d_inner (x and z branches)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv1d on x branch
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # left-pad for causal
            groups=d_inner,
            bias=True,
        )

        # SSM projections
        # x -> (dt_rank + 2*d_state) for dt_input, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # dt_input -> d_inner (dt projection with bias)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # SSM parameters
        # A is stored as log for numerical stability: A = -exp(A_log)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        # D is a skip connection parameter
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with residual already added
        """
        residual = x
        x = self.norm(x)

        # Input projection -> split into x and z branches
        xz = self.in_proj(x)                        # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)           # each (B, L, d_inner)

        # Causal conv1d on x branch
        # conv1d expects (B, C, L), so transpose
        x_branch = x_branch.transpose(1, 2)         # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)
        # Remove right padding to keep causal (conv adds d_conv-1 extra on right)
        x_branch = x_branch[:, :, :residual.shape[1]]
        x_branch = x_branch.transpose(1, 2)         # (B, L, d_inner)

        x_branch = F.silu(x_branch)

        # SSM computation
        y = self._ssm_scan(x_branch)                # (B, L, d_inner)

        # Gated output
        y = y * F.silu(z)                            # (B, L, d_inner)

        # Output projection + residual
        out = self.out_proj(y) + residual            # (B, L, d_model)
        return out

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM scan (sequential implementation).

        This is the correct reference implementation using a loop over the
        sequence dimension. It computes:
            dt = softplus(dt_proj(dt_input) + dt_bias)
            A = -exp(A_log)
            dA = exp(dt * A)
            dB = dt * B
            h[t] = dA * h[t-1] + dB * x[t]
            y[t] = C[t] . h[t] + D * x[t]

        TODO: Replace this sequential loop with a fused CUDA kernel
        (e.g. selective_scan_fn from mamba_ssm) for 3-5x training speedup.
        The sequential version is correct but memory-intensive due to the
        explicit loop over seq_len. For TPU, XLA may partially optimize this.

        Args:
            x: (batch, seq_len, d_inner) -- post-conv, post-SiLU
        Returns:
            y: (batch, seq_len, d_inner)
        """
        B_batch, L, D = x.shape

        # Project x to get dt_input, B_ssm, C_ssm
        x_dbl = self.x_proj(x)                      # (B, L, dt_rank + 2*d_state)
        dt_input = x_dbl[:, :, :self.dt_rank]        # (B, L, dt_rank)
        B_ssm = x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state]  # (B, L, d_state)
        C_ssm = x_dbl[:, :, self.dt_rank + self.d_state:]             # (B, L, d_state)

        # dt projection: dt_rank -> d_inner, then softplus
        dt = F.softplus(self.dt_proj(dt_input))      # (B, L, d_inner)

        # Discrete SSM parameters
        A = -torch.exp(self.A_log)                   # (d_inner, d_state)

        # Sequential scan over time
        h = torch.zeros(B_batch, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            dt_t = dt[:, t, :]                       # (B, d_inner)
            B_t = B_ssm[:, t, :]                     # (B, d_state)
            C_t = C_ssm[:, t, :]                     # (B, d_state)
            x_t = x[:, t, :]                         # (B, d_inner)

            # dA = exp(dt * A): (B, d_inner, d_state)
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))

            # dB = dt * B: (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            # State update: h = dA * h + dB * x
            h = dA * h + dB * x_t.unsqueeze(-1)

            # Output: y = C . h + D * x
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                   # (B, L, d_inner)
        return y


class MambaByte(nn.Module):
    """
    MambaByte: byte-level Mamba language model (96M parameters).

    Architecture:
        byte_id -> embedding(256, 768) -> 24x MambaBlock -> RMSNorm -> lm_head(768, 256)

    Embedding and lm_head weights are tied (shared).
    """

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

        # Tied lm_head: share weight with embedding
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following Mamba conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Special init for dt_proj bias: inverse softplus of uniform [0.001, 0.1]
        for layer in self.layers:
            dt_init = torch.exp(
                torch.rand(D_INNER) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            )
            inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
            with torch.no_grad():
                layer.dt_proj.bias.copy_(inv_softplus)

        # Scale down output projection for deep residual
        for layer in self.layers:
            with torch.no_grad():
                layer.out_proj.weight.mul_(1.0 / math.sqrt(2.0 * self.n_layer))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) long tensor of byte values [0, 255]
        Returns:
            logits: (batch, seq_len, vocab_size) float tensor
        """
        x = self.embedding(input_ids)                # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)                     # (B, L, vocab_size)
        return logits

    def num_params(self, non_embedding: bool = False) -> int:
        """Count total parameters (optionally excluding embedding)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract embedding (tied with lm_head, only counted once)
            n -= self.embedding.weight.numel()
        return n


# ---------------------------------------------------------------------------
# Section 4: Dataset
# ---------------------------------------------------------------------------


class ByteDataset(IterableDataset):
    """
    Streaming byte-level dataset for MambaByte training.

    Yields (input_ids, labels) pairs where:
      - input_ids: bytes[0..SEQ_LEN-1]
      - labels: bytes[1..SEQ_LEN]
    Both are long tensors of shape (SEQ_LEN,) with values in [0, 255].

    Data sources (tried in order):
      1. PG19 from HuggingFace (books, high quality text)
      2. Wikipedia from HuggingFace
      3. Synthetic random bytes (fallback for testing)
    """

    def __init__(self, seq_len: int = SEQ_LEN, split: str = "train",
                 worker_id: int = 0, num_workers: int = 1):
        super().__init__()
        self.seq_len = seq_len + 1  # +1 for the label shift
        self.split = split
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._source = None
        self._buffer = bytearray()

    def _init_source(self):
        """Initialize data source (lazy, called on first iteration)."""
        # Try PG19
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "deepmind/pg19",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            self._source = ("pg19", iter(ds))
            print(f"[DATA] Using PG19 (streaming)")
            return
        except Exception as e:
            print(f"[DATA] PG19 not available: {e}")

        # Try Wikipedia
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "wikipedia",
                "20220301.en",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            self._source = ("wiki", iter(ds))
            print(f"[DATA] Using Wikipedia (streaming)")
            return
        except Exception as e:
            print(f"[DATA] Wikipedia not available: {e}")

        # Fallback: synthetic
        print("[DATA] WARNING: Using synthetic random bytes (for testing only)")
        self._source = ("synthetic", None)

    def _fill_buffer(self):
        """Fill internal buffer with bytes from the source."""
        if self._source is None:
            self._init_source()

        source_name, source_iter = self._source

        if source_name == "synthetic":
            # Generate random bytes
            self._buffer.extend(os.urandom(self.seq_len * 64))
            return

        # Stream text from HuggingFace dataset
        attempts = 0
        while len(self._buffer) < self.seq_len and attempts < 1000:
            try:
                sample = next(source_iter)
                text = sample.get("text", sample.get("content", ""))
                encoded = text.encode("utf-8", errors="replace")
                self._buffer.extend(encoded)
                attempts += 1
            except StopIteration:
                # Re-initialize the iterator (loop dataset)
                print("[DATA] Dataset exhausted, restarting...")
                self._init_source()
                _, source_iter = self._source
                if source_iter is None:
                    break
                attempts += 1

    def __iter__(self):
        while True:
            # Ensure we have enough bytes
            while len(self._buffer) < self.seq_len:
                self._fill_buffer()

            # Extract one sequence
            seq_bytes = self._buffer[:self.seq_len]
            self._buffer = self._buffer[self.seq_len:]

            # Convert to tensor
            tokens = torch.tensor(list(seq_bytes), dtype=torch.long)
            input_ids = tokens[:-1]   # (seq_len,)
            labels = tokens[1:]       # (seq_len,)

            yield input_ids, labels


def create_dataloader(batch_size: int, seq_len: int) -> DataLoader:
    """Create a DataLoader for byte-level training data."""
    dataset = ByteDataset(seq_len=seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,     # streaming dataset, keep in main process
        pin_memory=USE_GPU,
        drop_last=True,
    )
    return loader


# ---------------------------------------------------------------------------
# Section 5: Learning Rate Schedule
# ---------------------------------------------------------------------------


def get_lr(step: int) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    - Linear warmup from 0 to LR over WARMUP_STEPS
    - Cosine decay from LR to MIN_LR over remaining steps
    """
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS

    if step >= TOTAL_STEPS:
        return MIN_LR

    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (LR - MIN_LR) * cosine


# ---------------------------------------------------------------------------
# Section 6: Evaluation
# ---------------------------------------------------------------------------

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_10K_PATH = os.path.join(DATA_DIR, "enwik8_10k.bin")


def download_enwik8_10k():
    """Download enwik8 and extract first 10KB for evaluation."""
    if os.path.exists(ENWIK8_10K_PATH):
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print("[EVAL] Downloading enwik8 for evaluation...")

    try:
        import urllib.request
        import zipfile
        import io

        zip_path = os.path.join(DATA_DIR, "enwik8.zip")
        urllib.request.urlretrieve(ENWIK8_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("enwik8") as f:
                data = f.read(10240)  # 10KB

        with open(ENWIK8_10K_PATH, "wb") as f:
            f.write(data)

        os.remove(zip_path)
        print(f"[EVAL] enwik8 10KB saved ({len(data)} bytes)")

    except Exception as e:
        print(f"[EVAL] Could not download enwik8: {e}")
        # Fallback: use first 10KB of training data
        fallback = os.urandom(10240)
        with open(ENWIK8_10K_PATH, "wb") as f:
            f.write(fallback)
        print("[EVAL] Using random fallback for evaluation")


@torch.no_grad()
def eval_bpb(model: nn.Module, step: int) -> float:
    """
    Evaluate model on enwik8 first 10KB.

    Computes cross-entropy loss and converts to BPB (bits per byte).
    BPB = cross_entropy_loss / ln(2)

    Reference targets:
      - Random: 8.0 BPB
      - gzip-9: ~3.0 BPB on enwik8
      - Redukt Mamba+CM: 0.961 BPB (actual AZTH compression)
      - Target for this model: 1.20-1.40 BPB
    """
    download_enwik8_10k()

    model.eval()

    with open(ENWIK8_10K_PATH, "rb") as f:
        data = f.read()

    tokens = torch.tensor(list(data), dtype=torch.long).unsqueeze(0)  # (1, N)
    tokens = tokens.to(DEVICE)

    # Process in chunks of SEQ_LEN to avoid OOM
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, tokens.shape[1] - 1, SEQ_LEN):
        end = min(start + SEQ_LEN, tokens.shape[1] - 1)
        input_ids = tokens[:, start:end]
        labels = tokens[:, start + 1:end + 1]

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            labels.reshape(-1),
            reduction="sum",
        )

        if USE_TPU:
            loss_val = loss.item()
        else:
            loss_val = loss.item()

        total_loss += loss_val
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    bpb = avg_loss / math.log(2)

    print(f"[EVAL] step={step} | enwik8-10K | loss={avg_loss:.4f} | BPB={bpb:.4f}")

    model.train()
    return bpb


# ---------------------------------------------------------------------------
# Section 7: Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, loss: float = 0.0):
    """Save training checkpoint."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"mambabyte_step{step:06d}.pt")

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": {
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "d_inner": D_INNER,
            "d_state": D_STATE,
            "d_conv": D_CONV,
            "dt_rank": DT_RANK,
            "n_layer": N_LAYER,
        },
    }

    if USE_TPU:
        xm.save(state, ckpt_path)
    else:
        torch.save(state, ckpt_path)

    print(f"[CKPT] Saved checkpoint: {ckpt_path}")

    # Keep only last 3 checkpoints to save disk space
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "mambabyte_step*.pt")))
    for old_ckpt in ckpts[:-3]:
        os.remove(old_ckpt)
        print(f"[CKPT] Removed old checkpoint: {old_ckpt}")


def maybe_load_checkpoint(model: nn.Module,
                          optimizer: torch.optim.Optimizer) -> int:
    """Load latest checkpoint if available. Returns the step to resume from."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "mambabyte_step*.pt")))

    if not ckpts:
        print("[CKPT] No checkpoint found, starting from scratch")
        return 0

    latest = ckpts[-1]
    print(f"[CKPT] Loading checkpoint: {latest}")

    if USE_TPU:
        state = torch.load(latest, map_location="cpu")
    else:
        state = torch.load(latest, map_location=DEVICE, weights_only=False)

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    step = state["step"]
    loss = state.get("loss", 0.0)

    print(f"[CKPT] Resumed from step {step} (loss={loss:.4f})")
    return step


# ---------------------------------------------------------------------------
# Section 8: SafeTensors Export
# ---------------------------------------------------------------------------


def export_safetensors(model: nn.Module, path: str):
    """
    Export model weights in SafeTensors format compatible with Redukt Rust loader.

    The naming convention follows HuggingFace Mamba format so that the existing
    Redukt infrastructure/mamba/model.rs weight loader can read the file directly.

    Tensor map:
        backbone.embeddings.weight          -> embedding (256, 768)
        backbone.layers.{i}.norm.weight     -> RMSNorm weight
        backbone.layers.{i}.mixer.in_proj.weight    -> in_proj (2*d_inner, d_model)
        backbone.layers.{i}.mixer.conv1d.weight     -> conv1d (d_inner, 1, d_conv)
        backbone.layers.{i}.mixer.conv1d.bias       -> conv1d bias (d_inner,)
        backbone.layers.{i}.mixer.x_proj.weight     -> x_proj (dt_rank+2*d_state, d_inner)
        backbone.layers.{i}.mixer.dt_proj.weight    -> dt_proj (d_inner, dt_rank)
        backbone.layers.{i}.mixer.dt_proj.bias      -> dt_proj bias (d_inner,)
        backbone.layers.{i}.mixer.A_log             -> SSM A parameter (d_inner, d_state)
        backbone.layers.{i}.mixer.D                 -> SSM D skip (d_inner,)
        backbone.layers.{i}.mixer.out_proj.weight   -> out_proj (d_model, d_inner)
        backbone.norm_f.weight                      -> final RMSNorm
        lm_head.weight                              -> lm_head (tied to embedding)
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("[EXPORT] safetensors not installed. Install with: pip install safetensors")
        print("[EXPORT] Falling back to raw PyTorch export.")
        torch.save(model.state_dict(), path.replace(".safetensors", ".pt"))
        return

    state: Dict[str, torch.Tensor] = {}

    # Embedding
    state["backbone.embeddings.weight"] = model.embedding.weight.data.cpu()

    # Layers
    for i, layer in enumerate(model.layers):
        pfx = f"backbone.layers.{i}"

        # RMSNorm
        state[f"{pfx}.norm.weight"] = layer.norm.weight.data.cpu()

        # Mixer
        state[f"{pfx}.mixer.in_proj.weight"] = layer.in_proj.weight.data.cpu()

        state[f"{pfx}.mixer.conv1d.weight"] = layer.conv1d.weight.data.cpu()
        state[f"{pfx}.mixer.conv1d.bias"] = layer.conv1d.bias.data.cpu()

        state[f"{pfx}.mixer.x_proj.weight"] = layer.x_proj.weight.data.cpu()

        state[f"{pfx}.mixer.dt_proj.weight"] = layer.dt_proj.weight.data.cpu()
        state[f"{pfx}.mixer.dt_proj.bias"] = layer.dt_proj.bias.data.cpu()

        state[f"{pfx}.mixer.A_log"] = layer.A_log.data.cpu()
        state[f"{pfx}.mixer.D"] = layer.D.data.cpu()

        state[f"{pfx}.mixer.out_proj.weight"] = layer.out_proj.weight.data.cpu()

    # Final norm
    state["backbone.norm_f.weight"] = model.norm_f.weight.data.cpu()

    # LM head (tied to embedding, but saved separately for Rust loader)
    state["lm_head.weight"] = model.embedding.weight.data.cpu()

    save_file(state, path)
    print(f"[EXPORT] SafeTensors saved: {path}")

    # Print tensor inventory
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(f"[EXPORT] {len(state)} tensors, {total_bytes / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Section 9: Training Loop
# ---------------------------------------------------------------------------


def train():
    """Main training loop for MambaByte-96M."""
    print("=" * 70)
    print("MambaByte-96M Training")
    print("=" * 70)

    # Build model
    model = MambaByte().to(DEVICE)
    n_params = model.num_params()
    n_params_ne = model.num_params(non_embedding=True)
    print(f"[MODEL] Total parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"[MODEL] Non-embedding parameters: {n_params_ne:,} ({n_params_ne / 1e6:.1f}M)")
    print(f"[MODEL] Config: d_model={D_MODEL}, d_inner={D_INNER}, n_layer={N_LAYER}")
    print(f"[MODEL] SSM: d_state={D_STATE}, d_conv={D_CONV}, dt_rank={DT_RANK}")
    print(f"[MODEL] Vocab: {VOCAB_SIZE} (byte-level), tied embeddings")

    # Optimizer: AdamW with Mamba-recommended betas
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    # Resume from checkpoint
    start_step = maybe_load_checkpoint(model, optimizer)

    # DataLoader
    print(f"[TRAIN] batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, total_steps={TOTAL_STEPS}")
    print(f"[TRAIN] lr={LR}, min_lr={MIN_LR}, warmup={WARMUP_STEPS}")
    print(f"[TRAIN] grad_clip={GRAD_CLIP}, weight_decay={WEIGHT_DECAY}")
    bytes_per_step = BATCH_SIZE * SEQ_LEN
    total_bytes = TOTAL_STEPS * bytes_per_step
    print(f"[TRAIN] Bytes per step: {bytes_per_step:,} ({bytes_per_step / 1e6:.1f}M)")
    print(f"[TRAIN] Total bytes: {total_bytes:,} ({total_bytes / 1e9:.1f}B)")
    print("-" * 70)

    loader = create_dataloader(PER_DEVICE_BATCH, SEQ_LEN)
    data_iter = iter(loader)

    model.train()
    running_loss = 0.0
    step_time_start = time.time()

    for step in range(start_step, TOTAL_STEPS):
        # Learning rate schedule
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Get batch
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_ids, labels = next(data_iter)

        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
        )

        # Backward
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Optimizer step
        if USE_TPU:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Accumulate loss for logging
        loss_val = loss.item()
        running_loss += loss_val

        # Logging
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            bpb = avg_loss / math.log(2)
            elapsed = time.time() - step_time_start
            steps_per_sec = LOG_EVERY / elapsed
            bytes_per_sec = steps_per_sec * bytes_per_step
            eta_hours = (TOTAL_STEPS - step) / steps_per_sec / 3600

            if isinstance(grad_norm, torch.Tensor):
                gn = grad_norm.item()
            else:
                gn = float(grad_norm)

            print(
                f"step {step + 1:>6d}/{TOTAL_STEPS} | "
                f"loss {avg_loss:.4f} | BPB {bpb:.4f} | "
                f"lr {lr:.2e} | grad_norm {gn:.2f} | "
                f"{steps_per_sec:.2f} steps/s | "
                f"{bytes_per_sec / 1e6:.1f} MB/s | "
                f"ETA {eta_hours:.1f}h"
            )

            running_loss = 0.0
            step_time_start = time.time()

        # Evaluation
        if (step + 1) % EVAL_EVERY == 0:
            eval_bpb(model, step + 1)

        # Checkpoint
        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step + 1, loss_val)

    # Final checkpoint + export
    print("=" * 70)
    print("[TRAIN] Training complete!")
    save_checkpoint(model, optimizer, TOTAL_STEPS, loss_val)

    # Final evaluation
    final_bpb = eval_bpb(model, TOTAL_STEPS)
    print(f"[TRAIN] Final BPB on enwik8-10K: {final_bpb:.4f}")

    # Export SafeTensors for Redukt integration
    st_path = os.path.join(WORK_DIR, "mambabyte-96m.safetensors")
    export_safetensors(model, st_path)

    print("=" * 70)
    print(f"[DONE] Model saved to: {st_path}")
    print(f"[DONE] Copy to Redukt weights/ directory for integration")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Section 10: Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if USE_TPU:
        # TPU: launch via xmp.spawn for multi-core
        # For single-core testing, call train() directly
        try:
            world_size = xm.xrt_world_size()
            if world_size > 1:
                print(f"[TPU] Launching on {world_size} cores")
                # On Kaggle TPU v5e-8, xmp.spawn handles distribution
                # For simplicity, run on single core (Kaggle handles SPMD)
                train()
            else:
                train()
        except Exception:
            train()
    else:
        train()
