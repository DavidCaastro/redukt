#!/usr/bin/env python3
"""
Convert MambaByte PyTorch checkpoint to SafeTensors for Redukt.

Usage:
    python convert-mambabyte-weights.py --input ckpt_step_76000.pt --output weights/mambabyte-96m/model.safetensors

Produces:
    - model.safetensors: weights in Redukt-compatible format
    - config.json: model configuration (same directory as output)
    - SHA-256 hash printed to stdout
"""
import argparse
import hashlib
import json
import os
import sys

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

try:
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: safetensors is required. Install with: pip install safetensors", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "96m": {
        "vocab_size": 256,
        "d_model": 768,
        "d_inner": 1536,
        "d_state": 16,
        "d_conv": 4,
        "dt_rank": 48,
        "n_layer": 24,
    },
    "353m": {
        "vocab_size": 256,
        "d_model": 1024,
        "d_inner": 2048,
        "d_state": 16,
        "d_conv": 4,
        "dt_rank": 64,
        "n_layer": 56,
    },
    "972m": {
        "vocab_size": 256,
        "d_model": 1792,
        "d_inner": 3584,
        "d_state": 16,
        "d_conv": 4,
        "dt_rank": 112,
        "n_layer": 48,
    },
}

DEFAULT_CONFIG = CONFIGS["96m"]


# ---------------------------------------------------------------------------
# Key mapping: training checkpoint -> Redukt SafeTensors
# ---------------------------------------------------------------------------

def build_key_map(n_layer: int, hf_format: bool = False) -> dict[str, str]:
    """Return {training_key: redukt_key} for all expected parameters.

    hf_format: if True, source keys use 'layers.{i}.mixer.xxx' (HuggingFace).
               if False, source keys use 'layers.{i}.xxx' (our training).
    """
    km = {
        "embedding.weight": "backbone.embeddings.weight",
        "norm_f.weight": "backbone.norm_f.weight",
        "norm_f.bias": "backbone.norm_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    # HuggingFace has mixer. prefix on per-layer SSM keys; our training doesn't
    mx = "mixer." if hf_format else ""
    for i in range(n_layer):
        layer_pairs = [
            ("norm.weight",          "norm.weight"),
            ("norm.bias",            "norm.bias"),
            (f"{mx}in_proj.weight",  "mixer.in_proj.weight"),
            (f"{mx}conv1d.weight",   "mixer.conv1d.weight"),
            (f"{mx}conv1d.bias",     "mixer.conv1d.bias"),
            (f"{mx}x_proj.weight",   "mixer.x_proj.weight"),
            (f"{mx}dt_proj.weight",  "mixer.dt_proj.weight"),
            (f"{mx}dt_proj.bias",    "mixer.dt_proj.bias"),
            (f"{mx}A_log",           "mixer.A_log"),
            (f"{mx}D",               "mixer.D"),
            (f"{mx}out_proj.weight", "mixer.out_proj.weight"),
        ]
        for src_suffix, dst_suffix in layer_pairs:
            src = f"layers.{i}.{src_suffix}"
            dst = f"backbone.layers.{i}.{dst_suffix}"
            km[src] = dst
    return km


# ---------------------------------------------------------------------------
# Expected shapes per key (relative to config)
# ---------------------------------------------------------------------------

def expected_shapes(cfg: dict) -> dict[str, tuple]:
    """Return {redukt_key: expected_shape} for validation."""
    v  = cfg["vocab_size"]
    d  = cfg["d_model"]
    di = cfg["d_inner"]
    ds = cfg["d_state"]
    dc = cfg["d_conv"]
    dr = cfg["dt_rank"]

    shapes = {
        "backbone.embeddings.weight": (v, d),
        "backbone.norm_f.weight": (d,),
        "backbone.norm_f.bias": (d,),
        "lm_head.weight": (v, d),
    }
    for i in range(cfg["n_layer"]):
        p = f"backbone.layers.{i}"
        shapes.update({
            f"{p}.norm.weight":            (d,),
            f"{p}.norm.bias":              (d,),
            f"{p}.mixer.in_proj.weight":   (2 * di, d),
            f"{p}.mixer.conv1d.weight":    (di, 1, dc),
            f"{p}.mixer.conv1d.bias":      (di,),
            f"{p}.mixer.x_proj.weight":    (dr + 2 * ds, di),
            f"{p}.mixer.dt_proj.weight":   (di, dr),
            f"{p}.mixer.dt_proj.bias":     (di,),
            f"{p}.mixer.A_log":            (di, ds),
            f"{p}.mixer.D":                (di,),
            f"{p}.mixer.out_proj.weight":  (d, di),
        })
    return shapes


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def strip_backbone_prefix(ckpt: dict) -> dict:
    """Strip 'backbone.' prefix from HuggingFace MambaByte checkpoints.

    HuggingFace format uses keys like:
        backbone.embedding.weight, backbone.layers.0.mixer.in_proj.weight
    Our training format uses:
        embedding.weight, layers.0.mixer.in_proj.weight

    Also normalizes 'backbone.embedding.weight' (singular, HF) to
    'embedding.weight' for compatibility with key_map.
    """
    has_prefix = any(k.startswith("backbone.") for k in ckpt if k != "lm_head.weight")
    if not has_prefix:
        return ckpt

    print("  Detected HuggingFace key prefix 'backbone.' — stripping...")
    new_ckpt = {}
    for k, v in ckpt.items():
        new_key = k
        if k.startswith("backbone."):
            new_key = k[len("backbone."):]
        # Normalize singular 'embedding' to match our key_map
        if new_key == "embedding.weight":
            new_key = "embedding.weight"  # already correct
        new_ckpt[new_key] = v
    return new_ckpt


def convert(input_path: str, output_path: str, cfg: dict) -> None:
    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # Handle common checkpoint wrappers
    if isinstance(ckpt, dict):
        for candidate in ("model_state_dict", "state_dict", "model"):
            if candidate in ckpt:
                print(f"  Unwrapping key '{candidate}' from checkpoint dict")
                ckpt = ckpt[candidate]
                break

    if not isinstance(ckpt, dict):
        print("ERROR: checkpoint is not a state_dict (dict). Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # Strip HuggingFace 'backbone.' prefix if present
    ckpt = strip_backbone_prefix(ckpt)

    # R-088: Keep LayerNorm bias keys (Redukt now supports both LayerNorm and RMSNorm)
    has_norm_bias = any(k.endswith(".bias") and (".norm." in k or k == "norm_f.bias") for k in ckpt)
    if has_norm_bias:
        print("  LayerNorm bias detected — will be preserved in SafeTensors")

    n_layer = cfg["n_layer"]

    # Auto-detect HuggingFace format (keys have 'mixer.' in layer paths)
    hf_format = any("mixer." in k for k in ckpt if k.startswith("layers."))
    if hf_format:
        print("  Detected HuggingFace key format (layers.{i}.mixer.xxx)")

    key_map = build_key_map(n_layer, hf_format=hf_format)
    shapes  = expected_shapes(cfg)

    # --- Detect available keys and resolve lm_head tying ---
    has_lm_head = "lm_head.weight" in ckpt
    if not has_lm_head:
        print("  lm_head.weight not found -- will tie to embedding.weight")

    # --- Map and validate ---
    out_tensors: dict[str, torch.Tensor] = {}
    missing_src = []
    shape_errors = []

    for src_key, dst_key in key_map.items():
        # Handle tied lm_head
        if src_key == "lm_head.weight" and not has_lm_head:
            continue  # handled after loop

        if src_key not in ckpt:
            missing_src.append(src_key)
            continue

        tensor = ckpt[src_key].float()  # ensure float32
        actual_shape = tuple(tensor.shape)
        expect_shape = shapes.get(dst_key)

        if expect_shape is not None and actual_shape != expect_shape:
            shape_errors.append(
                f"  {src_key}: expected {expect_shape}, got {actual_shape}"
            )
        else:
            out_tensors[dst_key] = tensor

    # Tie lm_head if absent
    if not has_lm_head and "backbone.embeddings.weight" in out_tensors:
        out_tensors["lm_head.weight"] = out_tensors["backbone.embeddings.weight"].clone()
        print("  Tied lm_head.weight = embedding.weight")
    elif has_lm_head and "lm_head.weight" not in out_tensors:
        # Was in key_map but missing from ckpt (already caught above)
        pass

    # --- Count skipped keys (extras in checkpoint not in our map) ---
    mapped_src = set(key_map.keys())
    extra_keys = [k for k in ckpt if k not in mapped_src]
    if extra_keys:
        print(f"  Skipping {len(extra_keys)} extra key(s) not in Redukt format:")
        for k in extra_keys[:10]:
            print(f"    - {k}  {tuple(ckpt[k].shape)}")
        if len(extra_keys) > 10:
            print(f"    ... and {len(extra_keys) - 10} more")

    # --- Report errors (bias keys are optional) ---
    optional_suffixes = (".norm.bias", "norm_f.bias")
    required_missing = [k for k in missing_src if not any(k.endswith(s) for s in optional_suffixes)]
    optional_missing = [k for k in missing_src if any(k.endswith(s) for s in optional_suffixes)]
    if optional_missing:
        print(f"  {len(optional_missing)} optional bias key(s) absent (RMSNorm mode)")
    if required_missing:
        print(f"\nERROR: {len(required_missing)} required key(s) missing from checkpoint:",
              file=sys.stderr)
        for k in required_missing:
            print(f"  - {k}", file=sys.stderr)
        print(f"\nCheckpoint contains {len(ckpt)} keys. First 30:", file=sys.stderr)
        for i, k in enumerate(sorted(ckpt.keys())):
            if i >= 30:
                print("  ...", file=sys.stderr)
                break
            print(f"  {k}  {tuple(ckpt[k].shape)}", file=sys.stderr)
        sys.exit(1)

    if shape_errors:
        print(f"\nERROR: {len(shape_errors)} shape mismatch(es):", file=sys.stderr)
        for e in shape_errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # --- Parameter count ---
    total_params = sum(t.numel() for t in out_tensors.values())
    # Subtract tied weights from count (lm_head duplicates embedding)
    emb_params = out_tensors["backbone.embeddings.weight"].numel()
    unique_params = total_params - emb_params  # lm_head is a copy
    print(f"\n  Total tensors:     {len(out_tensors)}")
    print(f"  Total parameters:  {total_params:,} ({total_params * 4 / 1024**2:.1f} MB float32)")
    print(f"  Unique parameters: {unique_params:,} (excluding tied lm_head)")

    # --- Save SafeTensors ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(out_tensors, output_path)
    print(f"\n  Saved: {output_path}")

    # --- SHA-256 ---
    sha = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    print(f"  SHA-256: {digest}")

    # --- Save config.json ---
    config_path = os.path.join(os.path.dirname(output_path) or ".", "config.json")
    config_out = {
        "model_type": "mambabyte",
        "vocab_size": cfg["vocab_size"],
        "d_model": cfg["d_model"],
        "d_inner": cfg["d_inner"],
        "d_state": cfg["d_state"],
        "d_conv": cfg["d_conv"],
        "dt_rank": cfg["dt_rank"],
        "n_layer": cfg["n_layer"],
        "total_params": total_params,
        "unique_params": unique_params,
        "sha256": digest,
    }
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
        f.write("\n")
    print(f"  Config:  {config_path}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert MambaByte PyTorch checkpoint to SafeTensors for Redukt."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to PyTorch checkpoint (.pt, .pth, or pytorch_model.bin)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for model.safetensors"
    )
    parser.add_argument(
        "--preset", default=None, choices=list(CONFIGS.keys()),
        help="Model preset: 96m (default), 353m (HuggingFace MambaByte_Arxiv), 972m"
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional JSON config file overriding default config"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Select base config from preset or default
    base = args.preset or "96m"
    cfg = dict(CONFIGS[base])

    if args.config:
        if not os.path.isfile(args.config):
            print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        with open(args.config) as f:
            overrides = json.load(f)
        for k in ("vocab_size", "d_model", "d_inner", "d_state", "d_conv", "dt_rank", "n_layer"):
            if k in overrides:
                cfg[k] = int(overrides[k])
        print(f"Config overrides from {args.config}: { {k: overrides[k] for k in overrides if k in cfg} }")

    print(f"MambaByte config (preset={base}):")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    convert(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
