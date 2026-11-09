#!/usr/bin/env python3
"""
Convert Mamba SafeTensors weights to .rbin binary format for Redukt GPU plugin.

Usage:
    python convert-weights-gpu.py --input weights/mamba-130m/model.safetensors --output weights/mamba-130m/model.rbin
    python convert-weights-gpu.py --input weights/mambabyte-353m/model.safetensors --output weights/mambabyte-353m/model.rbin

The .rbin format is a flat binary with a 44-byte header followed by all weight
tensors in sequential f32 layout. Designed for trivial GPU loading (no JSON parsing).

Copyright 2026 Redukt Contributors. Apache-2.0 License.
"""
import argparse
import hashlib
import os
import struct
import sys

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors required. Install with: pip install safetensors", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)


# --- Binary format constants ---

RBIN_MAGIC   = 0x55504752  # "RGPU" little-endian
RBIN_VERSION = 1
FLAG_NORM_BIAS   = 1 << 0
FLAG_UNTIED_HEAD = 1 << 1

HEADER_FMT = "<IIIIIIIIIII"  # 11 x uint32 = 44 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def detect_config(st):
    """Auto-detect model config from embedding tensor shape."""
    emb = st.get_tensor("backbone.embeddings.weight")
    vocab_size, d_model = emb.shape

    n_layer = 0
    while f"backbone.layers.{n_layer}.norm.weight" in st.keys():
        n_layer += 1

    d_inner = 2 * d_model
    d_state = 16
    d_conv = 4
    dt_rank = (d_model + 15) // 16

    # Verify d_inner from in_proj shape
    in_proj = st.get_tensor("backbone.layers.0.mixer.in_proj.weight")
    actual_2di = in_proj.shape[0]
    d_inner = actual_2di // 2

    # Verify dt_rank from dt_proj shape
    dt_proj = st.get_tensor("backbone.layers.0.mixer.dt_proj.weight")
    dt_rank = dt_proj.shape[1]

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "d_inner": d_inner,
        "d_state": d_state,
        "d_conv": d_conv,
        "dt_rank": dt_rank,
        "n_layer": n_layer,
    }


def convert(input_path, output_path):
    print(f"Loading: {input_path}")
    st = safe_open(input_path, framework="numpy")

    cfg = detect_config(st)
    vs = cfg["vocab_size"]
    d  = cfg["d_model"]
    di = cfg["d_inner"]
    ds = cfg["d_state"]
    dc = cfg["d_conv"]
    dr = cfg["dt_rank"]
    nl = cfg["n_layer"]

    print(f"Config: vocab={vs} d_model={d} d_inner={di} d_state={ds} "
          f"d_conv={dc} dt_rank={dr} n_layer={nl}")

    # Detect features
    has_norm_bias = "backbone.norm_f.bias" in st.keys()
    has_untied_head = "lm_head.weight" in st.keys()
    flags = 0
    if has_norm_bias:
        flags |= FLAG_NORM_BIAS
        print("  LayerNorm detected (norm bias present)")
    if has_untied_head:
        flags |= FLAG_UNTIED_HEAD
        print("  Untied lm_head detected")

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_floats = 0

    with open(output_path, "wb") as f:
        # Header
        header = struct.pack(HEADER_FMT,
                             RBIN_MAGIC, RBIN_VERSION,
                             vs, d, di, ds, dc, dr, nl, flags, 0)
        f.write(header)

        def write_tensor(name, expected_numel=None):
            nonlocal total_floats
            t = np.asarray(st.get_tensor(name), dtype=np.float32).flatten()
            if expected_numel is not None and t.size != expected_numel:
                print(f"  WARNING: {name} has {t.size} elements, "
                      f"expected {expected_numel}", file=sys.stderr)
            f.write(t.tobytes())
            total_floats += t.size

        # Embeddings
        write_tensor("backbone.embeddings.weight", vs * d)

        # Per-layer weights
        for i in range(nl):
            p = f"backbone.layers.{i}"
            write_tensor(f"{p}.norm.weight", d)
            if has_norm_bias:
                write_tensor(f"{p}.norm.bias", d)
            write_tensor(f"{p}.mixer.in_proj.weight", 2 * di * d)
            write_tensor(f"{p}.mixer.conv1d.weight", di * dc)
            write_tensor(f"{p}.mixer.conv1d.bias", di)
            write_tensor(f"{p}.mixer.x_proj.weight", (dr + 2 * ds) * di)
            write_tensor(f"{p}.mixer.dt_proj.weight", di * dr)
            write_tensor(f"{p}.mixer.dt_proj.bias", di)
            write_tensor(f"{p}.mixer.A_log", di * ds)
            write_tensor(f"{p}.mixer.D", di)
            write_tensor(f"{p}.mixer.out_proj.weight", d * di)

            if (i + 1) % 8 == 0 or i == nl - 1:
                print(f"  Written layer {i+1}/{nl}")

        # Final norm
        write_tensor("backbone.norm_f.weight", d)
        if has_norm_bias:
            write_tensor("backbone.norm_f.bias", d)

        # LM head
        if has_untied_head:
            write_tensor("lm_head.weight", vs * d)

    file_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"  Floats:  {total_floats:,} ({total_floats * 4 / 1024**2:.1f} MB)")
    print(f"  File:    {file_size / 1024**2:.1f} MB (header + data)")

    # SHA-256
    sha = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    print(f"  SHA-256: {sha.hexdigest()}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mamba SafeTensors to .rbin for Redukt GPU plugin."
    )
    parser.add_argument("--input", required=True,
                        help="Path to model.safetensors")
    parser.add_argument("--output", required=True,
                        help="Output .rbin path")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
