"""Strata ladder trainer: export the integer weights of every tier and check them against the sequential reference.

    python tools/strata/train/export.py --ckpt RUN/ckpt.pt --out RUN/weights-int.npz --check sel-zookeeper

The file holds, per tier, the integer tensors `bakeoff/ref_int.py` reads (the design-stage layout, not the weight
file of specification 5) and the boot calibration Q0. With --check, the file is read back and the byte-by-byte int64
reference (`ref_int.StrataRef`, an independent code path with its own sequential tables) codes the first --bytes
bytes of a selection source: every P(bit = 1) must equal the trainer's, for every tier, and a real range-coder
round trip of the first --roundtrip bytes must return the input. Exit status 0 only with 0 mismatches.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
import common  # noqa: E402
import ref_int  # noqa: E402
from features import features, taps  # noqa: E402
from model import Ladder, export_int  # noqa: E402

LISTS = ("w", "b", "wg", "bg")


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Ladder(ck["rung"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    model.check_exact()
    return model, ck


def save_export(model, q0, path):
    arrays = {"q0": q0, "widths": np.array(model.widths)}
    for k in range(len(model.widths)):
        for name, value in export_int(model, k).items():
            if name in LISTS:
                arrays.update({f"t{k}.{name}.{i}": a for i, a in enumerate(value)})
            else:
                arrays[f"t{k}.{name}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_export(path):
    """Per-tier parameter dictionaries in the layout of ref_int.StrataRef, and Q0."""
    z = np.load(path)
    tiers = []
    for k in range(len(z["widths"])):
        pre = f"t{k}."
        p = {n[len(pre):]: z[n] for n in z.files if n.startswith(pre) and n[len(pre):].split(".")[0] not in LISTS}
        p.update({n: [z[f"{pre}{n}.{i}"] for i in range(int(p["layers"]))] for n in LISTS})
        for n in ("width", "layers", "geared", "bc"):
            p[n] = p[n].item()
        tiers.append(p)
    return tiers, z["q0"]


def model_probs(model, q0, data):
    """P(bit = 1) of every tier for one stream coded from its first byte: int64 [K, N, 8], and the bits per tier."""
    xb = data[None, :]
    s, top = features(xb, q0, "cpu")
    y = torch.from_numpy(xb.astype(np.int64))
    with torch.no_grad():
        costs, _, p1 = model(taps(y), s.float(), top.long(), y, model.init_state(1))
    return p1[:, 0].numpy(), [float(c.sum()) for c in costs]


def reference_check(path, model, source, nbytes, roundtrip):
    if common.SOURCES[source][0] != "select":
        raise SystemExit(f"{source} is not a selection source")
    tiers, q0 = load_export(path)
    data = common.load_source(source)[:nbytes]
    probs, bits = model_probs(model, q0, data)
    ok = True
    for k, params in enumerate(tiers):
        ref_p, ref_bits = ref_int.code_stream(ref_int.StrataRef(params, q0), data)
        diff = int((ref_p != probs[k]).sum())
        n = min(roundtrip, data.size)
        enc = ref_int.RangeEncoder()
        ref_int.code_stream(ref_int.StrataRef(params, q0), data[:n], enc)
        coded = enc.finish_tight()
        back = ref_int.decode_stream(ref_int.StrataRef(params, q0), coded, n)
        ok = ok and diff == 0 and back == bytes(data[:n])
        print(f"C tier {k} (width {params['width']}): {data.size * 8} probabilities on {source}, mismatches trainer "
              f"vs integer reference = {diff}; bits trainer {bits[k]:.3f} / reference {ref_bits:.3f} "
              f"({ref_bits / data.size:.4f} bits/byte); range-coder round trip of {n} bytes "
              f"{'OK' if back == bytes(data[:n]) else 'FAILED'} ({len(coded)} B)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default="", help="selection source of bakeoff/common.py for the reference check")
    ap.add_argument("--bytes", type=int, default=12_000)
    ap.add_argument("--roundtrip", type=int, default=3_000)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, ck = load_model(args.ckpt)
    save_export(model, ck["q0"], args.out)
    print(f"wrote {args.out}: tiers {model.widths}, {ck['step']} updates, {os.path.getsize(args.out):,d} B", flush=True)
    if args.check:
        ok = reference_check(args.out, model, args.check, args.bytes, args.roundtrip)
        print("ALL CHECKS PASSED" if ok else "CHECK FAILED", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
