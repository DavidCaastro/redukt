"""Strata bake-off: length curves on 2 MiB single streams (axiom A2: no reset exists, nothing may collapse).

    python -I tools/strata/bakeoff/curve.py --prepare curve-dickens        # online features of one stream (~2 min)
    python -I tools/strata/bakeoff/curve.py --prepare curve-enwik8
    python -I tools/strata/bakeoff/curve.py --run strata_f [--minutes 2.5]  # repeat until DONE

Both streams are coded in one pass each, from the boot state, state carried to the end. Output:
data/strata/runs/<run>/eval-curve-on.csv with the cumulative bpb at each prefix length and the bpb of the
windows [0.5 MiB, 1 MiB) and [1.5 MiB, 2 MiB). `dickens` is a held-out source; `enwik8` is a SEEN source for
these models (Wikipedia is in the training set) and is included only because the repo's Mamba curve uses it.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate  # noqa: E402
from common import (BC_TAG, CACHE, RUNS, SOURCES, bitcell_evidence, evidence, load_source,  # noqa: E402
                    successor_features)
from models import build  # noqa: E402

STREAMS = ("curve-dickens", "curve-enwik8")
MIB = 1 << 20
GRID = (10_240, 51_200, 102_400, 512_000, MIB // 2, MIB, 3 * MIB // 2, 2 * MIB)


def feature_paths(name):
    return [os.path.join(CACHE, f"{name}-{k}.npy") for k in ("s", "top")]


def prepare(name):
    t0 = time.time()
    xb = load_source(name)[None, :]
    q0 = np.load(os.path.join(CACHE, "q0.npy"))
    pred, conf = successor_features(xb)
    s, top = evidence(xb, pred, conf, q0)
    for p, arr in zip(feature_paths(name), (s, top)):
        np.save(p + ".tmp.npy", arr)
        os.replace(p + ".tmp.npy", p)
    pbc = os.path.join(CACHE, f"{name}-{BC_TAG}.npy")
    if not os.path.isfile(pbc):
        np.save(pbc + ".tmp.npy", bitcell_evidence(xb))
        os.replace(pbc + ".tmp.npy", pbc)
    print(f"{name}: features ready in {time.time() - t0:.0f} s\nDONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare")
    ap.add_argument("--run")
    ap.add_argument("--online", default="on", choices=("on", "off"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--minutes", type=float, default=2.5)
    args = ap.parse_args()
    if args.prepare:
        prepare(args.prepare)
        return
    torch.set_num_threads(args.threads)
    run_dir = os.path.join(RUNS, args.run)
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), weights_only=False)
    model = build(ck["args"]["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    xb = np.stack([load_source(n) for n in STREAMS])
    lengths = np.array([xb.shape[1]] * len(STREAMS))
    feats = [[np.load(p, mmap_mode="r") for p in feature_paths(n)] for n in STREAMS]
    s_all = np.concatenate([f[0] for f in feats], axis=0)
    top_all = np.concatenate([f[1] for f in feats], axis=0)
    bc_all = None
    if getattr(model, "bc", False):
        bc_all = np.concatenate([np.load(os.path.join(CACHE, f"{n}-{BC_TAG}.npy"), mmap_mode="r") for n in STREAMS],
                                axis=0)
    evaluate.CURVE_POINTS = GRID
    out = evaluate.run(model, xb, lengths, s_all, top_all, args.online == "on", args.window, args.minutes,
                       os.path.join(run_dir, f"eval-curve-{args.online}.resume.pt"), bc_all=bc_all)
    if out is None:
        return
    bits, pref = out
    path = os.path.join(run_dir, f"eval-curve-{args.online}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "arith", "tier", "mac_per_byte", "online", "source", "role"]
                   + [f"bpb@{p}" for p in GRID] + ["bpb[0.5MiB,1MiB)", "bpb[1.5MiB,2MiB)", "late/early"])
        for k, (tier, macs, _) in enumerate(model.budget()):
            for i, name in enumerate(STREAMS):
                cum = pref[k, i]
                early = (cum[5] - cum[4]) / (MIB // 2)
                late = (cum[7] - cum[6]) / (MIB // 2)
                w.writerow([ck["args"]["model"], "int" if model.quant else "float", tier, macs, args.online, name,
                            SOURCES[name][1]] + [f"{v / p:.4f}" for v, p in zip(cum, GRID)]
                           + [f"{early:.4f}", f"{late:.4f}", f"{late / early:.3f}"])
                print(f"{ck['args']['model']:16s} tier {tier} {name}: "
                      + "  ".join(f"{p}={v / p:.4f}" for v, p in zip(cum, GRID))
                      + f"  | window early {early:.4f} late {late:.4f} ratio {late / early:.3f}", flush=True)
    print(f"wrote {path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
