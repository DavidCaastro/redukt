"""Strata bake-off evaluation on sources that training never saw (single pass, state carried, no resets).

    python -I tools/strata/bakeoff/evaluate.py --run strata_f --set select --online on
    sets:  select (model selection)   test (frozen test + the seen-source enwik8 slice)   curve (2 MiB streams)

Each source is one stream coded from its first byte with the boot state. Reported bits are ideal code lengths
of the probabilities the coder would receive (integer probabilities for quantised models), i.e. the range coder's
own overhead (< 0.01 %) and the container framing are not included; `ref_int.py` checks real coded bytes.
`--online off` disables the successor tables and freezes the mixer at its boot weights (ablation on the same weights).
`--trunk off` zeroes the trunk output: what remains is node biases + online tables + online mixer (ablation).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, CURVE_POINTS, RUNS, SOURCES, evidence, load_source, successor_features,  # noqa: E402
                    sources_with_role)
from models import build  # noqa: E402
from train import taps  # noqa: E402

SETS = {"select": ("select",), "test": ("test", "seen"), "curve": ("curve",)}


def load_set(name):
    names = [n for role in SETS[name] for n in sources_with_role(role)]
    data = [load_source(n) for n in names]
    lengths = np.array([len(d) for d in data])
    xb = np.zeros((len(data), int(lengths.max())), dtype=np.uint8)
    for i, d in enumerate(data):
        xb[i, :len(d)] = d
    return names, xb, lengths


def features(name, xb):
    os.makedirs(CACHE, exist_ok=True)
    ps, pt = (os.path.join(CACHE, f"eval-{name}-{k}.npy") for k in ("s", "top"))
    if os.path.isfile(ps) and os.path.isfile(pt):
        return np.load(ps, mmap_mode="r"), np.load(pt, mmap_mode="r")
    t0 = time.time()
    q0 = np.load(os.path.join(CACHE, "q0.npy"))
    pred, conf = successor_features(xb)
    s, top = evidence(xb, pred, conf, q0)
    for p, arr in ((ps, s), (pt, top)):
        np.save(p + ".tmp.npy", arr)
        os.replace(p + ".tmp.npy", p)
    print(f"features for set '{name}' computed in {time.time() - t0:.0f} s", flush=True)
    return s, top


def run(model, xb, lengths, s_all, top_all, online, window, minutes, resume_path, trunk_on=True):
    """Returns bits[tier][stream] and prefix bits[tier][stream][point]; resumable through `resume_path`."""
    bsz, n = xb.shape
    ntier = len(model.widths)
    xt_all = taps(xb)
    bits = np.zeros((ntier, bsz))
    pref = np.full((ntier, bsz, len(CURVE_POINTS)), np.nan)
    state = model.init_state(bsz)
    pos = 0
    if resume_path and os.path.isfile(resume_path):
        ck = torch.load(resume_path, weights_only=False)
        bits, pref, state, pos = ck["bits"], ck["pref"], ck["state"], ck["pos"]
        print(f"resumed evaluation at byte {pos}", flush=True)
    t0 = time.time()
    idx = np.arange(n)[None, :]
    with torch.no_grad():
        while pos < n:
            hi = min(n, pos + window)
            sl = slice(pos, hi)
            costs, state = model(torch.from_numpy(xt_all[:, sl].astype(np.int64)),
                                 torch.from_numpy(np.asarray(s_all[:, sl]).astype(np.float32)),
                                 torch.from_numpy(np.asarray(top_all[:, sl]).astype(np.int64)),
                                 torch.from_numpy(xb[:, sl].astype(np.int64)), state, online=online,
                                 trunk_on=trunk_on)
            valid = idx[:, sl] < lengths[:, None]
            for k in range(ntier):
                c = costs[k].numpy() * valid
                run_sum = bits[k][:, None] + np.cumsum(c, axis=1)
                for pi, point in enumerate(CURVE_POINTS):
                    if pos < point <= hi:
                        pref[k, :, pi] = np.where(lengths >= point, run_sum[:, point - pos - 1], np.nan)
                bits[k] = run_sum[:, -1]
            pos = hi
            if resume_path and time.time() - t0 > minutes * 60 and pos < n:
                torch.save({"bits": bits, "pref": pref, "state": state, "pos": pos}, resume_path)
                print(f"PAUSED at byte {pos}/{n}: run the same command again", flush=True)
                return None
    if resume_path and os.path.isfile(resume_path):
        os.remove(resume_path)
    return bits, pref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under data/strata/runs")
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--online", default="on", choices=("on", "off"))
    ap.add_argument("--trunk", default="on", choices=("on", "off"),
                    help="off = ablation: zero trunk output (biases + online tables + mixer only)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--minutes", type=float, default=8.0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run_dir = os.path.join(RUNS, args.run)
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), weights_only=False)
    model = build(ck["args"]["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    names, xb, lengths = load_set(args.set)
    s_all, top_all = features(args.set, xb)
    tag = f"{args.set}-{args.online}" + ("" if args.trunk == "on" else "-notrunk")
    out = run(model, xb, lengths, s_all, top_all, args.online == "on", args.window, args.minutes,
              os.path.join(run_dir, f"eval-{tag}.resume.pt"), args.trunk == "on")
    if out is None:
        return
    bits, pref = out
    budget = model.budget()
    path = os.path.join(run_dir, f"eval-{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "arith", "tier", "mac_per_byte", "params", "train_steps", "online", "trunk", "source", "role",
                    "bytes", "bits", "bpb"] + [f"bpb@{p}" for p in CURVE_POINTS])
        for k, (tier, macs, params) in enumerate(budget):
            for i, name in enumerate(names):
                row = [ck["args"]["model"], "int" if model.quant else "float", tier, macs, params, ck["step"],
                       args.online, args.trunk, name, SOURCES[name][0], int(lengths[i]), f"{bits[k, i]:.1f}",
                       f"{bits[k, i] / lengths[i]:.4f}"]
                row += ["" if np.isnan(v) else f"{v / p:.4f}" for v, p in zip(pref[k, i], CURVE_POINTS)]
                w.writerow(row)
            macro = np.mean(bits[k] / lengths)
            print(f"{ck['args']['model']:16s} tier {tier} ({macs:,d} MAC/B) online={args.online} trunk={args.trunk} "
                  f"set={args.set}: "
                  + "  ".join(f"{n}={bits[k, i] / lengths[i]:.4f}" for i, n in enumerate(names))
                  + f"  | macro {macro:.4f}", flush=True)
    print(f"wrote {path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
