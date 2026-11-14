"""Strata bake-off, POST-HOC EXPLORATION (not part of the pre-registered comparison, decides nothing).

    python -I tools/strata/bakeoff/explore_bitcells.py --run lstm --bytes 100000

Question raised by the bake-off: most of the bits are spent on bytes that no successor table predicted (the
"literals"), where the only predictor left is the static trunk. Classical context mixers code those bytes with
adaptive low-order BIT statistics. This script adds such statistics as extra mixer inputs to an already trained
model, WITHOUT retraining anything: adaptive cells P(bit = 1 | context, node) for orders 0, 1, 2 (exact index) and
3 (hashed to 2^22 cells), same 22-bit update rule as the calibration cells, boot value 1/2, boot mixer weight
0.3, weights then adapted by the same online mixer rule. Everything else (trunk, gains, successor evidence, boot
weights) is the trained model's. Arithmetic here is float64 (an estimate of headroom, not an integer contract).

The first row re-computes the model's own cost with this script's loop: it must agree with evaluate.py, which is
the sanity check of the re-implementation.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, MIX_SH, N_IN, NORD, Q_BITS, QSHIFT, RUNS, ST, ST_MAX, W_MAX, W_ONE,  # noqa: E402
                    evidence, load_source, successor_features, sources_with_role)
from models import ACT, build  # noqa: E402
from train import taps  # noqa: E402

BOOT_W = 0.3 * W_ONE
HALF = 1 << (Q_BITS - 1)


def head_inputs(model, k, xb, q0, trunk_on):
    """Trunk stretch [N, 8], gained successor evidence [N, 8, NORD], weight-set index [N, 8], bits, boot weights."""
    pred, conf = successor_features(xb)
    s, top = evidence(xb, pred, conf, q0)
    hd, dk = model.head, model.widths[k]
    with torch.no_grad():
        h, _ = model.trunk(torch.from_numpy(taps(xb).astype(np.int64)), model.trunk.init_state(1))
        if not trunk_on:
            h = torch.zeros_like(h)
        hk = h[:, :, :dk]
        y = torch.from_numpy(xb.astype(np.int64))
        shifts = torch.arange(7, -1, -1)
        ybit = (y[:, :, None] >> shifts) & 1
        node = ((1 << torch.arange(8)) | (y[:, :, None] >> (shifts + 1))) - 1
        z = hk @ hd.q(hd.hw[:, :dk], 64.0, ACT).t()
        z = torch.gather(z, 2, node) + hd.q(hd.hb[k], 2048.0)[node]
        st_net = torch.clamp(torch.floor((z + 4.0) / 8.0) if hd.quant else z / 8.0, -ST_MAX, ST_MAX)
        g = hk @ hd.q(hd.gw[:, :dk], 64.0, ACT).t() + hd.q(hd.gb[k], 2048.0)
        gain = torch.clamp(torch.floor((g + 32.0) / 64.0) if hd.quant else g / 64.0, 0.0, ACT)
        sp = torch.from_numpy(s.astype(np.float32)) * gain[:, :, None, :] / 32.0
        sp = torch.clamp(torch.floor(sp) if hd.quant else sp, -ST_MAX, ST_MAX)
        w0 = hd.q(hd.w0[k], float(W_ONE), W_MAX)
    setidx = np.arange(8)[None, :] * (NORD + 1) + top[0].astype(np.int64)
    return (st_net[0].double().numpy(), sp[0].double().numpy(), setidx, ybit[0].numpy(), node[0].numpy() + 1,
            w0.double().numpy())


def run_mixer(data, st_net, sp, setidx, ybit, node, w0, orders):
    """Online mixer over the trained inputs plus the adaptive bit cells of `orders` (subset of 0..3)."""
    n = len(data)
    nb = len(orders)
    w = np.concatenate([w0, np.full((w0.shape[0], nb), BOOT_W)], axis=1)
    cells = {0: np.full(256, HALF, dtype=np.int64), 1: np.full(1 << 16, HALF, dtype=np.int64),
             2: np.full(1 << 24, HALF, dtype=np.int32), 3: np.full(1 << 22, HALF, dtype=np.int32)}
    counts = {o: np.zeros(len(cells[o]), dtype=np.uint8) for o in orders}
    scale = 65536.0 / float(1 << MIX_SH)
    lo_p, hi_p = 1.0 / 65536.0, 1.0 - 1.0 / 65536.0
    d = data.astype(np.int64)
    bits = 0.0
    x = np.zeros((8, N_IN + nb))
    for t in range(n):
        c1 = int(d[t - 1]) if t >= 1 else 0
        c2 = int(d[t - 2]) if t >= 2 else 0
        c3 = int(d[t - 3]) if t >= 3 else 0
        nd = node[t]
        idx = {0: nd, 1: (c1 << 8) + nd, 2: (c2 << 16) + (c1 << 8) + nd,
               3: ((((((c3 * 0x9E3779B1 + c2) * 0x9E3779B1 + c1) * 0x85EBCA6B) & 0xFFFFFFFF) >> 18) << 8) + nd}
        x[:, 0] = st_net[t]
        x[:, 1:N_IN] = sp[t]
        for m, o in enumerate(orders):
            x[:, N_IN + m] = ST[cells[o][idx[o]] >> (Q_BITS - 12)]
        si = setidx[t]
        wt = w[si]
        st = np.clip((wt * x).sum(1) / W_ONE, -ST_MAX, ST_MAX)
        p = np.clip(1.0 / (1.0 + np.exp(-st / 256.0)), lo_p, hi_p)
        yb = ybit[t]
        bits -= np.log2(np.where(yb == 1, p, 1.0 - p)).sum()
        w[si] = np.clip(wt + x * ((yb - p) * scale)[:, None], -W_MAX, W_MAX)
        for o in orders:
            q = cells[o][idx[o]].astype(np.int64)
            cnt = counts[o][idx[o]]
            cells[o][idx[o]] = q + (((yb.astype(np.int64) << Q_BITS) - q) >> QSHIFT[cnt])
            counts[o][idx[o]] = np.minimum(cnt.astype(np.int64) + 1, 255)
    return bits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bytes", type=int, default=100_000)
    ap.add_argument("--sources", default=",".join(sources_with_role("select")))
    ap.add_argument("--tier", type=int, default=-1)
    args = ap.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(os.path.join(RUNS, args.run, "ckpt.pt"), weights_only=False)
    model = build(ck["args"]["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    k = args.tier if args.tier >= 0 else len(model.widths) - 1
    q0 = np.load(os.path.join(CACHE, "q0.npy"))
    variants = [("as trained (sanity)", True, ()), ("+ order 0", True, (0,)), ("+ orders 0-1", True, (0, 1)),
                ("+ orders 0-2", True, (0, 1, 2)), ("+ orders 0-3", True, (0, 1, 2, 3)),
                ("trunk off, as trained", False, ()), ("trunk off + orders 0-3", False, (0, 1, 2, 3))]
    names = args.sources.split(",")
    table = {v[0]: [] for v in variants}
    for src in names:
        data = load_source(src)[:args.bytes]
        xb = data[None, :]
        cache = {}
        for label, trunk_on, orders in variants:
            if trunk_on not in cache:
                cache[trunk_on] = head_inputs(model, k, xb, q0, trunk_on)
            bpb = run_mixer(data, *cache[trunk_on], orders)
            table[label].append(bpb)
            print(f"{src:16s} {label:26s} {bpb:.4f}", flush=True)
    print(f"\n{args.run} tier index {k}, first {args.bytes} bytes of each source, bpb")
    print(f"{'variant':26s} " + " ".join(f"{n:>14s}" for n in names) + f" {'macro':>8s}")
    for label, vals in table.items():
        print(f"{label:26s} " + " ".join(f"{v:14.4f}" for v in vals) + f" {np.mean(vals):8.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
