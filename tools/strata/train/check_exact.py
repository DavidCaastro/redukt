"""Strata ladder trainer: exactness tests of the device port.

    python tools/strata/train/check_exact.py --test A                       # vs bakeoff/models.py, CPU (local)
    python tools/strata/train/check_exact.py --test features --data DIR     # vs bakeoff/common.py
    python tools/strata/train/check_exact.py --test B --data DIR --rung ladder3 --device cuda

A  same weights, same inputs: per-position costs, P(bit = 1), trunk state and mixer state identical to the
   prototype over consecutive windows with carried state, every tier; gradients compared with a tolerance (the
   two backward passes sum the same terms in a different float order).
B  the same model on two devices (cpu and --device): integers identical; costs and gradients within float noise.
features  the torch feature pipeline (on --device) against the numpy prototype.
Exit status 0 only if every integer comparison has 0 mismatches and the gradients are within tolerance.
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
import common  # noqa: E402
from corpus import Corpus  # noqa: E402
from features import features, taps  # noqa: E402
from model import RUNGS, Ladder, export_int  # noqa: E402

WINDOW = 256
GRAD_RTOL = 1e-3      # of the largest gradient entry of the same tensor
G0_RUNS = ("strata_g_bc", "strata_g_bc_s1205")


def perturb(model, seed):
    """Random weights that exercise every clamp: non-zero biases, spread boot weights."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = {"b": 0.02, "bg": 0.5, "hb": 0.5, "gb": 0.5, "w0": 0.3}.get(name.split(".")[1], 0.0)
            p.add_(torch.randn(p.shape, generator=gen) * scale)


def batch_inputs(chunks, q0, device):
    """Model inputs of whole chunks (uint8 [B, N]): taps, features, set index, bytes, on `device`."""
    s, top = features(chunks, q0, device)
    y = torch.as_tensor(chunks).long().to(device)
    return taps(y), s.float().to(device), top.long().to(device), y


def run_windows(model, inputs, n_win, tiers):
    """Forward + backward over consecutive windows with carried state; returns everything to be compared."""
    xt, s, top, y = inputs
    state = model.init_state(y.shape[0])
    out = []
    for w in range(n_win):
        sl = slice(w * WINDOW, (w + 1) * WINDOW)
        model.zero_grad(set_to_none=True)
        costs, state, p1 = model(xt[:, sl], s[:, sl], top[:, sl], y[:, sl], state)
        (sum(c.mean() for c in costs) / tiers).backward()
        out.append(dict(costs=torch.stack([c.detach() for c in costs]).cpu(), p1=p1.cpu(),
                        trunk=torch.stack(state[0]).long().cpu(), mix=state[1].long().cpu(),
                        grads={n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}))
    return out


def compare(ref, new, exact_costs):
    """Mismatch counts of the integer outputs, worst cost difference, worst relative gradient difference."""
    bad = dict(p1=0, trunk=0, mix=0, costs=0)
    cost_diff, grad_rel = 0.0, 0.0
    for a, b in zip(ref, new):
        bad["p1"] += int((a["p1"] != b["p1"]).sum())
        bad["trunk"] += int((a["trunk"] != b["trunk"]).sum())
        bad["mix"] += int((a["mix"] != b["mix"]).sum())
        cost_diff = max(cost_diff, float((a["costs"] - b["costs"]).abs().max()))
        bad["costs"] += int((a["costs"] != b["costs"]).sum()) if exact_costs else 0
        for name, ga in a["grads"].items():
            scale = float(ga.abs().max())
            if scale > 0:
                grad_rel = max(grad_rel, float((ga - b["grads"][name]).abs().max()) / scale)
    ok = not any(bad.values()) and grad_rel <= GRAD_RTOL
    return ok, bad, cost_diff, grad_rel


def prototype_model(rung, state_dict):
    """The frozen prototype (bakeoff/models.py) with the widths of `rung`, bit cells on, carrying `state_dict`."""
    import models
    models.CONFIGS["ladder-check"] = dict(kind="strata", widths=RUNGS[rung]["widths"], layers=3, geared=True,
                                          quant=True, bc=True)
    proto = models.CodecModel("ladder-check")
    proto.load_state_dict(state_dict, strict=False)
    return proto


def proto_windows(proto, inputs, n_win, tiers):
    """The prototype returns P(bit = 1) only on request and keeps its mixer state as a list: same record."""
    xt, s, top, y = inputs
    state = proto.init_state(y.shape[0])
    out = []
    for w in range(n_win):
        sl = slice(w * WINDOW, (w + 1) * WINDOW)
        proto.zero_grad(set_to_none=True)
        costs, state, probs = proto(xt[:, sl], s[:, sl], top[:, sl], y[:, sl], state, want_p=True)
        (sum(c.mean() for c in costs) / tiers).backward()
        out.append(dict(costs=torch.stack([c.detach() for c in costs]), p1=torch.stack(probs),
                        trunk=torch.stack([t.detach().long() for t in state[0]]), mix=torch.stack(state[1]),
                        grads={n: p.grad.detach().clone() for n, p in proto.named_parameters()}))
        state = ([t.detach() for t in state[0]], state[1])
    return out


def same_export(proto, mine):
    """The integer tensors this trainer exports equal the prototype's export of the same weights, every tier."""
    import models
    for k in range(len(mine.widths)):
        ref, new = models.export_int(proto, k), export_int(mine, k)
        for name, value in new.items():
            pairs = zip(ref[name], value) if isinstance(value, list) else [(ref[name], value)]
            if not all(np.array_equal(a, b) for a, b in pairs):
                return False
    return True


def test_a(args):
    torch.set_num_threads(args.threads)
    chunks = common.Corpus().sample_chunks(args.seed, 7, args.streams, args.windows * WINDOW)
    q0 = np.load(os.path.join(common.CACHE, "q0.npy"))
    inputs = batch_inputs(chunks, q0, "cpu")
    cases = [("g0", run) for run in G0_RUNS] + [(r, None) for r in ("g0", "tiny", "ladder3")]
    ok_all = True
    for rung, run in cases:
        torch.manual_seed(args.seed)
        mine = Ladder(rung)
        if run:
            ck = torch.load(os.path.join(common.RUNS, run, "ckpt.pt"), weights_only=False)
            mine.load_state_dict({k: v for k, v in ck["model"].items() if k in dict(mine.named_parameters())},
                                 strict=False)
        else:
            perturb(mine, args.seed)
        proto = prototype_model(rung, {n: p.detach().clone() for n, p in mine.named_parameters()})
        tiers = len(mine.widths)
        ok, bad, cost_diff, grad_rel = compare(proto_windows(proto, inputs, args.windows, tiers),
                                               run_windows(mine, inputs, args.windows, tiers), True)
        ok = ok and same_export(proto, mine)
        ok_all = ok_all and ok
        n_p = chunks.size * 8 * tiers
        print(f"A {rung:8s} weights={run or 'random':18s} {tiers} tiers, {args.streams} streams x {args.windows} "
              f"windows: mismatches p1 {bad['p1']}/{n_p}, costs {bad['costs']}, trunk state {bad['trunk']}, mixer "
              f"state {bad['mix']}; max |cost diff| {cost_diff:.3g}; max gradient diff {grad_rel:.3g} of the "
              f"tensor's largest entry; exported integers equal the prototype's; exact bound "
              f"{mine.check_exact():,.0f} < 2^24 -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok_all


def test_features(args):
    corpus = Corpus(args.data)
    real = corpus.sample(args.seed, 11, args.streams, 4096)
    rng = np.random.default_rng(args.seed)
    cases = {"corpus": real, "random bytes": rng.integers(0, 256, (4, 2000), dtype=np.uint8),
             "two symbols": rng.integers(97, 99, (4, 2000), dtype=np.uint8), "short": real[:, :11].copy(),
             "many streams": corpus.sample(args.seed, 12, 72, 8192)}
    q0 = corpus.boot_q0(args.seed)
    ok_all = True
    for name, xb in cases.items():
        pred, conf = common.successor_features(xb)
        s_ref, top_ref = common.evidence(xb, pred, conf, q0)
        s_ref = np.concatenate([s_ref, common.bitcell_evidence(xb)], axis=-1)
        s, top = features(xb, q0, args.device, max_bytes=3 * xb.shape[1])
        bad = int((s.numpy() != s_ref).sum()) + int((top.numpy() != top_ref).sum())
        ok_all = ok_all and bad == 0
        print(f"features {name:13s} {xb.shape[0]} x {xb.shape[1]} bytes on {args.device}: {bad} mismatches "
              f"in {s_ref.size + top_ref.size} values -> {'PASS' if bad == 0 else 'FAIL'}", flush=True)
    return ok_all


def test_b(args):
    corpus = Corpus(args.data)
    chunks = corpus.sample(args.seed, 13, args.streams, args.windows * WINDOW)
    q0 = corpus.boot_q0(args.seed)
    torch.manual_seed(args.seed)
    cpu_model = Ladder(args.rung)
    if args.ckpt:
        cpu_model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    else:
        perturb(cpu_model, args.seed)
    dev_model = copy.deepcopy(cpu_model).to(args.device)
    tiers = len(cpu_model.widths)
    ref = run_windows(cpu_model, batch_inputs(chunks, q0, "cpu"), args.windows, tiers)
    new = run_windows(dev_model, batch_inputs(chunks, q0, args.device), args.windows, tiers)
    ok, bad, cost_diff, grad_rel = compare(ref, new, False)
    print(f"B {args.rung} weights={'checkpoint' if args.ckpt else 'random'} cpu vs {args.device}, {args.streams} "
          f"streams x {args.windows} windows: mismatches p1 {bad['p1']}/{chunks.size * 8 * tiers}, trunk state "
          f"{bad['trunk']}, mixer state {bad['mix']}; max |cost diff| {cost_diff:.3g} bits; max gradient diff "
          f"{grad_rel:.3g} of the tensor's largest entry; exact bound {cpu_model.check_exact():,.0f} < 2^24 -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=("A", "B", "features"))
    ap.add_argument("--data", default="", help="corpus root of corpus.py (tests B and features)")
    ap.add_argument("--rung", default="ladder3", choices=sorted(RUNGS))
    ap.add_argument("--ckpt", default="", help="test B: a checkpoint of run_train.py instead of random weights")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1204)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    ok = {"A": test_a, "B": test_b, "features": test_features}[args.test](args)
    print("ALL CHECKS PASSED" if ok else "CHECK FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
