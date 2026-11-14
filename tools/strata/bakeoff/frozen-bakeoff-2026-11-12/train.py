"""Strata bake-off trainer: one script for every model, resumable, with a wall-clock budget per invocation.

    python -I tools/strata/bakeoff/train.py --model strata_f --steps 3000 --minutes 8 [--threads 3]

Run the same command again until it prints DONE. Every model sees the same chunks in the same order (the sampler
is a pure function of --seed and the step), the same number of steps, the same optimiser and schedule.
A step = 64 streams x 256 bytes; a stream is a chunk of 8,192 consecutive bytes fed as 32 consecutive windows with
the recurrent state, the online tables and the online mixer carried across windows (and reset at chunk start).
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BOS, RUNS, Corpus, get_q0, superbatch  # noqa: E402
from models import build, check_budget, detach_state  # noqa: E402


def taps(xb):
    """Input byte indices for the four delay taps: taps[b, t, k] = x[t - 1 - k], BOS (256) before the start."""
    bsz, n = xb.shape
    pad = np.full((bsz, n + 4), BOS, dtype=np.int16)
    pad[:, 4:] = xb
    return np.stack([pad[:, 3 - k:3 - k + n] for k in range(4)], axis=2)


def lr_at(step, total, peak, warm=100, floor=0.05):
    if step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1204)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run = os.path.join(RUNS, args.model + args.tag)
    os.makedirs(run, exist_ok=True)
    ckpt_path = os.path.join(run, "ckpt.pt")
    log_path = os.path.join(run, "train_log.csv")

    model = build(args.model)
    for tier, macs, params in model.budget():
        print(f"budget tier {tier}: {macs:,d} MAC/byte, {params:,d} params", flush=True)
    if not check_budget(model):
        raise SystemExit("model exceeds the bake-off caps")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    step, spent = 0, 0.0
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, spent = ck["step"], ck["spent"]
        print(f"resumed at step {step} ({spent / 60:.1f} min of training so far)", flush=True)
    if step >= args.steps:
        print("DONE (already complete)")
        return

    corpus = Corpus()
    q0 = get_q0(corpus, args.seed)
    wins = args.chunk // args.window
    per_sb = args.groups * wins
    ntier = len(model.widths)
    recent = []
    new_log = not os.path.isfile(log_path)
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["step", "train_seconds", "lr"] + [f"bpb_tier{k}" for k in range(ntier)])

    def save():
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "spent": spent + time.time() - t_start, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)

    stop = False
    while step < args.steps and not stop:
        sb, within = divmod(step, per_sb)
        assert within % wins == 0, "checkpoints are only written at chunk boundaries"
        chunks, s_all, top_all = superbatch(corpus, q0, args.seed, sb, args.groups * args.batch, args.chunk)
        xt_all = taps(chunks)
        for g in range(within // wins, args.groups):
            rows = slice(g * args.batch, (g + 1) * args.batch)
            state = model.init_state(args.batch)
            for w in range(wins):
                sl = slice(w * args.window, (w + 1) * args.window)
                xt = torch.from_numpy(xt_all[rows, sl].astype(np.int64))
                s = torch.from_numpy(s_all[rows, sl].astype(np.float32))
                top = torch.from_numpy(top_all[rows, sl].astype(np.int64))
                y = torch.from_numpy(chunks[rows, sl].astype(np.int64))
                costs, state = model(xt, s, top, y, state)
                loss = sum(c.mean() for c in costs) / ntier
                for grp in opt.param_groups:
                    grp["lr"] = lr_at(step, args.steps, args.lr)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                state = detach_state(state)
                recent.append([float(c.mean().detach()) for c in costs])
                step += 1
                if step % 50 == 0:
                    avg = np.mean(recent, axis=0)
                    recent = []
                    el = spent + time.time() - t_start
                    logw.writerow([step, f"{el:.0f}", f"{lr_at(step, args.steps, args.lr):.6f}"]
                                  + [f"{a:.4f}" for a in avg])
                    logf.flush()
                    print(f"step {step:5d}  {el / 60:6.1f} min  bpb " + " ".join(f"{a:.4f}" for a in avg),
                          flush=True)
            if step >= args.steps or time.time() - t_start > args.minutes * 60:
                stop = True
                break
    save()
    logf.close()
    print("DONE" if step >= args.steps else f"PAUSED at step {step}/{args.steps}: run the same command again",
          flush=True)


if __name__ == "__main__":
    main()
