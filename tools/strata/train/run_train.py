"""Strata ladder trainer (gate V7, arm W): one process per device, resumable, with a wall-clock budget.

    python tools/strata/train/run_train.py --rung ladder3 --data DIR --out RUN --device cuda \
        --updates 7648 --batch 512 --minutes 300

Run the same command again until it prints DONE. An update = `--batch` streams x 256 bytes; a stream is a chunk of
`--chunk` consecutive bytes fed as consecutive windows with the trunk state, the online tables and the online
mixers carried across windows and reset at the chunk start. The chunks of a group are a pure function of
(--seed, group index), so a resumed run sees the bytes an uninterrupted run would. The run stops and checkpoints
(atomic rename) only between groups, so `--updates` is rounded up to a multiple of chunk / 256.
No mixed precision: the forward pass is the integer model (model.py).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import Corpus  # noqa: E402
from features import features, taps  # noqa: E402
from model import RUNGS, Ladder, detach_state  # noqa: E402

WINDOW = 256


def lr_at(step, total, peak, warm=100, floor=0.05):
    """The bake-off's schedule: linear warm-up, cosine to 5 % of the peak."""
    if step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def train_group(model, opt, chunks, q0, args, step):
    """One group of chunks: features, then one update per window. Returns the mean cost per tier and the seconds
    spent in the feature pipeline."""
    dev = args.device
    t0 = time.time()
    s_all, top_all = features(chunks, q0, dev, args.feat_mb << 20)
    feat_seconds = time.time() - t0
    y_all = torch.as_tensor(chunks).to(dev).long()
    xt_all = taps(y_all)
    state = model.init_state(args.batch)
    total = torch.zeros(len(model.widths), dtype=torch.float64, device=dev)
    wins = args.chunk // WINDOW
    for w in range(wins):
        sl = slice(w * WINDOW, (w + 1) * WINDOW)
        costs, state, _ = model(xt_all[:, sl], s_all[:, sl].to(dev).float(), top_all[:, sl].to(dev).long(),
                                y_all[:, sl], state)
        means = torch.stack([c.mean() for c in costs])
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step + w, args.updates, args.lr)
        opt.zero_grad(set_to_none=True)
        means.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        state = detach_state(state)
        total += means.detach()
    return (total / wins).tolist(), feat_seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--data", required=True, help="corpus root prepared by corpus.py")
    ap.add_argument("--out", required=True, help="run directory (checkpoint, log, summary)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--updates", type=int, required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--minutes", type=float, default=8.0, help="wall-clock budget of this invocation")
    ap.add_argument("--session-updates", type=int, default=0, help="pause after this many updates (0 = no limit)")
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1204)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--feat-mb", type=int, default=8, help="bytes of text per call of the feature pipeline")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    log_path = os.path.join(args.out, "train_log.csv")

    model = Ladder(args.rung)
    model.check_budget()
    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    corpus = Corpus(args.data)
    step, spent = 0, 0.0
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, spent, q0 = ck["step"], ck["spent"], ck["q0"]
        print(f"resumed at update {step} ({spent / 60:.1f} min of training so far)", flush=True)
    else:
        q0 = corpus.boot_q0(args.seed)
    if step >= args.updates:
        print("DONE (already complete)")
        return

    def save():
        bound = model.check_exact()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "q0": q0,
                    "rung": args.rung, "spent": spent + time.time() - t_start, "args": vars(args)},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        return bound

    wins = args.chunk // WINDOW
    first, feat_total, last_save = step, 0.0, time.time()
    new_log = not os.path.isfile(log_path)
    with open(log_path, "a", newline="") as logf:
        logw = csv.writer(logf)
        if new_log:
            logw.writerow(["update", "seconds", "lr"] + [f"bits_per_byte_w{w}" for w in model.widths])
        t_loop = time.time()
        while step < args.updates:
            chunks = corpus.sample(args.seed, step // wins, args.batch, args.chunk)
            costs, feat_seconds = train_group(model, opt, chunks, q0, args, step)
            step += wins
            feat_total += feat_seconds
            el = spent + time.time() - t_start
            logw.writerow([step, f"{el:.0f}", f"{lr_at(step - 1, args.updates, args.lr):.6f}"]
                          + [f"{c:.4f}" for c in costs])
            logf.flush()
            print(f"update {step:6d}  {el / 60:7.1f} min  bits/byte " + " ".join(f"{c:.4f}" for c in costs),
                  flush=True)
            if time.time() - last_save > args.ckpt_minutes * 60:
                save()
                last_save = time.time()
            if time.time() - t_start > args.minutes * 60 or 0 < args.session_updates <= step - first:
                break
        loop_seconds = time.time() - t_loop
    bound = save()
    seen = (step - first) * args.batch * WINDOW
    summary = {"rung": args.rung, "device": args.device, "batch": args.batch, "chunk": args.chunk, "lr": args.lr,
               "seed": args.seed, "updates_done": step, "updates_this_session": step - first,
               "bytes_this_session": seen, "loop_seconds": round(loop_seconds, 2),
               "feature_seconds": round(feat_total, 2), "kb_per_second": round(seen / loop_seconds / 1e3, 2),
               "updates_per_second": round((step - first) / loop_seconds, 4), "exact_bound": bound,
               "corpus_bytes": corpus.nbytes, "final_costs": costs,
               "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if args.device != "cpu" else 0}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary), flush=True)
    print("DONE" if step >= args.updates else f"PAUSED at update {step}/{args.updates}: run the same command again",
          flush=True)


if __name__ == "__main__":
    main()
