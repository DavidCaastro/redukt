"""Strata ladder evaluation: ideal code length per file and tier with the integer model.

    python tools/strata/train/run_eval.py --ckpt RUN/ckpt.pt --sources sel-webster,sel-xml,sel-zookeeper,sel-proxifier

One stream per file, coded from its first byte with the boot state, no reset, all files in one batch. The bits are
ideal code lengths of the integer probabilities the coder would receive (no range-coder overhead, no container).
Role gate: only the selection sources of `bakeoff/common.py` are accepted. Every other name (the frozen test set,
the seen-source and curve slices, the extended decision set of `viability/sources.py`) is a decision file and is
refused unless --allow-decision is given, which is the orchestrator's call after the ladder is registered.
Resumable: with --minutes, the pass stops between windows and the same command continues it.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VIABILITY = os.path.join(HERE, "..", "viability")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
import common  # noqa: E402
from export import load_model  # noqa: E402
from features import features, taps  # noqa: E402


def extended_sources():
    """The extended decision set, read at run time from viability/sources.py when that file exists."""
    if not os.path.isfile(os.path.join(VIABILITY, "sources.py")):
        return {}
    sys.path.insert(0, VIABILITY)
    import sources
    return {**sources.EXTENDED, **sources.ALTERNATIVES}


def eval_features(xb, q0, device):
    """On the CPU the prototype's numpy pipeline (identical values, `check_exact.py --test features`; six times
    faster than the torch port for a handful of streams); on a GPU the torch port."""
    if device != "cpu":
        return features(xb, q0, device)
    pred, conf = common.successor_features(xb)
    s, top = common.evidence(xb, pred, conf, q0)
    return torch.from_numpy(np.concatenate([s, common.bitcell_evidence(xb)], axis=-1)), torch.from_numpy(top)


def load(names, allow_decision, limit):
    """Bytes and role of every requested source; refuses decision files unless explicitly allowed."""
    ext = extended_sources()
    out = []
    for name in names:
        if name in common.SOURCES:
            role, data = common.SOURCES[name][0], common.load_source(name)
        elif name in ext:
            _, rel, off, length = ext[name]
            with open(os.path.join(common.ROOT, rel), "rb") as f:
                f.seek(off)
                role, data = "decision", np.frombuffer(f.read(length), dtype=np.uint8)
        else:
            raise SystemExit(f"unknown source {name}")
        if role != "select" and not allow_decision:
            raise SystemExit(f"REFUSED: {name} has role '{role}', not 'select'. Decision files are evaluated only "
                             "with --allow-decision, after the ladder configuration is registered.")
        out.append((name, role, data[:limit] if limit else data))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sources", required=True, help="comma-separated source names")
    ap.add_argument("--allow-decision", action="store_true")
    ap.add_argument("--bytes", type=int, default=0, help="evaluate only this prefix of every file (0 = whole)")
    ap.add_argument("--out", default="", help="CSV path (default: next to the checkpoint)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    files = load(args.sources.split(","), args.allow_decision, args.bytes)
    model, ck = load_model(args.ckpt, args.device)
    lengths = torch.tensor([len(d) for _, _, d in files])
    n = int(lengths.max())
    xb = np.zeros((len(files), n), dtype=np.uint8)
    for i, (_, _, d) in enumerate(files):
        xb[i, :len(d)] = d
    t0 = time.time()
    s_all, top_all = eval_features(xb, ck["q0"], args.device)
    print(f"features of {xb.size:,d} bytes in {time.time() - t0:.0f} s", flush=True)
    y_all = torch.from_numpy(xb).to(args.device).long()
    xt_all = taps(y_all)
    out_path = args.out or os.path.join(os.path.dirname(args.ckpt), "eval-" + "-".join(n for n, _, _ in files) + ".csv")
    resume = out_path + ".resume.pt"
    pos, state = 0, model.init_state(len(files))
    bits = torch.zeros(len(model.widths), len(files), dtype=torch.float64)
    if os.path.isfile(resume):
        saved = torch.load(resume, map_location=args.device, weights_only=False)
        pos, state, bits = saved["pos"], saved["state"], saved["bits"]
        print(f"resumed at byte {pos}", flush=True)
    idx = torch.arange(n)[None, :]
    with torch.no_grad():
        while pos < n:
            sl = slice(pos, min(n, pos + args.window))
            costs, state, _ = model(xt_all[:, sl], s_all[:, sl].to(args.device).float(),
                                    top_all[:, sl].to(args.device).long(), y_all[:, sl], state)
            valid = idx[:, sl] < lengths[:, None]
            bits += torch.stack([(c.cpu() * valid).sum(1) for c in costs])
            pos = sl.stop
            if time.time() - t0 > args.minutes * 60 and pos < n:
                torch.save({"pos": pos, "state": state, "bits": bits}, resume)
                print(f"PAUSED at byte {pos}/{n}: run the same command again", flush=True)
                return
    if os.path.isfile(resume):
        os.remove(resume)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "width", "mac_per_byte", "params", "updates", "source", "role", "bytes", "bits", "bpb"])
        for k, (width, macs, params, _) in enumerate(model.budget()):
            bpb = bits[k] / lengths
            for i, (name, role, d) in enumerate(files):
                w.writerow([ck["rung"], width, macs, params, ck["step"], name, role, len(d), f"{bits[k, i]:.1f}",
                            f"{bpb[i]:.4f}"])
            print(f"{ck['rung']} width {width} ({macs:,d} MAC/byte, {ck['step']} updates): "
                  + "  ".join(f"{name}={bpb[i]:.4f}" for i, (name, _, _) in enumerate(files))
                  + f"  | macro {bpb.mean():.4f}", flush=True)
    print(f"wrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
