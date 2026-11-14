"""Strata bake-off: fill the on-disk cache of training super-batches (chunks + online context features).

    python -I tools/strata/bakeoff/precompute.py --first 0 --last 49 [--bc]

Optional (train.py computes a missing super-batch itself); it only avoids recomputing the same features in every
run. A super-batch = 128 chunks of 8,192 bytes = 64 training steps.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Corpus, get_q0, superbatch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=49)
    ap.add_argument("--seed", type=int, default=1204)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--bc", action="store_true", help="also the bit-cell evidence of spec 3.3 (gate G0)")
    args = ap.parse_args()
    t0 = time.time()
    corpus = Corpus()
    q0 = get_q0(corpus, args.seed)
    for sb in range(args.first, args.last + 1):
        if time.time() - t0 > args.minutes * 60:
            print(f"PAUSED before super-batch {sb}: run again with --first {sb}", flush=True)
            return
        superbatch(corpus, q0, args.seed, sb, 128, 8192, bc=args.bc)
        print(f"super-batch {sb} ready ({time.time() - t0:.0f} s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
