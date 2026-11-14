"""Strata bake-off: classical codecs on exactly the bytes of the evaluation sets.

    python -I tools/strata/bakeoff/baselines.py

Writes data/strata/runs/baselines.csv. Codecs (Python bindings, single shot, whole stream): xz preset 9|EXTREME,
zstd level 19 and 22 (long-distance window = stream size), brotli quality 11 (lgwin 24), bz2 level 9, zlib level 9.
For the length curves the prefixes of CURVE_POINTS are compressed independently.
"""
import bz2
import csv
import lzma
import os
import sys
import zlib

import brotli
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CURVE_POINTS, RUNS, SOURCES, load_source  # noqa: E402

CODECS = {
    "xz-9e": lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME),
    "zstd-19": lambda b: zstandard.ZstdCompressor(level=19).compress(b),
    "zstd-22": lambda b: zstandard.ZstdCompressor(level=22).compress(b),
    "brotli-11": lambda b: brotli.compress(b, quality=11, lgwin=24),
    "bz2-9": lambda b: bz2.compress(b, 9),
    "gzip-9": lambda b: zlib.compress(b, 9),
}


def main():
    os.makedirs(RUNS, exist_ok=True)
    rows = []
    for name, (role, group, _, _, _) in SOURCES.items():
        if role == "train":
            continue
        data = bytes(load_source(name))
        cuts = [len(data)] if role != "curve" else [p for p in CURVE_POINTS if p <= len(data)]
        for cut in cuts:
            piece = data[:cut]
            best = None
            for codec, fnc in CODECS.items():
                out = len(fnc(piece))
                bpb = out * 8 / len(piece)
                rows.append([name, role, group, len(piece), codec, out, f"{bpb:.4f}"])
                best = bpb if best is None else min(best, bpb)
            print(f"{name:16s} {role:7s} {len(piece):>9,d} B  best classical {best:.4f} bpb  "
                  + "  ".join(f"{r[4]}={r[6]}" for r in rows[-len(CODECS):]), flush=True)
    path = os.path.join(RUNS, "baselines.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "role", "group", "bytes", "codec", "out_bytes", "bpb"])
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
