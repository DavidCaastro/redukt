"""Strata viability: PPMd (context-model family) baseline on the selection and frozen test bytes.

    python -I tools/strata/viability/ppmd_baseline.py [--sevenzip PATH]

The bake-off baselines (`tools/strata/bakeoff/baselines.py`) cover the LZ and BWT families only. The width-0 tier
`cm` is a context mixer, so its fair classical comparison includes a context-model codec. This script codes exactly
the bytes of every `select`, `test` and `seen` source with 7-Zip's PPMd (PPMd var.H) at several model orders and
reports the smallest packed stream (the "Packed Size" of the archive listing: container header excluded).
Writes data/strata/runs/ppmd-baseline.csv. Needs the 7-Zip command-line program; nothing is sent anywhere.
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bakeoff"))
from common import RUNS, SOURCES, load_source  # noqa: E402

ORDERS = (4, 6, 8, 12, 16, 24, 32)
ROLES = ("select", "test", "seen")


def find_sevenzip(argv):
    if "--sevenzip" in argv:
        return argv[argv.index("--sevenzip") + 1]
    for cand in ("7z", "7za", r"C:\Program Files\7-Zip\7z.exe"):
        path = shutil.which(cand) or (cand if os.path.isfile(cand) else None)
        if path:
            return path
    raise SystemExit("7-Zip not found: pass --sevenzip PATH")


def packed_size(sevenzip, src, order, tmp):
    arc = os.path.join(tmp, f"o{order}.7z")
    if os.path.exists(arc):
        os.remove(arc)
    subprocess.run([sevenzip, "a", "-t7z", f"-m0=PPMd:mem=256m:o={order}", "-ms=off", "-bd", arc, src],
                   check=True, stdout=subprocess.DEVNULL)
    listing = subprocess.run([sevenzip, "l", "-slt", arc], check=True, capture_output=True, text=True).stdout
    sizes = [int(line.split("=")[1]) for line in listing.splitlines() if line.startswith("Packed Size =")]
    subprocess.run([sevenzip, "t", arc], check=True, stdout=subprocess.DEVNULL)
    return sizes[-1]


def main():
    sevenzip = find_sevenzip(sys.argv)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (role, group, _, _, _) in SOURCES.items():
            if role not in ROLES:
                continue
            data = bytes(load_source(name))
            src = os.path.join(tmp, "input.bin")
            with open(src, "wb") as f:
                f.write(data)
            best = None
            for order in ORDERS:
                out = packed_size(sevenzip, src, order, tmp)
                bpb = out * 8 / len(data)
                rows.append([name, role, group, len(data), f"ppmd-o{order}", out, f"{bpb:.4f}"])
                best = (bpb, order) if best is None or bpb < best[0] else best
            print(f"{name:14s} {role:7s} {len(data):>9,d} B  best PPMd {best[0]:.4f} bpb (order {best[1]})",
                  flush=True)
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, "ppmd-baseline.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "role", "group", "bytes", "codec", "out_bytes", "bpb"])
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
