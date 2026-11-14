"""Strata viability: the classical set of amendment 4 on the extended decision set (`sources.py`).

    python -I tools/strata/viability/baselines_ext.py [--sevenzip PATH]
    python -I tools/strata/viability/baselines_ext.py --frozen     # PPMd memory check on the bake-off's sets

Codecs: xz preset 9|EXTREME, zstd 19 and 22 (one shot: the window covers the whole 1 MiB input), brotli quality 11
with lgwin 24, bz2 level 9 (the callables of `bakeoff/baselines.py`), and 7-Zip PPMd var.H at orders 4-32 with 256 MB,
packed stream without the container header, exactly as `ppmd_baseline.py` (`ppmd-oN`).
A second PPMd row per order, `ppmd-oN-1g`, feeds 7-Zip through stdin with 1 GB: given a file, 7-Zip lowers the model
memory to the power of two that holds 16x the file size whatever `mem` says (16 MB for a 1 MiB file), the model
restarts at high orders and PPMd is under-served. The best classical codec is reported with and without those rows.
Writes data/strata/runs/baselines-ext.csv; the proposed replacement slices of `sources.ALTERNATIVES` are coded too
and labelled `alternative`. `--frozen` codes the selection, frozen-test and seen sources of the bake-off with the
`-1g` rows only and writes data/strata/runs/ppmd-baseline-1g.csv, next to the registered `ppmd-baseline.csv`.
Classical codecs only: no Strata model is evaluated here.
"""
import csv
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
sys.path.insert(0, HERE)
from baselines import CODECS  # noqa: E402
from common import RUNS, SOURCES, load_source  # noqa: E402
from ppmd_baseline import ORDERS, ROLES, find_sevenzip, packed_size  # noqa: E402
from primer_baselines import ppmd_size  # noqa: E402
from sources import ALTERNATIVES, EXTENDED, check_roles, load_slice  # noqa: E402

CLASSICAL = ("xz-9e", "zstd-19", "zstd-22", "brotli-11", "bz2-9")


def registered_set(sevenzip, data, tmp):
    """Sizes of the classical set exactly as registered in amendment 4."""
    sizes = {codec: len(CODECS[codec](data)) for codec in CLASSICAL}
    src = os.path.join(tmp, "input.bin")
    with open(src, "wb") as f:
        f.write(data)
    sizes.update({f"ppmd-o{o}": packed_size(sevenzip, src, o, tmp) for o in ORDERS})
    return sizes


def main():
    sevenzip = find_sevenzip(sys.argv)
    check_roles()
    frozen = "--frozen" in sys.argv
    if frozen:
        slices = [(n, v[0], v[1], bytes(load_source(n))) for n, v in SOURCES.items() if v[0] in ROLES]
    else:
        slices = [(n, role, spec[0], bytes(load_slice(spec))) for role, table in
                  (("decision-ext", EXTENDED), ("alternative", ALTERNATIVES)) for n, spec in table.items()]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, role, group, data in slices:
            sizes = {} if frozen else registered_set(sevenzip, data, tmp)
            big = {f"ppmd-o{o}-1g": ppmd_size(sevenzip, data, o, tmp) for o in ORDERS}
            rows += [[name, role, group, len(data), codec, out, f"{out * 8 / len(data):.4f}"]
                     for codec, out in {**sizes, **big}.items()]
            best_big = min(big, key=big.get)
            line = f"{name:16s} {group:13s} {len(data):>9,d} B  "
            if sizes:
                best = min(sizes, key=sizes.get)
                line += f"best as registered {best} {sizes[best]:,d} B {sizes[best] * 8 / len(data):.4f} bpb | "
            print(line + f"PPMd through stdin, 1 GB: {best_big} {big[best_big]:,d} B "
                  f"{big[best_big] * 8 / len(data):.4f} bpb", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, "ppmd-baseline-1g.csv" if frozen else "baselines-ext.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "role", "group", "bytes", "codec", "out_bytes", "bpb"])
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
