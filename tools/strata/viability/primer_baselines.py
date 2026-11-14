"""Strata viability, arm P: classical codecs given the same prior as a primed Strata stream, and the result table.

    python -I tools/strata/viability/primer_baselines.py classical [--comp mixI] [--sizes 65536,1048576,16777216]
    python -I tools/strata/viability/primer_baselines.py summary  [--comp mixI]

`classical` (selection sources by default; resumable: run it again until it prints DONE) writes
data/strata/runs/arm-p-classical.csv with, for every primer size and input:
  zstd-19-dict, zstd-22-dict   the primer bytes as a raw-content dictionary, window >= primer + input: a deployable
                               mode, the counterpart of the primer (amendment 4)
  xz-9e-cond, ppmd-oN-cond     C(primer + input) - C(primer): what the codec would spend on the input if its model
                               had already read the primer. NOT a deployable mode (the primer would have to be
                               re-read for every file); a stricter reference, reported separately. PPMd var.H of
                               7-Zip with 1 GB through stdin so that the model is not restarted, orders 4-32.
The Python brotli binding takes no dictionary (`brotli.compress(string, mode, quality, lgwin, lgblock)` only), so
brotli has no same-prior row.
`summary` merges data/strata/runs/<run>/arm-p-{select,explore,ablate}.csv of both seeds into
data/strata/runs/arm-p-{selection,explore,ablate}.csv and prints the tables of `09a-arm-p-selection.md`.
"""
import argparse
import csv
import lzma
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
sys.path.insert(0, HERE)
from common import RUNS, SOURCES, load_source  # noqa: E402
from ppmd_baseline import ORDERS, find_sevenzip  # noqa: E402
from sources import EXTENDED, load_extended  # noqa: E402

SEEDS = {"strata_g_bc": 1204, "strata_g_bc_s1205": 1205}
OUT = os.path.join(RUNS, "arm-p-classical.csv")
PPMD_MEM = "1g"


def zstd_dict(level, primer, data):
    import zstandard
    window = max(10, (len(primer) + len(data) - 1).bit_length())
    params = zstandard.ZstdCompressionParameters.from_level(level, source_size=len(data), dict_size=len(primer),
                                                            window_log=window)
    prior = zstandard.ZstdCompressionDict(primer, dict_type=zstandard.DICT_TYPE_RAWCONTENT)
    out = zstandard.ZstdCompressor(compression_params=params, dict_data=prior).compress(data)
    back = zstandard.ZstdDecompressor(dict_data=prior, max_window_size=1 << window).decompress(out)
    if back != data:
        raise SystemExit("zstd dictionary round trip failed")
    return len(out)


def ppmd_size(sevenzip, blob, order, tmp):
    """Packed size of 7-Zip PPMd fed through stdin. With a file argument 7-Zip silently lowers the model memory
    to the power of two that holds 16x the file size (a 1 MiB file gets 16 MB whatever `mem` says), so the model
    restarts at high orders and C(primer + input) can come out smaller than C(primer); stdin has no known size."""
    arc = os.path.join(tmp, "out.7z")
    if os.path.exists(arc):
        os.remove(arc)
    subprocess.run([sevenzip, "a", "-t7z", f"-m0=PPMd:mem={PPMD_MEM}:o={order}", "-ms=off", "-bd", "-siinput.bin",
                    arc], input=blob, check=True, stdout=subprocess.DEVNULL)
    listing = subprocess.run([sevenzip, "l", "-slt", arc], check=True, capture_output=True, text=True).stdout
    subprocess.run([sevenzip, "t", arc], check=True, stdout=subprocess.DEVNULL)
    return [int(line.split("=")[1]) for line in listing.splitlines() if line.startswith("Packed Size =")][-1]


def classical(args):
    """One row per (primer size, input, codec), appended as soon as it is measured; rows of the pseudo-input
    `(primer)` hold C(primer) of the conditional codecs. Stops after --minutes: run again until DONE."""
    from primer_long import build_primer
    sevenzip = find_sevenzip(sys.argv)
    t0 = time.time()
    have = {}
    if os.path.isfile(OUT):
        have = {(r["comp"], r["primer_bytes"], r["source"], r["codec"]): int(r["out_bytes"])
                for r in csv.DictReader(open(OUT))}
    else:
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(["comp", "primer_bytes", "source", "bytes", "codec", "out_bytes", "bpb"])
    cond = {"xz-9e": lambda blob, tmp: len(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME))}
    cond.update({f"ppmd-o{o}": lambda blob, tmp, o=o: ppmd_size(sevenzip, blob, o, tmp) for o in ORDERS})

    def record(size, name, nbytes, codec, fnc):
        key = (args.comp, str(size), name, codec)
        if key not in have:
            if time.time() - t0 > args.minutes * 60:
                raise TimeoutError
            have[key] = fnc()
            with open(OUT, "a", newline="") as f:
                csv.writer(f).writerow([args.comp, size, name, nbytes, codec, have[key],
                                        f"{have[key] * 8 / nbytes:.4f}"])
        return have[key]

    try:
        with tempfile.TemporaryDirectory() as tmp:
            for size in [int(v) for v in args.sizes.split(",")]:
                primer = build_primer(args.comp, size)
                for name in args.inputs.split(","):
                    data = bytes(load_extended(name) if name in EXTENDED else load_source(name))
                    for level in (19, 22):
                        record(size, name, len(data), f"zstd-{level}-dict", lambda: zstd_dict(level, primer, data))
                    for codec, fnc in cond.items():
                        base = record(size, "(primer)", size, codec, lambda: fnc(primer, tmp))
                        record(size, name, len(data), codec + "-cond", lambda: fnc(primer + data, tmp) - base)
                    print(f"{args.comp} {size:>10,d} {name} done", flush=True)
    except TimeoutError:
        print("PAUSED: run the same command again", flush=True)
        return
    print("DONE", flush=True)


def no_prior():
    """Best classical bpb per selection source without any prior (amendment 4's set; existing CSVs)."""
    best = {}
    for path in ("baselines.csv", "ppmd-baseline.csv"):
        for r in csv.DictReader(open(os.path.join(RUNS, path))):
            if r["role"] == "select" and r["codec"] != "gzip-9":
                best[r["source"]] = min(best.get(r["source"], 99.0), int(r["out_bytes"]) * 8 / int(r["bytes"]))
    return best


def merged(kind, out):
    """Rows of data/strata/runs/<run>/arm-p-<kind>.csv of every seed, also written to data/strata/runs/<out>."""
    rows = []
    for run in SEEDS:
        path = os.path.join(RUNS, run, f"arm-p-{kind}.csv")
        if os.path.isfile(path):
            rows += list(csv.DictReader(open(path)))
    if rows:
        with open(os.path.join(RUNS, out), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def cell(v):
    return "-" if v is None else f"{v:.4f}"


def flat_table(title, rows, sources):
    """One line per (primer, tables, size, rung): mean over the seeds present, per source and macro."""
    acc = {}
    for r in rows:
        key = (r["comp"], r["tables"], int(r["primer_bytes"]), r["tier"])
        acc.setdefault(key, {}).setdefault(r["source"], []).append(float(r["bits"]) / int(r["bytes"]))
    print(f"\n{title}")
    print("| Primer | Tables | Bytes | Rung | Seeds | " + " | ".join(sources) + " | macro |")
    print("|---" * (len(sources) + 6) + "|")
    for (comp, tables, size, tier), per in sorted(acc.items(), key=lambda kv: (kv[0][3] != "cm", kv[0][1], kv[0][2])):
        vals = [sum(per[m]) / len(per[m]) for m in sources]
        print(f"| {comp} | {tables} | {size:,d} | {'R0' if tier == 'cm' else 'R1'} | {len(per[sources[0]])} | "
              + " | ".join(cell(v) for v in vals) + f" | {sum(vals) / len(vals):.4f} |")


def summary(args):
    rows = merged("select", "arm-p-selection.csv")
    if not rows:
        raise SystemExit("no arm-p-select.csv found")
    strata = {}
    for r in rows:
        if r["comp"] in (args.comp, "none"):
            key = (r["source"], int(r["primer_bytes"]), r["tier"], r["tables"])
            strata.setdefault(key, []).append(float(r["bits"]) / int(r["bytes"]))
    cls = {}
    if os.path.isfile(OUT):
        for r in csv.DictReader(open(OUT)):
            if r["comp"] == args.comp and r["source"] != "(primer)":
                kind = "ppmd-cond" if r["codec"].startswith("ppmd") else r["codec"]
                key = (r["source"], int(r["primer_bytes"]), kind)
                cls[key] = min(cls.get(key, 99.0), float(r["out_bytes"]) * 8 / int(r["bytes"]))
    plain = no_prior()
    sources = [n for n, v in SOURCES.items() if v[0] == "select"]
    sizes = sorted({k[1] for k in strata})
    cols = [(t, tb) for t in ("cm", "A") for tb in sorted({k[3] for k in strata})]
    kinds = ("zstd-19-dict", "xz-9e-cond", "ppmd-cond")

    def mean(members, pick):
        vals = [pick(m) for m in members]
        return None if None in vals else sum(vals) / len(vals)

    print(f"primer `{args.comp}`; Strata = mean of {len(SEEDS)} seeds, ideal bits of the integer model, bpb")
    print("| Source | Primer | " + " | ".join(f"{'R0' if t == 'cm' else 'R1'} tables {tb}" for t, tb in cols)
          + " | Best classical, no prior | " + " | ".join(kinds) + " |")
    print("|---" * (len(cols) + len(kinds) + 3) + "|")
    for src in sources + ["macro"]:
        members = sources if src == "macro" else [src]
        for size in sizes:
            vals = [mean(members, lambda m: mean(strata.get((m, size, t, tb), [None]), lambda v: v))
                    for t, tb in cols]
            refs = [mean(members, lambda m: cls.get((m, size, k))) for k in kinds]
            print(f"| {src} | {size:,d} | " + " | ".join(cell(v) for v in vals)
                  + f" | {mean(members, plain.get):.4f} | " + " | ".join(cell(v) for v in refs) + " |")
    flat_table("composition (selection evidence for the choice)", merged("explore", "arm-p-explore.csv"), sources)
    flat_table("parts of the snapshot (diagnostic)", merged("ablate", "arm-p-ablate.csv"), sources)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("classical", "summary"))
    ap.add_argument("--comp", default="mixI")
    ap.add_argument("--sizes", default="65536,1048576,16777216")
    ap.add_argument("--inputs", default=",".join(n for n, v in SOURCES.items() if v[0] == "select"))
    ap.add_argument("--sevenzip")
    ap.add_argument("--minutes", type=float, default=7.5)
    args = ap.parse_args()
    if args.cmd == "classical":
        classical(args)
    else:
        summary(args)


if __name__ == "__main__":
    main()
