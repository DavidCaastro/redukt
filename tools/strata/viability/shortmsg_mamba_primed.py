"""Strata viability, check E3b (amendment 8): Mamba-130M primed by difference on the R-113 short messages.

    python -I tools/strata/viability/shortmsg_mamba_primed.py [--neutral FILE] [--jobs 32]

The product has no primer option, so the primed payload is measured the same way as for the public CM compressors
in `shortmsg_cm.py`: size(azth(D || m)) - size(azth(D)), real archives of the CPU product path
(`redukt compress --input-dir`, Mamba-130M Q8 + CM, `--reset-every 0` so that no state reset falls between the
primer and the message). Header and CRC cancel. D is the last 8,192 bytes of the first half of the message source
(`same8k`) or of the foreign source (`foreign8k`): the model's usable context, not the 110,000 bytes given to the
online coders. Registered arms: logs same8k, json same8k, json foreign8k.
Rows are appended to data/strata/runs/shortmsg-cm.csv with tool `mamba130m+<primer>`; finished (source, tool)
pairs are skipped. `--neutral FILE` runs three messages cut from FILE and writes nothing.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
from common import RUNS  # noqa: E402
from shortmsg import FOREIGN, PATHS, PER_CELL, SIZES, cut_messages, read_source  # noqa: E402

OUT = os.path.join(RUNS, "shortmsg-cm.csv")
WORK = os.path.join(RUNS, "shortmsg-mamba-work")
REDUKT = os.path.join(ROOT, "target", "maxperf", "redukt.exe")
PRIMER = 8192
ARMS = [("log-apache", "same8k"), ("log-linux", "same8k"), ("log-openssh", "same8k"), ("json-events", "same8k"),
        ("json-events", "foreign8k")]


def azth_sizes(files, jobs):
    """{name: archive bytes} for {name: bytes}, one lockstep batch."""
    src, dst = os.path.join(WORK, "in"), os.path.join(WORK, "out")
    for d in (src, dst):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for name, blob in files.items():
        with open(os.path.join(src, name), "wb") as f:
            f.write(blob)
    subprocess.run([REDUKT, "compress", "--input-dir", src, "--output-dir", dst, "--jobs", str(jobs),
                    "--backbone", "mamba", "--quant", "q8", "--reset-every", "0", "--no-governor",
                    "--weights", os.path.join(ROOT, "weights", "mamba-130m")],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                   check=True)
    return {n: os.path.getsize(os.path.join(dst, n + ".rdkt")) for n in files}


def primer_of(src, kind):
    blob = read_source(PATHS[src] if kind.startswith("same") else FOREIGN[src])
    half = len(blob) // 2
    return blob[half - PRIMER:half]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    if args.neutral:
        blob = open(args.neutral, "rb").read()
        half = len(blob) // 2
        d, m = blob[half - PRIMER:half], blob[half:half + 512]
        t0 = time.time()
        s = azth_sizes({"p": d, "p0": d, "a": d + m[:128], "b": d + m, "m": m}, args.jobs)
        print(f"primer {s['p']} B (twice: {s['p0']}); +128 B: {s['a'] - s['p']}; +512 B: {s['b'] - s['p']}; "
              f"512 B alone: {s['m']} B archive; {time.time() - t0:.0f} s")
        return
    done = {(r["source"], r["tool"]) for r in csv.DictReader(open(OUT))}
    for src, kind in ARMS:
        tool = f"mamba130m+{kind}"
        if (src, tool) in done:
            continue
        d = primer_of(src, kind)
        blob = read_source(PATHS[src])
        files = {"primer": d}
        for size in SIZES:
            for i, msg in enumerate(cut_messages(blob, size, PER_CELL)):
                files[f"m_{size:04d}_{i:02d}"] = d + msg
        t0 = time.time()
        sizes = azth_sizes(files, args.jobs)
        rows = [[src, int(n[2:6]), int(n[7:9]), tool, b - sizes["primer"]] for n, b in sizes.items() if n != "primer"]
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"{src} {tool} primer -> {sizes['primer']} B; total payload {sum(r[4] for r in rows)}; "
              f"{time.time() - t0:.0f} s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
