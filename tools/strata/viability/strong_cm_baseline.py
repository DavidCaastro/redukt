"""Strata viability, check E2 (amendment 6): public context-mixing compressors on the 13 decision files.

    python -I tools/strata/viability/strong_cm_baseline.py [--only NAME]

Binaries are expected under data/strata/tools-ext/<name>/ (downloaded by hand; sources and sha256 of the archives in
`10-strong-cm-check.md`). One setting per compressor, as registered. Writes data/strata/runs/strong-cm.csv with the
output bytes and the wall seconds of each compression (one process at a time). Finished rows are kept on re-runs.
"""
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
from common import RUNS  # noqa: E402

TOOLS = os.path.join(RUNS, "..", "tools-ext")
INPUTS = os.path.join(RUNS, "mamba-h2h", "in")
WORK = os.path.join(TOOLS, "work")
OUT = os.path.join(RUNS, "strong-cm.csv")
FIELDS = ["source", "bytes", "codec", "out_bytes", "seconds"]


def commands(src, dst):
    """codec label -> (argv, path of the file the tool writes)."""
    t = lambda *p: os.path.join(TOOLS, *p)  # noqa: E731
    return {
        "zpaq-m5": ([t("zpaq", "zpaq64.exe"), "a", dst + ".zpaq", src, "-m5", "-t1"], dst + ".zpaq"),
        "lpaq1-9": ([t("lpaq1", "lpaq1.exe"), "9", src, dst + ".lpaq1"], dst + ".lpaq1"),
        "paq8l-8": ([t("paq8l", "paq8l.exe"), "-8", dst, src], dst + ".paq8l"),
        "paq8px217-8": ([t("paq8px", "paq8px.exe"), "-8", src, dst + ".paq8px217"], dst + ".paq8px217"),
        "bsc-e2": ([t("bsc", "bsc.exe"), "e", src, dst + ".bsc", "-e2"], dst + ".bsc"),
    }


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    os.makedirs(WORK, exist_ok=True)
    done = {}
    if os.path.exists(OUT):
        done = {(r["source"], r["codec"]): r for r in csv.DictReader(open(OUT))}
    for name in sorted(os.listdir(INPUTS)):
        src = os.path.join(INPUTS, name)
        source = name.replace(".bin", "")
        for codec, (argv, produced) in commands(src, os.path.join(WORK, source)).items():
            if (source, codec) in done or (only and codec != only):
                continue
            if os.path.exists(produced):
                os.remove(produced)
            t0 = time.perf_counter()
            subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                           check=True)
            dt = time.perf_counter() - t0
            done[(source, codec)] = {"source": source, "bytes": os.path.getsize(src), "codec": codec,
                                     "out_bytes": os.path.getsize(produced), "seconds": f"{dt:.3f}"}
            print(source, codec, done[(source, codec)]["out_bytes"], f"{dt:.1f}s", flush=True)
            with open(OUT, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(done[k] for k in sorted(done))
    print("DONE", len(done), "rows")


if __name__ == "__main__":
    main()
