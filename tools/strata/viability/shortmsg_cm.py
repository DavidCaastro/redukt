"""Strata viability, check E3 (amendment 7): primed public context-mixing compressors on the R-113 short messages.

    python -I tools/strata/viability/shortmsg_cm.py [--workers 4] [--only CODEC] [--neutral FILE]

The public compressors have no dictionary option, so a primed message is measured by difference: the payload of
message m after primer D is size(C(D || m)) - size(C(D)), the bytes an adaptive coder adds for m once its state has
read D (what a codec that ships the snapshot of that state would emit, up to the coder flush). With no primer D is
empty. Every run uses the same file names, so the header of the tool cancels.
Messages, sources and the foreign pairing are those of `tools/strata/bakeoff/shortmsg.py` (R-113 protocol).
Primers: `same110k` = last 110,000 bytes of the first half of the message source (the bytes the zstd dictionary
of R-113 is trained on, same budget); `foreign110k` = the same position in the foreign source; `samehalf` = the
whole first half; `zdict` = the content of the trained zstd dictionary itself.
Rows are appended to data/strata/runs/shortmsg-cm.csv (source, size, index, tool, bytes); finished
(source, tool) pairs are skipped. `--neutral FILE` runs every arm on one message cut from FILE and writes nothing.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
from common import RUNS  # noqa: E402
from shortmsg import FOREIGN, PATHS, PER_CELL, R113, SIZES, cut_messages, read_source  # noqa: E402

TOOLS = os.path.join(RUNS, "..", "tools-ext")
OUT = os.path.join(RUNS, "shortmsg-cm.csv")
PRIMER = 110_000
ARMS = [("lpaq1-9", "none"), ("lpaq1-9", "same110k"), ("lpaq1-9", "foreign110k"), ("lpaq1-9", "samehalf"),
        ("lpaq1-9", "zdict"), ("lpaq1-0", "same110k"), ("lpaq1-0", "foreign110k"),
        ("paq8l-8", "same110k"), ("paq8l-8", "foreign110k"),
        ("paq8px217-8", "same110k"), ("paq8px217-8", "foreign110k")]
LOCK = threading.Lock()


def compress(codec, data, keep=None):
    """Size in bytes of `data` compressed by `codec`; `keep` = path that receives the archive."""
    t = lambda *p: os.path.join(TOOLS, *p)  # noqa: E731
    work = tempfile.mkdtemp(prefix="smcm")
    try:
        with open(os.path.join(work, "in.bin"), "wb") as f:
            f.write(data)
        argv, produced = {
            "lpaq1-9": ([t("lpaq1", "lpaq1.exe"), "9", "in.bin", "out.bin"], "out.bin"),
            "lpaq1-0": ([t("lpaq1", "lpaq1.exe"), "0", "in.bin", "out.bin"], "out.bin"),
            "paq8l-8": ([t("paq8l", "paq8l.exe"), "-8", "out", "in.bin"], "out.paq8l"),
            "paq8px217-8": ([t("paq8px", "paq8px.exe"), "-8", "in.bin", "out.bin"], "out.bin"),
        }[codec]
        subprocess.run(argv, cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       stdin=subprocess.DEVNULL, check=True)
        if keep:
            shutil.copy(os.path.join(work, produced), keep)
        return os.path.getsize(os.path.join(work, produced))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def primer_bytes(kind, blob, fblob):
    half, fhalf = len(blob) // 2, len(fblob) // 2
    if kind == "none":
        return b""
    if kind == "same110k":
        return blob[max(0, half - PRIMER):half]
    if kind == "foreign110k":
        return fblob[max(0, fhalf - PRIMER):fhalf]
    if kind == "samehalf":
        return blob[:half]
    import zstandard
    from shortmsg import _dictionary
    return _dictionary(zstandard, blob[:half]).as_bytes()


def run_arm(pool, codec, kind, blob, fblob):
    """[(size, index, payload bytes)] for the 80 messages of one source."""
    primer = primer_bytes(kind, blob, fblob)
    base = compress(codec, primer)
    jobs = [(size, i, msg) for size in SIZES for i, msg in enumerate(cut_messages(blob, size, PER_CELL))]
    sizes = list(pool.map(lambda j: compress(codec, primer + j[2]), jobs))
    return [(size, i, total - base) for (size, i, _), total in zip(jobs, sizes)], len(primer), base


def neutral(path, pool):
    blob = open(path, "rb").read()
    half = len(blob) // 2
    msg = blob[half:half + 512]
    for codec, kind in ARMS:
        primer = b"" if kind == "none" else blob[max(0, half - PRIMER):half]
        base = compress(codec, primer)
        got = [compress(codec, primer + m) - base for m in (b"", msg[:128], msg)]
        print(f"{codec:12s} {kind:12s} primer {len(primer):>7,d} B -> {base:>6,d} B; empty / 128 / 512 B message: {got}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only")
    ap.add_argument("--neutral")
    args = ap.parse_args()
    pool = ThreadPoolExecutor(args.workers)
    if args.neutral:
        neutral(args.neutral, pool)
        return
    done = set()
    if os.path.isfile(OUT):
        done = {(r["source"], r["tool"]) for r in csv.DictReader(open(OUT))}
    else:
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(["source", "size", "index", "tool", "bytes"])
    for codec, kind in ARMS:
        tool = f"{codec}+{kind}"
        for src in R113:
            if (src, tool) in done or (args.only and codec != args.only):
                continue
            blob, fblob = read_source(PATHS[src]), read_source(FOREIGN[src])
            rows, plen, base = run_arm(pool, codec, kind, blob, fblob)
            with LOCK, open(OUT, "a", newline="") as f:
                csv.writer(f).writerows([src, size, i, tool, b] for size, i, b in rows)
            print(f"{src} {tool} primer {plen} B -> {base} B; total payload {sum(r[2] for r in rows)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
