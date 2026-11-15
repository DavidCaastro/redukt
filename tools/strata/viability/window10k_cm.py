"""Strata viability, check E4 (amendment 8): the 10 KB headline windows against public CM compressors.

    python -I tools/strata/viability/window10k_cm.py

The first 10,240 bytes of the twelve files of the headline table of `CLAUDE.md`. Redukt: the real `.azth` of the
current binary with its defaults (`--backbone auto --quant auto`, one lockstep batch). Public compressors: lpaq1
`9`, paq8l `-8`, paq8px v217 `-8`, no primer, whole archive as written and the archive minus the tool's size for an
empty input. Classical: brotli -11, xz -9e, gzip -9 from Python. Writes data/strata/runs/window10k-cm.csv and
prints the table.
"""
import csv
import gzip
import lzma
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
sys.path.insert(0, HERE)
from common import BENCH, RUNS  # noqa: E402
from shortmsg_cm import compress  # noqa: E402

FILES = ["industry/apache-access.log", "industry/facturae-batch.xml", "industry/ubl-invoice-batch.xml",
         "silesia/nci", "silesia/dickens", "enwik8", "silesia/samba", "silesia/mozilla", "silesia/mr",
         "silesia/ooffice", "silesia/x-ray", "silesia/sao"]
CODECS = ["lpaq1-9", "paq8l-8", "paq8px217-8"]
WORK = os.path.join(RUNS, "window10k-work")
N = 10240


def main():
    import brotli
    blobs = {os.path.basename(f): open(os.path.join(BENCH, f), "rb").read(N) for f in FILES}
    src, dst = os.path.join(WORK, "in"), os.path.join(WORK, "out")
    for d in (src, dst):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for name, blob in blobs.items():
        with open(os.path.join(src, name), "wb") as f:
            f.write(blob)
    subprocess.run([os.path.join(ROOT, "target", "maxperf", "redukt.exe"), "compress", "--input-dir", src,
                    "--output-dir", dst, "--jobs", "12", "--no-governor"], cwd=ROOT, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, check=True)
    empty = {c: compress(c, b"") for c in CODECS}
    rows = []
    for name, blob in blobs.items():
        row = {"file": name, "redukt-azth": os.path.getsize(os.path.join(dst, name + ".azth")),
               "brotli-11": len(brotli.compress(blob, quality=11)),
               "xz-9e": len(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)),
               "gzip-9": len(gzip.compress(blob, 9))}
        for c in CODECS:
            row[c] = compress(c, blob)
        rows.append(row)
        print(row, flush=True)
    cols = list(rows[0])
    with open(os.path.join(RUNS, "window10k-cm.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("empty-input archive bytes:", empty)
    print("| File | " + " | ".join(cols[1:]) + " | best CM | Redukt vs best CM | Redukt payload vs CM payload |")
    print("|---" * (len(cols) + 3) + "|")
    wins = wins_p = 0
    for r in rows:
        best = min(CODECS, key=lambda c: r[c])
        bestp = min(r[c] - empty[c] for c in CODECS)
        a, p = r["redukt-azth"] / r[best] - 1, (r["redukt-azth"] - 20) / bestp - 1
        wins += a < 0
        wins_p += p < 0
        print(f"| {r['file']} | " + " | ".join(f"{r[c]:,d}" for c in cols[1:]) + f" | {best} | {a:+.1%} | {p:+.1%} |")
    print(f"\nRedukt archive smaller than the best public CM archive: {wins} of {len(rows)}; "
          f"payload against payload: {wins_p} of {len(rows)}")


if __name__ == "__main__":
    main()
