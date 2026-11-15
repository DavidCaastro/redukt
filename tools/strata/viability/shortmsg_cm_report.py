"""Strata viability, check E3 (amendment 7): report of the primed public CM compressors on the R-113 messages.

    python -I tools/strata/viability/shortmsg_cm_report.py

Reads data/strata/runs/shortmsg-cm.csv (`shortmsg_cm.py`), the R-113 per-message file (Mamba-130M archives and the
classical codecs measured on Kaggle) and the Strata short-message files of the bake-off (classical codecs measured
here on the same cut, zstd with a foreign dictionary, Strata `cm` / `msg88` with a 65,536-byte primer). Applies the
registered rule: per class and pairing, B_cm = smallest class total among the members of S; Mamba KEEPS if its
payload total <= 0.90 B_cm, LOST if >= 1.00 B_cm, else MARGINAL.
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
from common import RUNS  # noqa: E402

R113_CSV = os.path.join(RUNS, "..", "..", "benchmark", "results", "short-message-bench", "run-1",
                        "short-message-results.csv")
SIZES = [128, 256, 512, 1024, 2048]
CLASSES = {"text": ["wikipedia", "prose"], "logs": ["log-apache", "log-linux", "log-openssh"],
           "json": ["json-events"]}
S = ["lpaq1-9", "lpaq1-0", "paq8l-8", "paq8px217-8"]
FRAMING = 20        # bytes of header + CRC-32 in the R-113 Mamba archives
CLASSICAL = ("gzip-9-raw", "xz-9e-raw", "brotli-11", "zstd-19")


def load():
    """{tool: {(source, size, index): bytes}}"""
    data, ref = {}, {}
    for r in csv.DictReader(open(R113_CSV)):
        key = (r["source"], int(r["size"]), int(r["index"]))
        ref[(key, r["tool"])] = int(r["out_bytes"])
        if r["tool"] == "redukt-azth":
            data.setdefault("Mamba-130M archive", {})[key] = int(r["out_bytes"])
            data.setdefault("Mamba-130M payload", {})[key] = int(r["out_bytes"]) - FRAMING
    same = diff = 0
    for path in sorted(glob.glob(os.path.join(RUNS, "strata_g_bc", "shortmsg-r113-t[01]-*.csv"))):
        for r in csv.DictReader(open(path)):
            key = (r["source"], int(r["size"]), int(r["index"]))
            if r["tool"] in CLASSICAL and "t1-same" in path:
                same += ref[(key, r["tool"])] == int(r["bytes"])
                diff += ref[(key, r["tool"])] != int(r["bytes"])
            data.setdefault(r["tool"], {})[key] = int(r["bytes"])
    for r in csv.DictReader(open(os.path.join(RUNS, "shortmsg-cm.csv"))):
        data.setdefault(r["tool"], {})[(r["source"], int(r["size"]), int(r["index"]))] = int(r["bytes"])
    print(f"alignment with the R-113 file (gzip, xz, brotli, zstd without dictionary, local against Kaggle): "
          f"{same} sizes identical, {diff} different")
    return data


def total(table, sources, size=None):
    keys = [k for k in table if k[0] in sources and (size is None or k[1] == size)]
    return sum(table[k] for k in keys), len(keys)


def verdict(mamba, b_cm):
    return "KEEPS" if mamba <= 0.90 * b_cm else "LOST" if mamba >= b_cm else "MARGINAL"


def main():
    data = load()
    mamba = data["Mamba-130M payload"]
    final = {}
    for cls, sources in CLASSES.items():
        m, n = total(mamba, sources)
        raw = sum(k[1] for k in mamba if k[0] in sources)
        print(f"\n### class `{cls}`: {', '.join(sources)}; {n} messages, {raw:,d} raw bytes")
        print("| Arm | Total bytes | % of raw | vs Mamba payload |")
        print("|---|---|---|---|")
        order = (["Mamba-130M payload", "Mamba-130M archive", "brotli-11", "zstd-19-dict", "zstd-19-fdict"]
                 + sorted(t for t in data if t.startswith("strata"))
                 + sorted(t for t in data if "+" in t and not t.endswith("8k")))
        for tool in order:
            if tool not in data:
                continue
            b, k = total(data[tool], sources)
            if k == n:
                print(f"| {tool} | {b:,d} | {b / raw:.1%} | {b / m - 1:+.1%} |")
        res = []
        for pairing in ("same110k", "foreign110k"):
            best = min(S, key=lambda c: total(data[f"{c}+{pairing}"], sources)[0])
            b_cm = total(data[f"{best}+{pairing}"], sources)[0]
            res.append(verdict(m, b_cm))
            print(f"\n{pairing}: B_cm = {b_cm:,d} ({best}); Mamba payload / B_cm = {m / b_cm:.3f} -> {res[-1]}")
            wins = [sum(total(mamba, sources, s)[0] < total(data[f"{best}+{pairing}"], sources, s)[0]
                        for s in SIZES)]
            by_size = ", ".join(f"{s}: {total(mamba, sources, s)[0] / total(data[f'{best}+{pairing}'], sources, s)[0]:.2f}"
                                for s in SIZES)
            print(f"  ratio by message size ({wins[0]} of 5 below 1): {by_size}")
            with_frame = total(data["Mamba-130M archive"], sources)[0] / (b_cm + 4 * n)
            print(f"  real archive against CM payload + 4 B of framing: {with_frame:.3f}")
        final[cls] = (res[0] if res[0] == res[1] and res[0] != "MARGINAL" else
                      "MARGINAL" if res[0] == res[1] else f"DEPENDS ON THE PRIMER ({res[0]} same / {res[1]} foreign)")
    print("\n### check E3")
    for cls, v in final.items():
        print(f"- `{cls}`: {v}")
    for pairing in ("same110k", "foreign110k"):
        print(f"\n### mean bytes per message, {pairing} (cells = source x size)")
        cols = (["Mamba-130M payload", "Mamba-130M archive", "zstd-19-dict" if pairing == "same110k" else "zstd-19-fdict"]
                + [f"{c}+{pairing}" for c in S])
        print("| Source | Size | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 2) + "|")
        cells = smallest = 0
        for sources in CLASSES.values():
            for src in sources:
                for size in SIZES:
                    vals = [total(data[c], [src], size) for c in cols]
                    means = [b / k for b, k in vals]
                    cells += 1
                    smallest += means[0] < min(means[3:])
                    print(f"| {src} | {size} | " + " | ".join(f"{v:.1f}" for v in means) + " |")
        print(f"\nMamba's payload is smaller than every member of S in {smallest} of {cells} cells")
    print("\n### lpaq1: memory and primer (total bytes over the 480 messages)")
    every = [s for v in CLASSES.values() for s in v]
    for tool in sorted(t for t in data if t.startswith("lpaq1") and not t.endswith("8k")):
        print(f"- {tool}: {total(data[tool], every)[0]:,d}")
    print(f"- Mamba-130M payload: {total(mamba, every)[0]:,d}")
    e3b(data)


def e3b(data):
    """Amendment 8: Mamba-130M primed by difference (8,192 bytes) on the arms it did not keep in E3."""
    mamba = data["Mamba-130M payload"]
    print("\n### check E3b: Mamba-130M primed with 8,192 bytes (payload by difference)")
    print("| Class, pairing | Mamba unprimed | Mamba primed | B_cm of E3 (110 KB) | primed / B_cm | Verdict | "
          "lpaq1-0, 8 KB primer | paq8px, 8 KB primer |")
    print("|---|---|---|---|---|---|---|---|")
    for cls, kind in (("logs", "same"), ("json", "same"), ("json", "foreign")):
        sources = CLASSES[cls]
        primed = data.get(f"mamba130m+{kind}8k", {})
        got, n = total(primed, sources)
        if n != total(mamba, sources)[1]:
            print(f"| {cls}, {kind} | not measured |")
            continue
        b_cm = min(total(data[f"{c}+{kind}110k"], sources)[0] for c in S)
        eq = [total(data[f"{c}+{kind}8k"], sources)[0] for c in ("lpaq1-0", "paq8px217-8")]
        print(f"| {cls}, {kind} | {total(mamba, sources)[0]:,d} | {got:,d} | {b_cm:,d} | {got / b_cm:.3f} | "
              f"{verdict(got, b_cm)} | {eq[0]:,d} | {eq[1]:,d} |")
        for src in sources:
            cells = ", ".join(f"{s}: {total(mamba, [src], s)[0] / 16:.1f} -> {total(primed, [src], s)[0] / 16:.1f}"
                              f" ({total(data[f'paq8px217-8+{kind}110k'], [src], s)[0] / 16:.1f})" for s in SIZES)
            print(f"  - {src}, mean bytes unprimed -> primed (paq8px with 110 KB): {cells}")


if __name__ == "__main__":
    main()
