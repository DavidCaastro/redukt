"""Strata viability, check E2 (amendment 6): tier `cm` against public context-mixing compressors.

    python -I tools/strata/viability/strong_cm_report.py

Reads data/strata/runs/strong-cm.csv (`strong_cm_baseline.py`), the classical CSVs, `arm-p-decision.csv` (R0 = tier
`cm`, G0 weights, ideal code lengths) and the Mamba-130M archives, and applies the registered rule: per set, on the
class `logs`, F = sum over the files of the smallest output among the fast compressors (existing classical set and
members of S measured at >= 200 KB/s); ASSET if B_cm <= 0.95 F on both sets, NO ASSET if B_cm > 1.05 F on either,
else TIE. Cost per raw GB = $0.014 per vCPU-hour x measured seconds per byte (one thread).
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
sys.path.insert(0, HERE)
from arm_p_decision import SETS, classical_no_prior  # noqa: E402
from common import RUNS  # noqa: E402
from ladder_verdict import mamba_bytes  # noqa: E402
from viability import P_CPU, value_per_gb  # noqa: E402

FAST_KBS = 200.0
CM_US = (0.53, 1.45)        # us per byte of the speed proxy: one thread alone, 12 threads loaded


def load():
    rows = list(csv.DictReader(open(os.path.join(RUNS, "strong-cm.csv"))))
    size = {(r["source"], r["codec"]): int(r["out_bytes"]) for r in rows}
    secs = {(r["source"], r["codec"]): float(r["seconds"]) for r in rows}
    raw = {r["source"]: int(r["bytes"]) for r in rows}
    cm = {}
    for r in csv.DictReader(open(os.path.join(RUNS, "arm-p-decision.csv"))):
        if r["arm"] == "none" and r["tables"] == "D" and r["rung"] == "R0":
            cm.setdefault(r["source"], []).append(float(r["strata_bytes"]))
    return size, secs, raw, {s: sum(v) / len(v) for s, v in cm.items()}


def main():
    size, secs, raw, cm = load()
    plain = classical_no_prior()
    codecs = sorted({c for _, c in size})
    big = [s for s in raw if raw[s] == 1 << 20]
    us = {c: sum(secs[(s, c)] for s in big) / sum(raw[s] for s in big) * 1e6 for c in codecs}
    fast = [c for c in codecs if 1e3 / us[c] >= FAST_KBS]
    print("### speed (seven 1 MiB files, one process at a time, wall time including start-up)")
    print("| Compressor | us per byte | KB/s | Cost $ per raw GB | Fast class |")
    print("|---|---|---|---|---|")
    for c in codecs:
        print(f"| {c} | {us[c]:.2f} | {1e3 / us[c]:,.0f} | {P_CPU[0] * 1e9 * us[c] * 1e-6 / 3600:.4f} | "
              f"{'yes' if c in fast else 'no'} |")
    print(f"| tier `cm` (speed proxy) | {CM_US[0]} - {CM_US[1]} | {1e3 / CM_US[1]:,.0f} - {1e3 / CM_US[0]:,.0f} | "
          f"{P_CPU[0] * 1e3 * CM_US[0] / 3600:.4f} - {P_CPU[0] * 1e3 * CM_US[1] / 3600:.4f} | - |")
    verdicts = []
    for set_name, table in SETS:
        for cls in dict.fromkeys(table.values()):
            members = [n for n, c in table.items() if c == cls]
            n = sum(raw[m] for m in members)
            b_cl = sum(plain[m][0] for m in members)
            b_cm = sum(cm[m] for m in members)
            b_mb = sum(mamba_bytes(m) for m in members)
            f_fast = sum(min([plain[m][0]] + [size[(m, c)] for c in fast]) for m in members)
            print(f"\n### {set_name}, class `{cls}`: {', '.join(members)}; {n:,d} raw bytes")
            print("| Compressor | Bytes | vs old best classical | vs tier `cm` | vs Mamba-130M | Value $/GB | "
                  "Cost $/GB | Value - cost |")
            print("|---|---|---|---|---|---|---|---|")

            def line(label, b, cost):
                value = value_per_gb(8 * (b_cl - b) / n)
                net = "-" if cost is None else f"{value - cost:+.4f}"
                shown = "-" if cost is None else f"{cost:.4f}"
                print(f"| {label} | {b:,.0f} | {b / b_cl - 1:+.1%} | {b / b_cm - 1:+.1%} | {b / b_mb - 1:+.1%} | "
                      f"{value:+.4f} | {shown} | {net} |")

            line("old best classical (per file)", b_cl, None)
            line("tier `cm` (R0, ideal code length)", b_cm, P_CPU[0] * 1e3 * CM_US[1] / 3600)
            for c in codecs:
                line(c, sum(size[(m, c)] for m in members), P_CPU[0] * 1e9 * us[c] * 1e-6 / 3600)
            line("best fast compressor per file (F)", f_fast, None)
            line("Mamba-130M Q8 (real `.azth`)", b_mb, None)
            ratio = b_cm / f_fast
            res = "ASSET" if ratio <= 0.95 else "NO ASSET" if ratio > 1.05 else "TIE"
            print(f"\nB_cm / F = {ratio:.3f} -> {res}" + ("" if cls == "logs" else " (reported, no vote)"))
            if cls == "logs":
                verdicts.append(res)
    final = "NO ASSET" if "NO ASSET" in verdicts else "ASSET" if all(v == "ASSET" for v in verdicts) else "TIE"
    print(f"\n### check E2, class `logs`: {' / '.join(verdicts)} -> {final}")
    print("\n### per file, bytes")
    names = [n for _, table in SETS for n in table]
    print("| Compressor | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    print("| old best classical | " + " | ".join(f"{plain[n][0]:,d}" for n in names) + " |")
    print("| tier `cm` | " + " | ".join(f"{cm[n]:,.0f}" for n in names) + " |")
    for c in codecs:
        print(f"| {c} | " + " | ".join(f"{size[(n, c)]:,d}" for n in names) + " |")
    print("| Mamba-130M | " + " | ".join(f"{mamba_bytes(n):,d}" for n in names) + " |")


if __name__ == "__main__":
    main()
