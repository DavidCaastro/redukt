"""Strata bake-off: print the result tables (markdown) from the CSV files written by evaluate.py / curve.py.

    python -I tools/strata/bakeoff/report.py [select|test|curve|train|all]

Reads data/strata/runs/<run>/eval-*.csv, train_log.csv and data/strata/runs/baselines.csv. Smoke runs are skipped.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RUNS  # noqa: E402

ORDER = ["strata_f", "strata_g", "strata_f_float", "strata_g_float", "lstm", "gru", "lstm_a", "gru_a", "lstm_sm"]


def rows_of(pattern):
    out = []
    for run in ORDER:
        for path in sorted(glob.glob(os.path.join(RUNS, run, pattern))):
            out += list(csv.DictReader(open(path)))
    return out


def best_classical(role_filter):
    best = {}
    path = os.path.join(RUNS, "baselines.csv")
    if not os.path.isfile(path):
        return best
    for r in csv.DictReader(open(path)):
        if r["role"] in role_filter:
            key = (r["source"], int(r["bytes"]))
            if key not in best or float(r["bpb"]) < best[key][0]:
                best[key] = (float(r["bpb"]), r["codec"])
    return best


def table(set_name, roles):
    rows = [r for r in rows_of(f"eval-{set_name}-*.csv")]
    if not rows:
        print(f"(no results for set {set_name})")
        return
    sources = []
    for r in rows:
        if r["source"] not in sources:
            sources.append(r["source"])
    groups = {}
    for r in rows:
        key = (r["model"], r["arith"], r["tier"], int(r["mac_per_byte"]), int(r["params"]), r["online"],
               r.get("trunk", "on"))
        groups.setdefault(key, {})[r["source"]] = float(r["bpb"])
    print(f"| model | arith | tier | MAC/byte | params | online | trunk | " + " | ".join(sources) + " | macro |")
    print("|" + "---|" * (8 + len(sources)))
    for key, vals in sorted(groups.items(), key=lambda kv: (kv[0][5] != "on", kv[0][6] != "on", kv[0][2] != "B",
                                                              ORDER.index(kv[0][0]))):
        cells = [f"{vals[s]:.4f}" if s in vals else "" for s in sources]
        macro = sum(vals.values()) / len(vals)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]:,} | {key[4]:,} | {key[5]} | {key[6]} | "
              + " | ".join(cells) + f" | {macro:.4f} |")
    best = best_classical(roles)
    if best:
        by_src = {k[0]: v for k, v in best.items()}
        cells = [f"{by_src[s][0]:.4f} ({by_src[s][1]})" if s in by_src else "" for s in sources]
        vals = [by_src[s][0] for s in sources if s in by_src]
        print(f"| best classical | | | | | | | " + " | ".join(cells) + f" | {sum(vals) / len(vals):.4f} |")


def curve():
    rows = rows_of("eval-curve-*.csv")
    if not rows:
        print("(no curve results)")
        return
    cols = [c for c in rows[0] if c.startswith("bpb")]
    print("| model | tier | source | " + " | ".join(c.replace("bpb@", "") for c in cols) + " | late/early |")
    print("|" + "---|" * (4 + len(cols)))
    for r in rows:
        print(f"| {r['model']} | {r['tier']} | {r['source']} | " + " | ".join(r[c] for c in cols)
              + f" | {r['late/early']} |")
    path = os.path.join(RUNS, "baselines.csv")
    if os.path.isfile(path):
        print("\nclassical codecs on the same prefixes (independent compressions):")
        acc = {}
        for r in csv.DictReader(open(path)):
            if r["role"] == "curve":
                acc.setdefault((r["source"], r["codec"]), {})[int(r["bytes"])] = r["bpb"]
        sizes = sorted({b for v in acc.values() for b in v})
        print("| source | codec | " + " | ".join(str(s) for s in sizes) + " |")
        print("|" + "---|" * (2 + len(sizes)))
        for (src, codec), v in sorted(acc.items()):
            print(f"| {src} | {codec} | " + " | ".join(v.get(s, "") for s in sizes) + " |")


def train():
    print("| run | steps logged | training seconds (wall, shared machine) | last 50-step train bpb per tier |")
    print("|---|---|---|---|")
    for run in ORDER:
        path = os.path.join(RUNS, run, "train_log.csv")
        if not os.path.isfile(path):
            continue
        rows = list(csv.reader(open(path)))[1:]
        last = {}
        for r in rows:
            last[int(r[0])] = r
        step = max(last)
        r = last[step]
        print(f"| {run} | {step} | {r[1]} | {' / '.join(r[3:])} |")
    print("\ntrain bpb (mean of the previous 50 steps) at common steps, largest tier:")
    steps = [250, 500, 1000, 1500, 2000]
    print("| run | " + " | ".join(str(s) for s in steps) + " |")
    print("|" + "---|" * (1 + len(steps)))
    for run in ORDER:
        path = os.path.join(RUNS, run, "train_log.csv")
        if not os.path.isfile(path):
            continue
        last = {int(r[0]): r for r in list(csv.reader(open(path)))[1:]}
        print(f"| {run} | " + " | ".join(last[s][-1] if s in last else "" for s in steps) + " |")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("select", "all"):
        print("### selection set\n")
        table("select", ("select",))
    if what in ("test", "all"):
        print("\n### frozen test set + seen-source slice\n")
        table("test", ("test", "seen"))
    if what in ("curve", "all"):
        print("\n### length curves\n")
        curve()
    if what in ("train", "all"):
        print("\n### training\n")
        train()


if __name__ == "__main__":
    main()
