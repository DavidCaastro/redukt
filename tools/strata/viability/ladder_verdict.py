"""Strata viability, gate V7: verdict of the capacity ladder (arm W, arm P, arm P-s) by the registered rule.

    python -I tools/strata/viability/ladder_verdict.py

Inputs (all under data/strata/, not in git):
  runs-ladder/kaggle-long-v3/run-ladder3-s<seed>/eval-decision.csv   arm W, rungs R1-R3 (`train/run_eval.py`)
  runs/arm-p-decision.csv                                            R0 and the G0 R1, arm P, arm P-s
  runs/mamba-h2h/out/<source>.bin.azth                               Mamba-130M Q8 on the identical bytes
  the classical CSVs read by `arm_p_decision.classical_no_prior`

Rule (05-gate-amendments.md, amendments 4 and 5): per class, B_r = mean over the seeds of the coded bytes, B_cl =
sum of the best classical codec of each file, delta = 8 (B_cl - B_r) / N, saving = 1 - B_r / B_cl, parity =
B_r / B_mamba; cost passes if delta >= needed_delta(MAC) at 10 G MAC/s and $0.014; material if saving >= 10 %;
parity if <= 1.10. O1 = all three, O2 = cost and material, O3 = otherwise; a class verdict is the worse of the frozen
test set and the extended set. Primed rungs are judged at their primary point only.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
sys.path.insert(0, HERE)
from arm_p_decision import PRIMARY, SETS, classical_no_prior  # noqa: E402
from common import RUNS  # noqa: E402
from viability import MATERIAL, P_CPU, PARITY, R_MAC, cost_per_gb, needed_delta, value_per_gb  # noqa: E402

LADDER = os.path.join(RUNS, "..", "runs-ladder", "kaggle-long-v3")
SEEDS = (1204, 1205)
WIDTH_RUNG = {88: "R1", 208: "R2", 456: "R3"}
RANK = {"O1": 0, "O2": 1, "CM only": 1, "O3": 2}


def load():
    """(config label, source) -> [coded bytes per seed]; config label -> (MAC/byte, kind)."""
    cells, meta = {}, {}
    for seed in SEEDS:
        path = os.path.join(LADDER, f"run-ladder3-s{seed}", "eval-decision.csv")
        for r in csv.DictReader(open(path)):
            label = f"W {WIDTH_RUNG[int(r['width'])]} width {r['width']}, retrained"
            meta[label] = (int(r["mac_per_byte"]), "trunk")
            cells.setdefault((label, r["source"]), []).append(float(r["bits"]) / 8)
    for r in csv.DictReader(open(os.path.join(RUNS, "arm-p-decision.csv"))):
        arm, size = r["arm"], int(r["primer_bytes"])
        if arm == "none" and r["tables"] == "D":
            label = "R0 cm, no primer" if r["rung"] == "R0" else "G0 R1 width 88 (33.55 MB training)"
            kind = "cm" if r["rung"] == "R0" else "reference"
        elif (arm, size) in PRIMARY:
            label, kind = f"{arm} {r['rung']}, primer {size:,d}", "primed"
        else:
            continue
        meta[label] = (int(r["mac_per_byte"]), kind)
        cells.setdefault((label, r["source"]), []).append(float(r["strata_bytes"]))
    return cells, meta


def mamba_bytes(source):
    return os.path.getsize(os.path.join(RUNS, "mamba-h2h", "out", f"{source}.bin.azth"))


def outcome(kind, cost, material, parity):
    if not (cost and material):
        return "O3"
    if kind == "cm":
        return "CM only"
    return "O1" if parity else "O2"


def main():
    cells, meta = load()
    plain = classical_no_prior()
    sizes = {}
    for r in csv.DictReader(open(os.path.join(RUNS, "arm-p-decision.csv"))):
        sizes[r["source"]] = int(r["bytes"])
    order = sorted(meta, key=lambda k: (("cm", "reference", "trunk", "primed").index(meta[k][1]), meta[k][0], k))
    worst, totals = {}, {}
    print(f"decision point {R_MAC[1] / 1e9:.0f} G MAC/s, ${P_CPU[0]} per vCPU-hour; materiality {MATERIAL:.0%}; "
          f"parity {PARITY}")
    for set_name, table in SETS:
        for cls in dict.fromkeys(table.values()):
            members = [n for n, c in table.items() if c == cls]
            raw = sum(sizes[m] for m in members)
            b_cl = sum(plain[m][0] for m in members)
            b_mb = sum(mamba_bytes(m) for m in members)
            print(f"\n### {set_name}, class `{cls}`: {', '.join(members)}; {raw:,d} raw bytes; B_cl {b_cl:,d}; "
                  f"Mamba-130M {b_mb:,d} ({b_mb / b_cl - 1:+.1%} vs classical)")
            print("| Configuration | MAC/byte | Coded bytes (1204 / 1205) | B_r | Saving | delta bpb | Needed | "
                  "Value $/GB | Cost $/GB | Cost | Material | Parity | Outcome |")
            print("|---" * 13 + "|")
            for label in order:
                mac, kind = meta[label]
                per_seed = [sum(cells[(label, m)][i] for m in members) for i in range(len(SEEDS))]
                b_r = sum(per_seed) / len(per_seed)
                delta, saving, par = 8 * (b_cl - b_r) / raw, 1 - b_r / b_cl, b_r / b_mb
                need = needed_delta(mac, R_MAC[1], P_CPU[0])
                res = outcome(kind, delta >= need, saving >= MATERIAL, par <= PARITY)
                totals[(set_name, cls, label)] = b_r
                if kind != "reference":
                    worst.setdefault((cls, label), []).append((set_name, res))
                print(f"| {label} | {mac:,d} | " + " / ".join(f"{v:,.0f}" for v in per_seed)
                      + f" | {b_r:,.0f} | {saving:+.1%} | {delta:+.4f} | {need:.4f} | {value_per_gb(delta):+.4f} | "
                      f"{cost_per_gb(mac, R_MAC[1], P_CPU[0]):.4f} | {'yes' if delta >= need else 'no'} | "
                      f"{'yes' if saving >= MATERIAL else 'no'} | {par:.2f} | {res} |")
    print("\n### class verdicts (worse of the sets a class appears in)")
    print("| Class | Configuration | Verdict | Per set |")
    print("|---|---|---|---|")
    for (cls, label), seen in sorted(worst.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1]))):
        res = max((r for _, r in seen), key=RANK.get)
        per_set = "; ".join(f"{n}: {r}" for n, r in seen)
        print(f"| {cls} | {label} | {res} | {per_set} |")
    print("\n### stop rule 2: R2 -> R3 change of the class total (R4 is trained only if one class gains >= 2 %)")
    r2, r3 = (next(k for k in order if f"W {r} " in k) for r in ("R2", "R3"))
    r1 = next(k for k in order if "W R1 " in k)
    g0 = next(k for k in order if k.startswith("G0 R1"))
    print("| Set | Class | R1 -> R2 | R2 -> R3 | G0 R1 -> retrained R1 (43x the training bytes) |")
    print("|---|---|---|---|---|")
    best = 0.0
    for set_name, table in SETS:
        for cls in dict.fromkeys(table.values()):
            t = {k: totals[(set_name, cls, k)] for k in (g0, r1, r2, r3)}
            step = t[r3] / t[r2] - 1
            best = min(best, step) if cls != "xml-synthetic" else best
            print(f"| {set_name} | {cls} | {t[r2] / t[r1] - 1:+.2%} | {step:+.2%} | {t[r1] / t[g0] - 1:+.2%} |")
    print(f"largest R2 -> R3 gain among voting classes: {-best:.2%} -> the condition for R4 is "
          f"{'MET' if -best >= 0.02 else 'NOT met'}")
    mac4 = 1_961_768
    print(f"R4 (`r4`, {mac4:,d} MAC/byte) would need delta >= {needed_delta(mac4, R_MAC[1], P_CPU[0]):.3f} bpb at "
          f"10 G MAC/s and {needed_delta(mac4, R_MAC[2], P_CPU[0]):.3f} at 20 G")


if __name__ == "__main__":
    main()
