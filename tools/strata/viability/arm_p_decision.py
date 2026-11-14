"""Strata viability, gate V7: arm P and arm P-s on the decision files, as registered by amendment 5.

    python -I tools/strata/viability/arm_p_decision.py

Reads the Strata rows that `primer_long.py code --allow-decision --out arm-p-decision-<...>.part.csv` wrote under
data/strata/runs/<run>/ and the classical CSVs (`baselines.csv`, `ppmd-baseline.csv`, `ppmd-baseline-1g.csv`,
`baselines-ext.csv`, `arm-p-classical.csv`), writes data/strata/runs/arm-p-decision.csv (one row per source x rung x
seed x configuration) and prints the per-class tables of `09c-arm-p-decision.md`.

Rules applied (05-gate-amendments.md, amendments 4 and 5):
  best classical of a file  = smallest of xz -9e, zstd -19 / -22, brotli -11, bz2 -9, PPMd (file mode and stdin 1 GB)
  classical side of a primed rung = per file, the smaller of that and zstd -19 / -22 with the primer as dictionary
  class totals: B_r = coded bytes of the rung (ideal bits / 8, mean of the seeds), B_cl, N raw bytes
  delta = 8 (B_cl - B_r) / N;  saving = 1 - B_r / B_cl;  cost, value and the viability inequality from `viability.py`
  at 10 G MAC/s and its first price. Parity against Mamba-130M is left blank (Mamba is run afterwards).
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
sys.path.insert(0, HERE)
from common import RUNS, SOURCES  # noqa: E402
from primer_baselines import SEEDS  # noqa: E402
from sources import EXTENDED  # noqa: E402
from viability import MATERIAL, P_CPU, R_MAC, cost_per_gb, needed_delta, value_per_gb  # noqa: E402

R_DECISION = R_MAC[1]                       # 10 G MAC/s, the decision point of V7
ARMS = {"none": "none", "mixI": "P", "mixI/keep-succ": "P-s"}
PRIMARY = {("P", 1_048_576), ("P-s", 16_777_216)}
SETS = (("frozen test", {n: v[1] for n, v in SOURCES.items() if v[0] == "test"}),
        ("extended", {n: v[0] for n, v in EXTENDED.items()}))


def classical_no_prior():
    """source -> (bytes, codec) of the registered classical set."""
    best = {}
    for name in ("baselines.csv", "ppmd-baseline.csv", "ppmd-baseline-1g.csv", "baselines-ext.csv"):
        for r in csv.DictReader(open(os.path.join(RUNS, name))):
            if r["codec"] != "gzip-9" and int(r["out_bytes"]) < best.get(r["source"], (1 << 60, ""))[0]:
                best[r["source"]] = (int(r["out_bytes"]), r["codec"])
    return best


def classical_same_prior():
    """(source, primer bytes) -> {"zstd": bytes, "xz": bytes, "ppmd": bytes} (dictionary; conditional sizes)."""
    out = {}
    for r in csv.DictReader(open(os.path.join(RUNS, "arm-p-classical.csv"))):
        if r["comp"] != "mixI" or r["source"] == "(primer)":
            continue
        kind = "zstd" if r["codec"].startswith("zstd") else "xz" if r["codec"].startswith("xz") else "ppmd"
        cell = out.setdefault((r["source"], int(r["primer_bytes"])), {})
        cell[kind] = min(cell.get(kind, 1 << 60), int(r["out_bytes"]))
    return out


def load_rows():
    plain, prior = classical_no_prior(), classical_same_prior()
    classes = {n: (s, c) for s, table in SETS for n, c in table.items()}
    rows, seen = [], set()
    for run in SEEDS:
        for path in sorted(glob.glob(os.path.join(RUNS, run, "arm-p-decision-*.part.csv"))):
            for r in csv.DictReader(open(path)):
                key = (run, r["tier"], r["comp"], r["tables"], r["primer_bytes"], r["source"])
                if key in seen:
                    continue
                seen.add(key)
                size, src = int(r["primer_bytes"]), r["source"]
                same = prior.get((src, size), {}) if size else {}
                side = min(plain[src][0], same.get("zstd", 1 << 60))
                rows.append({
                    "set": classes[src][0], "class": classes[src][1], "source": src, "bytes": int(r["bytes"]),
                    "rung": "R0" if r["tier"] == "cm" else "R1", "mac_per_byte": int(r["mac_per_byte"]),
                    "seed": int(r["seed"]), "arm": ARMS[r["comp"]], "tables": r["tables"], "primer_bytes": size,
                    "state_bytes": int(r["state_bytes"]), "bits": float(r["bits"]),
                    "bpb": float(r["bits"]) / int(r["bytes"]), "strata_bytes": float(r["bits"]) / 8,
                    "classical_bytes": plain[src][0], "classical_codec": plain[src][1],
                    "zstd_dict_bytes": same.get("zstd", ""), "classical_side_bytes": side,
                    "xz_cond_bytes": same.get("xz", ""), "ppmd_cond_bytes": same.get("ppmd", "")})
    return rows


def write_csv(rows):
    path = os.path.join(RUNS, "arm-p-decision.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "bits": f"{r['bits']:.1f}", "bpb": f"{r['bpb']:.4f}",
                        "strata_bytes": f"{r['strata_bytes']:.1f}"})
    return path


def class_tables(rows):
    configs = sorted({(r["arm"], r["tables"], r["primer_bytes"]) for r in rows},
                     key=lambda c: (("none", "P", "P-s").index(c[0]), c[1], c[2]))
    for set_name, table in SETS:
        for cls in dict.fromkeys(table.values()):
            members = [n for n, c in table.items() if c == cls]
            raw = sum(next(r["bytes"] for r in rows if r["source"] == m) for m in members)
            print(f"\n{set_name}, class `{cls}`: {', '.join(members)}; {raw:,d} raw bytes")
            print("| Arm | Tables | Primer | Rung | Coded bytes (seed 1204 / 1205) | B_r mean | B_cl | Saving "
                  "| delta bpb | Value $/GB | Cost $/GB | Cost <= value | Material | Conditional PPMd / xz "
                  "| Parity vs Mamba |")
            print("|---" * 15 + "|")
            for arm, tables, size in configs:
                for rung in ("R0", "R1"):
                    pick = [r for r in rows if r["source"] in members and r["rung"] == rung
                            and (r["arm"], r["tables"], r["primer_bytes"]) == (arm, tables, size)]
                    seeds = sorted({r["seed"] for r in pick})
                    per_seed = [sum(r["strata_bytes"] for r in pick if r["seed"] == s) for s in seeds]
                    if len(pick) != len(members) * len(seeds) or not pick:
                        print(f"| {arm} | {tables} | {size:,d} | {rung} | INCOMPLETE ({len(pick)} rows) |")
                        continue
                    one = [r for r in pick if r["seed"] == seeds[0]]
                    b_r = sum(per_seed) / len(per_seed)
                    b_cl = sum(r["classical_side_bytes"] for r in one)
                    delta = 8 * (b_cl - b_r) / raw
                    mac = one[0]["mac_per_byte"]
                    viable = delta >= needed_delta(mac, R_DECISION, P_CPU[0])
                    cond = "-"
                    if size:
                        cond = (f"{sum(r['ppmd_cond_bytes'] for r in one):,d} / "
                                f"{sum(r['xz_cond_bytes'] for r in one):,d}")
                    mark = " **(primary)**" if (arm, size) in PRIMARY else ""
                    print(f"| {arm}{mark} | {tables} | {size:,d} | {rung} | "
                          + " / ".join(f"{v:,.0f}" for v in per_seed)
                          + f" | {b_r:,.0f} | {b_cl:,d} | {1 - b_r / b_cl:+.1%} | {delta:+.4f} | "
                          f"{value_per_gb(delta):+.4f} | {cost_per_gb(mac, R_DECISION, P_CPU[0]):.4f} | "
                          f"{'yes' if viable else 'no'} | {'yes' if 1 - b_r / b_cl >= MATERIAL else 'no'} | "
                          f"{cond} | |")


def source_table(rows):
    print("\nper source, mean of the seeds, bpb (classical = registered set without prior)")
    configs = sorted({(r["arm"], r["tables"], r["primer_bytes"], r["rung"]) for r in rows},
                     key=lambda c: (c[3], ("none", "P", "P-s").index(c[0]), c[1], c[2]))
    names = [n for _, table in SETS for n in table]
    print("| Rung | Arm | Tables | Primer | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 4) + "|")
    plain = {r["source"]: r["classical_bytes"] * 8 / r["bytes"] for r in rows}
    print("| - | best classical | - | - | " + " | ".join(f"{plain[n]:.4f}" for n in names) + " |")
    for arm, tables, size, rung in configs:
        cells = []
        for n in names:
            v = [r["bpb"] for r in rows if (r["source"], r["arm"], r["tables"], r["primer_bytes"], r["rung"])
                 == (n, arm, tables, size, rung)]
            cells.append(f"{sum(v) / len(v):.4f}" if v else "-")
        print(f"| {rung} | {arm} | {tables} | {size:,d} | " + " | ".join(cells) + " |")


def main():
    rows = load_rows()
    print(f"{len(rows)} rows -> {write_csv(rows)}")
    print(f"decision point: {R_DECISION / 1e9:.0f} G MAC/s, ${P_CPU[0]} per vCPU-hour; materiality {MATERIAL:.0%}")
    class_tables(rows)
    source_table(rows)


if __name__ == "__main__":
    main()
