"""Gate G0: apply the rule of docs/internal/research/strata-backbone/05-gate-amendments.md (08:44 pre-registration
as changed by amendment 1 of 09:16). Selection data only.

    python -I tools/strata/bakeoff/g0_decide.py --g0b pass|fail

Inputs: data/strata/runs/<run>/eval-select-on.csv (metric L) for the candidate's two seeds and the four controls;
data/strata/runs/<candidate run>/shortmsg-sel-*.csv (metric S, per cell). G0b is checked with ref_int.py.
"""
import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RUNS  # noqa: E402

SEEDS = {"1204": "strata_g_bc", "1205": "strata_g_bc_s1205"}
CONTROLS = {"B": ("gru_bc", "lstm_bc"), "A": ("gru_a_bc", "lstm_a_bc")}
TIER_INDEX = {"cm": 0, "A": 1, "B": 2}
S_MIN, E_MIN = 0.05, 0.05          # 5 % saving; 5 % per 100,000 MAC/byte
TYPES = {"text+markup": ("sel-webster", "sel-xml"), "logs": ("sel-zookeeper", "sel-proxifier")}


def macro_l(run):
    rows = list(csv.DictReader(open(os.path.join(RUNS, run, "eval-select-on.csv"))))
    out = {}
    for r in rows:
        out.setdefault(r["tier"], {"mac": int(r["mac_per_byte"]), "bpb": []})["bpb"].append(float(r["bpb"]))
    return {t: (v["mac"], sum(v["bpb"]) / len(v["bpb"])) for t, v in out.items()}


def cells(run):
    """{tier: {(source, size, case): mean payload}} from the selection short-message files of one run."""
    acc = {}
    for path in glob.glob(os.path.join(RUNS, run, "shortmsg-sel-*.csv")):
        for r in csv.DictReader(open(path)):
            for tier, k in TIER_INDEX.items():
                for case, suffix in (("same", "primer65536"), ("foreign", "fprimer65536")):
                    if r["tool"] == f"strata-t{k}-payload-{suffix}":
                        acc.setdefault(tier, {}).setdefault((r["source"], int(r["size"]), case), []).append(
                            int(r["bytes"]))
    return {t: {c: sum(v) / len(v) for c, v in d.items()} for t, d in acc.items()}


def ratio_s(cell_k, cell_ref, sources=None):
    keys = [c for c in cell_ref if sources is None or c[0] in sources]
    return math.exp(sum(math.log(cell_k[c] / cell_ref[c]) for c in keys) / len(keys)), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0b", choices=("pass", "fail", "unknown"), default="unknown")
    args = ap.parse_args()
    keep = {"A": {"L": True, "S": True}, "B": {"L": True, "S": True}}
    rows, l_by_seed = [], {}
    for seed, run in SEEDS.items():
        lc = macro_l(run)
        mac = {t: lc[t][0] for t in lc}
        l_by_seed[seed] = {t: lc[t][1] for t in lc}
        cl = cells(run)
        if len(cl) != 3 or any(len(v) != 40 for v in cl.values()):
            raise SystemExit(f"seed {seed}: short-message cells incomplete: "
                             + str({t: len(v) for t, v in cl.items()}))
        print(f"seed {seed} ({run}): L cm / A / B = " + " / ".join(f"{l_by_seed[seed][t]:.4f}" for t in TIER_INDEX)
              + f"   MAC/byte = " + " / ".join(f"{mac[t]:,}" for t in TIER_INDEX))
        for x in ("L", "S"):
            if x == "L":
                r_a = l_by_seed[seed]["A"] / l_by_seed[seed]["cm"]
                r_b = l_by_seed[seed]["B"] / l_by_seed[seed]["cm"]
                r_ba = l_by_seed[seed]["B"] / l_by_seed[seed]["A"]
            else:
                r_a, n = ratio_s(cl["A"], cl["cm"])
                r_b, _ = ratio_s(cl["B"], cl["cm"])
                r_ba, _ = ratio_s(cl["B"], cl["A"])
            s_a, s_b, s_ba = 1 - r_a, 1 - r_b, 1 - r_ba
            e_a = s_a / ((mac["A"] - mac["cm"]) / 1e5)
            e_b = s_b / ((mac["B"] - mac["cm"]) / 1e5)
            m_b = s_ba / ((mac["B"] - mac["A"]) / 1e5)
            ok_a = s_a >= S_MIN and e_a >= E_MIN
            ok_b = s_b >= S_MIN and e_b >= E_MIN and m_b >= E_MIN
            keep["A"][x] &= ok_a
            keep["B"][x] &= ok_b
            rows.append((seed, x, "A", s_a, e_a, None, ok_a))
            rows.append((seed, x, "B", s_b, e_b, m_b, ok_b))
        for name, srcs in TYPES.items():
            ra, _ = ratio_s(cl["A"], cl["cm"], srcs)
            rb, _ = ratio_s(cl["B"], cl["cm"], srcs)
            print(f"   reported, S per type {name}: saving A {100 * (1 - ra):.2f} %, B {100 * (1 - rb):.2f} %")
        tot = {t: sum(cl[t].values()) * 16 for t in TIER_INDEX}
        print("   reported, S byte totals (both primer cases): " + ", ".join(f"{t} {tot[t]:,.0f}" for t in tot))
    print("\n| seed | metric | tier | saving s vs cm | e (% per 100K MAC) | marginal e over A | conditions met |")
    print("|---|---|---|---|---|---|---|")
    for seed, x, tier, s, e, m, ok in rows:
        print(f"| {seed} | {x} | {tier} | {100 * s:.2f} % | {100 * e:.2f} | "
              + ("—" if m is None else f"{100 * m:.2f}") + f" | {'yes' if ok else 'no'} |")
    for tier in ("A", "B"):
        for x in ("L", "S"):
            sp = [r[3] for r in rows if r[1] == x and r[2] == tier]
            print(f"spread between seeds, tier {tier}, {x}: {100 * abs(sp[0] - sp[1]):.2f} points")
    kept = {t: [x for x in ("L", "S") if keep[t][x]] for t in ("A", "B")}
    widest = "B" if kept["B"] else ("A" if kept["A"] else None)
    print(f"\nG0c: tier A kept on {kept['A'] or 'nothing'}; tier B kept on {kept['B'] or 'nothing'}"
          + ("" if widest else "  ->  NO TRUNK TIER KEPT: v1 ships the cm tier only"))
    order = {"WIN": 0, "TIE": 1, "LOSS": 2}
    verdict = "n/a"
    if widest:
        ctrl = {c: macro_l(c)[widest][1] for c in CONTROLS[widest]}
        best = min(ctrl.values())
        verdicts = []
        for seed in SEEDS:
            r = l_by_seed[seed][widest] / best
            v = "WIN" if r < 0.98 else ("TIE" if r <= 1.02 else "LOSS")
            verdicts.append(v)
            print(f"G0a seed {seed}: tier {widest} L {l_by_seed[seed][widest]:.4f} vs best control {best:.4f} "
                  f"({', '.join(f'{c} {v_:.4f}' for c, v_ in ctrl.items())}) -> ratio {r:.4f} -> {v}")
        verdict = max(verdicts, key=order.get)
        print(f"G0a (worse of the two seeds): {verdict}")
    else:
        print("G0a: not applicable (no trunk tier kept)")
    for tier in ("A", "B"):
        for c in CONTROLS[tier]:
            print(f"   reported: {c} (tier {tier}, seed 1204) L = {macro_l(c)[tier][1]:.4f}")
    passes = args.g0b == "pass" and (widest is None or verdict in ("WIN", "TIE"))
    print(f"G0b: {args.g0b}")
    print(f"G0: {'PASS' if passes else ('FAIL' if args.g0b != 'unknown' else 'undecided (G0b unknown)')}")


if __name__ == "__main__":
    main()
