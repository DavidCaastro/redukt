"""Strata viability frontier: is the compute cost of a rung below the storage value of the bytes it saves?

    python -I tools/strata/viability/viability.py            # tables for the measured rungs
    python -I tools/strata/viability/viability.py --self-test

Standard library only; no file is read and nothing is sent anywhere. Gate V7 of the specification
(`docs/api/BACKBONE-STRATA-SPEC.md` 9.2, amendment 4 of `05-gate-amendments.md`) is computed with this script.

Model (per raw byte; the same formula as `scripts/breakeven.py` and `02-evidence-and-constraints.md` C.4):
    value   = (bpb_classical - bpb_model) / 8 * P_ST * 12 * YEARS       $ per raw GB saved against the best
                                                                        classical codec of each file
    seconds = T_CM + MAC / R_MAC                                        cost hypothesis of specification 7.2
    cost    = P_CPU * 1e9 * seconds / 3600                              $ per raw GB, compression only
    viable  <=> cost <= value  <=>  bpb_classical - bpb_model >= K * P_CPU / R[KB/s],  K = 8e6 / (3600 * 1.932)

Inputs are MEASURED numbers copied from `03-bakeoff.md` (12 and 12.2: ideal code lengths of the integer reference,
mean of seeds 1204 and 1205) and from `ppmd_baseline.py` (run 2026-11-14). The throughput is a HYPOTHESIS until the
Rust implementation exists (gate V4): every verdict printed here is a projection.
"""
import sys

P_ST = 0.023            # $ per GB-month, hot object storage (scripts/breakeven.py)
YEARS = 7
P_CPU = (0.014, 0.022)  # $ per vCPU-hour: gate V4's price, and the spot price of scripts/breakeven.py
T_CM = 1.0e-6           # s per byte for tables, bit cells and bit steps (specification 7.2, HYPOTHESIS)
R_MAC = (2.5e9, 10e9, 20e9)   # MAC/s per vCPU: today's effective rate (02-evidence C.5), specification 7.2 x2
MATERIAL = 0.10         # V7: a rung must save at least 10 % of the best classical bytes of the class
PARITY = 1.10           # V7: "the same as Mamba-130M" = at most 10 % more bytes on identical bytes
STORAGE_7Y = P_ST * 12 * YEARS          # $ per GB kept for the horizon = 1.932
K = 8e6 / (3600 * STORAGE_7Y)           # needed delta bpb = K * P_CPU / R[KB/s]

RUNGS = {"R0 cm": 208, "R1 width 88": 27_752, "R2 width 208": 113_912}

# name: (class, bytes, best LZ/BWT bpb, best PPMd bpb, Mamba-130M bpb or None, {rung: (seed 1204, seed 1205)})
TEST = {
    "dickens": ("text", 500_000, 2.1960, 1.9681, 1.0620,
                {"R0 cm": (2.1268, 2.1270), "R1 width 88": (2.0662, 2.0644), "R2 width 208": (2.0398, 2.0418)}),
    "apache": ("logs", 171_239, 0.3132, 0.2576, 0.1849,
               {"R0 cm": (0.2055, 0.2052), "R1 width 88": (0.1964, 0.1932), "R2 width 208": (0.1882, 0.1839)}),
    "linux": ("logs", 216_485, 0.3697, 0.4039, 0.2449,
              {"R0 cm": (0.3496, 0.3494), "R1 width 88": (0.3212, 0.3194), "R2 width 208": (0.3200, 0.3072)}),
    "openssh": ("logs", 225_216, 0.3460, 0.3222, 0.2014,
                {"R0 cm": (0.2507, 0.2504), "R1 width 88": (0.2394, 0.2422), "R2 width 208": (0.2396, 0.2366)}),
    "facturae": ("xml-synthetic", 39_384, 0.3953, 0.3845, 0.2408,
                 {"R0 cm": (0.4785, 0.4785), "R1 width 88": (0.4713, 0.4723), "R2 width 208": (0.4846, 0.4906)}),
    "ubl": ("xml-synthetic", 83_260, 0.2470, 0.2490, 0.2392,
            {"R0 cm": (0.3150, 0.3149), "R1 width 88": (0.3212, 0.3156), "R2 width 208": (0.3313, 0.3243)}),
}
SELECT = {
    "webster": ("text", 300_000, 1.8547, 1.6681, None,
                {"R0 cm": (1.7387, 1.7388), "R1 width 88": (1.7279, 1.7308), "R2 width 208": (1.7171, 1.7255)}),
    "xml": ("markup", 300_000, 0.3219, 0.2778, None,
            {"R0 cm": (0.3872, 0.3872), "R1 width 88": (0.4086, 0.4008), "R2 width 208": (0.4061, 0.4054)}),
    "zookeeper": ("logs", 300_000, 0.2043, 0.1856, None,
                  {"R0 cm": (0.1491, 0.1489), "R1 width 88": (0.1390, 0.1373), "R2 width 208": (0.1340, 0.1372)}),
    "proxifier": ("logs", 300_000, 0.2416, 0.2201, None,
                  {"R0 cm": (0.2024, 0.2022), "R1 width 88": (0.2014, 0.1966), "R2 width 208": (0.1988, 0.1953)}),
}


def seconds_per_byte(mac, r_mac):
    return T_CM + mac / r_mac


def cost_per_gb(mac, r_mac, p_cpu):
    return p_cpu * 1e9 * seconds_per_byte(mac, r_mac) / 3600


def value_per_gb(delta_bpb):
    return delta_bpb / 8 * STORAGE_7Y


def needed_delta(mac, r_mac, p_cpu):
    return K * p_cpu * seconds_per_byte(mac, r_mac) * 1e3


def max_mac(delta_bpb, r_mac, p_cpu):
    """Largest MAC/byte whose cost is still covered by a saving of delta_bpb (0 if even T_CM is not)."""
    return max(0.0, (delta_bpb / (K * p_cpu * 1e3) - T_CM) * r_mac)


def class_totals(table, with_ppmd):
    """class -> (raw bytes, classical bytes, Mamba bytes or None, {rung: model bytes})."""
    out = {}
    for _, (cls, n, lz, ppmd, mamba, rungs) in table.items():
        raw, cl, mb, model = out.setdefault(cls, [0, 0.0, 0.0, {r: 0.0 for r in RUNGS}])
        best = min(lz, ppmd) if with_ppmd else lz
        out[cls][0] = raw + n
        out[cls][1] = cl + best * n / 8
        out[cls][2] = None if (mamba is None or mb is None) else mb + mamba * n / 8
        for rung, seeds in rungs.items():
            model[rung] += sum(seeds) / len(seeds) * n / 8
    return out


def verdict(delta, saving, mac, p_cpu):
    if delta <= 0:
        return "loses to classical"
    marks = "".join("Y" if delta >= needed_delta(mac, r, p_cpu) else "n" for r in R_MAC)
    return f"cost<=value at 2.5/10/20 G MAC/s: {marks}; material: {'Y' if saving >= MATERIAL else 'n'}"


def report(title, table, with_ppmd):
    label = "LZ/BWT + PPMd" if with_ppmd else "LZ/BWT only (the bake-off's set)"
    print(f"\n== {title}; classical = {label}; price ${P_CPU[0]} per vCPU-hour")
    for cls, (raw, cl, mamba, model) in class_totals(table, with_ppmd).items():
        cl_bpb = cl * 8 / raw
        line = f"  {cls:14s} {raw:>9,d} B  best classical {cl_bpb:.4f} bpb"
        print(line + (f"  Mamba-130M {mamba * 8 / raw:.4f}" if mamba else ""))
        for rung, mac in RUNGS.items():
            bpb = model[rung] * 8 / raw
            delta = cl_bpb - bpb
            saving = 1 - model[rung] / cl
            parity = f"  x{model[rung] / mamba:.2f} Mamba" if mamba else ""
            print(f"    {rung:13s} {bpb:.4f} bpb  vs classical {-saving:+.1%}  value ${value_per_gb(delta):+.4f}/GB"
                  f"  cost ${cost_per_gb(mac, R_MAC[0], P_CPU[0]):.4f} / ${cost_per_gb(mac, R_MAC[1], P_CPU[0]):.4f}"
                  f" / ${cost_per_gb(mac, R_MAC[2], P_CPU[0]):.4f}{parity}\n"
                  f"                  {verdict(delta, saving, mac, P_CPU[0])}")


def ceilings():
    print("\n== Ceilings (DERIVED): largest model, in MAC/byte, that storage economics can pay for")
    print("   class: at zero output bytes | at Mamba-130M's ratio | at the materiality floor (10 % saving)")
    for cls, (raw, cl, mamba, _) in class_totals(TEST, True).items():
        cl_bpb = cl * 8 / raw
        gains = [cl_bpb, cl_bpb - mamba * 8 / raw if mamba else 0.0, MATERIAL * cl_bpb]
        for p_cpu in P_CPU:
            cells = " | ".join("/".join(f"{max_mac(g, r, p_cpu) / 1e3:,.0f}K" for r in R_MAC) for g in gains)
            print(f"  {cls:14s} ${p_cpu}: {cells}   (2.5 / 10 / 20 G MAC/s)")


def self_test():
    assert abs(STORAGE_7Y - 1.932) < 1e-9
    assert abs(K * 0.014 - 16.10) < 0.01, K * 0.014          # gate V4: bpb <= bpb_classical - 16.1 / R[KB/s]
    assert abs(0.59 * 2.352 / 8 * STORAGE_7Y - 0.3351) < 1e-3  # scripts/breakeven.py, enwik8: $0.335 per raw GB
    assert abs(cost_per_gb(0, 1e9, 0.022) - 0.022 * 1e9 / (3600 * 1e6)) < 1e-12   # 1 us per byte = 1 MB/s
    mac = max_mac(0.5, 10e9, 0.014)
    assert abs(needed_delta(mac, 10e9, 0.014) - 0.5) < 1e-9   # max_mac inverts needed_delta
    assert abs(cost_per_gb(mac, 10e9, 0.014) - value_per_gb(0.5)) < 1e-9          # and sits on the frontier
    assert max_mac(0.001, 10e9, 0.014) == 0.0                 # a saving that does not pay even for the tables
    totals = class_totals(TEST, False)
    assert totals["logs"][0] == 171_239 + 216_485 + 225_216
    assert abs(totals["text"][1] * 8 / 500_000 - 2.1960) < 1e-9
    assert class_totals(TEST, True)["logs"][1] < totals["logs"][1]   # PPMd improves the log baseline
    print("self-test OK")


def main():
    if "--self-test" in sys.argv:
        return self_test()
    print(f"storage over the horizon ${STORAGE_7Y:.3f}/GB; needed delta bpb = {K * P_CPU[0]:.2f} / R[KB/s] at "
          f"${P_CPU[0]}, {K * P_CPU[1]:.2f} / R at ${P_CPU[1]}")
    print("throughput hypothesis, KB/s per vCPU at 2.5 / 10 / 20 G MAC/s: " + "; ".join(
        f"{rung} " + " / ".join(f"{1e-3 / seconds_per_byte(mac, r):,.0f}" for r in R_MAC)
        for rung, mac in RUNGS.items()))
    report("Frozen test set (held-out)", TEST, False)
    report("Frozen test set (held-out)", TEST, True)
    report("Selection set", SELECT, True)
    ceilings()


if __name__ == "__main__":
    main()
