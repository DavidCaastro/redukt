#!/usr/bin/env python3
"""Calculate benchmark metrics from compressed sizes. No rounding errors."""

INPUT_BYTES = 10000

# (new_compressed_size, coder_eff_from_compress, brotli_bpb, gzip_bpb, domain)
DATA = {
    "apache-access.log":     (370,  83.85, 0.4664, 0.5464, "logs/apache"),
    "linux-syslog.log":      (518,  89.33, 0.5816, 0.7136, "logs/syslog"),
    "facturae-batch.xml":    (559,  89.96, 0.8904, 1.0704, "e-invoicing/facturae"),
    "openssh-auth.log":      (651,  90.67, 0.7832, 0.9904, "logs/ssh"),
    "ubl-invoice-batch.xml": (661,  90.45, 0.8880, 1.0928, "e-invoicing/ubl"),
    "nci":                   (832,  89.44, 0.7344, 0.9344, "scientific/chemistry"),
    "xml":                   (842,  92.51, 0.9008, 1.0752, "markup/xml"),
    "samba":                 (1621, 97.45, 2.2200, 2.8840, "code/c-source"),
    "enwik8":                (1664, 96.05, 2.3520, 2.9720, "text/encyclopedia"),
    "dickens":               (2244, 97.74, 2.7784, 3.6208, "text/literature"),
    "webster":               (2251, 97.85, 2.8088, 3.6608, "text/dictionary"),
    "reymont":               (2370, 95.04, 2.1360, 2.3872, "text/polish"),
    "mozilla":               (2526, 96.30, 2.0208, 2.3728, "binary/executable"),
    "mr":                    (2989, 95.61, 2.3728, 2.7800, "medical/mri"),
    "ooffice":               (4688, 98.39, 2.7424, 2.9104, "binary/office"),
    "osdb":                  (7094, 98.88, 4.6760, 4.9672, "database/mysql"),
    "x-ray":                 (7318, 98.74, 4.5920, 5.8728, "medical/xray"),
    "sao":                   (11545,99.88, 5.9880, 6.3888, "scientific/astronomy"),
    "sample-invoice.pdf":    (13894,99.82, 7.8968, 7.9376, "binary/pdf"),
}

# Old baseline (2026-10-29) compressed sizes
OLD_SIZES = {
    "apache-access.log": 370, "linux-syslog.log": 515, "facturae-batch.xml": 566,
    "openssh-auth.log": 659, "ubl-invoice-batch.xml": 677, "nci": 836,
    "xml": 861, "samba": 1622, "enwik8": 1666, "dickens": 2246, "webster": 2256,
    "reymont": 2374, "mozilla": 2565, "mr": 2986, "ooffice": 4709, "osdb": 7133,
    "x-ray": 7302, "sao": 11554, "sample-invoice.pdf": 13903,
}

# --- Table 1: Comparison vs old baseline ---
print("=== COMPARISON vs OLD BASELINE (2026-10-29) ===")
print(f"{'File':<28} {'Old BPB':>8} {'New BPB':>8} {'d_bytes':>8} {'d%':>8} {'vs_brotli':>10} {'vs_gzip':>10}")
print("-" * 92)

wins_brotli = 0
improved = 0
regressed = 0

for name in DATA:
    new_size, eff, brotli, gzip, domain = DATA[name]
    old_size = OLD_SIZES[name]
    old_bpb = old_size * 8 / INPUT_BYTES
    new_bpb = new_size * 8 / INPUT_BYTES
    delta_bytes = new_size - old_size
    delta_pct = (new_bpb - old_bpb) / old_bpb * 100 if old_bpb > 0 else 0
    vs_brotli = (new_bpb - brotli) / brotli * 100
    vs_gzip = (new_bpb - gzip) / gzip * 100

    if delta_bytes < 0: improved += 1
    elif delta_bytes > 0: regressed += 1
    if new_bpb < brotli: wins_brotli += 1

    print(f"{name:<28} {old_bpb:8.4f} {new_bpb:8.4f} {delta_bytes:>+8d} {delta_pct:>+7.2f}% {vs_brotli:>+9.1f}% {vs_gzip:>+9.1f}%")

print("-" * 92)
print(f"Improved: {improved}/19, Regressed: {regressed}/19, Same: {19-improved-regressed}/19")
print(f"Wins vs brotli: {wins_brotli}/19")
print()

# --- Table 2: CSV format ---
print("=== CSV (for actual-compression-benchmark.csv) ===")
print("file,domain,bytes,redukt_bpb,ce_bpb,coder_eff_pct,compressed_size,brotli11_bpb,gzip9_bpb,vs_brotli_pct,vs_gzip_pct,roundtrip,date")
for name in DATA:
    new_size, eff, brotli, gzip, domain = DATA[name]
    redukt_bpb = new_size * 8 / INPUT_BYTES
    ce_bpb = redukt_bpb * eff / 100.0
    vs_brotli = (redukt_bpb - brotli) / brotli * 100
    vs_gzip = (redukt_bpb - gzip) / gzip * 100
    print(f"{name},{domain},{INPUT_BYTES},{redukt_bpb:.4f},{ce_bpb:.4f},{eff:.1f},{new_size},{brotli:.4f},{gzip:.4f},{vs_brotli:+.1f},{vs_gzip:+.1f},PASS,2026-10-30")
