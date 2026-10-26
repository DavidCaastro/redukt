#!/usr/bin/env python3
"""
Redukt Benchmark Suite — Reproducible compression benchmarks.

Compares Redukt against gzip-9, brotli-11, zstd-19 on standard and
industry-specific datasets. Generates JSON results + Markdown report.

Usage:
    python scripts/benchmark-suite.py                      # 10KB, all files
    python scripts/benchmark-suite.py --bytes 102400       # 100KB window
    python scripts/benchmark-suite.py --skip-redukt        # competitors only
    python scripts/benchmark-suite.py --files enwik8,xml   # specific files

Requirements:
    pip install brotli zstandard   (optional — gracefully skipped if absent)
"""

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Optional compressors
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

DOMAIN_MAP = {
    "enwik8": "text/encyclopedia",
    "dickens": "text/literature",
    "webster": "text/dictionary",
    "reymont": "text/polish",
    "samba": "code/c-source",
    "mozilla": "binary/executable",
    "ooffice": "binary/office",
    "mr": "medical/mri",
    "x-ray": "medical/xray",
    "nci": "scientific/chemistry",
    "sao": "scientific/astronomy",
    "osdb": "database/mysql",
    "xml": "markup/xml",
    "apache-access.log": "logs/apache",
    "hdfs-audit.log": "logs/hdfs",
    "linux-syslog.log": "logs/syslog",
    "openssh-auth.log": "logs/ssh",
    "ubl-invoice-batch.xml": "e-invoicing/ubl",
    "facturae-batch.xml": "e-invoicing/facturae",
}


def find_redukt_binary():
    """Auto-detect redukt binary in target/release/."""
    suffix = ".exe" if sys.platform == "win32" else ""
    candidates = [
        Path("target/release") / f"redukt{suffix}",
        Path("target/maxperf") / f"redukt{suffix}",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def discover_files(data_dir):
    """Find all benchmark files."""
    files = []
    data = Path(data_dir)

    # enwik8
    enwik = data / "enwik8"
    if enwik.exists():
        files.append(enwik)

    # Silesia
    silesia = data / "silesia"
    if silesia.is_dir():
        for f in sorted(silesia.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                files.append(f)

    # Industry
    industry = data / "industry"
    if industry.is_dir():
        for f in sorted(industry.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                files.append(f)

    return files


def classify(name):
    return DOMAIN_MAP.get(name, "unknown")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def bpb(compressed_size, original_size):
    if original_size == 0:
        return 0.0
    return (compressed_size * 8.0) / original_size


def bench_gzip(data):
    t0 = time.perf_counter()
    compressed = gzip.compress(data, compresslevel=9)
    elapsed = time.perf_counter() - t0
    return {"size": len(compressed), "bpb": bpb(len(compressed), len(data)),
            "time_s": elapsed}


def bench_brotli(data):
    if not HAS_BROTLI:
        return None
    t0 = time.perf_counter()
    compressed = brotli.compress(data, quality=11)
    elapsed = time.perf_counter() - t0
    return {"size": len(compressed), "bpb": bpb(len(compressed), len(data)),
            "time_s": elapsed}


def bench_zstd(data):
    if not HAS_ZSTD:
        return None
    cctx = zstandard.ZstdCompressor(level=19)
    t0 = time.perf_counter()
    compressed = cctx.compress(data)
    elapsed = time.perf_counter() - t0
    return {"size": len(compressed), "bpb": bpb(len(compressed), len(data)),
            "time_s": elapsed}


def bench_redukt(data, redukt_bin):
    """Compress with redukt, verify roundtrip, return results."""
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input.bin")
        out = os.path.join(td, "output.azth")
        restored = os.path.join(td, "restored.bin")

        with open(inp, "wb") as f:
            f.write(data)

        # Compress
        t0 = time.perf_counter()
        r = subprocess.run(
            [redukt_bin, "compress", "--input", inp, "--output", out],
            capture_output=True, text=True, timeout=600,
        )
        compress_time = time.perf_counter() - t0

        if r.returncode != 0:
            return {"error": f"compress failed: {r.stderr[-200:]}"}

        compressed_size = os.path.getsize(out)

        # Decompress + verify
        r2 = subprocess.run(
            [redukt_bin, "decompress", "--input", out, "--output", restored],
            capture_output=True, text=True, timeout=600,
        )
        if r2.returncode != 0:
            return {"size": compressed_size,
                    "bpb": bpb(compressed_size, len(data)),
                    "time_s": compress_time,
                    "roundtrip": False, "error": "decompress failed"}

        with open(restored, "rb") as f:
            restored_data = f.read()

        match = (restored_data == data)
        return {
            "size": compressed_size,
            "bpb": bpb(compressed_size, len(data)),
            "time_s": compress_time,
            "roundtrip": match,
            "restored_sha256": sha256(restored_data),
        }


def gather_context(redukt_bin):
    ctx = {
        "date": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "compressors": {
            "gzip": "9 (stdlib)",
            "brotli": f"11 (pip {brotli.__version__})" if HAS_BROTLI else "not installed",
            "zstd": f"19 (pip {zstandard.__version__})" if HAS_ZSTD else "not installed",
        },
    }
    # Git info
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        ctx["git_commit"] = sha
    except Exception:
        ctx["git_commit"] = "unknown"

    # Redukt version
    if redukt_bin:
        try:
            r = subprocess.run(
                [redukt_bin, "help"], capture_output=True, text=True, timeout=5,
            )
            ctx["redukt_bin"] = redukt_bin
        except Exception:
            pass

    return ctx


def write_json(results, context, window_bytes, output_dir):
    obj = {"context": context, "window_bytes": window_bytes, "results": results}
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(output_dir, f"benchmark-{date_str}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"\n[output] {path}")
    return path


def write_markdown(results, context, window_bytes, output_dir):
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(output_dir, f"benchmark-{date_str}.md")
    w = window_bytes

    lines = []
    lines.append("# Redukt Benchmark Report\n")
    lines.append(f"**Generated**: {context['date']}")
    lines.append(f"**Platform**: {context['platform']}")
    lines.append(f"**Processor**: {context['processor']}")
    lines.append(f"**Git commit**: {context.get('git_commit', 'N/A')}")
    lines.append(f"**Window**: {w:,} bytes per file\n")

    # Compressors
    lines.append("## Compressors Tested\n")
    lines.append("| Compressor | Level | Library |")
    lines.append("|---|---|---|")
    has_redukt = any(r.get("redukt") for r in results)
    if has_redukt:
        lines.append(f"| **Redukt** | default | `{context.get('redukt_bin', 'N/A')}` |")
    lines.append(f"| gzip | 9 | {context['compressors']['gzip']} |")
    lines.append(f"| brotli | 11 | {context['compressors']['brotli']} |")
    lines.append(f"| zstd | 19 | {context['compressors']['zstd']} |")
    lines.append("")

    # Summary table
    lines.append("## Results\n")
    header = "| File | Domain | Bytes |"
    sep = "|---|---|---|"
    if has_redukt:
        header += " Redukt BPB |"
        sep += "---|"
    header += " gzip-9 |"
    sep += "---|"
    if HAS_BROTLI:
        header += " brotli-11 |"
        sep += "---|"
    if HAS_ZSTD:
        header += " zstd-19 |"
        sep += "---|"
    if has_redukt and HAS_BROTLI:
        header += " vs brotli |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        row = f"| {r['file']} | {r['domain']} | {r['bytes']:,} |"
        redukt_bpb = None
        if has_redukt and r.get("redukt"):
            rb = r["redukt"]
            if "error" in rb and "bpb" not in rb:
                row += " ERROR |"
            else:
                redukt_bpb = rb["bpb"]
                row += f" **{redukt_bpb:.4f}** |"
        row += f" {r['gzip9']['bpb']:.4f} |"
        brotli_bpb = None
        if HAS_BROTLI and r.get("brotli11"):
            brotli_bpb = r["brotli11"]["bpb"]
            row += f" {brotli_bpb:.4f} |"
        if HAS_ZSTD and r.get("zstd19"):
            row += f" {r['zstd19']['bpb']:.4f} |"
        if has_redukt and HAS_BROTLI and redukt_bpb and brotli_bpb:
            delta = ((redukt_bpb - brotli_bpb) / brotli_bpb) * 100
            row += f" {delta:+.1f}% |"
        lines.append(row)

    # Averages
    lines.append("")

    if has_redukt:
        redukt_bpbs = [r["redukt"]["bpb"] for r in results
                       if r.get("redukt") and "bpb" in r["redukt"]]
        if redukt_bpbs:
            lines.append(f"**Redukt mean BPB**: {sum(redukt_bpbs)/len(redukt_bpbs):.4f} "
                         f"(across {len(redukt_bpbs)} files)\n")

    # Roundtrip verification
    if has_redukt:
        lines.append("## Roundtrip Verification\n")
        lines.append("| File | SHA-256 (original) | Roundtrip OK |")
        lines.append("|---|---|---|")
        for r in results:
            ok = "N/A"
            if r.get("redukt"):
                rt = r["redukt"].get("roundtrip")
                if rt is True:
                    ok = "PASS"
                elif rt is False:
                    ok = "**FAIL**"
            lines.append(f"| {r['file']} | `{r['sha256'][:16]}...` | {ok} |")
        lines.append("")

    # Input hashes
    lines.append("## Input File Hashes\n")
    lines.append("```")
    for r in results:
        lines.append(f"{r['sha256']}  {r['file']} ({r['bytes']:,}B window)")
    lines.append("```\n")

    # Methodology
    lines.append("## Methodology\n")
    lines.append(f"- Window: first {w:,} bytes of each file")
    lines.append("- SHA-256 computed on the windowed input (not full file)")
    lines.append("- Redukt: `compress --input FILE --output FILE.azth` (actual file size)")
    lines.append("- Roundtrip: decompress output and compare SHA-256 with original")
    lines.append("- Competitors: Python stdlib/pip libraries, in-memory compression")
    lines.append("- BPB = (compressed_size * 8) / original_size")
    lines.append("- Single run per file (deterministic compressors)")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[output] {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Redukt Benchmark Suite")
    parser.add_argument("--bytes", type=int, default=10240,
                        help="Window size in bytes (default: 10240)")
    parser.add_argument("--data-dir", default="data/benchmark",
                        help="Data directory (default: data/benchmark)")
    parser.add_argument("--output-dir", default="data/benchmark/results",
                        help="Output directory (default: data/benchmark/results)")
    parser.add_argument("--skip-redukt", action="store_true",
                        help="Skip Redukt (fast mode, competitors only)")
    parser.add_argument("--files", type=str, default=None,
                        help="Comma-separated file names to benchmark")
    args = parser.parse_args()

    redukt_bin = None if args.skip_redukt else find_redukt_binary()
    if not args.skip_redukt and redukt_bin is None:
        print("WARNING: redukt binary not found. Run `cargo build --release` first.")
        print("         Continuing with --skip-redukt mode.\n")
        args.skip_redukt = True

    context = gather_context(redukt_bin)
    all_files = discover_files(args.data_dir)

    if not all_files:
        print(f"ERROR: no benchmark files found in {args.data_dir}/")
        print("       Run: python scripts/prepare-benchmark-data.py")
        sys.exit(1)

    # Filter
    if args.files:
        names = set(args.files.split(","))
        all_files = [f for f in all_files if f.name in names or f.stem in names]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Redukt Benchmark Suite")
    print("=" * 60)
    print(f"  Files:    {len(all_files)}")
    print(f"  Window:   {args.bytes:,} bytes")
    print(f"  Redukt:   {'SKIP' if args.skip_redukt else redukt_bin}")
    print(f"  brotli:   {'YES' if HAS_BROTLI else 'NO (pip install brotli)'}")
    print(f"  zstd:     {'YES' if HAS_ZSTD else 'NO (pip install zstandard)'}")
    print("=" * 60)
    print()

    results = []
    total = len(all_files)

    for idx, fpath in enumerate(all_files, 1):
        name = fpath.name
        domain = classify(name)
        print(f"[{idx}/{total}] {name} ({domain})")

        with open(fpath, "rb") as f:
            raw = f.read(args.bytes)

        if len(raw) == 0:
            print(f"  SKIP (empty file)")
            continue

        entry = {
            "file": name,
            "domain": domain,
            "bytes": len(raw),
            "sha256": sha256(raw),
        }

        # Competitors (fast)
        entry["gzip9"] = bench_gzip(raw)
        print(f"  gzip-9:   {entry['gzip9']['bpb']:.4f} BPB "
              f"({entry['gzip9']['size']:,}B, {entry['gzip9']['time_s']:.3f}s)")

        entry["brotli11"] = bench_brotli(raw)
        if entry["brotli11"]:
            print(f"  brotli-11:{entry['brotli11']['bpb']:.4f} BPB "
                  f"({entry['brotli11']['size']:,}B, {entry['brotli11']['time_s']:.3f}s)")

        entry["zstd19"] = bench_zstd(raw)
        if entry["zstd19"]:
            print(f"  zstd-19:  {entry['zstd19']['bpb']:.4f} BPB "
                  f"({entry['zstd19']['size']:,}B, {entry['zstd19']['time_s']:.3f}s)")

        # Redukt (slow)
        if not args.skip_redukt:
            print(f"  redukt:   compressing ...", end="", flush=True)
            entry["redukt"] = bench_redukt(raw, redukt_bin)
            rb = entry["redukt"]
            if "error" in rb and "bpb" not in rb:
                print(f"\r  redukt:   ERROR — {rb['error'][:80]}")
            else:
                rt = "PASS" if rb.get("roundtrip") else "FAIL"
                print(f"\r  redukt:   {rb['bpb']:.4f} BPB "
                      f"({rb['size']:,}B, {rb['time_s']:.1f}s, roundtrip={rt})")

        results.append(entry)
        print()

    # Output
    write_json(results, context, args.bytes, args.output_dir)
    write_markdown(results, context, args.bytes, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
