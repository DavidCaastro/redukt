#!/usr/bin/env python3
"""
Redukt — full-size benchmark on Kaggle (T4 + 4 vCPU): the three measurements the competitive
report of 2026-11-10 found missing (docs/internal/research/competitive-position-2026-11-10/REPORT.md).

  A. Scale curve on enwik8 with the Mamba-130M backbone (10 KB -> 4 MB prefixes, single stream, GPU plugin;
     CPU F32/Q8 at 10 KB and 100 KB on the same machine). Answers: does BPB improve or worsen with size?
  B. Redukt on WHOLE industry files (facturae, UBL, apache, syslog, openssh) vs gzip -9, zstd -19/-22,
     xz -9e, brotli -11 on the same whole files and on every enwik8 prefix. Answers: how big is the
     real margin once classical codecs see the full file?
  C. Economic gate: measured bytes/s per vCPU (lockstep, 12 x 10 KB) and per T4, $/GB at Nov-2026 spot
     prices, value of the saved bytes (S3 Standard, 7 years) and the speed-up a fast tier would need.
     No fast-tier model exists yet (R-105), so this calibrates the gate; it does not measure a candidate.

Since R-109 the lockstep with `--backbone plugin` runs one FFI v2 `forward_batch` per step over per-stream
device slots (before R-109 it silently interleaved the single v1 state: 1.949 bpb). Phase C records the
aggregate throughput and checks batch == single-file sha256 on three slices.

Launch (repo root, after scripts/kaggle-setup.sh):
    cp notebooks/redukt-gpu-fullsize-bench.py notebooks/redukt-gpu-fullsize-bench/
    python -m kaggle kernels push -p notebooks/redukt-gpu-fullsize-bench
    python -m kaggle kernels status <user>/redukt-gpu-fullsize-bench
    python -m kaggle kernels output <user>/redukt-gpu-fullsize-bench -p data/benchmark/results/kaggle-fullsize

Outputs (/kaggle/working): fullsize-results.csv (long form), classical-fullsize.csv, gate-table.csv,
fullsize-report.txt. Every phase is deadline-aware (DEADLINE_HOURS, default 9) so results are always written.

Copyright 2026 Redukt Contributors. Apache-2.0 License.
"""

import argparse
import csv
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import time

REPO_URL = os.environ.get("REDUKT_REPO", "https://github.com/DavidCaastro/redukt.git")
REDUKT_REF = os.environ.get("REDUKT_REF", "b219870")  # origin/master 2026-11-10, R-108 + R-109 (plugin slots)
WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("fullsize-out")
DEADLINE_HOURS = float(os.environ.get("DEADLINE_HOURS", "9"))
PHASES = os.environ.get("PHASES", "C,A2,B,A").split(",")   # C lockstep, A2 CPU points, B industry, A GPU curve
T_START = time.time()
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
PREFIXES = [10_240, 51_200, 102_400, 512_000, 1_048_576, 2_097_152, 4_194_304]
BYTES_PER_TOKEN_TEXT = 3.8        # enwik8 BPE (repo convention, CLAUDE.md)
INDUSTRY = ["facturae-batch.xml", "ubl-invoice-batch.xml", "apache-access.log",
            "linux-syslog.log", "openssh-auth.log"]
LOCAL_SHA = {   # sha256 of the local copies (2026-11-10) — generator must be deterministic
    "facturae-batch.xml": "97086fd8296929e01f724b419414800ca901e438c2e72ea16c0c34c969a895b4",
    "ubl-invoice-batch.xml": "3522c908d6414642b01a7db23b367f68c4e0845862b87b29f1b84fcaa20f8f7d",
    "apache-access.log": "c7efa3eb686e3a96bd2f8f4457b2a7887e9cf2f3649327f1b4e87af841363ce8",
    "linux-syslog.log": "b3e20bc1afe732ab1bf3ed1de4bf9c809e4194e02f7dea911d918e5342e8e173",
    "openssh-auth.log": "1e4912727fa88245113d41b16a0cd25ceadba7f931e1c406542885b91254264f",
}
# Nov-2026 prices used by the gate (sources: research notes economia_mercado.md / coste_inferencia_*.md)
PRICES = {
    "vcpu_spot_c7i": 0.014, "vcpu_ondemand_c7i": 0.0446,          # $/vCPU-h, us-east-1
    "t4_spot_g4dn": 0.247, "t4_ondemand_g4dn": 0.526,            # $/h, g4dn.xlarge (T4 + 4 vCPU)
    "s3_standard_gb_month": 0.023, "b2_gb_month": 0.00695, "deep_archive_gb_month": 0.00099,
}
RETENTION_MONTHS = 84   # 7 years

RESULTS = []    # (phase, file, size_bytes, system, config, out_bytes, bpb, seconds, bytes_per_s, note)
CLASSICAL = []  # (file, size_bytes, tool, out_bytes, bpb, comp_s, comp_MBs, decomp_s, decomp_MBs)
REPORT = []


def log(msg):
    print(msg, flush=True)
    REPORT.append(msg)


def section(title):
    log(f"\n{'=' * 72}\n  {title}   [t+{(time.time() - T_START) / 60:.1f} min]\n{'=' * 72}")


def hours_left():
    return DEADLINE_HOURS - (time.time() - T_START) / 3600


def run(cmd, cwd=None, timeout=36000, check=True, quiet=False):
    if not quiet:
        print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, timeout=timeout, capture_output=True, text=True)
    if not quiet and r.stdout.strip():
        print(r.stdout[-2000:])
    if r.stderr.strip() and (not quiet or r.returncode != 0):
        print(r.stderr[-2000:], file=sys.stderr)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def record(phase, file, size, system, config, out_bytes, seconds, note=""):
    bpb = out_bytes * 8 / size if size else 0.0
    bps = size / seconds if seconds > 0 else 0.0
    RESULTS.append((phase, file, size, system, config, out_bytes, f"{bpb:.4f}", f"{seconds:.1f}",
                    f"{bps:.1f}", note))
    log(f"  [{phase}] {file:28s} {size:>10,d} B  {system:14s} {config:22s} bpb={bpb:.4f}  "
        f"{seconds:8.1f} s  {bps:10.1f} B/s  {note}")
    flush_outputs()


def flush_outputs():
    with open(os.path.join(WORK_DIR, "fullsize-results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "file", "size_bytes", "system", "config", "out_bytes", "bpb", "seconds",
                    "bytes_per_s", "note"])
        w.writerows(RESULTS)
    with open(os.path.join(WORK_DIR, "classical-fullsize.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "size_bytes", "tool", "out_bytes", "bpb", "comp_s", "comp_MBs", "decomp_s",
                    "decomp_MBs"])
        w.writerows(CLASSICAL)
    with open(os.path.join(WORK_DIR, "fullsize-report.txt"), "w") as f:
        f.write("\n".join(REPORT) + "\n")


# ---------------------------------------------------------------- 1. source + build

def get_repo(explicit):
    if explicit:
        return os.path.abspath(explicit)
    for cand in glob.glob("/kaggle/input/redukt-src*") + glob.glob("/kaggle/input/*/redukt"):
        if os.path.isfile(os.path.join(cand, "Cargo.toml")):
            dst = os.path.join(WORK_DIR, "redukt")
            if not os.path.isdir(dst):
                shutil.copytree(cand, dst)
            return dst
    dst = os.path.join(WORK_DIR, "redukt")
    if not os.path.isdir(dst):
        run(f"git clone {REPO_URL} {dst}")
        run(f"git checkout {REDUKT_REF}", cwd=dst)
    return dst


def build(repo):
    section("1. Build redukt (maxperf) + CUDA plugin")
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.path.expanduser("~/.cargo/bin:") + os.environ["PATH"]
    run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", check=False)
    run("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Flags' | cut -c1-160", check=False)
    run("free -g | head -2", check=False)
    if run("cargo --version", check=False).returncode != 0:
        run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    t0 = time.time()
    run("cargo build --profile maxperf", cwd=repo)
    log(f"  cargo build maxperf: {time.time() - t0:.0f} s")
    plugin_dir = os.path.join(repo, "plugins", "redukt-gpu-cuda")
    run("make clean all", cwd=plugin_dir)
    exe_dir = os.path.join(repo, "target", "maxperf")
    shutil.copy(os.path.join(plugin_dir, "libredukt_gpu.so"), exe_dir)
    return os.path.join(exe_dir, "redukt")


# ---------------------------------------------------------------- 2. weights + data

def prepare_weights(repo):
    section("2. mamba-130m weights -> model.rbin (plugin)")
    wdir = os.path.join(repo, "weights", "mamba-130m")
    if not os.path.isfile(os.path.join(wdir, "model.safetensors")):
        run("pip install -q huggingface_hub safetensors")
        from huggingface_hub import snapshot_download
        snapshot_download("state-spaces/mamba-130m-hf", local_dir=wdir,
                          allow_patterns=["*.safetensors", "*.json"])
    rbin = os.path.join(wdir, "model.rbin")
    if not os.path.isfile(rbin):
        run(f"{sys.executable} scripts/convert-weights-gpu.py --input {wdir}/model.safetensors "
            f"--output {rbin} --rbin-version 2 --tokenizer bpe", cwd=repo)
    return wdir


def prepare_data(repo):
    section("3. Data: enwik8 prefixes + industry files (regenerated, sha256-checked)")
    data = os.path.join(WORK_DIR, "data")
    os.makedirs(data, exist_ok=True)
    full = os.path.join(data, "enwik8")
    if not os.path.isfile(full):
        run(f"curl -sL {ENWIK8_URL} -o {data}/enwik8.zip && unzip -o -q {data}/enwik8.zip -d {data}")
    blob = open(full, "rb").read()
    prefixes = {}
    for n in PREFIXES:
        p = os.path.join(data, f"enwik8.{n // 1024}k" if n < 1_048_576 else f"enwik8.{n // 1_048_576}m")
        if not os.path.isfile(p):
            open(p, "wb").write(blob[:n])
        prefixes[n] = p
    # industry files: scripts/prepare-benchmark-data.py downloads Loghub logs and generates the XML batches
    run(f"{sys.executable} scripts/prepare-benchmark-data.py", cwd=repo, check=False)
    ind_dir = os.path.join(repo, "data", "benchmark", "industry")
    industry = {}
    for name in INDUSTRY:
        p = os.path.join(ind_dir, name)
        if not os.path.isfile(p):
            log(f"  MISSING industry file: {name} (prepare-benchmark-data.py failed?)")
            continue
        same = sha(p) == LOCAL_SHA[name]
        log(f"  {name:26s} {os.path.getsize(p):>8,d} B  sha256 == local copy: {'YES' if same else 'NO'}")
        industry[name] = p
    # 12 x 10 KB slices for the per-vCPU lockstep measurement (same shape as the local lockstep CSV)
    slices = os.path.join(data, "slices10k")
    os.makedirs(slices, exist_ok=True)
    for i in range(7):   # 7 slices of enwik8 at different offsets + 5 industry heads = 12 files
        open(os.path.join(slices, f"enwik8_{i}.10k"), "wb").write(blob[i * 1_000_000:i * 1_000_000 + 10_240])
    for name, p in industry.items():
        open(os.path.join(slices, name + ".10k"), "wb").write(open(p, "rb").read()[:10_240])
    log(f"  slices10k: {len(os.listdir(slices))} files")
    return full, prefixes, industry, slices


# ---------------------------------------------------------------- 4. classical codecs

def install_classical():
    r = run("apt-get install -y -q zstd xz-utils brotli > /dev/null 2>&1 || "
            "(apt-get update -q > /dev/null 2>&1 && apt-get install -y -q zstd xz-utils brotli > /dev/null 2>&1)",
            check=False)
    have = {t: shutil.which(t) is not None for t in ("gzip", "zstd", "xz", "brotli")}
    log(f"  classical tools: {have} (apt rc={r.returncode})")
    for t in ("zstd", "xz", "brotli", "gzip"):
        if have[t]:
            run(f"{t} --version 2>&1 | head -1", check=False)
    return have


def classical_one(path, name, tool, comp_cmd, decomp_cmd, out, back):
    t0 = time.time()
    run(comp_cmd, check=True, quiet=True)
    comp_s = time.time() - t0
    t0 = time.time()
    run(decomp_cmd, check=True, quiet=True)
    decomp_s = time.time() - t0
    size, osize = os.path.getsize(path), os.path.getsize(out)
    ok = open(back, "rb").read() == open(path, "rb").read()
    mbs = lambda s: size / 1e6 / s if s > 0 else 0.0
    CLASSICAL.append((name, size, tool, osize, f"{osize * 8 / size:.4f}", f"{comp_s:.2f}", f"{mbs(comp_s):.2f}",
                      f"{decomp_s:.2f}", f"{mbs(decomp_s):.1f}"))
    log(f"  [classical] {name:28s} {tool:14s} bpb={osize * 8 / size:.4f}  comp {mbs(comp_s):7.2f} MB/s  "
        f"decomp {mbs(decomp_s):8.1f} MB/s  roundtrip {'OK' if ok else 'FAIL'}")
    for p in (out, back):
        if os.path.exists(p):
            os.remove(p)


def classical(files, have):
    section("4. Classical codecs on whole files + enwik8 prefixes")
    tmp = os.path.join(WORK_DIR, "tmp")
    os.makedirs(tmp, exist_ok=True)
    for name, path in files.items():
        o, b = os.path.join(tmp, name + ".cmp"), os.path.join(tmp, name + ".back")
        cmds = []
        if have["gzip"]:
            cmds.append(("gzip-9", f"gzip -9 -c '{path}' > '{o}'", f"gzip -d -c '{o}' > '{b}'"))
        if have["zstd"]:
            cmds.append(("zstd-19", f"zstd -19 -q -f -T1 '{path}' -o '{o}'", f"zstd -d -q -f '{o}' -o '{b}'"))
            cmds.append(("zstd-22-ultra-long", f"zstd --ultra -22 --long=27 -q -f -T1 '{path}' -o '{o}'",
                         f"zstd -d -q -f --long=27 '{o}' -o '{b}'"))
        if have["xz"]:
            cmds.append(("xz-9e", f"xz -9e -k -c -T1 '{path}' > '{o}'", f"xz -d -c '{o}' > '{b}'"))
        if have["brotli"]:
            cmds.append(("brotli-11-w24", f"brotli -q 11 -w 24 -f -c '{path}' > '{o}'",
                         f"brotli -d -f -c '{o}' > '{b}'"))
        for tool, c, d in cmds:
            try:
                classical_one(path, name, tool, c, d, o, b)
            except Exception as e:   # noqa: BLE001 — one failing tool must not stop the benchmark
                log(f"  [classical] {name} {tool}: FAILED ({e})")
        flush_outputs()


# ---------------------------------------------------------------- 5. Redukt runs

def attempt(label, fn, *args):
    """Run one measurement; a failure is logged and never stops the remaining phases."""
    try:
        return fn(*args)
    except Exception as e:   # noqa: BLE001
        log(f"  FAILED {label}: {str(e)[:300]}")
        return None


def redukt_compress(redukt, repo, wdir, path, out, backbone, quant=None, extra=""):
    q = f" --quant {quant}" if quant else ""
    cmd = (f"{redukt} compress --input '{path}' --output '{out}' --backbone {backbone}{q} "
           f"--weights {wdir} --no-governor {extra}")
    t0 = time.time()
    run(cmd, cwd=repo, quiet=True)
    return time.time() - t0


def redukt_roundtrip(redukt, repo, wdir, azth, original):
    dec = azth + ".dec"
    t0 = time.time()
    run(f"{redukt} decompress --input '{azth}' --output '{dec}' --weights {wdir} --no-governor", cwd=repo, quiet=True)
    dt = time.time() - t0
    ok = open(dec, "rb").read() == open(original, "rb").read()
    os.remove(dec)
    return ok, dt


def estimate_seconds(size, rate_bps, default_bps):
    return size / (rate_bps or default_bps)


def scale_curve_gpu(redukt, repo, wdir, prefixes):
    section("A. Scale curve enwik8 — GPU plugin single stream (F32)")
    out = os.path.join(WORK_DIR, "azth")
    os.makedirs(out, exist_ok=True)
    rate = 0.0
    for n, path in sorted(prefixes.items()):
        need_h = estimate_seconds(n, rate, 450.0) / 3600 * 1.15
        if need_h > hours_left() - 1.5:   # keep 1.5 h for phases B/C
            log(f"  skip enwik8 {n:,} B: needs ~{need_h:.2f} h, only {hours_left():.2f} h left")
            continue
        azth = os.path.join(out, f"enwik8.{n}.plugin.azth")
        dt = attempt(f"gpu enwik8[{n}]", redukt_compress, redukt, repo, wdir, path, azth, "plugin")
        if dt is None:
            continue
        osz = os.path.getsize(azth)
        rate = n / dt
        note = f"sha256 {sha(azth)[:16]}"
        if n in (10_240, 102_400):
            rt = attempt(f"roundtrip enwik8[{n}]", redukt_roundtrip, redukt, repo, wdir, azth, path)
            if rt:
                note += f"; roundtrip {'OK' if rt[0] else 'FAIL'} decompress {rt[1]:.0f}s"
        record("A-scale", f"enwik8[{n}]", n, "redukt-T4", "plugin F32 single", osz, dt, note)


def scale_points_cpu(redukt, repo, wdir, prefixes):
    section("A'. Same machine, CPU Mamba F32 / Q8 at 10 KB and 100 KB")
    out = os.path.join(WORK_DIR, "azth")
    os.makedirs(out, exist_ok=True)
    for n in (10_240, 102_400):
        for quant in ("f32", "q8"):
            if hours_left() < 1.2:
                log("  skip CPU points: deadline")
                return
            path = prefixes[n]
            azth = os.path.join(out, f"enwik8.{n}.cpu.{quant}.azth")
            dt = attempt(f"cpu {quant} enwik8[{n}]", redukt_compress, redukt, repo, wdir, path, azth, "mamba", quant)
            if dt is None:
                continue
            record("A-scale", f"enwik8[{n}]", n, "redukt-CPU4", f"mamba {quant} single", os.path.getsize(azth), dt,
                   f"sha256 {sha(azth)[:16]}")


def industry_full(redukt, repo, wdir, industry):
    section("B. Whole industry files — GPU plugin single stream (F32)")
    out = os.path.join(WORK_DIR, "azth")
    os.makedirs(out, exist_ok=True)
    for name, path in industry.items():
        n = os.path.getsize(path)
        if estimate_seconds(n, 0, 450.0) / 3600 > hours_left() - 0.6:
            log(f"  skip {name}: deadline")
            continue
        azth = os.path.join(out, f"{name}.plugin.azth")
        dt = attempt(f"gpu {name}", redukt_compress, redukt, repo, wdir, path, azth, "plugin")
        if dt is None:
            continue
        note = f"sha256 {sha(azth)[:16]}"
        if name.endswith(".xml"):
            rt = attempt(f"roundtrip {name}", redukt_roundtrip, redukt, repo, wdir, azth, path)
            if rt:
                note += f"; roundtrip {'OK' if rt[0] else 'FAIL'} decompress {rt[1]:.0f}s"
        record("B-full", name, n, "redukt-T4", "plugin F32 single", os.path.getsize(azth), dt, note)


def lockstep_cpu(redukt, repo, wdir, slices):
    section("C. Lockstep 12 x 10 KB — CPU Q8 (per-vCPU throughput) and plugin (sequential fallback)")
    total = sum(os.path.getsize(os.path.join(slices, f)) for f in os.listdir(slices))
    nfiles = len(os.listdir(slices))
    for system, backbone, quant, jobs in (("redukt-CPU4", "mamba", "q8", 4), ("redukt-CPU4", "mamba", "q8", 12),
                                          ("redukt-CPU4", "mamba", "f32", 12), ("redukt-T4", "plugin", None, 12)):
        if hours_left() < 0.4:
            log("  skip lockstep: deadline")
            return
        outd = os.path.join(WORK_DIR, f"lockstep_{backbone}_{quant}_{jobs}")
        shutil.rmtree(outd, ignore_errors=True)
        os.makedirs(outd)
        q = f" --quant {quant}" if quant else ""
        t0 = time.time()
        r = attempt(f"lockstep {backbone} {quant} jobs={jobs}", run,
                    f"{redukt} compress --input-dir '{slices}' --output-dir '{outd}' --jobs {jobs} "
                    f"--backbone {backbone}{q} --weights {wdir} --no-governor", repo, 36000, True, True)
        if r is None:
            continue
        dt = time.time() - t0
        osz = sum(os.path.getsize(os.path.join(outd, f)) for f in os.listdir(outd) if f.endswith(".azth"))
        note = "aggregate incl. model load"
        if backbone == "plugin":
            # R-109 gate: lockstep over FFI v2 slots must be byte-identical to single-file plugin output.
            same = []
            for f in sorted(os.listdir(slices))[:3]:
                single = os.path.join(WORK_DIR, f"single_plugin_{f}.azth")
                one = attempt(f"single plugin {f}", redukt_compress, redukt, repo, wdir, os.path.join(slices, f),
                              single, "plugin")
                if one is not None:
                    same.append(sha(single) == sha(os.path.join(outd, f + ".azth")))
            note += f"; batch==single {sum(same)}/{len(same)} ({'PASS' if same and all(same) else 'FAIL'})"
        record("C-lockstep", f"slices10k[{nfiles}]", total, system, f"{backbone} {quant or 'f32'} jobs={jobs}", osz, dt,
               note)


# ---------------------------------------------------------------- 6. gate

def gate(industry, prefixes):
    section("D. Economic gate (Nov-2026 prices; formula: $/GB = price_h * 1e9 / (B/s * 3600))")
    rows = []

    def find(phase_prefix, file_key, system):
        for r in RESULTS:
            if r[0].startswith(phase_prefix) and r[1] == file_key and r[3] == system:
                return r
        return None

    def classical_bpb(file_key, tool):
        for r in CLASSICAL:
            if r[0] == file_key and r[2] == tool:
                return float(r[4])
        return None

    # throughput per vCPU and per T4
    cpu_rows = [r for r in RESULTS if r[0] == "C-lockstep" and r[3] == "redukt-CPU4"]
    best_cpu_bps = max((float(r[8]) for r in cpu_rows), default=0.0)
    gpu_rows = [r for r in RESULTS if r[3] == "redukt-T4" and r[0] == "A-scale"]
    best_gpu_bps = max((float(r[8]) for r in gpu_rows), default=0.0)
    per_vcpu = best_cpu_bps / 4.0 if best_cpu_bps else 0.0
    cost_cpu_spot = PRICES["vcpu_spot_c7i"] * 1e9 / (per_vcpu * 3600) if per_vcpu else float("inf")
    cost_gpu_spot = PRICES["t4_spot_g4dn"] * 1e9 / (best_gpu_bps * 3600) if best_gpu_bps else float("inf")
    log(f"  CPU lockstep best aggregate {best_cpu_bps:.0f} B/s on 4 vCPU = {per_vcpu:.0f} B/s per vCPU "
        f"-> ${cost_cpu_spot:,.1f}/GB (c7i spot)")
    log(f"  T4 single-stream best {best_gpu_bps:.0f} B/s -> ${cost_gpu_spot:,.1f}/GB (g4dn spot); "
        f"FFI batch N=128 would be ~7.4x faster (1,117 vs 151 tok/s, 2026-11-10)")

    # value of the margin per raw GB, against the best classical codec on the WHOLE file
    largest = max((r for r in RESULTS if r[0] == "A-scale" and r[3] == "redukt-T4"), key=lambda r: r[2], default=None)
    targets = []
    if largest:
        targets.append((largest[1], float(largest[6])))
    for name in industry:
        r = find("B-full", name, "redukt-T4")
        if r:
            targets.append((name, float(r[6])))
    for key, rbpb in targets:
        best_tool, best_bpb = None, None
        for tool in ("brotli-11-w24", "xz-9e", "zstd-22-ultra-long", "zstd-19", "gzip-9"):
            b = classical_bpb(key, tool)
            if b is not None and (best_bpb is None or b < best_bpb):
                best_tool, best_bpb = tool, b
        if best_bpb is None:
            continue
        delta_gb = max(best_bpb - rbpb, 0.0) / 8.0          # GB saved per raw GB vs best classical
        for store, price in (("S3 Standard", PRICES["s3_standard_gb_month"]), ("Backblaze B2", PRICES["b2_gb_month"]),
                             ("Deep Archive", PRICES["deep_archive_gb_month"])):
            value = delta_gb * price * RETENTION_MONTHS
            rstar = PRICES["vcpu_spot_c7i"] * 1e9 / (3600 * value) if value > 0 else float("inf")
            speedup = rstar / per_vcpu if per_vcpu else float("inf")
            rows.append((key, f"{rbpb:.4f}", best_tool, f"{best_bpb:.4f}", f"{delta_gb:.4f}", store,
                         f"{value:.4f}", f"{rstar / 1e3:.1f}", f"{per_vcpu / 1e3:.3f}", f"{speedup:.0f}",
                         f"{cost_cpu_spot:.1f}", f"{cost_gpu_spot:.1f}"))
            log(f"  {key:26s} redukt {rbpb:.4f} vs {best_tool} {best_bpb:.4f}: saves {delta_gb:.4f} GB/GB; "
                f"{store:12s} 7y value ${value:.4f}/GB; R*={rstar / 1e3:.1f} KB/s per vCPU; "
                f"measured {per_vcpu / 1e3:.3f} KB/s -> fast tier needs x{speedup:.0f}")
    with open(os.path.join(WORK_DIR, "gate-table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "redukt_bpb", "best_classical", "classical_bpb", "gb_saved_per_raw_gb", "storage",
                    "value_7y_usd_per_gb", "rstar_kbs_per_vcpu", "measured_kbs_per_vcpu", "speedup_needed",
                    "cost_cpu_spot_usd_per_gb", "cost_t4_spot_usd_per_gb"])
        w.writerows(rows)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Redukt full-size benchmark (Kaggle T4 + CPU)")
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.join(WORK_DIR, "azth"), exist_ok=True)
    log(f"deadline {DEADLINE_HOURS} h; ref {REDUKT_REF}; phases {PHASES}")
    repo = get_repo(args.repo)
    redukt = build(repo)
    wdir = prepare_weights(repo)
    full, prefixes, industry, slices = prepare_data(repo)
    have = install_classical()
    files = {f"enwik8[{n}]": p for n, p in prefixes.items()}
    files["enwik8[100000000]"] = full
    files.update(industry)
    classical(files, have)
    try:
        if "C" in PHASES:
            lockstep_cpu(redukt, repo, wdir, slices)      # short, gives the per-vCPU number early
        if "A2" in PHASES:
            scale_points_cpu(redukt, repo, wdir, prefixes)
        if "B" in PHASES:
            industry_full(redukt, repo, wdir, industry)
        if "A" in PHASES:
            scale_curve_gpu(redukt, repo, wdir, prefixes)  # longest, consumes whatever time is left
    finally:
        gate(industry, prefixes)
        flush_outputs()
        log(f"\nDONE in {(time.time() - T_START) / 3600:.2f} h")


if __name__ == "__main__":
    main()
