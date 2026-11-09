#!/usr/bin/env python3
"""
Redukt GPU CUDA Plugin — Validation & Benchmark (R-090 + R-096 FFI v2 batch)

Kaggle script (T4) / standalone Linux+CUDA script that:
  1. Gets the repo (Kaggle dataset `redukt-src`, a local checkout, or git clone of REDUKT_REF)
  2. Builds redukt (cargo --release) and the CUDA plugin (make -> libredukt_gpu.so)
  3. Downloads mamba-130m (BPE, vocab 50280) and converts it to .rbin v2 (tokenizer=bpe)
  4. FFI checks via ctypes: abi_version == 2, v1 logits vs a NumPy reference of the Mamba step,
     v1 vs forward_batch bit-exactness (N=8, arbitrary slots), slot independence, reset
  5. CLI: compress + decompress roundtrip with `--backbone plugin` on enwik8 10KB, determinism,
     BPB vs the CPU reference 0.9609 (and a same-machine CPU `--backbone mamba` run)
  6. Throughput: v1 tok/s and batch N = 1, 8, 32, 128 (aggregate tok/s)
  7. Writes /kaggle/working/cuda-plugin-results.csv and cuda-plugin-report.txt

Launch on Kaggle (from the repo root, after scripts/kaggle-setup.sh):
    cp notebooks/redukt-gpu-cuda-test.py notebooks/redukt-gpu-cuda-test/
    python -m kaggle kernels push -p notebooks/redukt-gpu-cuda-test
    python -m kaggle kernels status <user>/redukt-gpu-cuda-test
    python -m kaggle kernels output <user>/redukt-gpu-cuda-test -p data/benchmark/results/kaggle-cuda
The source must contain the R-096 changes: either push them to a branch and set REDUKT_REF
below (or env), or attach a Kaggle dataset named `redukt-src` holding the repo directory.

Local (Linux + NVIDIA GPU):  python notebooks/redukt-gpu-cuda-test.py --repo .

Copyright 2026 Redukt Contributors. Apache-2.0 License.
"""

import argparse
import csv
import ctypes
import glob
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

REPO_URL = os.environ.get("REDUKT_REPO", "https://github.com/DavidCaastro/redukt.git")
REDUKT_REF = os.environ.get("REDUKT_REF", "master")
WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("cuda-test-out")
CPU_REF_BPB = 0.9609          # CLAUDE.md: Mamba+CM AZTH v4, enwik8 10KB, CPU
BATCH_SIZES = (1, 8, 32, 128)
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
BACKEND_OK, BACKEND_INIT_FAILED, BACKEND_INVALID_ARGUMENT = 0, 1, 5

RESULTS = []   # (metric, value, note)
REPORT = []


def log(msg):
    print(msg, flush=True)
    REPORT.append(msg)


def record(metric, value, note=""):
    RESULTS.append((metric, value, note))
    log(f"  {metric:34s} {value}  {note}")


def run(cmd, cwd=None, timeout=3600, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, timeout=timeout, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout[-3000:])
    if r.stderr.strip():
        print(r.stderr[-3000:], file=sys.stderr)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def section(title):
    log(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


# ---------------------------------------------------------------- 1. source + build

def get_repo(explicit):
    if explicit:
        return os.path.abspath(explicit)
    for cand in glob.glob("/kaggle/input/redukt-src*") + glob.glob("/kaggle/input/*/redukt"):
        if os.path.isfile(os.path.join(cand, "Cargo.toml")):
            dst = os.path.join(WORK_DIR, "redukt")
            if not os.path.isdir(dst):
                shutil.copytree(cand, dst)   # Kaggle inputs are read-only
            return dst
    dst = os.path.join(WORK_DIR, "redukt")
    if not os.path.isdir(dst):
        run(f"git clone --depth 1 --branch {REDUKT_REF} {REPO_URL} {dst}")
    return dst


def build(repo):
    section("1. Build redukt + CUDA plugin")
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.path.expanduser("~/.cargo/bin:") + os.environ["PATH"]
    run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", check=False)
    if run("cargo --version", check=False).returncode != 0:
        run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    run("cargo build --release", cwd=repo)
    plugin_dir = os.path.join(repo, "plugins", "redukt-gpu-cuda")
    t0 = time.time()
    r = run("make clean all", cwd=plugin_dir)
    record("nvcc_build_seconds", f"{time.time() - t0:.1f}")
    warnings = [ln for ln in (r.stdout + r.stderr).splitlines() if "warning" in ln.lower()]
    record("nvcc_warnings", len(warnings))
    exe_dir = os.path.join(repo, "target", "release")
    # loader.rs looks for libredukt_gpu.so next to the real redukt executable
    shutil.copy(os.path.join(plugin_dir, "libredukt_gpu.so"), exe_dir)
    return os.path.join(exe_dir, "redukt"), os.path.join(exe_dir, "libredukt_gpu.so")


# ---------------------------------------------------------------- 2. weights + data

def prepare_weights(repo):
    section("2. mamba-130m -> model.rbin (v2, tokenizer=bpe)")
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
    hdr = struct.unpack("<11I", open(rbin, "rb").read(44))
    record("rbin_header", f"v{hdr[1]} vocab={hdr[2]} tok={hdr[10]}", "expect v2 vocab=50280 tok=1")
    assert hdr[1] == 2 and hdr[2] == 50280 and hdr[10] == 1, "unexpected .rbin header"
    return wdir, rbin


def prepare_enwik8():
    data_dir = os.path.join(WORK_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "enwik8.10k")
    if not os.path.isfile(path):
        run(f"curl -sL {ENWIK8_URL} -o {data_dir}/enwik8.zip && unzip -o -q {data_dir}/enwik8.zip -d {data_dir}")
        with open(os.path.join(data_dir, "enwik8"), "rb") as f, open(path, "wb") as g:
            g.write(f.read(10240))
    return path


# ---------------------------------------------------------------- 3. NumPy reference

class RefMamba:
    """Float32 NumPy Mamba-1 single-step reference reading the .rbin directly."""

    def __init__(self, rbin):
        h = struct.unpack("<11I", open(rbin, "rb").read(44))
        (_, _, self.vs, self.d, self.di, self.ds, self.dc, self.dr, self.nl, flags, _) = h
        self.ln = bool(flags & 1)
        mm = np.memmap(rbin, dtype=np.float32, mode="r", offset=44)
        self.off = 0

        def take(*shape):
            n = int(np.prod(shape))
            a = np.asarray(mm[self.off:self.off + n]).reshape(shape)
            self.off += n
            return a

        d, di, ds, dc, dr = self.d, self.di, self.ds, self.dc, self.dr
        self.emb = take(self.vs, d)
        self.layers = []
        for _ in range(self.nl):
            L = {"nw": take(d), "nb": take(d) if self.ln else None, "inp": take(2 * di, d),
                 "cw": take(di, dc), "cb": take(di), "xp": take(dr + 2 * ds, di), "dtw": take(di, dr),
                 "dtb": take(di), "alog": take(di, ds), "D": take(di), "out": take(d, di)}
            self.layers.append(L)
        self.nfw = take(d)
        self.nfb = take(d) if self.ln else None
        self.head = take(self.vs, d) if flags & 2 else self.emb
        self.reset()

    def reset(self):
        self.conv = [np.zeros((self.di, self.dc - 1), np.float32) for _ in range(self.nl)]
        self.ssm = [np.zeros((self.di, self.ds), np.float32) for _ in range(self.nl)]

    def norm(self, x, w, b):
        if b is None:
            return x / np.sqrt(np.mean(x * x) + 1e-5) * w
        m = x.mean()
        return (x - m) / np.sqrt(np.mean(x * x) - m * m + 1e-5) * w + b

    def step(self, tok):
        silu = lambda v: v / (1.0 + np.exp(-v))
        x = self.emb[tok].astype(np.float32).copy()
        for i, L in enumerate(self.layers):
            xz = L["inp"] @ self.norm(x, L["nw"], L["nb"])
            xb, z = xz[:self.di], xz[self.di:]
            st = self.conv[i]
            v = (st * L["cw"][:, :-1]).sum(1) + xb * L["cw"][:, -1] + L["cb"]
            st[:, :-1], st[:, -1] = st[:, 1:].copy(), xb
            xc = silu(v)
            zp = L["xp"] @ xc
            dtv = L["dtw"] @ zp[:self.dr] + L["dtb"]
            dt = np.where(dtv > 20, dtv, np.log1p(np.exp(np.minimum(dtv, 20))))
            B, C = zp[self.dr:self.dr + self.ds], zp[self.dr + self.ds:]
            dA = np.exp(dt[:, None] * -np.exp(L["alog"]))
            self.ssm[i] = self.ssm[i] * dA + (xc * dt)[:, None] * B[None, :]
            y = self.ssm[i] @ C + L["D"] * xc
            x = x + L["out"] @ (y * silu(z))
        return self.head @ self.norm(x, self.nfw, self.nfb)


# ---------------------------------------------------------------- 4. FFI checks

class BackendInfo(ctypes.Structure):
    _fields_ = [("ffi_version", ctypes.c_uint32), ("vocab_size", ctypes.c_uint32),
                ("hidden_dim", ctypes.c_uint32), ("name", ctypes.c_uint8 * 64)]


def load_plugin(lib_path, rbin):
    lib = ctypes.CDLL(lib_path)
    u32p, f32p = ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float)
    lib.redukt_backend_init.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(BackendInfo)]
    lib.redukt_backend_forward.argtypes = [ctypes.c_uint32, f32p, ctypes.c_uint32]
    lib.redukt_backend_abi_version.restype = ctypes.c_uint32
    lib.redukt_backend_batch_init.argtypes = [ctypes.c_uint32]
    lib.redukt_backend_batch_reset.argtypes = [u32p, ctypes.c_uint32]
    lib.redukt_backend_forward_batch.argtypes = [ctypes.c_uint32, u32p, u32p, f32p, ctypes.c_uint32]
    for fn in ("init", "forward"):
        getattr(lib, f"redukt_backend_{fn}").restype = ctypes.c_int
    for fn in ("batch_init", "batch_reset", "forward_batch"):
        getattr(lib, f"redukt_backend_{fn}").restype = ctypes.c_int32
    info = BackendInfo()
    rc = lib.redukt_backend_init(rbin.encode() + b"\0", 0, ctypes.byref(info))
    assert rc == BACKEND_OK, f"init failed: {rc}"
    return lib, info


def ptr(a, ct):
    return a.ctypes.data_as(ctypes.POINTER(ct))


def v1_forward(lib, tok, vs):
    out = np.empty(vs, np.float32)
    assert lib.redukt_backend_forward(tok, ptr(out, ctypes.c_float), vs) == BACKEND_OK
    return out


def batch_forward(lib, slots, toks, vs):
    s, t = np.asarray(slots, np.uint32), np.asarray(toks, np.uint32)
    out = np.empty((len(s), vs), np.float32)
    rc = lib.redukt_backend_forward_batch(len(s), ptr(s, ctypes.c_uint32), ptr(t, ctypes.c_uint32),
                                          ptr(out, ctypes.c_float), vs)
    return rc, out


def ffi_checks(lib, info, rbin, tokens):
    section("3. FFI v1/v2 correctness")
    vs = info.vocab_size
    record("ffi_version", info.ffi_version, "expect 1 (base contract unchanged)")
    record("abi_version", lib.redukt_backend_abi_version(), "expect 2")
    seq = [int(t) for t in tokens[:16]]

    ref = RefMamba(rbin)
    lib.redukt_backend_reset_state()
    v1 = [v1_forward(lib, t, vs) for t in seq]
    worst, argmax_ok = 0.0, 0
    for g, t in zip(v1[:8], seq[:8]):
        r = ref.step(t)
        worst = max(worst, float(np.max(np.abs(g - r)) / (np.max(np.abs(r)) + 1e-9)))
        argmax_ok += int(np.argmax(g) == np.argmax(r))
    record("v1_vs_numpy_max_rel_err", f"{worst:.2e}", "8 steps, expect < 1e-3")
    record("v1_vs_numpy_argmax_match", f"{argmax_ok}/8")

    assert lib.redukt_backend_batch_init(8) == BACKEND_OK
    slots = [5, 0, 7, 2, 1, 3, 6, 4]   # stream of interest lives in slot 5, batch position 0
    other = np.random.default_rng(0).integers(0, 50000, size=(len(seq), 7))
    exact = True
    for i, t in enumerate(seq):
        rc, out = batch_forward(lib, slots, [t] + [int(x) for x in other[i]], vs)
        assert rc == BACKEND_OK, f"forward_batch rc={rc}"
        exact &= bool(np.array_equal(out[0], v1[i]))
    record("v1_vs_batch8_bit_exact", "PASS" if exact else "FAIL", "16 steps, slot 5")

    s = np.array([5], np.uint32)
    assert lib.redukt_backend_batch_reset(ptr(s, ctypes.c_uint32), 1) == BACKEND_OK
    rc, out = batch_forward(lib, [5], [seq[0]], vs)
    record("batch_reset_slot_then_step0", "PASS" if np.array_equal(out[0], v1[0]) else "FAIL")
    rc, _ = batch_forward(lib, [1, 1], [0, 0], vs)
    record("duplicate_slot_rejected", "PASS" if rc == BACKEND_INVALID_ARGUMENT else f"FAIL rc={rc}")
    if worst > 1e-3:
        record("v1_vs_numpy_tolerance", "FAIL", f"{worst:.2e} > 1e-3")


# ---------------------------------------------------------------- 5. CLI roundtrip

def cli_roundtrip(redukt, repo, wdir, data):
    section("4. CLI roundtrip --backbone plugin (enwik8 10KB)")
    out = os.path.join(WORK_DIR, "data")
    runs = {}
    for tag, extra in (("gpu1", "--backbone plugin"), ("gpu2", "--backbone plugin"), ("cpu", "--backbone mamba")):
        azth = f"{out}/enwik8.{tag}.azth"
        t0 = time.time()
        run(f"{redukt} compress --input {data} --output {azth} {extra} --weights {wdir} --bytes 10240", cwd=repo)
        runs[tag] = (azth, time.time() - t0)
    size = os.path.getsize(runs["gpu1"][0])
    bpb = size * 8 / os.path.getsize(data)
    cpu_bpb = os.path.getsize(runs["cpu"][0]) * 8 / os.path.getsize(data)
    record("bpb_plugin", f"{bpb:.4f}", f"CPU ref {CPU_REF_BPB} (delta {100 * (bpb / CPU_REF_BPB - 1):+.2f}%)")
    record("bpb_cpu_same_machine", f"{cpu_bpb:.4f}")
    record("compress_seconds_plugin", f"{runs['gpu1'][1]:.1f}", f"cpu {runs['cpu'][1]:.1f}")
    same = open(runs["gpu1"][0], "rb").read() == open(runs["gpu2"][0], "rb").read()
    record("determinism_plugin_2_runs", "PASS" if same else "FAIL")
    dec = f"{out}/enwik8.gpu1.dec"
    run(f"{redukt} decompress --input {runs['gpu1'][0]} --output {dec} --weights {wdir}", cwd=repo)
    ok = open(dec, "rb").read() == open(data, "rb").read()
    record("roundtrip_plugin", "PASS" if ok else "FAIL", "CRC-32 verified by decoder")


# ---------------------------------------------------------------- 6. throughput

def throughput(lib, info, tokens):
    section("5. Throughput (T4)")
    vs = info.vocab_size
    toks = [int(t) for t in tokens]
    lib.redukt_backend_reset_state()
    for t in toks[:20]:
        v1_forward(lib, t, vs)
    n = 500
    t0 = time.perf_counter()
    for i in range(n):
        v1_forward(lib, toks[i % len(toks)], vs)
    record("v1_tok_per_s", f"{n / (time.perf_counter() - t0):.1f}", "single stream")
    for nb in BATCH_SIZES:
        assert lib.redukt_backend_batch_init(nb) == BACKEND_OK
        slots = list(range(nb))
        steps = 200 if nb <= 32 else 60
        batch_forward(lib, slots, toks[:nb] if nb <= len(toks) else [1] * nb, vs)  # warm-up
        t0 = time.perf_counter()
        for i in range(steps):
            step_toks = [toks[(i + s) % len(toks)] for s in range(nb)]
            rc, _ = batch_forward(lib, slots, step_toks, vs)
            assert rc == BACKEND_OK
        dt = time.perf_counter() - t0
        record(f"batch{nb}_tok_per_s", f"{nb * steps / dt:.1f}", f"{1000 * dt / steps:.2f} ms/step")
    lib.redukt_backend_batch_init(0)


# ---------------------------------------------------------------- main

def sample_tokens(wdir, data, n=512):
    """Realistic BPE ids of enwik8 (HF `tokenizers` if available, else a fixed pseudo-random draw)."""
    try:
        from tokenizers import Tokenizer
        text = open(data, "rb").read().decode("utf-8", "replace")
        ids = Tokenizer.from_file(os.path.join(wdir, "tokenizer.json")).encode(text).ids[:n]
        if ids:
            return ids
    except ImportError:
        pass
    return [int(x) for x in np.random.default_rng(1).integers(0, 50000, size=n)]


def main():
    ap = argparse.ArgumentParser(description="Redukt CUDA plugin validation (R-090/R-096)")
    ap.add_argument("--repo", default=None, help="existing checkout (default: Kaggle dataset or clone)")
    args = ap.parse_args()
    os.makedirs(WORK_DIR, exist_ok=True)
    repo = get_repo(args.repo)
    redukt, lib_path = build(repo)
    wdir, rbin = prepare_weights(repo)
    data = prepare_enwik8()

    tokens = sample_tokens(wdir, data)
    lib, info = load_plugin(lib_path, rbin)
    try:
        ffi_checks(lib, info, rbin, tokens)
        throughput(lib, info, tokens)
    finally:
        lib.redukt_backend_free()
    cli_roundtrip(redukt, repo, wdir, data)
    ok = not any(str(v).startswith("FAIL") for _, v, _ in RESULTS)
    ok &= all(v == 0 for m, v, _ in RESULTS if m == "nvcc_warnings")

    with open(os.path.join(WORK_DIR, "cuda-plugin-results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "note"])
        w.writerows(RESULTS)
    with open(os.path.join(WORK_DIR, "cuda-plugin-report.txt"), "w") as f:
        f.write("\n".join(REPORT) + "\n")
    log("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED (see report)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
