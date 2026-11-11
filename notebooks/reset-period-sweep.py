#!/usr/bin/env python3
"""
Redukt — reset period sweep on whole files (R-111, Fase 20), Kaggle T4.

R-108 fixed the Mamba state reset at 4096 tokens using enwik8 100 KB. The small-backbone probe (2026-11-11) showed,
backbone-only, that halving the period costs 18-75% bpb on XML/logs and only 1.5-3% on text. This run measures the
other direction with the real product path:

  A. Real `.azth` (backbone + CM + coder) of the five whole industry files, enwik8 500 KB and dickens 500 KB with
     `--reset-every` 2048 / 4096 / 8192 / 16384 / 32768 through the plugin lockstep (one batch per period), plus a
     batch decompress roundtrip of the 8192 archives.
  B. Overlapped reset, backbone-only estimate (HF Mamba-130M, no CM): the state is re-primed with the last K tokens
     before each period of P tokens (K = 0 is the plain reset). Costs (P + K) / P forwards per token. This is an
     estimate to decide whether the format change is worth it; nothing in the Rust code implements it yet.

Outputs (/kaggle/working): reset-sweep-azth.csv, reset-sweep-overlap.csv, reset-sweep-report.txt.

Launch (repo root):
    cp notebooks/reset-period-sweep.py notebooks/reset-period-sweep/
    python -m kaggle kernels push -p notebooks/reset-period-sweep

Copyright 2026 Redukt Contributors. Apache-2.0 License.
"""

import bz2
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import time

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("sweep-out")
REPO_URL = os.environ.get("REDUKT_REPO", "https://github.com/DavidCaastro/redukt.git")
REDUKT_REF = os.environ.get("REDUKT_REF", "e932f38")
PERIODS = [int(p) for p in os.environ.get("SWEEP_PERIODS", "4096,8192,16384,2048,32768").split(",")]
OVERLAPS = [(4096, 0), (4096, 512), (4096, 1024), (4096, 2048), (8192, 0), (8192, 1024), (16384, 0)]
DEADLINE_HOURS = float(os.environ.get("DEADLINE_HOURS", "8"))
BIG = 512_000
INDUSTRY = ["facturae-batch.xml", "ubl-invoice-batch.xml", "apache-access.log", "linux-syslog.log",
            "openssh-auth.log"]
T_START = time.time()
LN2 = math.log(2.0)
AZTH = []       # period, file, bytes, out_bytes, bpb
OVERLAP = []    # period, warmup, file, bytes, tokens, bits, bpb, forwards_per_token
REPORT = []


def log(msg):
    print(msg, flush=True)
    REPORT.append(msg)


def section(title):
    log(f"\n{'=' * 72}\n  {title}   [t+{(time.time() - T_START) / 60:.1f} min]\n{'=' * 72}")


def hours_left():
    return DEADLINE_HOURS - (time.time() - T_START) / 3600


def sh(cmd, cwd=None, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-2000:], file=sys.stderr)
        if check:
            raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def flush_outputs():
    for name, rows, header in (
        ("reset-sweep-azth.csv", AZTH, ["reset_every", "file", "bytes", "out_bytes", "bpb"]),
        ("reset-sweep-overlap.csv", OVERLAP, ["period", "warmup", "file", "bytes", "tokens", "bits", "bpb",
                                              "forwards_per_token"]),
    ):
        with open(os.path.join(WORK_DIR, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    with open(os.path.join(WORK_DIR, "reset-sweep-report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT) + "\n")


def prepare():
    section("1. Source, build, weights, data")
    repo = os.path.join(WORK_DIR, "redukt")
    if not os.path.isdir(repo):
        sh(f"git clone -q {REPO_URL} {repo}")
        sh(f"git checkout -q {REDUKT_REF}", cwd=repo)
    spec = importlib.util.spec_from_file_location(
        "fullsize", os.path.join(repo, "notebooks", "redukt-gpu-fullsize-bench.py"))
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)
    redukt = fb.build(repo)
    wdir = fb.prepare_weights(repo)
    data = os.path.join(WORK_DIR, "data")
    src = os.path.join(WORK_DIR, "sweep_in")
    os.makedirs(data, exist_ok=True)
    os.makedirs(src, exist_ok=True)
    sh(f"curl -sL http://mattmahoney.net/dc/enwik8.zip -o {data}/enwik8.zip && unzip -o -q {data}/enwik8.zip -d {data}")
    open(os.path.join(src, "enwik8.500k"), "wb").write(open(os.path.join(data, "enwik8"), "rb").read(BIG))
    try:
        sh(f"curl -sL https://sun.aei.polsl.pl/~sdeor/corpus/dickens.bz2 -o {data}/dickens.bz2")
        dk = bz2.decompress(open(os.path.join(data, "dickens.bz2"), "rb").read())
        open(os.path.join(src, "dickens.500k"), "wb").write(dk[:BIG])
    except Exception as e:   # noqa: BLE001
        log(f"  dickens unavailable: {e}")
    sh(f"{sys.executable} scripts/prepare-benchmark-data.py", cwd=repo, check=False)
    for name in INDUSTRY:
        p = os.path.join(repo, "data", "benchmark", "industry", name)
        if os.path.isfile(p):
            shutil.copy(p, src)
        else:
            log(f"  MISSING {name}")
    for f in sorted(os.listdir(src)):
        log(f"  {f:26s} {os.path.getsize(os.path.join(src, f)):>9,d} B")
    return repo, redukt, wdir, src


def sweep_azth(repo, redukt, wdir, src):
    section("A. Real .azth per reset period — plugin lockstep, one batch per period")
    files = sorted(os.listdir(src))
    for period in PERIODS:
        if hours_left() < 2.0:
            log(f"  skip period {period}: deadline")
            continue
        outd = os.path.join(WORK_DIR, f"azth_{period}")
        shutil.rmtree(outd, ignore_errors=True)
        os.makedirs(outd)
        t0 = time.time()
        try:
            r = sh(f"{redukt} compress --input-dir '{src}' --output-dir '{outd}' --jobs {len(files)} "
                   f"--backbone plugin --reset-every {period} --weights {wdir} --no-governor", cwd=repo)
        except Exception as e:   # noqa: BLE001
            log(f"  FAILED period {period}: {str(e)[:200]}")
            continue
        tail = [ln for ln in r.stderr.splitlines() if "of lockstep" in ln]
        log(f"  reset-every {period}: {time.time() - t0:.0f} s  {tail[-1].strip() if tail else ''}")
        for f in files:
            a = os.path.join(outd, f + ".azth")
            if os.path.isfile(a):
                n, o = os.path.getsize(os.path.join(src, f)), os.path.getsize(a)
                AZTH.append((period, f, n, o, f"{o * 8 / n:.4f}"))
                log(f"    {f:26s} {n:>9,d} -> {o:>8,d} B  bpb={o * 8 / n:.4f}")
        flush_outputs()
    if 8192 in PERIODS and hours_left() > 1.5 and os.path.isdir(os.path.join(WORK_DIR, "azth_8192")):
        back = os.path.join(WORK_DIR, "back_8192")
        shutil.rmtree(back, ignore_errors=True)
        os.makedirs(back)
        t0 = time.time()
        try:
            sh(f"{redukt} decompress --input-dir '{WORK_DIR}/azth_8192' --output-dir '{back}' --jobs {len(files)} "
               f"--weights {wdir} --no-governor", cwd=repo)
            good = sum(1 for f in files if os.path.isfile(os.path.join(back, f))
                       and open(os.path.join(back, f), "rb").read() == open(os.path.join(src, f), "rb").read())
            log(f"  roundtrip reset-every 8192: {good}/{len(files)} byte-identical "
                f"({'PASS' if good == len(files) else 'FAIL'}) in {time.time() - t0:.0f} s")
        except Exception as e:   # noqa: BLE001
            log(f"  roundtrip FAILED: {str(e)[:200]}")
    table("A", AZTH, 0, 1, 4, [str(p) for p in sorted(PERIODS)])


def overlap_bits(model, ids, period, warm, bos, device):
    """Backbone-only bits with the state re-primed by the last `warm` tokens before every period."""
    import torch
    import torch.nn.functional as F
    bits, forwards = 0.0, 0
    for i in range(0, len(ids), period):
        ctx = ids[max(0, i - warm):i]
        new = ids[i:i + period]
        x = torch.tensor([[bos] + ctx + new], device=device)
        with torch.no_grad():
            logits = model(input_ids=x[:, :-1]).logits[0, len(ctx):].float()
            nll = F.cross_entropy(logits, x[0, 1 + len(ctx):], reduction="sum")
        bits += float(nll.item()) / LN2
        forwards += len(ctx) + len(new)
        del logits
    return bits, forwards


def sweep_overlap(src):
    section("B. Overlapped reset — backbone-only estimate (HF Mamba-130M, no CM)")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", torch_dtype=torch.float32)
    model = model.eval().to(device)
    bos = tok.bos_token_id if tok.bos_token_id is not None else 0
    files = [f for f in sorted(os.listdir(src)) if not f.startswith("dickens")]
    texts = {f: open(os.path.join(src, f), "rb").read().decode("utf-8", errors="ignore") for f in files}
    ids = {f: tok(t, add_special_tokens=False)["input_ids"] for f, t in texts.items()}
    for period, warm in OVERLAPS:
        if hours_left() < 0.3:
            log(f"  skip P={period} K={warm}: deadline")
            continue
        for f in files:
            n = len(texts[f].encode("utf-8"))
            try:
                bits, fw = overlap_bits(model, ids[f], period, warm, bos, device)
            except Exception as e:   # noqa: BLE001
                log(f"  FAILED P={period} K={warm} {f}: {str(e)[:160]}")
                continue
            OVERLAP.append((period, warm, f, n, len(ids[f]), f"{bits:.1f}", f"{bits / n:.4f}",
                            f"{fw / len(ids[f]):.3f}"))
            log(f"  P={period:<6d} K={warm:<5d} {f:26s} bpb={bits / n:.4f}  forwards/token={fw / len(ids[f]):.3f}")
        flush_outputs()
    rows = [(f"P{p}+K{k}", f, 0, 0, b) for p, k, f, _, _, _, b, _ in OVERLAP]
    table("B", rows, 0, 1, 4, [f"P{p}+K{k}" for p, k in OVERLAPS])


def table(tag, rows, ci, fi, vi, columns):
    files = []
    for r in rows:
        if r[fi] not in files:
            files.append(r[fi])
    log(f"\n  [{tag}] bpb by file (rows) and setting (columns)")
    log("  " + f"{'file':26s}" + "".join(f"{c:>12s}" for c in columns))
    for f in files:
        cells = []
        for c in columns:
            v = [r[vi] for r in rows if r[fi] == f and str(r[ci]) == c]
            cells.append(f"{float(v[0]):12.4f}" if v else f"{'-':>12s}")
        log("  " + f"{f:26s}" + "".join(cells))
    flush_outputs()


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    log(f"ref {REDUKT_REF}; periods {PERIODS}; overlaps {OVERLAPS}; deadline {DEADLINE_HOURS} h")
    repo, redukt, wdir, src = prepare()
    try:
        sweep_azth(repo, redukt, wdir, src)
        sweep_overlap(src)
    except Exception as e:   # noqa: BLE001
        log(f"FAILED: {str(e)[:400]}")
    finally:
        log(f"\nDONE in {(time.time() - T_START) / 3600:.2f} h")
        flush_outputs()


if __name__ == "__main__":
    main()
