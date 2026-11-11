#!/usr/bin/env python3
"""
Redukt — short-message benchmark (R-113, Fase 20), Kaggle CPU (no GPU needed).

Path 3 of the commercial target: links where a transmitted byte is worth far more than the compute to shrink it
(narrowband satellite: Iridium SBD/IMT $0.3-3 per KB, Certus/BGAN $3.5-15 per MB). Traffic there is made of SHORT
messages, the regime where classical codecs have no history to exploit and a pretrained model has its whole prior.
Nothing in the repo measured it: every figure so far is a 10 KB window or a whole file.

Measured here, with the real CPU product path (`redukt compress --input-dir`, Mamba-130M Q8 + CM, AZTH v5):
  messages of 128 / 256 / 512 / 1024 / 2048 bytes cut from real data — Wikipedia text (enwik8), English prose
  (dickens), three Loghub logs (Apache, Linux, OpenSSH) and GitHub event JSON (GH Archive) — 16 messages per
  source and size, taken from the second half of each source.

Baselines per message, each with its smallest framing:
  gzip -9 (raw deflate), xz -9e (raw LZMA2), brotli -11 (built-in dictionary), zstd -19 (magicless frame, no
  checksum), and zstd -19 with a 110 KB dictionary trained on the FIRST half of the same source — the fair
  classical baseline for a fleet that sends the same kind of message over and over.

Redukt is reported twice: the real `.azth` size (15-byte header + 4-byte CRC for Q8 with the R-108 byte) and the
payload alone (file minus 19 bytes), which is what a compact message framing would cost. The header is a format
decision, not a model limit, so both numbers matter.

Outputs (/kaggle/working): short-message-results.csv (per message), short-message-summary.csv (per source and
size), short-message-report.txt.

Launch (repo root):
    cp notebooks/short-message-bench.py notebooks/short-message-bench/
    python -m kaggle kernels push -p notebooks/short-message-bench

Copyright 2026 Redukt Contributors. Apache-2.0 License.
"""

import bz2
import csv
import gzip
import lzma
import os
import re
import shutil
import subprocess
import sys
import time
import zlib

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("shortmsg-out")
REPO_URL = os.environ.get("REDUKT_REPO", "https://github.com/DavidCaastro/redukt.git")
REDUKT_REF = os.environ.get("REDUKT_REF", "e932f38")
SIZES = [128, 256, 512, 1024, 2048]
PER_CELL = 16
AZTH_OVERHEAD = 19          # Q8 Mamba, AZTH v5 with RESET_LOG2: 15-byte header + 4-byte CRC-32
GITHUB = "https://raw.githubusercontent.com/logpai/loghub/master"
T_START = time.time()
ROWS = []       # source, size, index, tool, out_bytes
REPORT = []


def log(msg):
    print(msg, flush=True)
    REPORT.append(msg)


def section(title):
    log(f"\n{'=' * 72}\n  {title}   [t+{(time.time() - T_START) / 60:.1f} min]\n{'=' * 72}")


def sh(cmd, cwd=None, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-2000:], file=sys.stderr)
        if check:
            raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def flush_outputs():
    with open(os.path.join(WORK_DIR, "short-message-results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "size", "index", "tool", "out_bytes"])
        w.writerows(ROWS)
    with open(os.path.join(WORK_DIR, "short-message-report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT) + "\n")


# ---------------------------------------------------------------- data

def sources():
    section("1. Sources (real data)")
    data = os.path.join(WORK_DIR, "data")
    os.makedirs(data, exist_ok=True)
    out = {}
    sh(f"curl -sL http://mattmahoney.net/dc/enwik8.zip -o {data}/enwik8.zip && unzip -o -q {data}/enwik8.zip -d {data}")
    out["wikipedia"] = open(os.path.join(data, "enwik8"), "rb").read(4_000_000)
    try:
        sh(f"curl -sL https://sun.aei.polsl.pl/~sdeor/corpus/dickens.bz2 -o {data}/dickens.bz2")
        out["prose"] = bz2.decompress(open(os.path.join(data, "dickens.bz2"), "rb").read())[:4_000_000]
    except Exception as e:   # noqa: BLE001
        log(f"  dickens unavailable: {e}")
    for name, path in (("log-apache", "Apache/Apache_2k.log"), ("log-linux", "Linux/Linux_2k.log"),
                       ("log-openssh", "OpenSSH/OpenSSH_2k.log")):
        try:
            sh(f"curl -sL --retry 3 '{GITHUB}/{path}' -o {data}/{name}")
            out[name] = open(os.path.join(data, name), "rb").read()
        except Exception as e:   # noqa: BLE001
            log(f"  {name} unavailable: {e}")
    try:
        sh(f"curl -sL --retry 3 https://data.gharchive.org/2015-01-01-15.json.gz -o {data}/gh.json.gz")
        out["json-events"] = gzip.decompress(open(os.path.join(data, "gh.json.gz"), "rb").read())[:4_000_000]
    except Exception as e:   # noqa: BLE001
        log(f"  GH Archive unavailable: {e}")
    for k, v in out.items():
        log(f"  {k:14s} {len(v):>10,d} B")
    return out


def cut_messages(blob, size, count):
    """`count` messages of `size` bytes from the second half of `blob`, each starting at a line start."""
    half = len(blob) // 2
    starts = [m.end() for m in re.finditer(b"\n", blob[half:])]
    usable = [half + s for s in starts if half + s + size <= len(blob)]
    if len(usable) < count:
        usable = list(range(half, len(blob) - size, max(1, (len(blob) - half - size) // count)))
    step = max(1, len(usable) // count)
    return [blob[o:o + size] for o in usable[::step][:count]]


# ---------------------------------------------------------------- classical

def classical_tools(train_half):
    import brotli
    import zstandard
    params = zstandard.ZstdCompressionParameters.from_level(
        19, format=zstandard.FORMAT_ZSTD1_MAGICLESS, write_checksum=False, write_content_size=False,
        write_dict_id=False)
    plain = zstandard.ZstdCompressor(compression_params=params)
    tools = {
        "gzip-9-raw": lambda b: zlib.compress(b, 9, wbits=-15),
        "xz-9e-raw": lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=[
            {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]),
        "brotli-11": lambda b: brotli.compress(b, quality=11),
        "zstd-19": plain.compress,
    }
    try:
        step = max(256, len(train_half) // 4000)
        samples = [train_half[i:i + 1024] for i in range(0, len(train_half) - 1024, step)]
        dict_data = zstandard.train_dictionary(110_000, samples)
        with_dict = zstandard.ZstdCompressor(compression_params=params, dict_data=dict_data)
        tools["zstd-19-dict"] = with_dict.compress
    except Exception as e:   # noqa: BLE001 — tiny sources cannot train a 110 KB dictionary
        try:
            dict_data = zstandard.ZstdCompressionDict(train_half[-110_000:], dict_type=zstandard.DICT_TYPE_RAWCONTENT)
            with_dict = zstandard.ZstdCompressor(compression_params=params, dict_data=dict_data)
            tools["zstd-19-dict"] = with_dict.compress
            log(f"    dictionary training failed ({str(e)[:60]}); using raw-content prefix dictionary")
        except Exception as e2:   # noqa: BLE001
            log(f"    no dictionary baseline: {e2}")
    return tools


# ---------------------------------------------------------------- redukt

def build_redukt():
    section("2. Build redukt (CPU, maxperf) + mamba-130m weights")
    repo = os.path.join(WORK_DIR, "redukt")
    if not os.path.isdir(repo):
        sh(f"git clone -q {REPO_URL} {repo}")
        sh(f"git checkout -q {REDUKT_REF}", cwd=repo)
    os.environ["PATH"] = os.path.expanduser("~/.cargo/bin:") + os.environ["PATH"]
    if sh("cargo --version", check=False).returncode != 0:
        sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    sh("cargo build --profile maxperf", cwd=repo)
    wdir = os.path.join(repo, "weights", "mamba-130m")
    sh(f"{sys.executable} -m pip install -q huggingface_hub brotli zstandard", check=False)
    from huggingface_hub import snapshot_download
    snapshot_download("state-spaces/mamba-130m-hf", local_dir=wdir, allow_patterns=["*.safetensors", "*.json"])
    sh("lscpu | grep -E 'Model name|^CPU\\(s\\)' | cut -c1-120", check=False)
    return repo, os.path.join(repo, "target", "maxperf", "redukt"), wdir


def redukt_batch(repo, redukt, wdir, messages):
    """messages: {name: bytes}. One lockstep batch (batch == single-file output is a release gate)."""
    src, dst = os.path.join(WORK_DIR, "msg_in"), os.path.join(WORK_DIR, "msg_out")
    for d in (src, dst):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for name, blob in messages.items():
        open(os.path.join(src, name), "wb").write(blob)
    t0 = time.time()
    r = sh(f"{redukt} compress --input-dir '{src}' --output-dir '{dst}' --jobs 32 --backbone mamba --quant q8 "
           f"--weights {wdir} --no-governor", cwd=repo)
    tail = [ln for ln in r.stderr.splitlines() if "of lockstep" in ln]
    log(f"  redukt batch of {len(messages)} messages: {time.time() - t0:.0f} s  {tail[-1].strip() if tail else ''}")
    sizes = {n: os.path.getsize(os.path.join(dst, n + ".azth")) for n in messages
             if os.path.isfile(os.path.join(dst, n + ".azth"))}
    back = os.path.join(WORK_DIR, "msg_back")
    shutil.rmtree(back, ignore_errors=True)
    os.makedirs(back)
    sample = sorted(sizes)[::max(1, len(sizes) // 24)][:24]      # roundtrip a spread of 24 messages
    tmp = os.path.join(WORK_DIR, "msg_rt")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for n in sample:
        shutil.copy(os.path.join(dst, n + ".azth"), tmp)
    sh(f"{redukt} decompress --input-dir '{tmp}' --output-dir '{back}' --jobs 24 --weights {wdir} --no-governor",
       cwd=repo, check=False)
    good = sum(1 for n in sample if os.path.isfile(os.path.join(back, n))
               and open(os.path.join(back, n), "rb").read() == messages[n])
    log(f"  roundtrip of {len(sample)} sampled messages: {good}/{len(sample)} byte-identical "
        f"({'PASS' if good == len(sample) else 'FAIL'})")
    return sizes


# ---------------------------------------------------------------- summary

def summarize():
    section("4. Summary — mean compressed size in bytes per message (and % of the original)")
    tools = []
    for r in ROWS:
        if r[3] not in tools:
            tools.append(r[3])
    srcs = []
    for r in ROWS:
        if r[0] not in srcs:
            srcs.append(r[0])
    out = []
    for src in srcs + ["ALL"]:
        log(f"\n  {src}")
        log("  " + f"{'size':>6s}" + "".join(f"{t:>20s}" for t in tools))
        for size in SIZES:
            cells = []
            for t in tools:
                v = [r[4] for r in ROWS if (src == "ALL" or r[0] == src) and r[1] == size and r[3] == t]
                if v:
                    mean = sum(v) / len(v)
                    cells.append(f"{mean:10.1f} ({100 * mean / size:5.1f}%)")
                    out.append((src, size, t, len(v), f"{mean:.2f}", f"{100 * mean / size:.2f}"))
                else:
                    cells.append(f"{'-':>20s}")
            log("  " + f"{size:>6d}" + "".join(f"{c:>20s}" for c in cells))
    with open(os.path.join(WORK_DIR, "short-message-summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "size", "tool", "messages", "mean_out_bytes", "percent_of_original"])
        w.writerows(out)
    flush_outputs()


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    log(f"ref {REDUKT_REF}; sizes {SIZES}; {PER_CELL} messages per source and size")
    src = sources()
    repo, redukt, wdir = build_redukt()
    section("3. Messages: classical baselines and Redukt")
    messages = {}
    try:
        for name, blob in src.items():
            tools = classical_tools(blob[:len(blob) // 2])
            for size in SIZES:
                for i, msg in enumerate(cut_messages(blob, size, PER_CELL)):
                    key = f"{name}_{size:04d}_{i:02d}"
                    messages[key] = msg
                    ROWS.append((name, size, i, "raw", len(msg)))
                    for tool, fn in tools.items():
                        ROWS.append((name, size, i, tool, min(len(fn(msg)), len(msg) + 1)))
            log(f"  {name:14s} classical done ({', '.join(tools)})")
        flush_outputs()
        sizes = redukt_batch(repo, redukt, wdir, messages)
        for key, out_bytes in sizes.items():
            name, size, i = key.rsplit("_", 2)
            ROWS.append((name, int(size), int(i), "redukt-azth", out_bytes))
            ROWS.append((name, int(size), int(i), "redukt-payload", out_bytes - AZTH_OVERHEAD))
    finally:
        summarize()
        log(f"\nDONE in {(time.time() - T_START) / 3600:.2f} h")
        flush_outputs()


if __name__ == "__main__":
    main()
