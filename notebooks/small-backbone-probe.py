#!/usr/bin/env python3
"""
Redukt — small-backbone probe (R-105, step 1): is there a smaller pretrained backbone that keeps the ratio?

Measures, on the SAME data and with the SAME protocol, the backbone-only cross-entropy (bits per input byte,
no CM ensemble, no mixer, no coder overhead) of:

  reference   state-spaces/mamba-130m-hf          129M  GPT-NeoX BPE 50K   Apache-2.0   (the product backbone)
  same data   EleutherAI/pythia-14m/31m/70m/160m  14-160M  same tokenizer, same training set (the Pile), Apache-2.0
  controls    distilgpt2 (82M), gpt2 (124M), HuggingFaceTB/SmolLM2-135M — other tokenizers / training recipes
  byte-level  pszemraj/nGPT-enwiki8               dim 512 x 8, vocab 256, MIT, trained on enwik8 (best effort)
  specialist  L3TC 200K / 800K / 3.2M / 12M       RWKV-v4 + 16K SPM, trained on enwik8, NO LICENSE in the repo
                                                  (reference for the size/ratio trade-off only; best effort)

Data: 8 x 10 KB enwik8 windows, enwik8 first 500 KB, 500 KB of enwik9 beyond the enwik8 prefix (unseen by the
enwik8-trained models), dickens first 500 KB (Silesia), and the five whole industry files of the benchmark
(facturae, UBL, apache, syslog, openssh). Classical codecs on the same bytes for the margin.

Protocol: the token stream of a file is cut into independent chunks of C tokens (a BOS token + C-1 tokens), which
is what the product does with the R-108 state reset (C = 4096 for Mamba). Transformers are limited by their
context (2048 for Pythia, 1024 for GPT-2), so every model is also reported at the largest C it supports and the
reference at C = 2048 for a like-for-like row. bpb = sum of -log2 p(token) / bytes of the file.

Speed: (a) analytic — parameters touched per token (all weights except the input embedding lookup) and
2 x that as MFLOP/token, relative to mamba-130m; (b) measured — PyTorch CPU, ONE thread, stateful decoding with
the framework cache when the model supports it. (b) is PyTorch, not the Redukt runtime: only the ratio between
models is meaningful.

Outputs (/kaggle/working): small-backbone-results.csv, small-backbone-speed.csv, small-backbone-classical.csv,
small-backbone-report.txt.

Launch (repo root):
    cp notebooks/small-backbone-probe.py notebooks/small-backbone-probe/
    python -m kaggle kernels push -p notebooks/small-backbone-probe

Copyright 2026 Redukt Contributors. Apache-2.0 License.
"""

import bz2
import csv
import glob
import hashlib
import importlib
import lzma
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import types
import zlib

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("probe-out")
REPO_URL = os.environ.get("REDUKT_REPO", "https://github.com/DavidCaastro/redukt.git")
REDUKT_REF = os.environ.get("REDUKT_REF", "d7add6d")
SMOKE = os.environ.get("PROBE_SMOKE", "") == "1"          # tiny local run: one model, small data
ONLY = [m for m in os.environ.get("PROBE_ONLY", "").split(",") if m]
T_START = time.time()
LN2 = math.log(2.0)
WINDOW = 10_240
BIG = 512_000
MAX_BATCH_TOKENS = 8192                                    # logits of B x T x 50K F32 must fit the T4
INDUSTRY = ["facturae-batch.xml", "ubl-invoice-batch.xml", "apache-access.log", "linux-syslog.log",
            "openssh-auth.log"]
L3TC_DRIVE_ID = "1LibLdeHTi3Io0H-ZiYZ6AhYUYKUkOpXz"
L3TC_REPO = "https://github.com/alipay/L3TC-leveraging-rwkv-for-learned-lossless-low-complexity-text-compression.git"

# (hf id, label, family, chunk sizes to evaluate, license)
HF_MODELS = [
    ("state-spaces/mamba-130m-hf", "mamba-130m", "reference", [4096, 2048], "Apache-2.0"),
    ("EleutherAI/pythia-14m", "pythia-14m", "pile-neox", [2048], "Apache-2.0"),
    ("EleutherAI/pythia-31m", "pythia-31m", "pile-neox", [2048], "Apache-2.0"),
    ("EleutherAI/pythia-70m", "pythia-70m", "pile-neox", [2048], "Apache-2.0"),
    ("EleutherAI/pythia-160m", "pythia-160m", "pile-neox", [2048], "Apache-2.0"),
    ("distilgpt2", "distilgpt2", "control", [1024], "Apache-2.0"),
    ("gpt2", "gpt2", "control", [1024], "MIT"),
    ("HuggingFaceTB/SmolLM2-135M", "smollm2-135m", "control", [4096, 2048], "Apache-2.0"),
]

RESULTS = []    # model, family, license, dataset, chunk, bytes, tokens, bits, bpb, seconds, note
SPEED = []      # model, total_params, active_params, mflop_per_token, rel_cost, cpu1_tok_s, bytes_per_token, note
CLASSICAL = []  # dataset, bytes, tool, out_bytes, bpb
REPORT = []


def log(msg):
    print(msg, flush=True)
    REPORT.append(msg)


def section(title):
    log(f"\n{'=' * 72}\n  {title}   [t+{(time.time() - T_START) / 60:.1f} min]\n{'=' * 72}")


def run(cmd, cwd=None, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr)
        if check:
            raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def flush_outputs():
    os.makedirs(WORK_DIR, exist_ok=True)
    tables = (
        ("small-backbone-results.csv", RESULTS, ["model", "family", "license", "dataset", "chunk_tokens",
                                                 "bytes", "tokens", "bits", "bpb", "seconds", "note"]),
        ("small-backbone-speed.csv", SPEED, ["model", "total_params", "active_params", "mflop_per_token",
                                             "cost_vs_mamba130m", "cpu1_tok_s", "bytes_per_token", "note"]),
        ("small-backbone-classical.csv", CLASSICAL, ["dataset", "bytes", "tool", "out_bytes", "bpb"]),
    )
    for name, rows, header in tables:
        with open(os.path.join(WORK_DIR, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    with open(os.path.join(WORK_DIR, "small-backbone-report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT) + "\n")


def record(model, family, lic, dataset, chunk, nbytes, ntok, bits, seconds, note=""):
    bpb = bits / nbytes if nbytes else 0.0
    RESULTS.append((model, family, lic, dataset, chunk, nbytes, ntok, f"{bits:.1f}", f"{bpb:.4f}", f"{seconds:.1f}",
                    note))
    log(f"  {model:16s} C={chunk:<5d} {dataset:22s} {nbytes:>9,d} B {ntok:>8,d} tok  bpb={bpb:.4f}  "
        f"{seconds:6.1f} s  {note}")
    flush_outputs()


# ---------------------------------------------------------------- data

def fetch(url, dst):
    if not os.path.isfile(dst):
        run(f"curl -sL --retry 3 '{url}' -o '{dst}'")
    return dst


def utf8_text(blob):
    """Decode a byte window; a window cut inside a UTF-8 sequence loses at most 3 bytes at each end."""
    return blob.decode("utf-8", errors="ignore")


def prepare_data():
    """dataset name -> list of text pieces (each piece is coded independently; bytes = sum of UTF-8 lengths)."""
    section("1. Data")
    data = os.path.join(WORK_DIR, "data")
    os.makedirs(data, exist_ok=True)
    sets = {}
    fetch("http://mattmahoney.net/dc/enwik8.zip", os.path.join(data, "enwik8.zip"))
    run(f"unzip -o -q {data}/enwik8.zip -d {data}")
    e8 = open(os.path.join(data, "enwik8"), "rb").read()
    sets["enwik8-8x10k"] = [utf8_text(e8[i * 12_000_000:i * 12_000_000 + WINDOW]) for i in range(8)]
    if SMOKE:
        sets = {"enwik8-8x10k": sets["enwik8-8x10k"][:2]}
        return sets
    sets["enwik8-500k"] = [utf8_text(e8[:BIG])]
    try:    # enwik9 beyond the enwik8 prefix: HTTP range request of 500 KB at offset 600 MB of the raw file
        fetch("http://mattmahoney.net/dc/enwik9.zip", os.path.join(data, "enwik9.zip"))
        run(f"unzip -o -q {data}/enwik9.zip -d {data}")
        with open(os.path.join(data, "enwik9"), "rb") as f:
            f.seek(600_000_000)
            sets["enwik9-tail-500k"] = [utf8_text(f.read(BIG))]
        os.remove(os.path.join(data, "enwik9"))
        os.remove(os.path.join(data, "enwik9.zip"))
    except Exception as e:   # noqa: BLE001
        log(f"  enwik9 unavailable: {e}")
    try:
        fetch("https://sun.aei.polsl.pl/~sdeor/corpus/dickens.bz2", os.path.join(data, "dickens.bz2"))
        dk = bz2.decompress(open(os.path.join(data, "dickens.bz2"), "rb").read())
        sets["dickens-500k"] = [dk[:BIG].decode("latin-1")]
    except Exception as e:   # noqa: BLE001
        log(f"  dickens unavailable: {e}")
    repo = os.path.join(WORK_DIR, "redukt")
    if not os.path.isdir(repo):
        run(f"git clone -q {REPO_URL} {repo}")
        run(f"git checkout -q {REDUKT_REF}", cwd=repo)
    run(f"{sys.executable} scripts/prepare-benchmark-data.py", cwd=repo, check=False)
    for name in INDUSTRY:
        p = os.path.join(repo, "data", "benchmark", "industry", name)
        if os.path.isfile(p):
            blob = open(p, "rb").read()
            sets[name] = [utf8_text(blob)]
            log(f"  {name:24s} {len(blob):>9,d} B sha256 {hashlib.sha256(blob).hexdigest()[:16]}")
        else:
            log(f"  MISSING {name}")
    for k, v in sets.items():
        log(f"  dataset {k:22s} pieces={len(v)} bytes={sum(len(t.encode('utf-8')) for t in v):,d}")
    return sets


def set_bytes(pieces):
    return sum(len(t.encode("utf-8")) for t in pieces)


def classical(sets):
    section("2. Classical codecs on the same bytes (per piece, summed)")
    tools = [("gzip-9", lambda b: zlib.compress(b, 9)), ("bz2-9", lambda b: bz2.compress(b, 9)),
             ("xz-9e", lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))]
    try:
        run(f"{sys.executable} -m pip install -q brotli zstandard", check=False)
        import brotli
        import zstandard
        tools.append(("brotli-11-w24", lambda b: brotli.compress(b, quality=11, lgwin=24)))
        tools.append(("zstd-19", lambda b: zstandard.ZstdCompressor(level=19).compress(b)))
    except Exception as e:   # noqa: BLE001
        log(f"  brotli/zstd unavailable: {e}")
    for name, pieces in sets.items():
        n = set_bytes(pieces)
        for tool, fn in tools:
            out = sum(len(fn(t.encode("utf-8"))) for t in pieces)
            CLASSICAL.append((name, n, tool, out, f"{out * 8 / n:.4f}"))
        best = min((r for r in CLASSICAL if r[0] == name), key=lambda r: r[3])
        log(f"  {name:22s} {n:>9,d} B  best classical {best[2]:14s} bpb={best[4]}")
    flush_outputs()


# ---------------------------------------------------------------- HF causal LMs

def chunk_ids(ids, chunk, bos):
    """Independent chunks of `chunk` tokens: BOS + up to chunk-1 tokens (the R-108 reset protocol)."""
    step = chunk - 1
    return [[bos] + ids[i:i + step] for i in range(0, len(ids), step)]


def hf_bits(model, chunks, device):
    import torch
    import torch.nn.functional as F
    chunks = sorted(chunks, key=len, reverse=True)
    bits, i = 0.0, 0
    while i < len(chunks):
        width = len(chunks[i])
        b = max(1, MAX_BATCH_TOKENS // width)
        batch = chunks[i:i + b]
        i += b
        x = torch.zeros((len(batch), width), dtype=torch.long)
        mask = torch.zeros((len(batch), width - 1), dtype=torch.bool)
        for r, c in enumerate(batch):
            x[r, :len(c)] = torch.tensor(c)
            mask[r, :len(c) - 1] = True
        x, mask = x.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=x[:, :-1]).logits.float()
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
        bits += float((nll.reshape(mask.shape) * mask).sum().item()) / LN2
        del logits, nll
    return bits


def param_counts(model):
    """(total, active): active = weights touched per token = everything except the input embedding lookup
    (the output head is counted once even when tied to the input embedding)."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    tied = head is not None and head.weight is emb
    active = total if tied else total - emb.numel()
    return total, active


def hf_speed(model, ids, bos):
    """Tokens/s on one CPU thread with the framework cache (stateful decoding), after a 256-token context."""
    import torch
    torch.set_num_threads(1)
    model = model.to("cpu").float()
    ctx = torch.tensor([[bos] + ids[:255]])
    steps = ids[255:255 + (16 if SMOKE else 96)]
    with torch.no_grad():
        out = model(input_ids=ctx, use_cache=True)
        key = "cache_params" if hasattr(out, "cache_params") else "past_key_values"
        state = getattr(out, key)
        t0 = time.time()
        for n, tok in enumerate(steps):
            kwargs = {key: state, "use_cache": True}
            if key == "cache_params":
                kwargs["cache_position"] = torch.tensor([256 + n])
            out = model(input_ids=torch.tensor([[tok]]), **kwargs)
            state = getattr(out, key)
        dt = time.time() - t0
    return len(steps) / dt


def eval_hf(hf_id, label, family, chunk_sizes, lic, sets, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    section(f"HF model {label} ({hf_id})")
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).eval()
    total, active = param_counts(model)
    bos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 0)
    log(f"  params total={total:,d} active/token={active:,d} bos={bos} vocab={len(tok)}")
    ids_cache = {name: [tok(t, add_special_tokens=False)["input_ids"] for t in pieces]
                 for name, pieces in sets.items()}
    model = model.to(device)
    for chunk in chunk_sizes:
        for name, pieces in sets.items():
            t0 = time.time()
            try:
                chunks = [c for ids in ids_cache[name] for c in chunk_ids(ids, chunk, bos)]
                bits = hf_bits(model, chunks, device)
            except Exception as e:   # noqa: BLE001
                log(f"  FAILED {label} C={chunk} {name}: {str(e)[:200]}")
                continue
            ntok = sum(len(i) for i in ids_cache[name])
            record(label, family, lic, name, chunk, set_bytes(pieces), ntok, bits, time.time() - t0)
    first = next(iter(sets))
    bpt = set_bytes(sets[first]) / max(1, sum(len(i) for i in ids_cache[first]))
    rate, note = "", "pytorch cpu 1 thread, cached decoding"
    try:
        rate = f"{hf_speed(model, ids_cache[first][0], bos):.1f}"
    except Exception as e:   # noqa: BLE001
        note = f"speed probe failed: {str(e)[:120]}"
    SPEED.append([label, total, active, f"{2 * active / 1e6:.1f}", "", rate, f"{bpt:.3f}", note])
    log(f"  speed {label}: {2 * active / 1e6:.1f} MFLOP/token, cpu1 {rate or 'n/a'} tok/s, {bpt:.2f} B/token")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    flush_outputs()


# ---------------------------------------------------------------- nGPT (byte level)

def eval_ngpt(sets, device):
    import torch
    import torch.nn.functional as F
    section("nGPT-enwiki8 (byte level, dim 512 x 8, trained on enwik8, MIT)")
    run(f"{sys.executable} -m pip install -q nGPT-pytorch")
    from nGPT_pytorch import nGPT
    ckpt_path = fetch("https://huggingface.co/pszemraj/nGPT-enwiki8/resolve/main/nGPT_best.pt",
                      os.path.join(WORK_DIR, "data", "nGPT_best.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    use_parametrize = ckpt.get("config", {}).get("use_parametrize", True)
    model = nGPT(num_tokens=256, dim=512, depth=8, tied_embedding=True, add_value_residual=True,
                 attn_norm_qk=False, manual_norm_weights=not use_parametrize)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval().to(device)
    total = sum(p.numel() for p in model.parameters())
    log(f"  params total={total:,d}")
    seq, stride = 512, 256      # trained with seq_len 512; score the last 256 bytes of every window
    for name, pieces in sets.items():
        t0, bits, nbytes = time.time(), 0.0, 0
        for text in pieces:
            raw = text.encode("utf-8")
            nbytes += len(raw)
            data = torch.tensor(list(raw), dtype=torch.long)
            bits += 8.0                                         # first byte of the piece: no context
            n = len(data)
            windows = []                                        # (start, end, first scored target index)
            end = min(n, seq)
            if end > 1:
                windows.append((0, end, 1))
            while end < n:                                      # later blocks: 256 new targets, 256 of context
                first, end = end, min(n, end + stride)
                windows.append((end - seq, end, first))
            for i in range(0, len(windows), 64):
                group = windows[i:i + 64]
                width = max(e - s for s, e, _ in group)
                x = torch.zeros((len(group), width), dtype=torch.long)
                mask = torch.zeros((len(group), width - 1), dtype=torch.bool)
                for r, (s, e, first_target) in enumerate(group):
                    x[r, :e - s] = data[s:e]
                    mask[r, first_target - s - 1:e - s - 1] = True
                x, mask = x.to(device), mask.to(device)
                with torch.no_grad():
                    logits = model(x[:, :-1]).float()
                    nll = F.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none")
                bits += float((nll.reshape(mask.shape) * mask).sum().item()) / LN2
        record("ngpt-enwik8-25m", "byte-level", "MIT", name, seq, nbytes, nbytes, bits, time.time() - t0,
               "sliding 512-byte context, stride 256")
    SPEED.append(["ngpt-enwik8-25m", total, total, f"{2 * total / 1e6:.1f}", "", "", "1.000",
                  "no cache API: analytic only; byte level = 1 forward per byte"])
    flush_outputs()


# ---------------------------------------------------------------- L3TC (RWKV-v4 + 16K SPM)

def l3tc_import(repo):
    """Import only the inference module; the package __init__ pulls the training code (CUDA extension)."""
    models = types.ModuleType("l3tc_models")
    models.__path__ = [os.path.join(repo, "models")]
    sub = types.ModuleType("l3tc_models.RWKV_V4")
    sub.__path__ = [os.path.join(repo, "models", "RWKV_V4")]
    sys.modules["l3tc_models"], sys.modules["l3tc_models.RWKV_V4"] = models, sub
    importlib.import_module("l3tc_models.registry")
    return importlib.import_module("l3tc_models.RWKV_V4.rwkv_tc_hira_infer")


def l3tc_config(repo, tag):
    cfg = {}
    exec(open(os.path.join(repo, "config", "l3tc", f"l3tc_{tag}.py"), encoding="utf-8").read(), {}, cfg)
    return cfg


def l3tc_state(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    st = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}
    for key in list(st):
        if "att.time_mix" in key or "ffn.time_mix" in key:
            st[key] = st[key][0]
        for name in ("key", "value", "receptance"):          # merge the HiRA branches into the main weights
            if key.endswith(f"{name}.weight"):
                a, b = key.replace(f"{name}.weight", f"{name}_A.weight"), key.replace(f"{name}.weight",
                                                                                       f"{name}_B.weight")
                if a in st and b in st:
                    st[key] = st[key] + st[b] @ st[a]
    st["ln0.weight"], st["ln0.bias"] = st.pop("blocks.0.ln0.weight"), st.pop("blocks.0.ln0.bias")
    return st


def l3tc_bits(model, segments, device, batch=256):
    import torch
    import torch.nn.functional as F
    bits = 0.0
    segments = sorted(segments, key=len, reverse=True)
    for i in range(0, len(segments), batch):
        group = segments[i:i + batch]
        width = len(group[0])
        x = torch.zeros((len(group), width), dtype=torch.long)
        mask = torch.zeros((len(group), width), dtype=torch.bool)
        for r, s in enumerate(group):
            x[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = True
        x, mask = x.to(device), mask.to(device)
        with torch.no_grad():
            hidden = model.forward_initialzation(len(group), device)
            for t in range(width - 1):
                logits, hidden = model(x[:, t], hidden)
                nll = F.cross_entropy(logits.float(), x[:, t + 1], reduction="none")
                bits += float((nll * mask[:, t + 1]).sum().item()) / LN2
    return bits


def eval_l3tc(sets, device):
    import torch
    section("L3TC (RWKV-v4 HiRA + 16K SentencePiece, trained on enwik8; repo has no license)")
    run(f"{sys.executable} -m pip install -q gdown sentencepiece addict yapf")
    import sentencepiece as spm
    base = os.path.join(WORK_DIR, "l3tc")
    repo = os.path.join(base, "repo")
    os.makedirs(base, exist_ok=True)
    if not os.path.isdir(repo):
        run(f"git clone -q --depth 1 {L3TC_REPO} {repo}")
    bundle = os.path.join(base, "bundle")
    if not os.path.isfile(bundle):
        run(f"{sys.executable} -m gdown {L3TC_DRIVE_ID} -O {bundle}")
    out = os.path.join(base, "files")
    os.makedirs(out, exist_ok=True)
    try:
        for fmt in ("zip", "gztar", "tar", "bztar", "xztar"):
            try:
                shutil.unpack_archive(bundle, out, fmt)
                break
            except Exception:   # noqa: BLE001
                continue
    finally:
        listing = [os.path.relpath(p, out) + f" ({os.path.getsize(p):,d} B)"
                   for p in glob.glob(os.path.join(out, "**", "*"), recursive=True) if os.path.isfile(p)]
        log(f"  bundle size {os.path.getsize(bundle):,d} B; files: {listing[:40]}")
    spm_models = glob.glob(os.path.join(out, "**", "*.model"), recursive=True)
    ckpts = [p for p in glob.glob(os.path.join(out, "**", "*"), recursive=True)
             if os.path.isfile(p) and p.endswith((".pth", ".pt", ".ckpt"))]
    if not ckpts and os.path.isfile(bundle) and not listing:
        ckpts = [bundle]            # the drive file may be a single checkpoint
    if not spm_models:
        log("  no SentencePiece model in the bundle: L3TC cannot be evaluated without its tokenizer (skipped)")
        return
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_models[0])
    log(f"  tokenizer {os.path.basename(spm_models[0])}: {sp.GetPieceSize()} pieces")
    module = l3tc_import(repo)
    seg_chars = 2048
    for tag, label in (("200k", "l3tc-200k"), ("800k", "l3tc-800k"), ("3m2", "l3tc-3.2m"), ("12m", "l3tc-12m")):
        cand = [p for p in ckpts if tag in os.path.basename(p).lower() or tag in os.path.dirname(p).lower()]
        if not cand:
            log(f"  no checkpoint matching '{tag}'")
            continue
        try:
            cfg = l3tc_config(repo, tag)
            st = l3tc_state(cand[0])
            vocab = st["emb.weight"].shape[0]
            model = module.RWKV_TC_HIRA_Infer_For_Script(
                vocab_size=vocab, hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layer"],
                intermediate_size=cfg["intermediate_size"], rwkv_rank=cfg["rwkv_rank"])
            log(f"  {label}: {os.path.basename(cand[0])} load -> {model.load_state_dict(st, strict=False)}")
            model = model.eval().to(device)
            total = sum(p.numel() for n, p in st.items() if not n.endswith(("_A.weight", "_B.weight")))
            active = total - st["emb.weight"].numel()
            bpt = 0.0
            for name, pieces in sets.items():
                t0, segments, unk_bits, ntok, nbytes = time.time(), [], 0.0, 0, 0
                for text in pieces:
                    nbytes += len(text.encode("utf-8"))
                    for s in range(0, len(text), seg_chars):
                        proto = sp.encode(text[s:s + seg_chars], out_type="immutable_proto")
                        ids = [2]
                        for piece in proto.pieces:
                            if piece.begin == piece.end:
                                continue
                            if piece.id == 1:               # outlier: bypasses the model, charged as raw bytes
                                unk_bits += 8.0 * (len(piece.surface.encode("utf-8")) + 1)
                            ids.append(piece.id)
                        ntok += len(ids) - 1
                        segments.append(ids)
                bits = l3tc_bits(model, segments, device) + unk_bits
                record(label, "specialist-enwik8", "none (unlicensed)", name, seg_chars, nbytes, ntok, bits,
                       time.time() - t0, f"2048-char segments; outliers raw = {unk_bits / 8:.0f} B")
                bpt = bpt or nbytes / max(1, ntok)
            torch.set_num_threads(1)
            cpu = model.to("cpu")
            ids = sp.encode(sets[next(iter(sets))][0][:4000])
            with torch.no_grad():
                hidden = cpu.forward_initialzation(1, "cpu")
                t0 = time.time()
                for tok in ids[:400]:
                    _, hidden = cpu(torch.tensor([tok]), hidden)
                rate = min(400, len(ids)) / (time.time() - t0)
            SPEED.append([label, total, active, f"{2 * active / 1e6:.2f}", "", f"{rate:.1f}", f"{bpt:.3f}",
                          "pytorch cpu 1 thread, stateful step"])
            log(f"  speed {label}: params {total:,d}, cpu1 {rate:.1f} tok/s, {bpt:.2f} B/token")
        except Exception as e:   # noqa: BLE001
            log(f"  FAILED {label}: {str(e)[:300]}\n{traceback.format_exc()[-600:]}")
        flush_outputs()


# ---------------------------------------------------------------- summary

def summary(sets):
    section("Summary — bpb per dataset (best chunk per model) and cost per token vs mamba-130m")
    ref = next((float(s[3]) for s in SPEED if s[0] == "mamba-130m"), None)
    for s in SPEED:
        if ref:
            s[4] = f"{float(s[3]) / ref:.4f}"
    names = list(sets)
    models = []
    for r in RESULTS:
        if r[0] not in models:
            models.append(r[0])
    log("  " + f"{'model':18s}" + "".join(f"{n[:14]:>15s}" for n in names))
    for m in models:
        cells = []
        for n in names:
            vals = [float(r[8]) for r in RESULTS if r[0] == m and r[3] == n]
            cells.append(f"{min(vals):15.4f}" if vals else f"{'-':>15s}")
        log("  " + f"{m:18s}" + "".join(cells))
    cells = []
    for n in names:
        vals = [float(r[4]) for r in CLASSICAL if r[0] == n]
        cells.append(f"{min(vals):15.4f}" if vals else f"{'-':>15s}")
    log("  " + f"{'best classical':18s}" + "".join(cells))
    log("")
    for s in SPEED:
        log(f"  {s[0]:18s} params {int(s[1]):>12,d}  active {int(s[2]):>12,d}  {s[3]:>8s} MFLOP/tok  "
            f"x{s[4] or '?':>7s} of mamba-130m  cpu1 {s[5] or 'n/a':>8s} tok/s  {s[6]} B/tok  {s[7]}")
    flush_outputs()


def main():
    import torch
    os.makedirs(WORK_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {device}; torch {torch.__version__}; smoke={SMOKE}; only={ONLY or 'all'}")
    sets = prepare_data()
    try:
        classical(sets)
        for hf_id, label, family, chunks, lic in HF_MODELS:
            if ONLY and label not in ONLY:
                continue
            try:
                eval_hf(hf_id, label, family, chunks, lic, sets, device)
            except Exception as e:   # noqa: BLE001
                log(f"  FAILED {label}: {str(e)[:300]}\n{traceback.format_exc()[-600:]}")
        for label, fn in (("ngpt", eval_ngpt), ("l3tc", eval_l3tc)):
            if ONLY and label not in ONLY:
                continue
            try:
                fn(sets, device)
            except Exception as e:   # noqa: BLE001
                log(f"  FAILED {label}: {str(e)[:300]}\n{traceback.format_exc()[-800:]}")
    finally:
        summary(sets)
        log(f"\nDONE in {(time.time() - T_START) / 3600:.2f} h")
        flush_outputs()


if __name__ == "__main__":
    main()
