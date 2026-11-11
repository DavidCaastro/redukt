#!/usr/bin/env python3
"""
Redukt — domain specialist for logs (R-112, Fase 20; R-105 step 2), Kaggle T4.

Question: does a small byte-level model trained by us on real logs of SOME systems compress the logs of systems
it has never seen? (L3TC showed that 3-12M parameters match a 130M generalist inside the training domain and are
useless outside it; nobody measured the middle ground "same kind of data, different source".)

Corpus: Loghub (He et al., CC-BY-4.0, Zenodo record 3227177). Split by SYSTEM:
  train     BGL, HDFS, Hadoop, Spark, HPC, Mac, Android, Windows, Thunderbird, OpenStack, HealthApp
            (first TRAIN_CAP bytes of each; a 300 KB slice past the training range is kept as "seen system" eval)
  held out  Apache, Linux, OpenSSH (the three log files of the Redukt benchmark, byte-identical to them),
            Zookeeper, Proxifier — never used for training

Models (byte level, vocab 256, so there is no 50K output head): LSTM XS/S/M and a small causal transformer.
The architecture is whatever trains fast and reliably on one T4 in pure PyTorch; the question is generalization
at a given size, not the best architecture. Every model has a wall-clock training budget.

Evaluation (bits per byte, backbone only, no CM):
  - static: independent chunks of C bytes from a zero state (C = 1024 / 4096 / 16384 / whole file) for the LSTMs,
    sliding 1024-byte context for the transformer;
  - dynamic: the LSTM keeps training on the bytes it has just coded (one Adam step per 256 bytes), the way
    NNCP/cmix-style compressors adapt. A decoder can mirror it, at ~3x compute and a hard determinism contract;
  - references on the same bytes: xz -9e, brotli -11, zstd -19, bzip2 -9 and Mamba-130M (HF, C = 4096 tokens).

Cost: 2 x (parameters touched per byte) = MFLOP/byte, against Mamba-130M's 71.35 MFLOP/byte (258.3 MFLOP/token
at 3.62 B/token). A byte-level model runs one forward per BYTE.

Outputs (/kaggle/working): specialist-results.csv, specialist-models.csv, specialist-train.csv,
specialist-classical.csv, specialist-report.txt, models/*.pt.

Launch (repo root):
    cp notebooks/log-specialist-train.py notebooks/log-specialist-train/
    python -m kaggle kernels push -p notebooks/log-specialist-train

Copyright 2026 Redukt Contributors. Apache-2.0 License.
"""

import bz2
import copy
import csv
import lzma
import math
import os
import subprocess
import sys
import time
import zlib

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("specialist-out")
SMOKE = os.environ.get("SPEC_SMOKE", "") == "1"
ZENODO = "https://zenodo.org/api/records/3227177/files/{}.tar.gz/content"
GITHUB = "https://raw.githubusercontent.com/logpai/loghub/master"
TRAIN_SYSTEMS = ["BGL", "HDFS_1", "Hadoop", "Spark", "HPC", "Mac", "Android", "Windows", "Thunderbird",
                 "OpenStack", "HealthApp"]
HELD_ZENODO = ["Zookeeper", "Proxifier"]
HELD_BENCH = {"apache": f"{GITHUB}/Apache/Apache_2k.log", "linux": f"{GITHUB}/Linux/Linux_2k.log",
              "openssh": f"{GITHUB}/OpenSSH/OpenSSH_2k.log"}
TRAIN_CAP = 48_000_000
EVAL_BYTES = 300_000
SEQ = 1024
# name, kind, config, training minutes
MODELS = [
    ("lstm-xs", "lstm", {"emb": 64, "hidden": 128, "layers": 2}, 22),
    ("lstm-s", "lstm", {"emb": 64, "hidden": 256, "layers": 2}, 30),
    ("lstm-m", "lstm", {"emb": 128, "hidden": 512, "layers": 2}, 45),
    ("tf-s", "tf", {"dim": 128, "layers": 4, "heads": 4}, 30),
]
MAMBA_MFLOP_PER_BYTE = 258.3 / 3.62
DEADLINE_HOURS = float(os.environ.get("DEADLINE_HOURS", "7"))
T_START = time.time()
LN2 = math.log(2.0)
RESULTS = []    # model, eval_set, kind, mode, bytes, bits, bpb
MODELS_OUT = []  # model, params, active_params, mflop_per_byte, cost_vs_mamba, train_min, steps, bytes_seen
TRAIN_LOG = []  # model, step, minutes, train_bpb, heldout_bpb
CLASSICAL = []  # eval_set, kind, bytes, tool, out_bytes, bpb
REPORT = []


def log(msg):
    print(msg, flush=True)
    REPORT.append(msg)


def section(title):
    log(f"\n{'=' * 72}\n  {title}   [t+{(time.time() - T_START) / 60:.1f} min]\n{'=' * 72}")


def hours_left():
    return DEADLINE_HOURS - (time.time() - T_START) / 3600


def flush_outputs():
    os.makedirs(WORK_DIR, exist_ok=True)
    for name, rows, header in (
        ("specialist-results.csv", RESULTS, ["model", "eval_set", "kind", "mode", "bytes", "bits", "bpb"]),
        ("specialist-models.csv", MODELS_OUT, ["model", "params", "active_params", "mflop_per_byte",
                                               "cost_vs_mamba130m", "train_min", "steps", "bytes_seen"]),
        ("specialist-train.csv", TRAIN_LOG, ["model", "step", "minutes", "train_bpb", "heldout_bpb"]),
        ("specialist-classical.csv", CLASSICAL, ["eval_set", "kind", "bytes", "tool", "out_bytes", "bpb"]),
    ):
        with open(os.path.join(WORK_DIR, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    with open(os.path.join(WORK_DIR, "specialist-report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT) + "\n")


def record(model, eval_set, kind, mode, nbytes, bits):
    RESULTS.append((model, eval_set, kind, mode, nbytes, f"{bits:.1f}", f"{bits / nbytes:.4f}"))


# ---------------------------------------------------------------- data

def stream_tar(system, limit, dst):
    """First `limit` bytes of the concatenated *.log members of a Loghub tarball (streamed, never stored whole)."""
    url = ZENODO.format(system)
    for pattern in ("--wildcards '*.log'", ""):
        cmd = f"curl -sL '{url}' | tar -xzOf - {pattern} 2>/dev/null | head -c {limit} > '{dst}'"
        subprocess.run(cmd, shell=True, check=False)
        if os.path.isfile(dst) and os.path.getsize(dst) > 200_000:
            return os.path.getsize(dst)
    return os.path.getsize(dst) if os.path.isfile(dst) else 0


def synthetic(seed, n):
    import random
    rnd = random.Random(seed)
    words = [f"svc{seed}-{i}" for i in range(20)]
    out = []
    while sum(len(x) for x in out) < n:
        out.append(f"2026-11-{rnd.randint(1, 28):02d} 12:{rnd.randint(0, 59):02d} {rnd.choice(words)} "
                   f"pid={rnd.randint(100, 999)} status={rnd.choice(['ok', 'fail', 'retry'])}\n")
    return "".join(out).encode()[:n]


def prepare_data():
    """Returns (train: {system: bytes}, evals: [(name, kind, bytes)])."""
    section("1. Loghub corpus — split by system")
    data = os.path.join(WORK_DIR, "data")
    os.makedirs(data, exist_ok=True)
    train, evals = {}, []
    if SMOKE:
        for i in range(3):
            blob = synthetic(i, 400_000)
            train[f"sys{i}"] = blob[:300_000]
            evals.append((f"sys{i}", "seen", blob[340_000:400_000]))
        evals.append(("unseen", "heldout", synthetic(9, 60_000)))
        return train, evals
    for system in TRAIN_SYSTEMS:
        dst = os.path.join(data, system + ".log")
        size = stream_tar(system, TRAIN_CAP + 2_000_000, dst)
        if size < 2_000_000:
            log(f"  {system:12s} only {size:,d} B fetched: skipped")
            continue
        blob = open(dst, "rb").read()
        os.remove(dst)
        if size >= TRAIN_CAP + 2_000_000:
            cut, ev = TRAIN_CAP, blob[TRAIN_CAP + 1_500_000:TRAIN_CAP + 1_500_000 + EVAL_BYTES]
        else:   # small system: train on the head, evaluate on the tail (500 KB gap is never trained on)
            cut, ev = size - 500_000, blob[size - EVAL_BYTES:]
        train[system] = blob[:cut]
        evals.append((system, "seen", ev))
        log(f"  train {system:12s} {cut:>11,d} B   seen-eval {len(ev):,d} B")
    for name, url in HELD_BENCH.items():
        dst = os.path.join(data, name + ".log")
        subprocess.run(f"curl -sL --retry 3 '{url}' -o '{dst}'", shell=True, check=False)
        blob = open(dst, "rb").read()
        evals.append((name, "heldout", blob))
        log(f"  held-out {name:10s} {len(blob):>9,d} B (benchmark file)")
    for system in HELD_ZENODO:
        dst = os.path.join(data, system + ".log")
        stream_tar(system, EVAL_BYTES, dst)
        blob = open(dst, "rb").read()
        if len(blob) > 50_000:
            evals.append((system.lower(), "heldout", blob))
            log(f"  held-out {system.lower():10s} {len(blob):>9,d} B")
    log(f"  training bytes: {sum(len(b) for b in train.values()):,d} in {len(train)} systems; "
        f"eval sets: {len(evals)}")
    return train, evals


def classical(evals):
    section("2. Classical codecs on the evaluation sets")
    tools = [("gzip-9", lambda b: zlib.compress(b, 9)), ("bz2-9", lambda b: bz2.compress(b, 9)),
             ("xz-9e", lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))]
    try:
        subprocess.run(f"{sys.executable} -m pip install -q brotli zstandard", shell=True, check=False)
        import brotli
        import zstandard
        tools.append(("brotli-11-w24", lambda b: brotli.compress(b, quality=11, lgwin=24)))
        tools.append(("zstd-19", lambda b: zstandard.ZstdCompressor(level=19).compress(b)))
    except Exception as e:   # noqa: BLE001
        log(f"  brotli/zstd unavailable: {e}")
    for name, kind, blob in evals:
        for tool, fn in tools:
            out = len(fn(blob))
            CLASSICAL.append((name, kind, len(blob), tool, out, f"{out * 8 / len(blob):.4f}"))
        best = min((r for r in CLASSICAL if r[0] == name), key=lambda r: r[4])
        log(f"  {name:12s} {kind:8s} {len(blob):>9,d} B  best {best[3]:14s} bpb={best[5]}")
    flush_outputs()


# ---------------------------------------------------------------- models

def build_model(kind, cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LstmLM(nn.Module):
        def __init__(self, emb, hidden, layers):
            super().__init__()
            self.emb = nn.Embedding(256, emb)
            self.rnn = nn.LSTM(emb, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, 256)

        def forward(self, x, state=None):
            y, state = self.rnn(self.emb(x), state)
            return self.head(y), state

    class Block(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        def forward(self, x):
            b, t, d = x.shape
            q, k, v = self.qkv(self.n1(x)).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
            return x + self.mlp(self.n2(x))

    class TfLM(nn.Module):
        def __init__(self, dim, layers, heads):
            super().__init__()
            self.emb, self.pos = nn.Embedding(256, dim), nn.Embedding(SEQ, dim)
            self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
            self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, 256)

        def forward(self, x, state=None):
            h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))
            for blk in self.blocks:
                h = blk(h)
            return self.head(self.norm(h)), None

    return LstmLM(**cfg) if kind == "lstm" else TfLM(**cfg)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    lookup = model.emb.weight.numel() + (model.pos.weight.numel() if hasattr(model, "pos") else 0)
    return total, total - lookup


def sample_batch(tensors, batch, device):
    import torch
    rows = []
    for _ in range(batch):
        t = tensors[int(torch.randint(len(tensors), (1,)))]
        o = int(torch.randint(len(t) - SEQ - 1, (1,)))
        rows.append(t[o:o + SEQ + 1])
    return torch.stack(rows).to(device).long()


def train_model(name, kind, cfg, minutes, train, quick_eval, device):
    import torch
    import torch.nn.functional as F
    section(f"Train {name} ({kind} {cfg}) — budget {minutes} min")
    torch.manual_seed(1204)
    model = build_model(kind, cfg).to(device)
    total, active = count_params(model)
    log(f"  params {total:,d}; touched per byte {active:,d}; {2 * active / 1e6:.3f} MFLOP/byte = "
        f"{2 * active / 1e6 / MAMBA_MFLOP_PER_BYTE:.4f}x Mamba-130M")
    tensors = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in train.values()]
    batch = (8 if SMOKE else 128) if kind == "lstm" else (8 if SMOKE else 64)
    peak = 2e-3 if kind == "lstm" else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.01)
    budget = minutes * 60.0
    t0, step, acc, acc_n, next_log = time.time(), 0, 0.0, 0, 0.0
    use_amp = kind == "tf" and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    while True:
        frac = (time.time() - t0) / budget
        if frac >= 1.0 or hours_left() < 1.2:
            break
        lr = peak * min(1.0, (step + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = sample_batch(tensors, batch, device)
        with torch.autocast("cuda", enabled=use_amp):
            logits, _ = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, 256).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        acc, acc_n = acc + float(loss.item()), acc_n + 1
        if frac >= next_log:
            next_log += 0.1
            held = quick_eval(model)
            mins = (time.time() - t0) / 60
            TRAIN_LOG.append((name, step, f"{mins:.1f}", f"{acc / acc_n / LN2:.4f}", f"{held:.4f}"))
            log(f"  step {step:>6d}  {mins:5.1f} min  lr {lr:.2e}  train {acc / acc_n / LN2:.4f} bpb  "
                f"held-out(static 4096) {held:.4f} bpb")
            acc, acc_n = 0.0, 0
            flush_outputs()
    mins = (time.time() - t0) / 60
    MODELS_OUT.append((name, total, active, f"{2 * active / 1e6:.3f}",
                       f"{2 * active / 1e6 / MAMBA_MFLOP_PER_BYTE:.4f}", f"{mins:.1f}", step, step * batch * SEQ))
    os.makedirs(os.path.join(WORK_DIR, "models"), exist_ok=True)
    torch.save({"kind": kind, "cfg": cfg, "state": model.state_dict()},
               os.path.join(WORK_DIR, "models", f"{name}.pt"))
    return model.eval()


# ---------------------------------------------------------------- evaluation

def static_bits(model, kind, blob, chunk, device):
    """Bits to code `blob` with a frozen model. LSTM: independent chunks of `chunk` bytes from a zero state
    (0 = whole file). Transformer: sliding 1024-byte context, 512 new bytes per window. The first byte of every
    independent piece has no context and is charged 8 bits."""
    import torch
    import torch.nn.functional as F
    data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).long().to(device)
    n, bits = len(data), 0.0
    with torch.no_grad():
        if kind == "lstm":
            size = chunk or n
            for s in range(0, n, size * 64):
                group = [data[i:i + size] for i in range(s, min(n, s + size * 64), size)]
                full = [g for g in group if len(g) == size and len(g) > 1]
                rest = [g for g in group if len(g) != size and len(g) > 1]
                bits += 8.0 * len(group)
                for part in ([torch.stack(full)] if full else []) + [r[None] for r in rest]:
                    logits, _ = model(part[:, :-1])
                    bits += float(F.cross_entropy(logits.reshape(-1, 256).float(), part[:, 1:].reshape(-1),
                                                  reduction="sum").item()) / LN2
        else:
            bits += 8.0
            end = min(n, SEQ)
            windows = [(0, end, 1)] if end > 1 else []
            while end < n:
                first, end = end, min(n, end + SEQ // 2)
                windows.append((end - SEQ, end, first))
            for s, e, first in windows:
                logits, _ = model(data[None, s:e - 1])
                nll = F.cross_entropy(logits[0].float(), data[s + 1:e], reduction="none")
                bits += float(nll[first - s - 1:].sum().item()) / LN2
    return bits


def dynamic_bits(model, blob, device, seg=256, lr=5e-4):
    """Dynamic evaluation: code a segment with the current weights, then take one Adam step on it. The state is
    carried (detached) across segments. Predictions never use bytes that have not been coded yet."""
    import torch
    import torch.nn.functional as F
    m = copy.deepcopy(model).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).long().to(device)
    bits, state = 8.0, None
    for s in range(0, len(data) - 1, seg):
        x = data[s:s + seg + 1]
        if len(x) < 2:
            break
        logits, state = m(x[None, :-1], state)
        loss = F.cross_entropy(logits[0].float(), x[1:], reduction="sum")
        bits += float(loss.item()) / LN2
        opt.zero_grad(set_to_none=True)
        (loss / (len(x) - 1)).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        state = tuple(t.detach() for t in state)
    return bits


def evaluate(name, kind, model, evals, device):
    section(f"Evaluate {name}")
    modes = ([("static-1024", 1024), ("static-4096", 4096), ("static-16384", 16384), ("static-whole", 0)]
             if kind == "lstm" else [("sliding-1024", 0)])
    for ev_name, ev_kind, blob in evals:
        cells = []
        for mode, chunk in modes:
            bits = static_bits(model, kind, blob, chunk, device)
            record(name, ev_name, ev_kind, mode, len(blob), bits)
            cells.append(f"{mode} {bits / len(blob):.4f}")
        if kind == "lstm":
            bits = dynamic_bits(model, blob, device)
            record(name, ev_name, ev_kind, "dynamic-256", len(blob), bits)
            cells.append(f"dynamic {bits / len(blob):.4f}")
        log(f"  {ev_name:12s} {ev_kind:8s} " + "  ".join(cells))
    flush_outputs()


def mamba_reference(evals, device):
    section("Reference: Mamba-130M (HF, backbone only, independent chunks of 4096 tokens)")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", torch_dtype=torch.float32)
    model = model.eval().to(device)
    bos = tok.bos_token_id if tok.bos_token_id is not None else 0
    for ev_name, ev_kind, blob in evals:
        if hours_left() < 0.25:
            log("  deadline: remaining reference rows skipped")
            break
        text = blob.decode("utf-8", errors="ignore")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        bits = 0.0
        for i in range(0, len(ids), 4095):
            x = torch.tensor([[bos] + ids[i:i + 4095]], device=device)
            with torch.no_grad():
                logits = model(input_ids=x[:, :-1]).logits[0].float()
                bits += float(F.cross_entropy(logits, x[0, 1:], reduction="sum").item()) / LN2
        nbytes = len(text.encode("utf-8"))
        record("mamba-130m", ev_name, ev_kind, "chunks-4096-tokens", nbytes, bits)
        log(f"  {ev_name:12s} {ev_kind:8s} bpb={bits / nbytes:.4f} ({len(ids):,d} tokens)")
        flush_outputs()


def summary(evals):
    section("Summary — bpb per evaluation set")
    cols = []
    for r in RESULTS:
        key = (r[0], r[3])
        if key not in cols:
            cols.append(key)
    log("  " + f"{'set':12s}{'kind':9s}{'classical':>10s}" + "".join(f"{m[:9] + ':' + md[:8]:>19s}" for m, md in cols))
    for ev_name, ev_kind, _ in evals:
        best = min((float(r[5]) for r in CLASSICAL if r[0] == ev_name), default=float("nan"))
        cells = []
        for m, md in cols:
            v = [r[6] for r in RESULTS if r[0] == m and r[3] == md and r[1] == ev_name]
            cells.append(f"{float(v[0]):19.4f}" if v else f"{'-':>19s}")
        log("  " + f"{ev_name:12s}{ev_kind:9s}{best:10.4f}" + "".join(cells))
    for kind in ("heldout", "seen"):
        log(f"\n  mean bpb over {kind} sets:")
        names = [e[0] for e in evals if e[1] == kind]
        best = [min(float(r[5]) for r in CLASSICAL if r[0] == n) for n in names if any(r[0] == n for r in CLASSICAL)]
        if best:
            log(f"    {'best classical':34s} {sum(best) / len(best):.4f}")
        for m, md in cols:
            v = [float(r[6]) for r in RESULTS if r[0] == m and r[3] == md and r[1] in names]
            if len(v) == len(names) and v:
                log(f"    {m + ' ' + md:34s} {sum(v) / len(v):.4f}")
    flush_outputs()


def main():
    import torch
    os.makedirs(WORK_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {device}; torch {torch.__version__}; smoke={SMOKE}; deadline {DEADLINE_HOURS} h")
    train, evals = prepare_data()
    held = [b for _, k, b in evals if k == "heldout"]
    try:
        classical(evals)
        for name, kind, cfg, minutes in MODELS:
            minutes = 0.4 if SMOKE else minutes

            def quick_eval(model, kind=kind):
                model.eval()
                v = sum(static_bits(model, kind, b[:100_000], 4096, device) for b in held)
                model.train()
                return v / sum(min(len(b), 100_000) for b in held)

            try:
                model = train_model(name, kind, cfg, minutes, train, quick_eval, device)
                evaluate(name, kind, model, evals, device)
            except Exception as e:   # noqa: BLE001
                log(f"  FAILED {name}: {str(e)[:400]}")
            flush_outputs()
        if not SMOKE:
            try:
                mamba_reference(evals, device)
            except Exception as e:   # noqa: BLE001
                log(f"  FAILED mamba reference: {str(e)[:300]}")
    finally:
        summary(evals)
        log(f"\nDONE in {(time.time() - T_START) / 3600:.2f} h")
        flush_outputs()


if __name__ == "__main__":
    main()
