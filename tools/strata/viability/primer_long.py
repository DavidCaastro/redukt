"""Strata viability, arm P of gate V7: a general primer (specification 3.4) on long inputs, SELECTION set only.

    python -I tools/strata/viability/primer_long.py verify --run strata_g_bc --tables D
    python -I tools/strata/viability/primer_long.py prime  --run strata_g_bc,strata_g_bc_s1205 --tables D
    python -I tools/strata/viability/primer_long.py code   --run strata_g_bc --comp mixI --tables D --size 1048576

`prime` reads the primer as one stream from boot (tables, calibration cells, bit cells, online mixer, trunk) and
writes a snapshot at every size of --sizes (the primers are prefixes of one another, so one pass serves all sizes).
It is resumable: run the same command again until it prints DONE. `code` codes each input from the snapshot (trunk,
delay taps, history, byte count and time runs back to boot; tables, cells and mixer kept) and appends ideal code
lengths of the integer model to data/strata/runs/<run>/arm-p-select.csv; finished rows are skipped, inputs of
equal length are coded as one batch, and a call pauses after --minutes (run it again until it prints DONE).
`verify` compares every P(bit = 1) of a primed stream with the sequential integer reference `bakeoff/ref_int.py`.
`occupancy --size 65536,1048576,16777216` prints how full the tables of each snapshot are.

Inputs are coded by the frozen bake-off simulation (`bakeoff/models.py`, the one `evaluate.py` runs). The online
features are recomputed here because `bakeoff/common.py` only builds them from boot, in one piece and at fixed
table sizes: `Online` carries the same state across chunks and takes the table bits as configuration. While the
primer is read, the frozen trunk runs as it is and the mixers of all models and tiers advance in one batched loop
(`prime_chunk`); `verify` goes through that same path.

Data roles: the primer is built from training sources only (`build_primer` refuses any other role); inputs must be
selection sources. A decision file (frozen test set, extended set) is refused unless --allow-decision is given,
which is not to be done before the arm-P configuration is registered in `08-capacity-ladder.md`.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bakeoff"))
sys.path.insert(0, HERE)
from common import (BC_BOOT, BC_LEN, BC_TB, BITLEN, BOS, CACHE, HASH_M1, HASH_M2, MIX_SH, NBC, NC,  # noqa: E402
                    NORD, NSET, ORDERS, Q_BITS, QSHIFT, ROOT, RUNS, SOURCES, SQ, ST, ST_MAX, TABLE_BITS, W_MAX,
                    ctx_slot_check, load_source)
from models import build, export_int  # noqa: E402
from sources import EXTENDED, is_decision, load_extended  # noqa: E402

# table bits: successor tables (orders 1, 2, 3, 4, 6, 8, 12, 16) and bit cells (orders 0-3)
TABLES = {
    "D": (TABLE_BITS, BC_TB),                                   # v1 default (specification 7.2)
    "L": ((8, 16, 22, 22, 22, 22, 22, 22), (8, 16, 24, 24)),    # the largest the specification allows (2.1)
    "V": ((8, 16, 22, 22, 22, 22, 22, 22), (8, 16, 23, 24)),    # `verify` only: L without the exact order-2 index
}
SIZES = (65_536, 1_048_576, 16_777_216)      # 2^24 is the largest primer of specification 3.4
CHUNK = 65_536
HIST = 32
PREFIX = (10_240, 102_400)
PRIMER_DIR = os.path.join(CACHE, "primer")
KEEP = ("succ", "cal", "bc", "mix")          # the four parts of a snapshot (specification 3.4)
BUDGET = ("cm", "A", "B")                    # tier names of the G0 weights (widths 0, 88, 208)

# general primer: blocks of whole lines (at most BLOCK bytes) from the training sources, by a fixed pattern
BLOCK = 4096
PATTERN = "TLTLTLTLMTLTLTLTLTLM"                 # 9 text, 9 log, 2 misc blocks: the 45 / 45 / 10 mix of training
COMPS = {"mixI": (PATTERN, None), "catTL": (PATTERN, "TML"), "catLT": (PATTERN, "LMT"), "text": ("T", None),
         "logs": ("L", None)}


def _train_bytes(name, limit):
    role, _, rel, _, _ = SOURCES[name]
    if role != "train":
        raise SystemExit(f"primer source {name} has role {role}: only training sources may enter a primer")
    with open(os.path.join(ROOT, rel), "rb") as f:
        return f.read(limit)


def build_primer(comp, size):
    """The first `size` bytes of composition `comp` (a prefix of every larger primer of an interleaved one)."""
    pattern, regroup = COMPS[comp]
    groups = {"T": ["wiki"], "L": [n for n, v in SOURCES.items() if v[0] == "train" and v[1] == "logs"],
              "M": [n for n, v in SOURCES.items() if v[0] == "train" and v[1] == "misc"]}
    streams = {"T": [_train_bytes("wiki", size + BLOCK)], "L": [_train_bytes(n, size + BLOCK) for n in groups["L"]],
               "M": [b"".join(_train_bytes(n, size) for n in groups["M"])]}
    cursor = {k: [0] * len(v) for k, v in streams.items()}
    turn = {k: 0 for k in streams}
    blocks, total, i = [], 0, 0
    while total < size:
        kind = pattern[i % len(pattern)]
        j = turn[kind] % len(streams[kind])
        data, cur = streams[kind][j], cursor[kind][j]
        piece = data[cur:cur + BLOCK]
        cut = piece.rfind(b"\n")
        if cut >= BLOCK // 2:
            piece = piece[:cut + 1]
        if not piece:
            raise SystemExit(f"primer {comp}: training stream {kind}{j} is exhausted at {total:,d} bytes")
        piece = piece[:size - total]
        cursor[kind][j] = cur + len(piece)
        turn[kind] += 1
        blocks.append((kind, piece))
        total += len(piece)
        i += 1
    if regroup:
        blocks.sort(key=lambda b: regroup.index(b[0]))        # stable: the same bytes, grouped by kind
    return b"".join(p for _, p in blocks)


def state_bytes(tables, width, layers=3):
    """Per-stream state of specification 7.1 (bytes)."""
    tb, btb = TABLES[tables]
    return (4 * (layers + 1) * width + 4 * sum(1 << b for b in tb) + 4 * NORD * NC * 8
            + 4 * NSET * (1 + NORD + NBC) + 4 * sum(1 << b for b in btb))


class Online:
    """Online state of B streams that is a pure function of the bytes: successor tables and time runs, calibration
    cells, bit cells, byte history (specification 2.2, 2.5, 3.2, 3.3). `step` returns the mixer inputs of a chunk
    and leaves the state as after its last byte, so a stream can be fed in pieces of any size."""

    def __init__(self, bsz, tables, q0):
        self.tb, self.btb = TABLES[tables]
        self.bsz = bsz
        self.succ = [[np.full(1 << b, -1, dtype=np.int32) for b in self.tb] for _ in range(bsz)]
        self.trun = np.zeros((bsz, NORD), dtype=np.int64)
        self.cal = np.tile((q0.reshape(NORD * NC, 8) << 8).astype(np.int32), (bsz, 1))   # (q << 8) | count
        self.bc_off = np.concatenate([[0], np.cumsum([1 << b for b in self.btb])]).astype(np.int64)
        self.bc = np.full((bsz, int(self.bc_off[-1])), BC_BOOT << 8, dtype=np.int32)
        self.hist = np.zeros((bsz, 0), dtype=np.uint8)

    def snapshot(self):
        """What specification 3.4 keeps (stream 0); the trunk part and the time runs are not in it."""
        snap = {f"succ{i}": t for i, t in enumerate(self.succ[0])}
        snap.update(cal=self.cal[:NORD * NC], bc=self.bc[0])
        return snap

    def load_snapshot(self, snap):
        for b in range(self.bsz):
            self.succ[b] = [snap[f"succ{i}"].copy() for i in range(NORD)]
            self.bc[b] = snap["bc"]
        self.cal = np.tile(snap["cal"], (self.bsz, 1))

    def step(self, xb):
        bsz, n = xb.shape
        pred = np.full((bsz, n, NORD), -1, dtype=np.int16)
        conf = np.zeros((bsz, n, NORD), dtype=np.uint8)
        for b in range(bsz):
            for oi in range(NORD):
                self._order(b, oi, xb[b], pred[b, :, oi], conf[b, :, oi])
        s = np.empty((bsz, n, 8, NORD + NBC), dtype=np.int16)
        top = self._evidence(xb, pred, conf, s)
        self._bitcells(xb, s)
        self.hist = np.concatenate([self.hist, xb], axis=1)[:, -HIST:]
        return s, top

    def _order(self, b, oi, x, pred, conf):
        """One successor table over one chunk, exact: events sorted by slot, the slot's content before the chunk
        (check, successor, run) standing in for the event that precedes the first one of each slot."""
        length, table, n = ORDERS[oi], self.succ[b][oi], len(x)
        have = min(self.hist.shape[1], length)
        first = length - have                      # first position of the chunk whose context is complete
        m = n - first
        if m <= 0:
            self.trun[b, oi] = 0
            return
        ext = np.concatenate([self.hist[b, self.hist.shape[1] - have:], x])
        slot, check = ctx_slot_check(ext[None, :], length, self.tb[oi])
        order = np.argsort(slot[0], kind="stable")
        ss, cs, us = slot[0][order], check[0][order].astype(np.int32), x[first:][order].astype(np.int32)
        idx = np.arange(m)
        new = np.ones(m, dtype=bool)
        new[1:] = ss[1:] != ss[:-1]
        entry = table[ss]                          # -1 = empty: its check field (-1) matches nothing
        pchk = np.where(new, entry >> 10, np.concatenate([[0], cs[:-1]]))
        psuc = np.where(new, (entry >> 2) & 0xFF, np.concatenate([[0], us[:-1]]))
        match = pchk == cs
        pred_s = np.where(match, psuc, -1)
        hit = pred_s == us
        gs = np.maximum.accumulate(np.where(new, idx, 0))
        last0 = np.maximum.accumulate(np.where(~hit, idx, -1))
        run_after = np.where(last0 < gs, np.minimum(idx - gs + 1 + (entry & 3)[gs], 3), np.minimum(idx - last0, 3))
        run_look = np.where(match, np.where(new, entry & 3, np.concatenate([[0], run_after[:-1]])), 0)
        last = np.ones(m, dtype=bool)
        last[:-1] = new[1:]
        table[ss[last]] = ((cs << 10) | (us << 2) | run_after)[last]
        pred_t, run_t, hit_t = np.empty(m, dtype=np.int16), np.empty(m, dtype=np.int64), np.zeros(n, dtype=bool)
        pred_t[order], run_t[order], hit_t[first + order] = pred_s, run_look, hit
        t = np.arange(n)
        last0 = np.maximum.accumulate(np.where(~hit_t, t, -1))
        run_end = np.where(last0 < 0, t + 1 + self.trun[b, oi], t - last0)
        trun = np.concatenate([[self.trun[b, oi]], np.minimum(run_end[:-1], 255)])
        self.trun[b, oi] = min(int(run_end[-1]), 255)
        pred[first:] = pred_t
        conf[first:] = 8 * run_t + np.minimum(BITLEN[trun[first:]], 7)

    def _evidence(self, xb, pred, conf, s_out, block=1024):
        """`common.evidence` with the calibration cells carried in `self.cal`; fills s_out[..., :NORD]."""
        bsz, n = xb.shape
        top_out = np.empty((bsz, n, 8), dtype=np.uint8)
        cells = self.cal
        base = ((np.arange(bsz)[:, None] * NORD + np.arange(NORD)[None, :]) * NC).astype(np.int32)
        shifts = np.arange(7, -1, -1, dtype=np.int16)
        order_rank = np.arange(1, NORD + 1, dtype=np.uint8)[None, None, :, None]
        st32, qsh32 = ST.astype(np.int32), QSHIFT.astype(np.int32)
        for lo in range(0, n, block):
            hi = min(n, lo + block)
            y = xb[:, lo:hi].astype(np.int16)
            p = pred[:, lo:hi]
            has = p >= 0
            pz = np.where(has, p, np.int16(0))[:, :, :, None]
            ebit = ((pz >> shifts) & 1).astype(np.int8)
            ybit = ((y[:, :, None] >> shifts) & 1).astype(np.int8)
            live = has[:, :, :, None] & ((pz >> (shifts + 1)) == (y[:, :, None] >> (shifts + 1))[:, :, None, :])
            hit = (ebit == ybit[:, :, None, :]).astype(np.int32) << Q_BITS
            sgn = np.where(live, 2 * ebit - 1, np.int8(0))
            top_out[:, lo:hi] = np.max(np.where(live, order_rank, np.uint8(0)), axis=2)
            idx_all = base[:, None, :] + conf[:, lo:hi]
            ev = np.empty(live.shape, dtype=np.int16)
            for t in range(hi - lo):
                idx = idx_all[:, t]
                cell = cells[idx]
                qs = cell >> 8
                ns = cell & 255
                ev[:, t] = sgn[:, t] * st32[qs >> (Q_BITS - 12)]
                qn = qs + ((hit[:, t] - qs) >> qsh32[ns])
                cells[idx] = np.where(live[:, t], (qn << 8) | np.minimum(ns + 1, 255), cell)
            s_out[:, lo:hi, :, :NORD] = ev.transpose(0, 1, 3, 2)
        return top_out

    def bitcell_index(self, xb):
        """Cell of every (byte, cell order, bit) inside `self.bc` flattened: int64 [B, N, NBC, 8] (spec 3.3)."""
        bsz, n = xb.shape
        pad = np.zeros((bsz, n + 3), dtype=np.uint32)          # BOS reads as 0
        k = min(3, self.hist.shape[1])
        pad[:, 3 - k:3] = self.hist[:, self.hist.shape[1] - k:]
        pad[:, 3:] = xb
        c1, c2, c3 = pad[:, 2:n + 2], pad[:, 1:n + 1], pad[:, 0:n]
        j = np.arange(8, dtype=np.uint32)
        node = (np.uint32(1) << j)[None, None, :] | (pad[:, 3:, None] >> (np.uint32(8) - j)[None, None, :])
        out = np.empty((bsz, n, NBC, 8), dtype=np.int64)
        m1, m2 = np.uint32(HASH_M1), np.uint32(HASH_M2)
        for m, (ln, tb) in enumerate(zip(BC_LEN, self.btb)):
            if ln == 0:
                ctx = np.zeros((bsz, n), dtype=np.uint32)
            elif ln == 1:
                ctx = c1
            elif ln == 2 and tb == 24:
                ctx = (c2 << np.uint32(8)) | c1                # exact index
            else:
                h = (c2 * m1 + c1 if ln == 2 else (c3 * m1 + c2) * m1 + c1) * m2
                ctx = h >> np.uint32(40 - tb)
            out[:, :, m] = ((ctx << np.uint32(8))[:, :, None] | node).astype(np.int64) + self.bc_off[m]
        return out + (np.arange(bsz, dtype=np.int64) * self.bc.shape[1])[:, None, None, None]

    def _bitcells(self, xb, s_out):
        """`common.bitcell_evidence` on the persistent tables `self.bc`; fills s_out[..., NORD:]."""
        flat = self.bitcell_index(xb)
        cells = self.bc.reshape(-1)
        st32, qsh32 = ST.astype(np.int32), QSHIFT.astype(np.int32)
        shifts = np.arange(7, -1, -1, dtype=np.int32)
        tgt = ((xb[:, :, None].astype(np.int32) >> shifts) & 1) << Q_BITS
        for t in range(xb.shape[1]):
            ix = flat[:, t]
            cell = cells[ix]
            qs = cell >> 8
            ns = cell & 255
            s_out[:, t, :, NORD:] = st32[qs >> (Q_BITS - 12)].transpose(0, 2, 1)
            qn = qs + ((tgt[:, t][:, None, :] - qs) >> qsh32[ns])
            cells[ix] = (qn << 8) | np.minimum(ns + 1, 255)


class Coder:
    """Trunk and head of the frozen bake-off simulation for chosen tiers, with the stream state made explicit."""

    def __init__(self, run, tiers):
        ck = torch.load(os.path.join(RUNS, run, "ckpt.pt"), weights_only=False)
        self.model = build(ck["args"]["model"])
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        if not self.model.quant or not self.model.bc:
            raise SystemExit("arm P needs an integer Strata model with bit cells")
        self.run, self.seed, self.tiers = run, ck["args"]["seed"], tuple(tiers)
        self.trunk_on = any(self.model.widths[k] > 0 for k in tiers)
        self.budget = self.model.budget()
        self.ints = {k: export_int(self.model, k) for k in tiers}

    def boot(self, bsz, deltas=None):
        """Stream start: trunk at rest, BOS in the delay taps; mixer at boot or at the primer's weights."""
        head = self.model.head.init_state(bsz)
        delta = {k: head[k] if deltas is None else torch.from_numpy(np.tile(deltas[k], (bsz, 1, 1)))
                 for k in self.tiers}
        return {"trunk": self.model.trunk.init_state(bsz), "delta": delta,
                "tail": np.full((bsz, 4), BOS, dtype=np.int16)}

    def hidden(self, st, xb, window=4096):
        """Trunk output of a chunk [B, N, d] (integers held in float32); advances the trunk and the delay taps."""
        bsz, n = xb.shape
        pad = np.concatenate([st["tail"], xb.astype(np.int16)], axis=1)
        st["tail"] = pad[:, -4:]
        if not self.trunk_on:
            return torch.zeros(bsz, n, 0)
        xt = np.stack([pad[:, 3 - k:3 - k + n] for k in range(4)], axis=2)
        out = []
        with torch.no_grad():
            for lo in range(0, n, window):
                h, st["trunk"] = self.model.trunk(torch.from_numpy(xt[:, lo:lo + window].astype(np.int64)),
                                                  st["trunk"])
                out.append(h)
        return torch.cat(out, dim=1)

    def code(self, st, xb, s, top, window=4096, want_p=False):
        """Cost in bits per byte {tier: [B, N]} (and P(bit = 1) {tier: [B, N, 8]}); advances `st`."""
        bsz, n = xb.shape
        h = self.hidden(st, xb, window)
        cost = {k: np.empty((bsz, n)) for k in self.tiers}
        prob = {k: np.empty((bsz, n, 8), dtype=np.int64) for k in self.tiers} if want_p else None
        with torch.no_grad():
            for lo in range(0, n, window):
                sl = slice(lo, min(n, lo + window))
                y = torch.from_numpy(xb[:, sl].astype(np.int64))
                sw = torch.from_numpy(s[:, sl].astype(np.float32))
                tw = torch.from_numpy(top[:, sl].astype(np.int64))
                for k in self.tiers:
                    c, st["delta"][k], p1 = self.model.head.tier(k, h[:, sl], sw, tw, y, st["delta"][k])
                    cost[k][:, sl] = c.numpy()
                    if want_p:
                        prob[k][:, sl] = p1.numpy()
        return (cost, prob) if want_p else cost

    def mixer_inputs(self, k, h, x, s):
        """Mixer inputs of tier k for one stream, int64 [N, 8, n_in]: the part of `Head.tier` before its mixer loop
        (node logit, gains, scaled evidence, bit cells), from the exported integers (specification 2.5)."""
        p = self.ints[k]
        hk = h[:, :p["width"]].astype(np.float64)                  # |sums| < 2^53: float64 products are exact
        shifts = np.arange(7, -1, -1)
        node = ((1 << np.arange(8)) | (x[:, None].astype(np.int64) >> (shifts + 1))) - 1
        z = np.take_along_axis((hk @ p["hw"].T.astype(np.float64)).astype(np.int64), node, axis=1) + p["hb"][node]
        st_net = np.clip((z + 4) >> 3, -ST_MAX, ST_MAX)
        gain = np.clip(((hk @ p["gw"].T.astype(np.float64)).astype(np.int64) + p["gb"] + 32) >> 6, 0, 127)
        sp = np.clip((s[:, :, :NORD].astype(np.int64) * gain[:, None, :]) >> 5, -ST_MAX, ST_MAX)
        return np.concatenate([st_net[:, :, None], sp, s[:, :, NORD:].astype(np.int64)], axis=2)


def prime_chunk(online, coders, sts, xb):
    """Advance a primer by one chunk: online state, every trunk, and the mixers of all (model, tier) pairs in ONE
    sequential loop. The loop is `Head.mix_loop` (integer branch) over absolute weights; it is batched over the
    models because per-step interpreter overhead, not arithmetic, is what a primer of megabytes costs here."""
    s, top = online.step(xb)
    pairs = [(c, k) for c in coders for k in c.tiers]
    hs = {c.run: c.hidden(sts[c.run], xb)[0].numpy() for c in coders}
    x = np.stack([c.mixer_inputs(k, hs[c.run], xb[0], s[0]) for c, k in pairs], axis=1)       # [N, S, 8, n_in]
    w0 = np.stack([c.ints[k]["w0"] for c, k in pairs])
    w = (w0 + np.stack([sts[c.run]["delta"][k][0].numpy() for c, k in pairs])).reshape(-1, x.shape[-1])
    si = np.arange(8)[None, :] * (NORD + 1) + top[0]                                           # [N, 8]
    si = si[:, None, :] + (np.arange(len(pairs)) * NSET)[None, :, None]
    yb = (((xb[0][:, None].astype(np.int64) >> np.arange(7, -1, -1)) & 1) << 16)[:, None, :]
    half = 1 << (MIX_SH - 1)
    for t in range(xb.shape[1]):
        i, xt = si[t], x[t]
        wt = w[i]
        st = np.minimum(np.maximum((wt * xt).sum(-1) >> 16, -ST_MAX), ST_MAX)
        err = yb[t] - SQ[st + ST_MAX]
        w[i] = np.minimum(np.maximum(wt + ((xt * err[..., None] + half) >> MIX_SH), -W_MAX), W_MAX)
    delta = w.reshape(w0.shape) - w0
    for j, (c, k) in enumerate(pairs):
        sts[c.run]["delta"][k] = torch.from_numpy(delta[j][None])


def snapshot_path(coder, comp, tables, size):
    tag = "".join(str(k) for k in coder.tiers)
    return os.path.join(PRIMER_DIR, f"{coder.run}-{comp}-{tables}-t{tag}-{size}.npz")


def prime(coders, comp, tables, sizes, minutes, chunk=CHUNK):
    """Read the primer from boot with every model; write one snapshot per model at every size. Returns False when
    paused (resumable)."""
    os.makedirs(PRIMER_DIR, exist_ok=True)
    sizes = sorted(sizes)
    if all(os.path.isfile(snapshot_path(c, comp, tables, z)) for c in coders for z in sizes):
        return True
    primer = np.frombuffer(build_primer(comp, sizes[-1]), dtype=np.uint8)
    resume = snapshot_path(coders[0], comp, tables, 0) + f".x{len(coders)}.resume.pt"
    if os.path.isfile(resume):
        ck = torch.load(resume, weights_only=False)
        online, sts, pos, spent = ck["online"], ck["sts"], ck["pos"], ck["spent"]
        print(f"resumed at byte {pos:,d} ({spent / 60:.1f} min so far)", flush=True)
    else:
        online = Online(1, tables, np.load(os.path.join(CACHE, "q0.npy")))
        sts, pos, spent = {c.run: c.boot(1) for c in coders}, 0, 0.0
    t0 = time.time()
    while pos < sizes[-1]:
        hi = min([z for z in sizes if z > pos] + [pos + chunk])
        prime_chunk(online, coders, sts, primer[None, pos:hi])
        pos = hi
        if pos in sizes:
            for c in coders:
                snap = online.snapshot()
                snap.update({f"delta{k}": sts[c.run]["delta"][k][0].numpy() for k in c.tiers})
                snap["seconds"] = np.float64(spent + time.time() - t0)
                tmp = snapshot_path(c, comp, tables, pos) + ".tmp.npz"
                np.savez_compressed(tmp, **snap)
                os.replace(tmp, snapshot_path(c, comp, tables, pos))
            print(f"snapshot at {pos:,d} bytes after {(spent + time.time() - t0) / 60:.1f} min", flush=True)
        if time.time() - t0 > minutes * 60 and pos < sizes[-1]:
            torch.save({"online": online, "sts": sts, "pos": pos, "spent": spent + time.time() - t0},
                       resume + ".tmp")
            os.replace(resume + ".tmp", resume)
            print(f"PAUSED at byte {pos:,d}/{sizes[-1]:,d}: run the same command again", flush=True)
            return False
    if os.path.isfile(resume):
        os.remove(resume)
    return True


def load_inputs(names, allow_decision):
    """Inputs grouped by length: [(names, uint8 [B, N])]; each group is coded as one batch of streams."""
    groups = {}
    for name in names:
        if is_decision(name) and not allow_decision:
            raise SystemExit(f"refusing to code {name}: not a selection source (decision files need "
                             "--allow-decision, after the arm-P configuration is registered)")
        data = load_extended(name) if name in EXTENDED else load_source(name)
        groups.setdefault(len(data), []).append((name, data))
    return [([n for n, _ in g], np.stack([d for _, d in g])) for g in groups.values()]


def code_inputs(coder, comp, tables, size, names, xb, keep=KEEP, resume=None, minutes=1e9, chunk=CHUNK // 4):
    """Ideal bits of every input coded from the snapshot of (comp, tables, size); size 0 = boot.
    `keep` names the parts of the snapshot that are used; the others start from boot (specification 3.4 keeps
    all four; arm P-s of amendment 5 keeps `succ` only). With `resume`, pauses after `minutes` and returns None."""
    if resume and os.path.isfile(resume):
        ck = torch.load(resume, weights_only=False)
        online, st, parts, lo = ck["online"], ck["st"], ck["parts"], ck["lo"]
    else:
        q0 = np.load(os.path.join(CACHE, "q0.npy"))
        online = Online(len(names), tables, q0)
        deltas = None
        if size:
            with np.load(snapshot_path(coder, comp, tables, size)) as f:
                snap = {k: f[k] for k in f.files}
            boot = Online(1, tables, q0).snapshot()
            snap.update({k: v for k, v in boot.items() if not any(k.startswith(part) for part in keep)})
            online.load_snapshot(snap)
            if "mix" in keep:
                deltas = {k: snap[f"delta{k}"] for k in coder.tiers}
        st, parts, lo = coder.boot(len(names), deltas), {k: [] for k in coder.tiers}, 0
    t0 = time.time()
    while lo < xb.shape[1]:
        if resume and time.time() - t0 > minutes * 60:
            torch.save({"online": online, "st": st, "parts": parts, "lo": lo}, resume + ".tmp")
            os.replace(resume + ".tmp", resume)
            print(f"at byte {lo:,d} of {xb.shape[1]:,d} ({len(names)} streams)", flush=True)
            return None
        piece = xb[:, lo:lo + chunk]
        s, top = online.step(piece)
        cost = coder.code(st, piece, s, top)
        for k in coder.tiers:
            parts[k].append(cost[k])
        lo += chunk
    if resume and os.path.isfile(resume):
        os.remove(resume)
    return {k: np.concatenate(v, axis=1) for k, v in parts.items()}


def cmd_code(args, coder):
    groups = load_inputs(args.inputs.split(","), args.allow_decision)
    out = os.path.join(RUNS, args.run, args.out)
    done = set()
    if os.path.isfile(out):
        done = {(r["comp"], r["tables"], r["primer_bytes"], r["tier"], r["source"]) for r in csv.DictReader(open(out))}
    else:
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(["run", "seed", "tier", "width", "mac_per_byte", "comp", "tables", "primer_bytes",
                                    "state_bytes", "source", "bytes", "bits", "bpb"] + [f"bpb@{p}" for p in PREFIX])
    keep = tuple(args.keep.split(","))
    if not set(keep) <= set(KEEP):
        raise SystemExit(f"--keep takes a subset of {','.join(KEEP)}")
    t0 = time.time()
    for size in [int(v) for v in args.size.split(",")]:
        comp = (args.comp if set(keep) == set(KEEP) else f"{args.comp}/keep-{'-'.join(keep)}") if size else "none"
        for names, xb in groups:
            keys = [(comp, args.tables, str(size), BUDGET[k], n) for k in coder.tiers for n in names]
            if all(k in done for k in keys):
                continue
            left = args.minutes - (time.time() - t0) / 60
            cost = code_inputs(coder, args.comp, args.tables, size, names, xb, keep,
                               f"{out}.{size}.{xb.shape[1]}.resume.pt", left)
            if cost is None:
                print("PAUSED: run the same command again", flush=True)
                return
            rows = []
            for k in coder.tiers:
                width = coder.model.widths[k]
                for i, name in enumerate(names):
                    bits = float(cost[k][i].sum())
                    rows.append([args.run, coder.seed, BUDGET[k], width, coder.budget[k][1], comp, args.tables, size,
                                 state_bytes(args.tables, width), name, xb.shape[1], f"{bits:.1f}",
                                 f"{bits / xb.shape[1]:.4f}"]
                                + [f"{cost[k][i, :p].sum() / p:.4f}" if p <= xb.shape[1] else "" for p in PREFIX])
                print(f"{args.run} tier {BUDGET[k]} primer {comp} {size:>10,d} tables {args.tables}: "
                      + "  ".join(f"{n}={cost[k][i].sum() / xb.shape[1]:.4f}" for i, n in enumerate(names)),
                      flush=True)
            with open(out, "a", newline="") as f:
                csv.writer(f).writerows(r for r, key in zip(rows, keys) if key not in done)
    print("DONE", flush=True)


def cmd_occupancy(args, coder):
    """How full the tables of each snapshot are: valid successor slots and visited bit cells, per order."""
    tb, btb = TABLES[args.tables]
    off = np.concatenate([[0], np.cumsum([1 << b for b in btb])])
    print(f"tables {args.tables}: successor bits {tb}, bit-cell bits {btb}; state per stream "
          + " / ".join(f"{state_bytes(args.tables, coder.model.widths[k]):,d} B ({BUDGET[k]})" for k in coder.tiers))
    for size in [int(v) for v in args.size.split(",")]:
        with np.load(snapshot_path(coder, args.comp, args.tables, size)) as snap:
            succ = [float((snap[f"succ{i}"] >= 0).mean()) for i in range(NORD)]
            bc = [float(((snap["bc"][off[m]:off[m + 1]] & 255) > 0).mean()) for m in range(NBC)]
            full = [float(((snap["bc"][off[m]:off[m + 1]] & 255) == 255).mean()) for m in range(NBC)]
            print(f"{args.comp} {size:>10,d}: successor slots valid " + " ".join(f"{v:.3f}" for v in succ)
                  + " | bit cells visited " + " ".join(f"{v:.3f}" for v in bc)
                  + " | count saturated " + " ".join(f"{v:.3f}" for v in full)
                  + f" | priming {float(snap['seconds']) / 60:.1f} min", flush=True)


def cmd_verify(args, coder):
    """A primed stream against the sequential integer reference (independent code path), every probability."""
    import ref_int
    ref_int.TABLE_BITS, ref_int.BC_TB = TABLES[args.tables]
    q0 = np.load(os.path.join(CACHE, "q0.npy"))
    primer = np.frombuffer(build_primer(args.comp, args.primer_bytes), dtype=np.uint8)
    data = load_inputs([args.source], False)[0][1][:, :args.bytes]
    online = Online(1, args.tables, q0)
    sts = {coder.run: coder.boot(1)}
    for lo in range(0, len(primer), args.chunk):         # small chunks: every carry across chunks is exercised
        prime_chunk(online, [coder], sts, primer[None, lo:lo + args.chunk])
    snap = online.snapshot()
    deltas = {k: sts[coder.run]["delta"][k][0].numpy() for k in coder.tiers}
    online = Online(1, args.tables, q0)
    online.load_snapshot(snap)
    st = coder.boot(1, deltas)
    probs = {k: [] for k in coder.tiers}
    bits = {k: 0.0 for k in coder.tiers}
    for lo in range(0, data.shape[1], args.chunk):
        piece = data[:, lo:lo + args.chunk]
        s, top = online.step(piece)
        cost, prob = coder.code(st, piece, s, top, want_p=True)
        for k in coder.tiers:
            probs[k].append(prob[k][0])
            bits[k] += float(cost[k].sum())
    ok = True
    for k in coder.tiers:
        ref = ref_int.StrataRef(export_int(coder.model, k), q0)
        ref_int.code_stream(ref, primer)
        ref.hist, ref.ctx, ref.n, ref.trun = [BOS] * 4, [], 0, [0] * NORD      # specification 3.4
        ref.s_in[:] = 0
        for acc in ref.s:
            acc[:] = 0
        ref_p, ref_bits = ref_int.code_stream(ref, data[0])
        diff = int((ref_p != np.concatenate(probs[k])).sum())
        ok = ok and diff == 0
        print(f"tier {BUDGET[k]} tables {args.tables}: primer {len(primer):,d} B + input {data.shape[1]:,d} B, "
              f"{ref_p.size} probabilities, mismatches vs the integer reference = {diff}; bits here {bits[k]:.3f} / "
              f"reference {ref_bits:.3f}", flush=True)
    print("ALL CHECKS PASSED" if ok else "CHECK FAILED", flush=True)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prime", "code", "verify", "occupancy"))
    ap.add_argument("--run", required=True, help="run directory under data/strata/runs (G0 weights); `prime` "
                    "takes several, comma-separated, and reads the primer once for all of them")
    ap.add_argument("--comp", default="mixI", choices=sorted(COMPS))
    ap.add_argument("--tables", default="D", choices=sorted(TABLES))
    ap.add_argument("--tiers", default="0,1")
    ap.add_argument("--sizes", default=",".join(str(z) for z in SIZES), help="prime: snapshot sizes")
    ap.add_argument("--size", default="0", help="code, occupancy: primer sizes, 0 = no primer")
    ap.add_argument("--inputs", default=",".join(n for n, v in SOURCES.items() if v[0] == "select"))
    ap.add_argument("--keep", default=",".join(KEEP), help="code: parts of the snapshot to use (diagnostic)")
    ap.add_argument("--allow-decision", action="store_true")
    ap.add_argument("--out", default="arm-p-select.csv")
    ap.add_argument("--minutes", type=float, default=7.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--source", default="sel-webster", help="verify: input")
    ap.add_argument("--bytes", type=int, default=3000)
    ap.add_argument("--primer-bytes", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=1000, help="verify: chunk size")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    coders = [Coder(run, [int(k) for k in args.tiers.split(",")]) for run in args.run.split(",")]
    coder = coders[0]
    if args.cmd == "prime":
        done = prime(coders, args.comp, args.tables, [int(v) for v in args.sizes.split(",")], args.minutes)
        print("DONE" if done else "", flush=True)
    elif args.cmd == "code":
        cmd_code(args, coder)
    elif args.cmd == "occupancy":
        cmd_occupancy(args, coder)
    else:
        cmd_verify(args, coder)


if __name__ == "__main__":
    main()
