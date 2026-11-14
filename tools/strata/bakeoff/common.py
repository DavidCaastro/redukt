"""Strata bake-off: constants, integer tables, source split, online context features (numpy, exact integers).

Everything here that is called "integer" is computed with integer numpy dtypes only. The two lookup tables
(squash / stretch) are generated once with float64 by the trainer side and then treated as data: in the product
they ship inside the weight file, no implementation regenerates them.
"""
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "strata", "raw")
CACHE = os.path.join(ROOT, "data", "strata", "cache")
RUNS = os.path.join(ROOT, "data", "strata", "runs")
BENCH = os.path.join(ROOT, "data", "benchmark")

# ---------------------------------------------------------------- fixed-point contract of the bake-off
A_BITS = 5            # activations: real = int / 32, int8 range [-127, 127]
F_BITS = 12           # trace accumulators carry 12 fractional bits
ST_MAX = 3071         # stretch domain: Q8 nats, clamped to [-3071, 3071]
Q_BITS = 22           # adaptive probabilities of the context evidence: 22-bit
MIX_SH = 17           # mixer learning rate 2^(8-17) = 2^-9
W_ONE = 1 << 16       # mixer weights are Q16
W_MAX = (1 << 19) - 1
ORDERS = (1, 2, 3, 4, 6, 8, 12, 16)      # context lengths of the successor tables
TABLE_BITS = (8, 16, 16, 16, 18, 18, 18, 18)
NORD = len(ORDERS)
N_IN = NORD + 1       # mixer inputs: the trunk + one per order
NC = 32               # confidence buckets per order: 8 * min(slot_run, 3) + min(bit_length(time_run), 7)
NSET = 8 * (NORD + 1)  # mixer weight sets: bit position x (highest live order + 1)
HASH_M1 = 0x9E3779B1
HASH_M2 = 0x85EBCA6B
MASK32 = 0xFFFFFFFF
BOS = 256

BITLEN = np.array([int(n).bit_length() for n in range(256)], dtype=np.int64)
QSHIFT = np.array([min(7, max(2, int(n + 2).bit_length())) for n in range(256)], dtype=np.int64)


def squash_table():
    """SQ[st + ST_MAX] = P(bit = 1) in 1..65535 for a stretch value st (Q8 nats)."""
    i = np.arange(2 * ST_MAX + 1, dtype=np.float64) - ST_MAX
    p = 65536.0 / (1.0 + np.exp(-i / 256.0))
    return np.clip(np.rint(p), 1, 65535).astype(np.int64)


def stretch_table():
    """ST[p12] = stretch (Q8 nats) of the probability (p12 + 0.5) / 4096."""
    k = np.arange(4096, dtype=np.float64)
    x = (k + 0.5) / 4096.0
    return np.clip(np.rint(256.0 * np.log(x / (1.0 - x))), -ST_MAX, ST_MAX).astype(np.int64)


SQ = squash_table()
ST = stretch_table()

# ---------------------------------------------------------------- sources and roles (leak-free by source)
# role: train | select (architecture / hyper-parameter choice) | test (frozen, report-only) | seen (same source as
# a training source, disjoint bytes; never called held-out)
SOURCES = {
    # name: (role, group, path relative to ROOT, offset, length or None)
    "wiki": ("train", "text", "data/strata/raw/wiki/enwik9-100M-300M.bin", 0, None),
    "world192": ("train", "misc", "data/strata/raw/canterbury/world192.txt", 0, None),
    "lcet10": ("train", "misc", "data/strata/raw/canterbury/lcet10.txt", 0, None),
    "cp.html": ("train", "misc", "data/strata/raw/canterbury/cp.html", 0, None),
    "fields.c": ("train", "misc", "data/strata/raw/canterbury/fields.c", 0, None),
    "grammar.lsp": ("train", "misc", "data/strata/raw/canterbury/grammar.lsp", 0, None),
    "xargs.1": ("train", "misc", "data/strata/raw/canterbury/xargs.1", 0, None),
    **{f"log-{s}": ("train", "logs", f"data/strata/raw/loghub/{s}.log", 0, None)
       for s in ("BGL", "HDFS_1", "Hadoop", "Spark", "HPC", "Mac", "Android", "Windows", "Thunderbird",
                 "OpenStack", "HealthApp")},
    # selection set: sources never trained on, distinct from the frozen test set
    "sel-webster": ("select", "text", "data/benchmark/silesia/webster", 0, 300_000),
    "sel-xml": ("select", "markup", "data/benchmark/silesia/xml", 0, 300_000),
    "sel-zookeeper": ("select", "logs", "data/strata/raw/loghub/Zookeeper.log", 0, 300_000),
    "sel-proxifier": ("select", "logs", "data/strata/raw/loghub/Proxifier.log", 0, 300_000),
    # frozen test set (report-only; carries the Mamba-130M reference numbers of the repo)
    "test-dickens": ("test", "text", "data/benchmark/silesia/dickens", 0, 500_000),
    "test-apache": ("test", "logs", "data/benchmark/industry/apache-access.log", 0, None),
    "test-linux": ("test", "logs", "data/benchmark/industry/linux-syslog.log", 0, None),
    "test-openssh": ("test", "logs", "data/benchmark/industry/openssh-auth.log", 0, None),
    "test-facturae": ("test", "xml-synthetic", "data/benchmark/industry/facturae-batch.xml", 0, None),
    "test-ubl": ("test", "xml-synthetic", "data/benchmark/industry/ubl-invoice-batch.xml", 0, None),
    # Wikipedia is a training source here, so enwik8 is NOT held-out for these models: same source, disjoint bytes
    "seen-enwik8": ("seen", "text", "data/benchmark/enwik8", 0, 500_000),
    # length curves (A2): 2 MiB single streams, resets do not exist
    "curve-dickens": ("curve", "text", "data/benchmark/silesia/dickens", 0, 2_097_152),
    "curve-enwik8": ("curve", "text-seen", "data/benchmark/enwik8", 0, 2_097_152),
}
TRAIN_GROUP_WEIGHT = {"text": 0.45, "logs": 0.45, "misc": 0.10}
CURVE_POINTS = (10_240, 51_200, 102_400, 512_000, 1_048_576, 2_097_152)


def load_source(name):
    role, group, rel, off, length = SOURCES[name]
    path = os.path.join(ROOT, rel)
    if length is None:
        data = np.fromfile(path, dtype=np.uint8)
        return data[off:]
    with open(path, "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(length), dtype=np.uint8).copy()


def sources_with_role(role):
    return [n for n, v in SOURCES.items() if v[0] == role]


# ---------------------------------------------------------------- context hashing and successor tables
def ctx_slot_check(xb, length, table_bits):
    """Slot index and 8-bit check of the context x[t-length .. t-1] for t = length .. N-1.

    xb: uint8 [B, N]. Returns two uint32 arrays [B, N - length].
    order 1: slot = previous byte; order 2: slot = previous two bytes (big-endian); otherwise
    h = 0; for each context byte, oldest first: h = ((h + byte + 1) * M1) mod 2^32;
    h ^= h >> 15; h = (h * M2) mod 2^32; h ^= h >> 13; slot = h >> (32 - table_bits); check = h & 0xFF.
    (uint32 numpy arithmetic wraps modulo 2^32, which is exactly the definition.)
    """
    n = xb.shape[1] - length
    xs = xb.astype(np.uint32)
    if length == 1:
        return xs[:, 0:n].copy(), np.zeros((xb.shape[0], n), dtype=np.uint32)
    if length == 2:
        return (xs[:, 0:n] << np.uint32(8)) | xs[:, 1:n + 1], np.zeros((xb.shape[0], n), dtype=np.uint32)
    h = np.zeros((xb.shape[0], n), dtype=np.uint32)
    for k in range(length):
        h = (h + xs[:, k:k + n] + np.uint32(1)) * np.uint32(HASH_M1)
    h ^= h >> np.uint32(15)
    h = h * np.uint32(HASH_M2)
    h ^= h >> np.uint32(13)
    return h >> np.uint32(32 - table_bits), h & np.uint32(0xFF)


def successor_features(xb):
    """Vectorised, exact emulation of the per-stream successor tables.

    xb: uint8 [B, N] (B independent streams). Returns pred int16 [B, N, NORD] (-1 = no prediction) and
    conf uint8 [B, N, NORD] (bucket 0..NC-1) for predicting xb[:, t] from the bytes before t.
    Table semantics per (stream, order): slot -> (check, successor, hit-run 0..3); lookup with the context ending
    at t-1, then update with the byte at t (hit: run+1 capped at 3; miss or empty: store byte, run 0).
    """
    bsz, n = xb.shape
    pred = np.full((bsz, n, NORD), -1, dtype=np.int16)
    conf = np.zeros((bsz, n, NORD), dtype=np.uint8)
    tidx = np.arange(n, dtype=np.int32)[None, :]
    for oi, (length, tb) in enumerate(zip(ORDERS, TABLE_BITS)):
        m = n - length
        if m <= 0:
            continue
        slot, check = ctx_slot_check(xb, length, tb)
        order = np.argsort(slot, axis=1, kind="stable")    # per stream: by slot, time order kept inside a slot
        slot_s = np.take_along_axis(slot, order, axis=1)
        check_s = np.take_along_axis(check, order, axis=1)
        succ = np.take_along_axis(xb[:, length:], order, axis=1).astype(np.int16)   # byte after the context = x[t]
        same = np.zeros((bsz, m), dtype=bool)
        same[:, 1:] = (slot_s[:, 1:] == slot_s[:, :-1]) & (check_s[:, 1:] == check_s[:, :-1])
        pred_s = np.full((bsz, m), -1, dtype=np.int16)
        pred_s[:, 1:] = np.where(same[:, 1:], succ[:, :-1], -1)
        hit_s = same & (pred_s == succ)
        idx = tidx[:, :m]
        last0 = np.maximum.accumulate(np.where(~hit_s, idx, 0), axis=1)
        run_after = np.minimum(idx - last0, 3)             # hit-run stored in the slot after this visit
        srun_s = np.zeros((bsz, m), dtype=np.int32)
        srun_s[:, 1:] = np.where(same[:, 1:], run_after[:, :-1], 0)
        pred_t = np.empty((bsz, m), dtype=np.int16)
        srun_t = np.empty((bsz, m), dtype=np.int32)
        hit_t = np.empty((bsz, m), dtype=bool)
        np.put_along_axis(pred_t, order, pred_s, axis=1)
        np.put_along_axis(srun_t, order, srun_s, axis=1)
        np.put_along_axis(hit_t, order, hit_s, axis=1)
        # time-run: consecutive hits of this order immediately before t (saturating at 255)
        hit_full = np.zeros((bsz, n), dtype=bool)
        hit_full[:, length:] = hit_t
        last0 = np.maximum.accumulate(np.where(~hit_full, tidx, -1), axis=1)
        run_end = tidx - last0
        trun = np.zeros((bsz, n), dtype=np.int32)
        trun[:, 1:] = np.minimum(run_end[:, :-1], 255)
        pred[:, length:, oi] = pred_t
        conf[:, length:, oi] = (8 * srun_t + np.minimum(BITLEN[trun[:, length:]], 7)).astype(np.uint8)
    return pred, conf


def evidence(xb, pred, conf, q0, block=1024):
    """Bit-level evidence of every order, with the online calibration cells q[order][conf][bit] (22-bit) + counts.

    Returns s int16 [B, N, 8, NORD] (signed stretch toward P(bit = 1), 0 when the order is not live) and
    top uint8 [B, N, 8] (1 + highest live order, 0 = none). q0: int64 [NORD, NC, 8] boot values.
    Cell update when the order is live at that bit: q += ((hit << 22) - q) >> QSHIFT[count]; count saturates at 255.
    """
    bsz, n = xb.shape
    s_out = np.empty((bsz, n, 8, NORD), dtype=np.int16)
    top_out = np.empty((bsz, n, 8), dtype=np.uint8)
    # packed (q << 8) | count in int32 (22 + 8 bits), one row of 8 bit positions per (stream, order, bucket)
    cells = np.tile((q0.reshape(NORD * NC, 8) << 8).astype(np.int32), (bsz, 1))
    base = ((np.arange(bsz)[:, None] * NORD + np.arange(NORD)[None, :]) * NC).astype(np.int32)
    shifts = np.arange(7, -1, -1, dtype=np.int16)            # bit j (MSB first) = (v >> (7 - j)) & 1
    order_rank = np.arange(1, NORD + 1, dtype=np.uint8)[None, None, :, None]
    st32 = ST.astype(np.int32)
    qsh32 = QSHIFT.astype(np.int32)
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        y = xb[:, lo:hi].astype(np.int16)                    # [B, T]
        p = pred[:, lo:hi]                                   # [B, T, NORD] int16, -1 = none
        has = p >= 0
        pz = np.where(has, p, np.int16(0))
        # bit j of the prediction and of the byte; "live at bit j" = the j bits already coded agree
        ebit = ((pz[:, :, :, None] >> shifts) & 1).astype(np.int8)               # [B, T, NORD, 8]
        ybit = ((y[:, :, None] >> shifts) & 1).astype(np.int8)                   # [B, T, 8]
        live = has[:, :, :, None] & ((pz[:, :, :, None] >> (shifts + 1)) == (y[:, :, None] >> (shifts + 1))[:, :, None, :])
        hit = ebit == ybit[:, :, None, :]
        sgn = np.where(live, 2 * ebit - 1, np.int8(0))
        top_out[:, lo:hi] = np.max(np.where(live, order_rank, np.uint8(0)), axis=2)
        idx_all = base[:, None, :] + conf[:, lo:hi]
        ev = np.empty(live.shape, dtype=np.int16)
        for t in range(hi - lo):
            idx = idx_all[:, t]
            cell = cells[idx]                                # [B, NORD, 8]
            qs = cell >> 8
            ns = cell & 255
            ev[:, t] = sgn[:, t] * st32[qs >> (Q_BITS - 12)]
            qn = qs + (((hit[:, t].astype(np.int32) << Q_BITS) - qs) >> qsh32[ns])
            cells[idx] = np.where(live[:, t], (qn << 8) | np.minimum(ns + 1, 255), cell)
        s_out[:, lo:hi] = ev.transpose(0, 1, 3, 2)
    return s_out, top_out


def boot_calibration(sample_chunks):
    """Data-derived boot values q0[order][conf][bit]: empirical frequency of 'bit equals the expected bit'."""
    pred, conf = successor_features(sample_chunks)
    y = sample_chunks.astype(np.int64)
    shifts = np.arange(7, -1, -1, dtype=np.int64)
    hits = np.zeros((NORD, NC, 8), dtype=np.float64)
    tot = np.zeros((NORD, NC, 8), dtype=np.float64)
    for oi in range(NORD):
        p = pred[:, :, oi].astype(np.int64)
        has = p >= 0
        pz = np.where(has, p, 0)
        live = has[:, :, None] & ((pz[:, :, None] >> (shifts + 1)) == (y[:, :, None] >> (shifts + 1)))
        hitb = ((pz[:, :, None] >> shifts) & 1) == ((y[:, :, None] >> shifts) & 1)
        c = conf[:, :, oi].astype(np.int64)
        for j in range(8):
            np.add.at(tot[oi, :, j], c[live[:, :, j]], 1.0)
            np.add.at(hits[oi, :, j], c[live[:, :, j] & hitb[:, :, j]], 1.0)
    frac = (hits + 1.0) / (tot + 2.0)
    q0 = np.clip(np.rint(frac * (1 << Q_BITS)), 1 << 10, (1 << Q_BITS) - (1 << 10)).astype(np.int64)
    return q0


# ---------------------------------------------------------------- training data: deterministic chunk sampler
class Corpus:
    def __init__(self):
        self.groups = {}
        for name in sources_with_role("train"):
            group = SOURCES[name][1]
            self.groups.setdefault(group, []).append((name, load_source(name)))
        self.group_names = sorted(self.groups)
        w = np.array([TRAIN_GROUP_WEIGHT[g] for g in self.group_names], dtype=np.float64)
        self.group_p = w / w.sum()

    def sample_chunks(self, seed, index, count, chunk):
        """`count` chunks of `chunk` bytes; chunk starts are moved to the next line start when one is near."""
        rng = np.random.default_rng([seed, index])
        out = np.empty((count, chunk), dtype=np.uint8)
        for i in range(count):
            g = self.group_names[rng.choice(len(self.group_names), p=self.group_p)]
            members = self.groups[g]
            if g == "misc":       # by size, so the tiny files are not repeated hundreds of times
                sizes = np.array([len(d) for _, d in members], dtype=np.float64)
                name, data = members[rng.choice(len(members), p=sizes / sizes.sum())]
            else:
                name, data = members[rng.integers(len(members))]
            if len(data) <= chunk + 600:
                off = 0
            else:
                off = int(rng.integers(0, len(data) - chunk - 600))
                nl = np.flatnonzero(data[off:off + 512] == 10)
                if len(nl):
                    off += int(nl[0]) + 1
            piece = data[off:off + chunk]
            if len(piece) < chunk:
                piece = np.concatenate([piece, np.zeros(chunk - len(piece), dtype=np.uint8)])
            out[i] = piece
        return out


def get_q0(corpus, seed=1204):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "q0.npy")
    if os.path.isfile(path):
        return np.load(path)
    sample = corpus.sample_chunks(seed, 10_000_019, 256, 8192)     # index outside the training range
    q0 = boot_calibration(sample)
    np.save(path, q0)
    return q0


def _save(path, arr):
    tmp = path + f".{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def superbatch(corpus, q0, seed, index, count, chunk, bc=False):
    """Chunks + online features of training super-batch `index`, cached on disk (identical for every model).

    Returns (chunks, s, top), plus the bit-cell evidence (spec 3.3) as a fourth array when `bc` is true."""
    d = os.path.join(CACHE, f"train-s{seed}-{count}x{chunk}")
    os.makedirs(d, exist_ok=True)
    paths = [os.path.join(d, f"sb{index:04d}-{k}.npy") for k in ("x", "s", "top")]
    if all(os.path.isfile(p) for p in paths):
        chunks, s, top = (np.load(p) for p in paths)
    else:
        chunks = corpus.sample_chunks(seed, index, count, chunk)
        pred, conf = successor_features(chunks)
        s, top = evidence(chunks, pred, conf, q0)
        for p, arr in zip(paths, (chunks, s, top)):
            _save(p, arr)
    if not bc:
        return chunks, s, top
    pbc = os.path.join(d, f"sb{index:04d}-{BC_TAG}.npy")
    if os.path.isfile(pbc):
        return chunks, s, top, np.load(pbc)
    bcev = bitcell_evidence(chunks)
    _save(pbc, bcev)
    return chunks, s, top, bcev


# ---------------------------------------------------------------- bit cells (specification 3.3), gate G0
NBC = 4
BC_LEN = (0, 1, 2, 3)
BC_TB = (8, 16, 22, 22)
BC_BOOT = 1 << (Q_BITS - 1)
BC_TAG = "bc" + "-".join(str(t) for t in BC_TB)


def bitcell_index(xb):
    """Cell index of every (byte t, bit j) and bit-cell order m: uint32 [B, N, NBC, 8].

    node = (1 << j) | prefix_j(x[t]) in 1..255; c1, c2, c3 = the three previous bytes (0 before the start).
    order 0: node; order 1: (c1 << 8) | node; order 2: h = c2 * M1 + c1; order 3: h = (c3 * M1 + c2) * M1 + c1;
    for orders 2-3 (all mod 2^32): h = h * M2; idx = ((h >> (40 - tb)) << 8) | node.
    """
    bsz, n = xb.shape
    x = xb.astype(np.uint32)
    pad = np.zeros((bsz, n + 3), dtype=np.uint32)
    pad[:, 3:] = x
    c1, c2, c3 = pad[:, 2:n + 2], pad[:, 1:n + 1], pad[:, 0:n]
    j = np.arange(8, dtype=np.uint32)
    node = (np.uint32(1) << j)[None, None, :] | (x[:, :, None] >> (np.uint32(8) - j)[None, None, :])
    out = np.empty((bsz, n, NBC, 8), dtype=np.uint32)
    m1, m2 = np.uint32(HASH_M1), np.uint32(HASH_M2)
    for m, (ln, tb) in enumerate(zip(BC_LEN, BC_TB)):
        if ln == 0:
            out[:, :, m] = node
        elif ln == 1:
            out[:, :, m] = (c1[:, :, None] << np.uint32(8)) | node
        else:
            h = c2 * m1 + c1 if ln == 2 else (c3 * m1 + c2) * m1 + c1
            h = h * m2
            out[:, :, m] = ((h >> np.uint32(40 - tb)) << np.uint32(8))[:, :, None] | node
    return out


def bitcell_evidence(xb):
    """Exact emulation of the bit cells: int16 [B, N, 8, NBC] = ST[q >> 10] fed to the mixer at every bit.

    Cell update after the bit b is known: q += ((b << 22) - q) >> QSHIFT[count]; count saturates at 255; boot
    (2^21, 0). The visited cells of a chunk are compacted (np.unique), which keeps the hashed-table semantics."""
    bsz, n = xb.shape
    idx = bitcell_index(xb)
    flat = np.empty((bsz, n, NBC, 8), dtype=np.int32)
    tag = (np.arange(NBC, dtype=np.uint32) << np.uint32(24))[None, :, None]
    base = 0
    for b in range(bsz):
        uniq, inv = np.unique((idx[b] | tag).ravel(), return_inverse=True)
        flat[b] = inv.reshape(n, NBC, 8).astype(np.int32) + base
        base += len(uniq)
    del idx
    cells = np.full(base, BC_BOOT << 8, dtype=np.int32)       # packed (q << 8) | count
    out = np.empty((bsz, n, 8, NBC), dtype=np.int16)
    st32 = ST.astype(np.int32)
    qsh32 = QSHIFT.astype(np.int32)
    shifts = np.arange(7, -1, -1, dtype=np.int32)
    tgt_all = ((xb[:, :, None].astype(np.int32) >> shifts) & 1) << Q_BITS      # [B, N, 8]
    for t in range(n):
        ix = flat[:, t]                                         # [B, NBC, 8], all distinct
        cell = cells[ix]
        qs = cell >> 8
        ns = cell & 255
        out[:, t] = st32[qs >> (Q_BITS - 12)].transpose(0, 2, 1)
        qn = qs + ((tgt_all[:, t][:, None, :] - qs) >> qsh32[ns])
        cells[ix] = (qn << 8) | np.minimum(ns + 1, 255)
    return out
