"""Strata bake-off: sequential integer reference of the candidate (numpy int64 only) + real range coding.

    python -I tools/strata/bakeoff/ref_int.py --run strata_f --source sel-webster --bytes 20000 --roundtrip 3000

Checks, on bytes the model never saw:
  1. every P(bit = 1) of the PyTorch integer simulation (vectorised over time, features from the vectorised
     table emulation in common.py) equals the value of this byte-by-byte, bit-by-bit integer implementation
     (independent code path: sequential tables, sequential mixer, int64 arithmetic with shifts);
  2. a real compress -> decompress round trip through a port of the AZTH range coder (binary decisions with
     total 2^16) using only this reference: decoded bytes must equal the input; coded size vs ideal bits.
This is a design-stage check written by the architect. The oracle of test rule T2 is a separate implementation
written from the specification alone.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BC_BOOT, BC_LEN, BC_TB, BITLEN, BOS, CACHE, F_BITS, HASH_M1, HASH_M2, MASK32,  # noqa: E402
                    MIX_SH, NC, NORD, ORDERS, Q_BITS, QSHIFT, RUNS, SQ, ST, ST_MAX, TABLE_BITS, W_MAX, load_source)

TOP = 1 << 24
P_TOTAL = 1 << 16


class RangeEncoder:
    """Port of src/domain/coder.rs RangeEncoder (carry propagation, u64 low, u32 range)."""

    def __init__(self):
        self.low, self.range, self.cache, self.ff, self.first, self.out = 0, 0xFFFFFFFF, 0, 0, True, bytearray()

    def shift_low(self):
        low32 = self.low & 0xFFFFFFFF
        if low32 < 0xFF000000 or (self.low >> 32) != 0:
            carry = (self.low >> 32) & 0xFF
            if not self.first:
                self.out.append((self.cache + carry) & 0xFF)
            self.first = False
            self.out.extend([(0xFF + carry) & 0xFF] * self.ff)
            self.ff = 0
            self.cache = (self.low >> 24) & 0xFF
        else:
            self.ff += 1
        self.low = (self.low & 0x00FFFFFF) << 8

    def encode(self, lo, hi, total):
        r = self.range // total
        self.low += lo * r
        self.range = r * (hi - lo) if hi < total else self.range - r * lo
        while self.range < TOP:
            self.shift_low()
            self.range = (self.range << 8) & 0xFFFFFFFF

    def encode_bit(self, bit, p1):
        """bit 0 owns [0, 65536 - p1), bit 1 owns [65536 - p1, 65536)."""
        split = P_TOTAL - p1
        if bit:
            self.encode(split, P_TOTAL, P_TOTAL)
        else:
            self.encode(0, split, P_TOTAL)

    def finish(self):
        """Legacy termination of the AZTH coder: five flush bytes."""
        for _ in range(5):
            self.shift_low()
        if not self.first:
            self.out.append(self.cache)
            self.out.extend([0xFF] * self.ff)
        return bytes(self.out)

    def finish_tight(self):
        """Proposed Strata termination: round low up to a multiple of 2^24 (always inside the final interval
        because range >= 2^24), emit its top byte only; the decoder reads zeros past the end of the stream."""
        self.low = (self.low + 0x00FFFFFF) & ~0x00FFFFFF
        self.shift_low()
        if not self.first:          # the initial cache is a dummy that is never written (same rule as finish)
            self.out.append(self.cache)
        self.out.extend([0xFF] * self.ff)
        return bytes(self.out)


class RangeDecoder:
    """Port of src/domain/coder.rs RangeDecoder."""

    def __init__(self, data):
        self.low, self.range, self.code, self.data, self.pos = 0, 0xFFFFFFFF, 0, data, 0
        for _ in range(4):
            self.code = ((self.code << 8) | self.next_byte()) & 0xFFFFFFFF

    def next_byte(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def decode_bit(self, p1):
        split = P_TOTAL - p1
        r = self.range // P_TOTAL
        cum = min(((self.code - self.low) & 0xFFFFFFFF) // r, P_TOTAL - 1)
        bit = 1 if cum >= split else 0
        lo, hi = (split, P_TOTAL) if bit else (0, split)
        self.low = (self.low + r * lo) & 0xFFFFFFFF
        self.range = r * (hi - lo) if hi < P_TOTAL else self.range - r * lo
        while self.range < TOP:
            self.code = ((self.code << 8) | self.next_byte()) & 0xFFFFFFFF
            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range = (self.range << 8) & 0xFFFFFFFF
        return bit


def ctx_hash(ctx, length, table_bits):
    """Slot and check of the last `length` bytes (see common.ctx_slot_check for the definition)."""
    if length == 1:
        return ctx[-1], 0
    if length == 2:
        return (ctx[-2] << 8) | ctx[-1], 0
    h = 0
    for b in ctx[-length:]:
        h = ((h + b + 1) * HASH_M1) & MASK32
    h ^= h >> 15
    h = (h * HASH_M2) & MASK32
    h ^= h >> 13
    return h >> (32 - table_bits), h & 0xFF


class StrataRef:
    """One stream of one tier, integer arithmetic only. Usage per byte: begin(); 8 x (prob(j, prefix); update(bit));
    end(byte)."""

    def __init__(self, p, q0):
        self.p = p
        d = p["width"]
        self.hist = [BOS, BOS, BOS, BOS]
        self.s_in = np.zeros(d, dtype=np.int64)
        self.s = [np.zeros(d, dtype=np.int64) for _ in range(p["layers"])]
        self.tables = [np.full(1 << tb, -1, dtype=np.int64) for tb in TABLE_BITS]
        self.trun = [0] * NORD
        self.cq = q0.astype(np.int64).copy()
        self.cn = np.zeros((NORD, NC, 8), dtype=np.int64)
        self.delta = np.zeros_like(p["w0"], dtype=np.int64)      # [NSET, n_in]
        self.bc = bool(p.get("bc", False))
        self.bcells = [np.full(1 << tb, BC_BOOT << 8, dtype=np.int64) for tb in BC_TB] if self.bc else []
        self.ctx = []
        self.n = 0
        self.cols = np.arange(d)

    def begin(self):
        p = self.p
        v0 = p["emb"][np.array(self.hist)[p["tap"]], self.cols]
        self.s_in = self.s_in - ((self.s_in - (v0 << F_BITS)) >> p["kappa_in"])
        h = self.s_in >> F_BITS
        for li in range(p["layers"]):
            a = p["w"][li] @ h + p["b"][li]
            u = np.clip((a + 64) >> 7, -127, 127)
            if p["geared"]:
                gate = np.repeat((p["wg"][li] @ h + p["bg"][li]) >= 0, 8)
                kap = np.where(gate, p["k_fast"], p["k_slow"])
            else:
                kap = p["kappa"]
            self.s[li] = self.s[li] - ((self.s[li] - (u << F_BITS)) >> kap)
            h = np.clip(h + (self.s[li] >> F_BITS), -127, 127)
        self.h = h
        self.gain = np.clip((p["gw"] @ h + p["gb"] + 32) >> 6, 0, 127)
        self.look = []
        for oi, (length, tb) in enumerate(zip(ORDERS, TABLE_BITS)):
            if self.n < length:
                self.look.append(None)
                continue
            slot, check = ctx_hash(self.ctx, length, tb)
            entry = int(self.tables[oi][slot])
            if entry >= 0 and (entry >> 10) == check:
                pred, run = (entry >> 2) & 0xFF, entry & 3
                conf = 8 * run + min(int(BITLEN[self.trun[oi]]), 7)
            else:
                pred, run, conf = -1, 0, 0
            self.look.append((slot, check, pred, run, conf))

    def prob(self, j, prefix):
        p = self.p
        node = ((1 << j) | prefix) - 1
        st_net = int(np.clip((int(p["hw"][node] @ self.h) + int(p["hb"][node]) + 4) >> 3, -ST_MAX, ST_MAX))
        inp = [st_net] + [0] * NORD
        live, top = [], 0
        for oi, lk in enumerate(self.look):
            if lk is None or lk[2] < 0 or (lk[2] >> (8 - j)) != prefix:
                continue
            e = (lk[2] >> (7 - j)) & 1
            ev = (2 * e - 1) * int(ST[int(self.cq[oi, lk[4], j]) >> (Q_BITS - 12)])
            inp[oi + 1] = max(-ST_MAX, min(ST_MAX, (ev * int(self.gain[oi])) >> 5))
            live.append((oi, lk[4], e))
            top = oi + 1
        bidx = []
        if self.bc:                                   # bit cells, spec 3.3
            c1, c2, c3 = (v if v < 256 else 0 for v in self.hist[:3])
            node1 = (1 << j) | prefix
            for m, (ln, tb) in enumerate(zip(BC_LEN, BC_TB)):
                if ln == 0:
                    idx = node1
                elif ln == 1:
                    idx = (c1 << 8) | node1
                else:
                    h = (c2 * HASH_M1 + c1) & MASK32 if ln == 2 else \
                        ((((c3 * HASH_M1 + c2) & MASK32) * HASH_M1) + c1) & MASK32
                    h = (h * HASH_M2) & MASK32
                    idx = ((h >> (40 - tb)) << 8) | node1
                bidx.append(idx)
                inp.append(int(ST[(int(self.bcells[m][idx]) >> 8) >> (Q_BITS - 12)]))
        sidx = j * (NORD + 1) + top
        w = p["w0"][sidx] + self.delta[sidx]
        x = np.array(inp, dtype=np.int64)
        st = max(-ST_MAX, min(ST_MAX, int((w * x).sum()) >> 16))
        p1 = int(SQ[st + ST_MAX])
        self.pending = (j, sidx, w, x, p1, live, bidx)
        return p1

    def update(self, bit):
        j, sidx, w, x, p1, live, bidx = self.pending
        for m, idx in enumerate(bidx):
            cell = int(self.bcells[m][idx])
            q, cnt = cell >> 8, cell & 255
            q = q + (((bit << Q_BITS) - q) >> int(QSHIFT[cnt]))
            self.bcells[m][idx] = (q << 8) | min(cnt + 1, 255)
        err = (bit << 16) - p1
        wn = np.clip(w + ((x * err + (1 << (MIX_SH - 1))) >> MIX_SH), -W_MAX, W_MAX)
        self.delta[sidx] = wn - self.p["w0"][sidx]
        for oi, conf, e in live:
            q, n = int(self.cq[oi, conf, j]), int(self.cn[oi, conf, j])
            tgt = (1 << Q_BITS) if e == bit else 0
            self.cq[oi, conf, j] = q + ((tgt - q) >> int(QSHIFT[n]))
            self.cn[oi, conf, j] = min(n + 1, 255)

    def end(self, byte):
        for oi, lk in enumerate(self.look):
            if lk is None:
                self.trun[oi] = 0
                continue
            slot, check, pred, run, _ = lk
            hit = pred == byte
            run = min(run + 1, 3) if hit else 0
            self.tables[oi][slot] = (check << 10) | (byte << 2) | run
            self.trun[oi] = min(self.trun[oi] + 1, 255) if hit else 0
        self.hist = [byte] + self.hist[:3]
        self.ctx.append(byte)
        if len(self.ctx) > 64:
            del self.ctx[:32]
        self.n += 1


def code_stream(ref, data, enc=None):
    """Encoder side: returns all P(bit=1) [N, 8] and the ideal bits; feeds `enc` when given."""
    probs = np.zeros((len(data), 8), dtype=np.int64)
    bits = 0.0
    for t, byte in enumerate(data):
        byte = int(byte)
        ref.begin()
        for j in range(8):
            prefix = byte >> (8 - j)
            bit = (byte >> (7 - j)) & 1
            p1 = ref.prob(j, prefix)
            probs[t, j] = p1
            bits -= np.log2((p1 if bit else P_TOTAL - p1) / P_TOTAL)
            if enc is not None:
                enc.encode_bit(bit, p1)
            ref.update(bit)
        ref.end(byte)
    return probs, bits


def decode_stream(ref, stream, n):
    dec = RangeDecoder(stream)
    out = bytearray()
    for _ in range(n):
        ref.begin()
        prefix = 0
        for j in range(8):
            bit = dec.decode_bit(ref.prob(j, prefix))
            ref.update(bit)
            prefix = (prefix << 1) | bit
        ref.end(prefix)
        out.append(prefix)
    return bytes(out)


def main():
    import torch
    from common import evidence, successor_features
    from models import build, export_int
    from train import taps

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", default="sel-webster")
    ap.add_argument("--bytes", type=int, default=20000)
    ap.add_argument("--roundtrip", type=int, default=3000)
    args = ap.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(os.path.join(RUNS, args.run, "ckpt.pt"), weights_only=False)
    model = build(ck["args"]["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    if not model.quant:
        raise SystemExit("the integer reference only applies to quantised Strata models")
    q0 = np.load(os.path.join(CACHE, "q0.npy"))
    data = load_source(args.source)[:args.bytes]
    xb = data[None, :]
    pred, conf = successor_features(xb)
    s, top = evidence(xb, pred, conf, q0)
    if getattr(model, "bc", False):
        from common import bitcell_evidence
        s = np.concatenate([s, bitcell_evidence(xb)], axis=-1)
    with torch.no_grad():
        costs, _, probs = model(torch.from_numpy(taps(xb).astype(np.int64)), torch.from_numpy(s.astype(np.float32)),
                                torch.from_numpy(top.astype(np.int64)), torch.from_numpy(xb.astype(np.int64)),
                                model.init_state(1), want_p=True)
    ok = True
    for k in range(len(model.widths)):
        params = export_int(model, k)
        t0 = time.time()
        ref_p, ref_bits = code_stream(StrataRef(params, q0), data)
        tor_p = probs[k][0].numpy()
        diff = int((ref_p != tor_p).sum())
        ok = ok and diff == 0
        print(f"tier {k} (width {params['width']}): {data.size * 8} probabilities, mismatches torch-sim vs "
              f"integer reference = {diff}; bits torch {float(costs[k].sum()):.3f} / reference {ref_bits:.3f} "
              f"({ref_bits / data.size:.4f} bpb) [{time.time() - t0:.0f} s]", flush=True)
        n = min(args.roundtrip, data.size)
        enc_l, enc_t = RangeEncoder(), RangeEncoder()
        _, ideal = code_stream(StrataRef(params, q0), data[:n], enc_l)
        code_stream(StrataRef(params, q0), data[:n], enc_t)
        legacy, tight = enc_l.finish(), enc_t.finish_tight()
        back_t = decode_stream(StrataRef(params, q0), tight, n)
        back_l = decode_stream(StrataRef(params, q0), legacy, n)
        rt = back_t == bytes(data[:n]) and back_l == bytes(data[:n])
        ok = ok and rt
        print(f"tier {k}: round trip of {n} bytes {'OK' if rt else 'FAILED'}; ideal {ideal / 8:.2f} B, coded "
              f"{len(tight)} B with the tight flush, {len(legacy)} B with the legacy flush", flush=True)
    print("ALL CHECKS PASSED" if ok else "CHECK FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
