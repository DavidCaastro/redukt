"""Strata ladder trainer: the online context features of `bakeoff/common.py` as torch code, on any device.

Same definitions, same integers (checked by `check_exact.py --test features`): successor-table evidence with its
calibration cells and the bit cells of specification 3.3. Everything is integer arithmetic; hashes are computed
modulo 2^32 inside int64 without ever overflowing it. The two sequential loops of the prototype (calibration cells,
bit cells) are one loop here, over time, vectorised over the streams.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bakeoff"))
from common import (BC_BOOT, BC_LEN, BC_TB, BITLEN, BOS, HASH_M1, HASH_M2, MASK32, NBC, NC, NORD,  # noqa: E402
                    ORDERS, Q_BITS, QSHIFT, ST, TABLE_BITS)

BLOCK = 512                    # time steps prepared at once for the cell loop
UNIQUE_ELEMS = 1 << 24         # bit-cell keys compacted per call of torch.unique


def mul32(a, m):
    """(a * m) mod 2^32 for a non-negative int64 tensor a < 2^33 and a constant m < 2^32, without int64 overflow."""
    return (a * (m & 0xFFFF) + (((a * (m >> 16)) & 0xFFFF) << 16)) & MASK32


def ctx_slot_check(x, length, table_bits):
    """Slot and 8-bit check of the context x[t-length .. t-1], t = length .. N-1 (common.ctx_slot_check)."""
    n = x.shape[1] - length
    if length == 1:
        return x[:, 0:n], torch.zeros_like(x[:, 0:n])
    if length == 2:
        return (x[:, 0:n] << 8) | x[:, 1:n + 1], torch.zeros_like(x[:, 0:n])
    h = torch.zeros_like(x[:, 0:n])
    for k in range(length):
        h = mul32(h + x[:, k:k + n] + 1, HASH_M1)
    h = h ^ (h >> 15)
    h = mul32(h, HASH_M2)
    h = h ^ (h >> 13)
    return h >> (32 - table_bits), h & 0xFF


def successor_features(x):
    """Successor tables of every order (common.successor_features). x: int64 [B, N] bytes.

    Returns pred int16 [B, N, NORD] (-1 = no prediction) and conf int16 [B, N, NORD] (bucket 0..NC-1)."""
    bsz, n = x.shape
    dev = x.device
    pred = torch.full((bsz, n, NORD), -1, dtype=torch.int16, device=dev)
    conf = torch.zeros((bsz, n, NORD), dtype=torch.int16, device=dev)
    tidx = torch.arange(n, device=dev)[None, :]
    bitlen = torch.as_tensor(BITLEN, device=dev)
    for oi, (length, tb) in enumerate(zip(ORDERS, TABLE_BITS)):
        m = n - length
        if m <= 0:
            continue
        slot, check = ctx_slot_check(x, length, tb)
        slot_s, order = torch.sort(slot, dim=1, stable=True)     # by slot, time order kept inside a slot
        check_s = check.gather(1, order)
        succ = x[:, length:].gather(1, order)
        same = torch.zeros((bsz, m), dtype=torch.bool, device=dev)
        same[:, 1:] = (slot_s[:, 1:] == slot_s[:, :-1]) & (check_s[:, 1:] == check_s[:, :-1])
        pred_s = torch.full((bsz, m), -1, dtype=torch.int64, device=dev)
        pred_s[:, 1:] = torch.where(same[:, 1:], succ[:, :-1], -1)
        hit_s = same & (pred_s == succ)
        idx = tidx[:, :m]
        last0 = torch.cummax(torch.where(~hit_s, idx, 0), dim=1).values
        run_after = torch.clamp(idx - last0, max=3)               # hit-run stored in the slot after this visit
        srun_s = torch.zeros((bsz, m), dtype=torch.int64, device=dev)
        srun_s[:, 1:] = torch.where(same[:, 1:], run_after[:, :-1], 0)
        pred_t = torch.empty_like(pred_s).scatter_(1, order, pred_s)
        srun_t = torch.empty_like(srun_s).scatter_(1, order, srun_s)
        hit_full = torch.zeros((bsz, n), dtype=torch.bool, device=dev)
        hit_full[:, length:] = torch.empty_like(order).scatter_(1, order, hit_s.long()) > 0
        last0 = torch.cummax(torch.where(~hit_full, tidx, -1), dim=1).values
        trun = torch.zeros((bsz, n), dtype=torch.int64, device=dev)
        trun[:, 1:] = torch.clamp((tidx - last0)[:, :-1], max=255)
        pred[:, length:, oi] = pred_t.to(torch.int16)
        conf[:, length:, oi] = (8 * srun_t + torch.clamp(bitlen[trun[:, length:]], max=7)).to(torch.int16)
    return pred, conf


def bitcell_index(x):
    """Cell index of every (byte, bit) and bit-cell order (common.bitcell_index): int64 [B, N, NBC, 8]."""
    bsz, n = x.shape
    pad = torch.zeros((bsz, n + 3), dtype=torch.int64, device=x.device)
    pad[:, 3:] = x
    c1, c2, c3 = pad[:, 2:n + 2], pad[:, 1:n + 1], pad[:, 0:n]
    j = torch.arange(8, device=x.device)
    node = (1 << j) | (x[:, :, None] >> (8 - j))
    out = []
    for ln, tb in zip(BC_LEN, BC_TB):
        if ln == 0:
            out.append(node)
        elif ln == 1:
            out.append((c1[:, :, None] << 8) | node)
        else:
            h = mul32(c2, HASH_M1) + c1 if ln == 2 else mul32(mul32(c3, HASH_M1) + c2, HASH_M1) + c1
            h = mul32(h, HASH_M2)
            out.append(((h >> (40 - tb)) << 8)[:, :, None] | node)
    return torch.stack(out, dim=2)


def compact_cells(key):
    """Dense numbering of the bit cells each stream visits: int32 [B, N, NBC, 8] and the number of cells.
    `key` is the output of bitcell_index and is overwritten."""
    bsz, n = key.shape[:2]
    key |= (torch.arange(NBC, device=key.device) << 24)[None, None, :, None]
    flat = torch.empty(key.shape, dtype=torch.int32, device=key.device)
    rows = max(1, UNIQUE_ELEMS // (n * NBC * 8))
    base = 0
    for lo in range(0, bsz, rows):
        part = key[lo:lo + rows] | (torch.arange(lo, min(bsz, lo + rows), device=key.device) << 26)[:, None, None, None]
        uniq, inv = torch.unique(part, return_inverse=True)
        flat[lo:lo + rows] = (inv + base).to(torch.int32)
        base += uniq.numel()
    return flat, base


def cell_loop(cells, idx, tgt, live):
    """q += ((target << 22) - q) >> QSHIFT[count] on the cells `idx` wherever `live`; returns ST[q >> 10] before it.

    cells: int32 packed (q << 8) | count. idx, tgt, live: [B, T, C, 8]; the cells of one time step are distinct."""
    dev = cells.device
    st = torch.as_tensor(ST, dtype=torch.int32, device=dev)
    qsh = torch.as_tensor(QSHIFT, dtype=torch.int32, device=dev)
    nxt = torch.clamp(torch.arange(256, dtype=torch.int32, device=dev) + 1, max=255)
    out = []
    steps = zip(idx.transpose(0, 1).unbind(0), (tgt.to(torch.int32) << Q_BITS).transpose(0, 1).unbind(0),
                live.transpose(0, 1).unbind(0))
    for ix, tg, lv in steps:
        cell = cells[ix]
        qs = cell >> 8
        ns = (cell & 255).long()
        out.append(st[(qs >> (Q_BITS - 12)).long()])
        qn = qs + ((tg - qs) >> qsh[ns])
        cells[ix] = torch.where(lv, (qn << 8) | nxt[ns], cell)
    return torch.stack(out, dim=1)


def features_rows(x, q0):
    """s int16 [B, N, 8, NORD + NBC] and top uint8 [B, N, 8], on the CPU, of streams x (int64 [B, N], any device)."""
    bsz, n = x.shape
    dev = x.device
    pred, conf = successor_features(x)
    flat, n_bc = compact_cells(bitcell_index(x))
    n_ev = bsz * NORD * NC * 8
    boot = (torch.as_tensor(q0, device=dev).reshape(NORD * NC * 8) << 8).to(torch.int32)
    cells = torch.cat([boot.repeat(bsz), torch.full((n_bc,), BC_BOOT << 8, dtype=torch.int32, device=dev)])
    base = (torch.arange(bsz, device=dev)[:, None] * NORD + torch.arange(NORD, device=dev)[None, :]) * NC
    shifts = torch.arange(7, -1, -1, device=dev)
    rank = torch.arange(1, NORD + 1, device=dev)[None, None, :, None]
    bit8 = torch.arange(8, device=dev)
    s_out, top_out = [], []
    for lo in range(0, n, BLOCK):
        y = x[:, lo:lo + BLOCK]
        p = pred[:, lo:lo + BLOCK].long()
        has = p >= 0
        pz = torch.where(has, p, 0)[:, :, :, None]
        ybit = ((y[:, :, None] >> shifts) & 1)[:, :, None, :]                       # [B, T, 1, 8]
        ebit = (pz >> shifts) & 1                                                 # [B, T, NORD, 8]
        live = has[:, :, :, None] & ((pz >> (shifts + 1)) == (y[:, :, None] >> (shifts + 1))[:, :, None, :])
        top_out.append(torch.where(live, rank, 0).amax(dim=2).to(torch.uint8).cpu())
        ev_idx = ((base[:, None, :] + conf[:, lo:lo + BLOCK].long()) * 8)[:, :, :, None] + bit8
        idx = torch.cat([ev_idx, flat[:, lo:lo + BLOCK] + n_ev], dim=2)
        tgt = torch.cat([ebit == ybit, ybit.expand(-1, -1, NBC, -1) == 1], dim=2)
        ones = torch.ones_like(live[:, :, :NBC])
        sign = torch.cat([torch.where(live, 2 * ebit - 1, 0), ones.long()], dim=2)
        ev = cell_loop(cells, idx, tgt, torch.cat([live, ones], dim=2)) * sign
        s_out.append(ev.transpose(2, 3).to(torch.int16).cpu())
    return torch.cat(s_out, dim=1), torch.cat(top_out, dim=1)


def features(xb, q0, device, max_bytes=8 << 20):
    """Features of uint8 chunks xb [B, N] (numpy or tensor), computed on `device` in groups of streams of at most
    `max_bytes` bytes, returned on the CPU: s int16 [B, N, 8, NORD + NBC], top uint8 [B, N, 8]."""
    xb = torch.as_tensor(xb)
    rows = max(1, max_bytes // xb.shape[1])
    s_parts, top_parts = [], []
    for lo in range(0, xb.shape[0], rows):
        s, top = features_rows(xb[lo:lo + rows].to(device).long(), q0)
        s_parts.append(s)
        top_parts.append(top)
    return torch.cat(s_parts), torch.cat(top_parts)


def taps(x):
    """Input byte indices of the four delay taps: taps[b, t, k] = x[t - 1 - k], BOS before the start (int64)."""
    bsz, n = x.shape
    pad = torch.full((bsz, n + 4), BOS, dtype=torch.int64, device=x.device)
    pad[:, 4:] = x
    return torch.stack([pad[:, 3 - k:3 - k + n] for k in range(4)], dim=2)
