"""Strata bake-off: candidate trunks (original), control trunks (published cells) and the shared coder-domain head.

All "quant=True" paths compute exact integers (int32 / int64 numpy in the sequential loops, integer-valued float32 /
float64 tensors elsewhere: every intermediate is below 2^24 in float32 and below 2^53 in float64), with
straight-through gradients. The model that is evaluated is therefore the integer model; `ref_int.py` re-implements
it in pure integer numpy, sequentially, and must match bit for bit.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

from common import (F_BITS, MIX_SH, N_IN, NBC, NORD, NSET, SQ, ST_MAX, W_MAX, W_ONE)

LN2 = math.log(2.0)
ACT = 127.0
F_ONE = float(1 << F_BITS)

# budget caps of the bake-off (multiply-accumulates per input byte, parameters)
CAPS = {"cm": (1_000, 4_000), "A": (32_000, 96_000), "B": (128_000, 256_000)}

CONFIGS = {
    # original candidates (integer, nested tiers A and B trained together)
    "strata_f": dict(kind="strata", widths=(96, 224), layers=3, geared=False, quant=True),
    "strata_g": dict(kind="strata", widths=(88, 208), layers=3, geared=True, quant=True),
    # same candidates without the integer simulation (continuous arithmetic), to measure the cost of integers
    "strata_f_float": dict(kind="strata", widths=(96, 224), layers=3, geared=False, quant=False),
    "strata_g_float": dict(kind="strata", widths=(88, 208), layers=3, geared=True, quant=False),
    # published controls inside the same coder-domain head (float = their best case), one model per budget
    "lstm": dict(kind="lstm", emb=64, hidden=147),
    "gru": dict(kind="gru", emb=64, hidden=174),
    "lstm_a": dict(kind="lstm", emb=32, hidden=70),
    "gru_a": dict(kind="gru", emb=32, hidden=82),
    # published control in its textbook form: LSTM + 256-way softmax, static (no online component)
    "lstm_sm": dict(kind="lstm_sm", emb=64, hidden=125),
    # gate G0: the same harness with the bit cells of spec 3.3; the candidate gets a trained width-0 tier
    "strata_g_bc": dict(kind="strata", widths=(0, 88, 208), layers=3, geared=True, quant=True, bc=True),
    "lstm_bc": dict(kind="lstm", emb=64, hidden=147, bc=True),
    "gru_bc": dict(kind="gru", emb=64, hidden=174, bc=True),
    "lstm_a_bc": dict(kind="lstm", emb=32, hidden=70, bc=True),
    "gru_a_bc": dict(kind="gru", emb=32, hidden=82, bc=True),
}
BUDGET_OF = {"strata_f": ("A", "B"), "strata_g": ("A", "B"), "strata_f_float": ("A", "B"),
             "strata_g_float": ("A", "B"), "lstm": ("B",), "gru": ("B",), "lstm_a": ("A",), "gru_a": ("A",),
             "lstm_sm": ("B",), "strata_g_bc": ("cm", "A", "B"), "lstm_bc": ("B",), "gru_bc": ("B",),
             "lstm_a_bc": ("A",), "gru_a_bc": ("A",)}

TAP_PATTERN = (0, 1, 2, 3, 0, 0, 1, 0)       # which previous byte feeds an input channel (channel index mod 8)
KAPPA_IN = (0, 0, 0, 0, 1, 2, 3, 5)          # trace shift of the input bank
KAPPA_FIXED = (0, 0, 1, 2, 3, 4, 6, 8)       # trace shift of a layer channel (fixed variant)
KAPPA_FAST = (0, 0, 0, 0, 1, 1, 2, 2)        # geared variant: shift when the gate is open
KAPPA_SLOW = (2, 3, 4, 5, 6, 7, 8, 10)       # geared variant: shift when the gate is closed
GATE_GROUP = 8


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def ste_floor(x):
    return x + (torch.floor(x) - x).detach()


def _tmajor(x):
    """[B, T, ...] tensor -> contiguous time-major numpy array [T, B, ...] (the loops index the first axis)."""
    return np.ascontiguousarray(np.swapaxes(x.detach().numpy(), 0, 1))


def _bmajor(a, dtype):
    return torch.from_numpy(np.ascontiguousarray(np.swapaxes(a, 0, 1)).astype(dtype, copy=False))


class TraceFn(torch.autograd.Function):
    """S <- S - ((S - (u << F)) >> kappa);  r = S >> F   (arithmetic shifts), per channel, over time.

    Forward is exact (int32 numpy when quant, float64 otherwise); backward is the gradient of the linear filter
    r_t = (1 - a) r_{t-1} + a u_t with a = 2^-kappa.
    """

    @staticmethod
    def forward(ctx, u, s_init, kappa, quant):
        kap = kappa.numpy()
        slow = np.flatnonzero(kap > 0)
        r = u.clone()
        s_fin = (u[:, -1] * F_ONE).clone()
        if len(slow):
            un = np.ascontiguousarray(np.swapaxes(u.detach().numpy()[:, :, slow], 0, 1))
            if quant:
                # float32 holds these integers exactly (|S| < 2^20, |S - U| < 2^21): floor(x * 2^-k) == x >> k
                us = un * np.float32(F_ONE)
                ss = s_init.detach().numpy()[:, slow].astype(np.float32)
                inv = np.ascontiguousarray(np.broadcast_to((0.5 ** kap[slow]).astype(np.float32), ss.shape))
                out = np.empty_like(us)
                tmp = np.empty_like(ss)
                for t in range(us.shape[0]):
                    np.subtract(ss, us[t], out=tmp)
                    np.multiply(tmp, inv, out=tmp)
                    np.floor(tmp, out=tmp)
                    np.subtract(ss, tmp, out=ss)
                    out[t] = ss
                rs = np.floor(out * np.float32(1.0 / F_ONE))
            else:
                us = un.astype(np.float64) * F_ONE
                ss = s_init.detach().numpy()[:, slow].astype(np.float64)
                inv = 0.5 ** kap[slow].astype(np.float64)
                out = np.empty_like(us)
                for t in range(us.shape[0]):
                    ss = ss - (ss - us[t]) * inv
                    out[t] = ss
                rs = out / F_ONE
            idx = torch.from_numpy(slow)
            r[:, :, idx] = _bmajor(rs, np.float32)
            s_fin[:, idx] = torch.from_numpy(ss.astype(np.float32))
        ctx.kap = kap
        return r, s_fin

    @staticmethod
    def backward(ctx, g_r, _g_s):
        kap = ctx.kap
        slow = np.flatnonzero(kap > 0)
        g_u = g_r.clone()
        if len(slow):
            a = (0.5 ** kap[slow]).astype(np.float32)
            gs = np.ascontiguousarray(np.swapaxes(g_r.detach().numpy()[:, :, slow], 0, 1))
            acc = np.zeros_like(gs[0])
            out = np.empty_like(gs)
            keep = 1.0 - a
            for t in range(gs.shape[0] - 1, -1, -1):
                acc = gs[t] + keep * acc
                out[t] = acc
            g_u[:, :, torch.from_numpy(slow)] = _bmajor(out * a, np.float32)
        return g_u, None, None, None


class GearFn(torch.autograd.Function):
    """Same accumulator with a per-step choice between two shifts (gate = 1 -> fast, gate = 0 -> slow)."""

    @staticmethod
    def forward(ctx, u, gate, s_init, k_fast, k_slow, quant):
        kf, ks = k_fast.numpy(), k_slow.numpy()
        un = _tmajor(u)
        open_ = _tmajor(gate) > 0.5
        if quant:
            us = un * np.float32(F_ONE)
            ss = s_init.detach().numpy().astype(np.float32)
            inv = np.where(open_, (0.5 ** kf).astype(np.float32), (0.5 ** ks).astype(np.float32))
            out = np.empty_like(us)
            tmp = np.empty_like(ss)
            for t in range(us.shape[0]):
                np.subtract(ss, us[t], out=tmp)
                np.multiply(tmp, inv[t], out=tmp)
                np.floor(tmp, out=tmp)
                np.subtract(ss, tmp, out=ss)
                out[t] = ss
            rs = np.floor(out * np.float32(1.0 / F_ONE))
        else:
            us = un.astype(np.float64) * F_ONE
            ss = s_init.detach().numpy().astype(np.float64)
            inv = np.where(open_, 0.5 ** kf.astype(np.float64), 0.5 ** ks.astype(np.float64))
            out = np.empty_like(us)
            for t in range(us.shape[0]):
                ss = ss - (ss - us[t]) * inv[t]
                out[t] = ss
            rs = out / F_ONE
        r = _bmajor(rs, np.float32)
        ctx.save_for_backward(u, r, gate, s_init)
        ctx.shifts = (kf, ks)
        return r, torch.from_numpy(ss.astype(np.float32))

    @staticmethod
    def backward(ctx, g_r, _g_s):
        u, r, gate, s_init = ctx.saved_tensors
        kf, ks = ctx.shifts
        a_f = (0.5 ** kf).astype(np.float32)
        a_s = (0.5 ** ks).astype(np.float32)
        gn, un, rn = _tmajor(g_r), _tmajor(u), _tmajor(r)
        alpha = np.where(_tmajor(gate) > 0.5, a_f, a_s)
        keep = 1.0 - alpha
        g_acc = np.empty_like(gn)
        acc = np.zeros_like(gn[0])
        for t in range(gn.shape[0] - 1, -1, -1):
            acc = gn[t] + acc
            g_acc[t] = acc
            acc = keep[t] * acc
        prev = np.concatenate([(s_init.numpy() / F_ONE)[None].astype(np.float32), rn[:-1]], axis=0)
        g_u = _bmajor(alpha * g_acc, np.float32)
        g_gate = _bmajor(g_acc * (a_f - a_s) * (un - prev), np.float32)
        return g_u, g_gate, None, None, None, None


def pattern(values, width):
    return torch.tensor([values[c % 8] for c in range(width)], dtype=torch.long)


class StrataTrunk(nn.Module):
    """Original candidate: stratified integer trace network (see docs/api/BACKBONE-STRATA-SPEC.md)."""

    def __init__(self, widths, layers, geared, quant):
        super().__init__()
        self.widths, self.layers, self.geared, self.quant = tuple(widths), layers, geared, quant
        d = widths[-1]
        self.d = d
        stratum = torch.zeros(d, dtype=torch.long)
        for k in range(1, len(widths)):
            stratum[widths[k - 1]:widths[k]] = k
        tri = (stratum[:, None] >= stratum[None, :]).float()          # row (out) reads col (in) of a lower stratum
        self.register_buffer("tri", tri)
        self.register_buffer("tapmask", torch.stack([(pattern(TAP_PATTERN, d) == k).float() for k in range(4)]))
        self.register_buffer("kappa_in", pattern(KAPPA_IN, d))
        self.register_buffer("kappa", pattern(KAPPA_FIXED, d))
        self.register_buffer("k_fast", pattern(KAPPA_FAST, d))
        self.register_buffer("k_slow", pattern(KAPPA_SLOW, d))
        fan = tri.sum(1)
        self.emb = nn.Parameter(torch.randn(257, d))
        self.w = nn.ParameterList([nn.Parameter(torch.randn(d, d) / fan.sqrt()[:, None]) for _ in range(layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(layers)])
        if geared:
            groups = d // GATE_GROUP
            gtri = tri[::GATE_GROUP]                                    # a gate reads what its channels read
            self.register_buffer("gtri", gtri)
            self.wg = nn.ParameterList([nn.Parameter(torch.randn(groups, d) / gtri.sum(1).sqrt()[:, None])
                                        for _ in range(layers)])
            self.bg = nn.ParameterList([nn.Parameter(torch.zeros(groups)) for _ in range(layers)])

    def qmat(self, w, mask):
        if self.quant:
            return torch.clamp(ste_round(w * 128.0), -ACT, ACT) * mask
        return torch.clamp(w * 128.0, -ACT, ACT) * mask

    def qemb(self):
        e = self.emb * 32.0
        return torch.clamp(ste_round(e) if self.quant else e, -ACT, ACT)

    def init_state(self, bsz):
        return [torch.zeros(bsz, self.d) for _ in range(self.layers + 1)]

    def forward(self, xt, state):
        eq = self.qemb().view(257, self.d // 8, 8)
        # channel c reads the byte of tap TAP_PATTERN[c % 8]: eight narrow gathers instead of four full ones
        v0 = torch.stack([eq[xt[:, :, TAP_PATTERN[p]], :, p] for p in range(8)], dim=3).flatten(2)
        h, s0 = TraceFn.apply(v0, state[0], self.kappa_in, self.quant)
        new_state = [s0]
        for li in range(self.layers):
            wq = self.qmat(self.w[li], self.tri)
            bq = self.b[li] * 4096.0
            a = h @ wq.t() + (ste_round(bq) if self.quant else bq)
            u = torch.clamp(ste_floor((a + 64.0) / 128.0) if self.quant else a / 128.0, -ACT, ACT)
            if self.geared:
                wgq = self.qmat(self.wg[li], self.gtri)
                bgq = self.bg[li] * 4096.0
                ag = h @ wgq.t() + (ste_round(bgq) if self.quant else bgq)
                soft = torch.clamp(ag / 8192.0 + 0.5, 0.0, 1.0)
                hard = (ag >= 0).to(ag.dtype)
                gate = (soft + (hard - soft).detach()).repeat_interleave(GATE_GROUP, dim=2)
                tr, s = GearFn.apply(u, gate, state[li + 1], self.k_fast, self.k_slow, self.quant)
            else:
                tr, s = TraceFn.apply(u, state[li + 1], self.kappa, self.quant)
            h = torch.clamp(h + tr, -ACT, ACT)
            new_state.append(s)
        return h, new_state

    def macs(self, width):
        m = int(self.tri[:width, :width].sum().item()) * self.layers
        if self.geared:
            m += int(self.gtri[:width // GATE_GROUP, :width].sum().item()) * self.layers
        return m

    def params(self, width):
        p = 257 * width + self.macs(width) + self.layers * width
        if self.geared:
            p += self.layers * (width // GATE_GROUP)
        return p


class RnnTrunk(nn.Module):
    """Published control: one-layer LSTM or GRU over a byte embedding (PyTorch reference implementation)."""

    def __init__(self, kind, emb, hidden):
        super().__init__()
        self.kind, self.e, self.d = kind, emb, hidden
        self.emb = nn.Embedding(257, emb)
        self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(emb, hidden, batch_first=True)

    def init_state(self, bsz):
        return None

    def forward(self, xt, state):
        out, st = self.rnn(self.emb(xt[:, :, 0]), state)
        return out * 32.0, st

    def macs(self, width):
        gates = 4 if self.kind == "lstm" else 3
        return gates * self.d * (self.e + self.d)

    def params(self, width):
        gates = 4 if self.kind == "lstm" else 3
        return 257 * self.e + gates * self.d * (self.e + self.d) + 2 * gates * self.d


class Head(nn.Module):
    """Coder-domain head shared by every trunk: 8 binary decisions per byte in the stretch domain, mixed with the
    evidence of the online successor tables by trunk-driven gains and an online mixer with boot weights."""

    def __init__(self, widths, quant, bc=False):
        super().__init__()
        self.widths, self.quant, self.bc = tuple(widths), quant, bc
        self.n_in = N_IN + (NBC if bc else 0)
        d, k = widths[-1], len(widths)
        self.hw = nn.Parameter(torch.randn(255, d) * (0.5 / math.sqrt(d)))
        self.hb = nn.Parameter(torch.zeros(k, 255))
        self.gw = nn.Parameter(torch.randn(NORD, d) * (0.1 / math.sqrt(d)))
        self.gb = nn.Parameter(torch.ones(k, NORD))
        w0 = torch.full((k, NSET, self.n_in), 0.25)
        w0[:, :, 0] = 1.0
        self.w0 = nn.Parameter(w0)
        self.register_buffer("sq", torch.from_numpy(SQ))

    def init_state(self, bsz):
        dt = torch.int64 if self.quant else torch.float64
        return [torch.zeros(bsz, NSET, self.n_in, dtype=dt) for _ in self.widths]

    def q(self, x, scale, lim=None):
        x = x * scale
        if self.quant:
            x = ste_round(x)
        return x if lim is None else torch.clamp(x, -lim, lim)

    def mix_loop(self, inp, setidx, ybit, w0q, delta):
        """Online mixer, sequential over time (numpy; int64 exact when quant). Returns the weight deltas that were
        in force at every step [B, T, 8, N_IN] and the final per-stream deltas [B, NSET, N_IN]."""
        xs = _tmajor(inp)
        si_all = _tmajor(setidx)
        yb = _tmajor(ybit)
        steps, bsz = xs.shape[:2]
        bar = np.arange(bsz)[:, None]
        used = np.empty((steps, bsz, 8, self.n_in), dtype=np.float64)
        dl = delta.detach().numpy().copy()
        w0q = w0q.detach()
        if self.quant:
            xs = xs.astype(np.int64)
            w0 = w0q.numpy().astype(np.int64)
            half = 1 << (MIX_SH - 1)
            for t in range(steps):
                si = si_all[t]
                d = dl[bar, si]
                w = w0[si] + d
                x = xs[t]
                st = np.clip((w * x).sum(-1) >> 16, -ST_MAX, ST_MAX)
                err = (yb[t] << 16) - SQ[st + ST_MAX]
                wn = np.clip(w + ((x * err[..., None] + half) >> MIX_SH), -W_MAX, W_MAX)
                dl[bar, si] = wn - w0[si]
                used[t] = d
        else:
            xs = xs.astype(np.float64)
            w0 = w0q.numpy().astype(np.float64)
            ybf = yb.astype(np.float64)
            scale = 65536.0 / float(1 << MIX_SH)
            for t in range(steps):
                si = si_all[t]
                d = dl[bar, si]
                w = w0[si] + d
                x = xs[t]
                st = np.clip((w * x).sum(-1) / W_ONE, -ST_MAX, ST_MAX)
                err = ybf[t] - 1.0 / (1.0 + np.exp(-st / 256.0))
                wn = np.clip(w + x * (err[..., None] * scale), -W_MAX, W_MAX)
                dl[bar, si] = wn - w0[si]
                used[t] = d
        return _bmajor(used, np.float64), torch.from_numpy(dl)

    def tier(self, k, h, s, top, y, delta, online=True):
        """Cost in bits per byte position [B, T] of tier k, the new mixer state and P(bit = 1) (integer mode)."""
        dk = self.widths[k]
        hk = h[:, :, :dk]
        shifts = torch.arange(7, -1, -1)
        ybit = (y[:, :, None] >> shifts) & 1                                   # [B, T, 8]
        node = ((1 << torch.arange(8)) | (y[:, :, None] >> (shifts + 1))) - 1    # [B, T, 8] in 0..254
        z = hk @ self.q(self.hw[:, :dk], 64.0, ACT).t()
        z = torch.gather(z, 2, node) + self.q(self.hb[k], 2048.0)[node]
        st_net = torch.clamp(ste_floor((z + 4.0) / 8.0) if self.quant else z / 8.0, -ST_MAX, ST_MAX)
        g = hk @ self.q(self.gw[:, :dk], 64.0, ACT).t() + self.q(self.gb[k], 2048.0)
        gain = torch.clamp(ste_floor((g + 32.0) / 64.0) if self.quant else g / 64.0, 0.0, ACT)
        sp = s[..., :NORD] * gain[:, :, None, :] / 32.0
        sp = torch.clamp(ste_floor(sp) if self.quant else sp, -ST_MAX, ST_MAX)
        parts = [st_net[..., None], sp]
        if self.bc:                                   # bit-cell stretch, always live, no gain (spec 3.3)
            parts.append(s[..., NORD:NORD + NBC])
        if not online:
            parts = [parts[0]] + [torch.zeros_like(q) for q in parts[1:]]
            top = torch.zeros_like(top)
        inp = torch.cat(parts, dim=-1)                                           # [B, T, 8, n_in]
        setidx = torch.arange(8)[None, None, :] * (NORD + 1) + top
        w0q = self.q(self.w0[k], float(W_ONE), W_MAX)
        if online:
            with torch.no_grad():
                used, delta = self.mix_loop(inp, setidx, ybit, w0q, delta)
        else:
            used = torch.zeros(inp.shape, dtype=torch.float64)
        w = w0q[setidx].double() + used
        dot = (w * inp.double()).sum(-1) / W_ONE
        st = torch.clamp(ste_floor(dot) if self.quant else dot, -ST_MAX, ST_MAX)
        sign = (1 - 2 * ybit).double()
        bce = fn.softplus(sign * st / 256.0) / LN2
        p1 = None
        if self.quant:
            p1 = self.sq[st.detach().long() + ST_MAX]
            pb = torch.where(ybit.bool(), p1, 65536 - p1).double()
            exact = -torch.log2(pb / 65536.0)
            bce = bce + (exact - bce).detach()
        return bce.sum(-1), delta, p1

    def macs(self, width):
        return 8 * width + NORD * width + 8 * 2 * self.n_in

    def params(self, width):
        return 255 * width + 255 + NORD * width + NORD + NSET * self.n_in


class CodecModel(nn.Module):
    """Trunk + coder-domain head; `tiers` are width prefixes of the same trunk (one tier for the controls)."""

    def __init__(self, name):
        super().__init__()
        cfg = CONFIGS[name]
        self.name, self.cfg = name, cfg
        if cfg["kind"] == "strata":
            self.trunk = StrataTrunk(cfg["widths"], cfg["layers"], cfg["geared"], cfg["quant"])
            self.widths, quant = tuple(cfg["widths"]), cfg["quant"]
        else:
            self.trunk = RnnTrunk(cfg["kind"], cfg["emb"], cfg["hidden"])
            self.widths, quant = (cfg["hidden"],), False
        self.quant = quant
        self.bc = cfg.get("bc", False)
        self.head = Head(self.widths, quant, self.bc)

    def init_state(self, bsz):
        return self.trunk.init_state(bsz), self.head.init_state(bsz)

    def forward(self, xt, s, top, y, state, online=True, want_p=False, trunk_on=True):
        h, tstate = self.trunk(xt, state[0])
        if not trunk_on:              # ablation: the head only sees its biases, the tables and the mixer
            h = torch.zeros_like(h)
        costs, mstates, probs = [], [], []
        for k in range(len(self.widths)):
            c, m, p1 = self.head.tier(k, h, s, top, y, state[1][k], online)
            costs.append(c)
            mstates.append(m)
            probs.append(p1)
        if want_p:
            return costs, (tstate, mstates), probs
        return costs, (tstate, mstates)

    def budget(self):
        rows = []
        for k, width in enumerate(self.widths):
            rows.append((BUDGET_OF[self.name][k], self.trunk.macs(width) + self.head.macs(width),
                         self.trunk.params(width) + self.head.params(width)))
        return rows


class SoftmaxLstm(nn.Module):
    """Textbook control: embedding -> LSTM -> 256-way softmax; static, no online component."""

    def __init__(self, name):
        super().__init__()
        cfg = CONFIGS[name]
        self.name, self.cfg, self.quant, self.widths = name, cfg, False, (cfg["hidden"],)
        self.emb = nn.Embedding(257, cfg["emb"])
        self.rnn = nn.LSTM(cfg["emb"], cfg["hidden"], batch_first=True)
        self.out = nn.Linear(cfg["hidden"], 256)

    def init_state(self, bsz):
        return None

    def forward(self, xt, s, top, y, state, online=True, want_p=False, trunk_on=True):
        out, st = self.rnn(self.emb(xt[:, :, 0]), state)
        logits = self.out(out)
        cost = fn.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").reshape(y.shape) / LN2
        return [cost], st

    def budget(self):
        e, h = self.cfg["emb"], self.cfg["hidden"]
        return [("B", 4 * h * (e + h) + 256 * h, 257 * e + 4 * h * (e + h) + 8 * h + 257 * h)]


def build(name):
    return SoftmaxLstm(name) if CONFIGS[name]["kind"] == "lstm_sm" else CodecModel(name)


def detach_state(state):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    return type(state)(detach_state(s) for s in state)


def check_budget(model):
    ok = True
    for tier, macs, params in model.budget():
        cap_m, cap_p = CAPS[tier]
        ok = ok and macs <= cap_m and params <= cap_p
    return ok


def export_int(model, k):
    """Integer tensors of tier k of a quantised Strata model (what a weight file would carry)."""
    tr, hd = model.trunk, model.head
    dk = model.widths[k]
    out = {"width": dk, "layers": tr.layers, "geared": tr.geared, "bc": hd.bc}
    with torch.no_grad():
        out["emb"] = tr.qemb()[:, :dk].round().long().numpy()
        out["tap"] = np.array([TAP_PATTERN[c % 8] for c in range(dk)], dtype=np.int64)
        out["kappa_in"] = tr.kappa_in[:dk].numpy()
        out["kappa"] = tr.kappa[:dk].numpy()
        out["k_fast"] = tr.k_fast[:dk].numpy()
        out["k_slow"] = tr.k_slow[:dk].numpy()
        out["w"] = [tr.qmat(tr.w[i], tr.tri)[:dk, :dk].round().long().numpy() for i in range(tr.layers)]
        out["b"] = [torch.round(tr.b[i] * 4096.0)[:dk].long().numpy() for i in range(tr.layers)]
        if tr.geared:
            ng = dk // GATE_GROUP
            out["wg"] = [tr.qmat(tr.wg[i], tr.gtri)[:ng, :dk].round().long().numpy() for i in range(tr.layers)]
            out["bg"] = [torch.round(tr.bg[i] * 4096.0)[:ng].long().numpy() for i in range(tr.layers)]
        out["hw"] = hd.q(hd.hw[:, :dk], 64.0, ACT).round().long().numpy()
        out["hb"] = hd.q(hd.hb[k], 2048.0).round().long().numpy()
        out["gw"] = hd.q(hd.gw[:, :dk], 64.0, ACT).round().long().numpy()
        out["gb"] = hd.q(hd.gb[k], 2048.0).round().long().numpy()
        out["w0"] = hd.q(hd.w0[k], float(W_ONE), W_MAX).round().long().numpy()
    return out
