"""Strata ladder trainer: the geared integer trunk and the coder-domain head of `bakeoff/models.py`, with the three
sequential parts (input accumulators, geared accumulators, online mixers) running on the model's device.

The forward pass is the integer model: accumulators are int32 with arithmetic shifts, the mixers of all tiers are
one int32 loop over time (products summed in int64), everything else is integer-valued float32 below 2^24
(`exact_bound` checks the trained weights against that limit). Parameter names and shapes are those of the
prototype, so a prototype checkpoint loads here and `check_exact.py` compares the two bit for bit.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as fn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bakeoff"))
from common import F_BITS, MIX_SH, N_IN, NBC, NORD, NSET, SQ, ST_MAX, W_MAX, W_ONE  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False     # TF32 would break the exact integer matrix products
LN2 = math.log(2.0)
ACT = 127.0
F_ONE = float(1 << F_BITS)
LAYERS = 3
GATE_GROUP = 8
N_MIX = N_IN + NBC                           # mixer inputs: trunk + successor orders + bit cells
TAP_PATTERN = (0, 1, 2, 3, 0, 0, 1, 0)       # which previous byte feeds an input channel (channel index mod 8)
KAPPA_IN = (0, 0, 0, 0, 1, 2, 3, 5)          # shift of the input accumulators
KAPPA_FAST = (0, 0, 0, 0, 1, 1, 2, 2)        # shift of a layer channel when its gate is open
KAPPA_SLOW = (2, 3, 4, 5, 6, 7, 8, 10)       # shift when the gate is closed

# nested tiers (width prefixes of one trunk) and the cap on multiply-accumulates per byte of each tier:
# gate V7 rungs R0 = 208, R1 <= 32,000, R2 <= 128,000, R3 <= 512,000, R4 <= 2,000,000 (amendment 4)
RUNGS = {
    "g0": dict(widths=(0, 88, 208), caps=(208, 32_000, 128_000)),             # the G0 model (strata_g_bc)
    "ladder3": dict(widths=(88, 208, 456), caps=(32_000, 128_000, 512_000)),
    "ladder4": dict(widths=(88, 208, 456, 928), caps=(32_000, 128_000, 512_000, 2_000_000)),
    "r4": dict(widths=(760,), caps=(2_000_000,)),
    "tiny": dict(widths=(16, 40), caps=(32_000, 32_000)),                     # CPU end-to-end proof only
}


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def ste_floor(x):
    return x + (torch.floor(x) - x).detach()


def pattern(values, width):
    return torch.tensor([values[c % 8] for c in range(width)], dtype=torch.int32)


def accumulate(u, s, shifts):
    """S <- S - ((S - (u << F)) >> shift), r = S >> F, over the time axis of u [B, T, d] (integer-valued float).

    s: int32 [B, d]; shifts: one int32 tensor per time step ([d] or [B, d]). Returns r (float, [B, T, d]) and S."""
    out = []
    for ut, kt in zip((u.to(torch.int32) << F_BITS).transpose(0, 1).unbind(0), shifts):
        s = s - ((s - ut) >> kt)
        out.append(s)
    return (torch.stack(out, dim=1) >> F_BITS).to(u.dtype), s


def filter_back(g, keeps):
    """Adjoint of r_t = keep_t r_{t-1} + (1 - keep_t) u_t:  A_t = g_t + keep_{t+1} A_{t+1}, over [B, T, d]."""
    gl = g.transpose(0, 1).unbind(0)
    acc = gl[-1]
    out = [acc]
    for t in range(len(gl) - 2, -1, -1):
        acc = torch.addcmul(gl[t], keeps[t + 1], acc)
        out.append(acc)
    out.reverse()
    return torch.stack(out, dim=1)


class TraceFn(torch.autograd.Function):
    """Accumulators with one fixed shift per channel; backward = gradient of the linear filter with a = 2^-shift."""

    @staticmethod
    def forward(ctx, u, s_init, kappa):
        r, s = accumulate(u, s_init, [kappa] * u.shape[1])
        ctx.alpha = 0.5 ** kappa.to(u.dtype)
        ctx.mark_non_differentiable(s)
        return r, s

    @staticmethod
    def backward(ctx, g_r, _g_s):
        return filter_back(g_r, [1.0 - ctx.alpha] * g_r.shape[1]) * ctx.alpha, None, None


class GearFn(torch.autograd.Function):
    """Accumulators with a per-step choice between two shifts; one gate per group of GATE_GROUP channels
    (gate [B, T, d / GATE_GROUP]: above 0.5 = open = fast)."""

    @staticmethod
    def forward(ctx, u, gate, s_init, k_fast, k_slow):
        open_ = (gate > 0.5).repeat_interleave(GATE_GROUP, dim=2)
        r, s = accumulate(u, s_init, torch.where(open_, k_fast, k_slow).transpose(0, 1).unbind(0))
        ctx.save_for_backward(u, r, gate, s_init, k_fast, k_slow)
        ctx.mark_non_differentiable(s)
        return r, s

    @staticmethod
    def backward(ctx, g_r, _g_s):
        u, r, gate, s_init, k_fast, k_slow = ctx.saved_tensors
        a_f, a_s = 0.5 ** k_fast.to(u.dtype), 0.5 ** k_slow.to(u.dtype)
        alpha = torch.where((gate > 0.5).repeat_interleave(GATE_GROUP, dim=2), a_f, a_s)
        g_acc = filter_back(g_r, (1.0 - alpha).transpose(0, 1).unbind(0))
        prev = torch.cat([(s_init.to(u.dtype) / F_ONE)[:, None], r[:, :-1]], dim=1)
        g_gate = (g_acc * (a_f - a_s) * (u - prev)).unflatten(2, (-1, GATE_GROUP)).sum(-1)
        return alpha * g_acc, g_gate, None, None, None


class Trunk(nn.Module):
    """Stratified geared integer trace network (docs/api/BACKBONE-STRATA-SPEC.md 2.3-2.4)."""

    def __init__(self, widths):
        super().__init__()
        self.widths = tuple(widths)
        d = self.d = widths[-1]
        stratum = torch.zeros(d, dtype=torch.long)
        for k in range(1, len(widths)):
            stratum[widths[k - 1]:widths[k]] = k
        tri = (stratum[:, None] >= stratum[None, :]).float()          # row (out) reads col (in) of a lower stratum
        self.register_buffer("tri", tri)
        self.register_buffer("gtri", tri[::GATE_GROUP].clone())       # a gate reads what its channels read
        self.register_buffer("kappa_in", pattern(KAPPA_IN, d))
        self.register_buffer("k_fast", pattern(KAPPA_FAST, d))
        self.register_buffer("k_slow", pattern(KAPPA_SLOW, d))
        self.emb = nn.Parameter(torch.randn(257, d))
        self.w = nn.ParameterList([nn.Parameter(torch.randn(d, d) / tri.sum(1).sqrt()[:, None])
                                   for _ in range(LAYERS)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(LAYERS)])
        self.wg = nn.ParameterList([nn.Parameter(torch.randn(d // GATE_GROUP, d) / self.gtri.sum(1).sqrt()[:, None])
                                    for _ in range(LAYERS)])
        self.bg = nn.ParameterList([nn.Parameter(torch.zeros(d // GATE_GROUP)) for _ in range(LAYERS)])

    @staticmethod
    def qmat(w, mask):
        return torch.clamp(ste_round(w * 128.0), -ACT, ACT) * mask

    def qemb(self):
        return torch.clamp(ste_round(self.emb * 32.0), -ACT, ACT)

    def init_state(self, bsz):
        return [torch.zeros(bsz, self.d, dtype=torch.int32, device=self.emb.device) for _ in range(LAYERS + 1)]

    def forward(self, xt, state):
        eq = self.qemb().view(257, self.d // 8, 8)
        # channel c reads the byte of tap TAP_PATTERN[c % 8]: eight narrow gathers instead of four full ones
        v0 = torch.stack([eq[xt[:, :, TAP_PATTERN[p]], :, p] for p in range(8)], dim=3).flatten(2)
        h, s0 = TraceFn.apply(v0, state[0], self.kappa_in)
        new_state = [s0]
        for li in range(LAYERS):
            a = h @ self.qmat(self.w[li], self.tri).t() + ste_round(self.b[li] * 4096.0)
            u = torch.clamp(ste_floor((a + 64.0) / 128.0), -ACT, ACT)
            ag = h @ self.qmat(self.wg[li], self.gtri).t() + ste_round(self.bg[li] * 4096.0)
            soft = torch.clamp(ag / 8192.0 + 0.5, 0.0, 1.0)
            gate = soft + ((ag >= 0).to(ag.dtype) - soft).detach()
            tr, s = GearFn.apply(u, gate, state[li + 1], self.k_fast, self.k_slow)
            h = torch.clamp(h + tr, -ACT, ACT)
            new_state.append(s)
        return h, new_state

    def macs(self, width):
        full = int(self.tri[:width, :width].sum().item()) + int(self.gtri[:width // GATE_GROUP, :width].sum().item())
        return full * LAYERS

    def params(self, width):
        return 257 * width + self.macs(width) + LAYERS * width + LAYERS * (width // GATE_GROUP)


class MixDot(torch.autograd.Function):
    """Value: the exact stretch the online mixer produced (clamped integer, as float64). Gradient: that of
    clamp(floor(sum(w * x) / 2^16)) with straight-through floor, w = boot weights + the online deltas in force."""

    @staticmethod
    def forward(ctx, inp, w0_sel, w_used, raw):
        ctx.save_for_backward(inp, w_used, raw)
        return raw.clamp(-ST_MAX, ST_MAX).double()

    @staticmethod
    def backward(ctx, g):
        inp, w_used, raw = ctx.saved_tensors
        inside = (raw >= -ST_MAX) & (raw <= ST_MAX)
        gw = (g * inside / W_ONE)[..., None]
        return (gw * w_used.double()).to(inp.dtype), (gw * inp.double()).to(inp.dtype), None, None


def mix_loop(inp, setidx, ybit, w0, delta, sq):
    """Online mixers of all tiers, sequential over time, exact integers (int32; the dot is summed in int64).

    inp [K, B, T, 8, n] integer-valued; setidx, ybit [B, T, 8]; w0 int32 [K, NSET, n]; delta int32 [K, B, NSET, n]
    (weights minus boot weights). Returns the weights in force at every step (int32 [K, B, T, 8, n]), the unclamped
    stretch (int64 [K, B, T, 8]) and the new deltas."""
    bar = torch.arange(inp.shape[1], device=inp.device)[:, None]
    weights = w0[:, None] + delta
    half = 1 << (MIX_SH - 1)
    used, raws = [], []
    steps = zip(inp.to(torch.int32).permute(2, 0, 1, 3, 4).contiguous().unbind(0), setidx.transpose(0, 1).unbind(0),
                (ybit.to(torch.int32) << 16).transpose(0, 1).unbind(0))
    for x, si, y16 in steps:
        w = weights[:, bar, si]
        raw = (w * x).sum(-1, dtype=torch.int64) >> 16
        err = y16 - sq[raw.clamp(-ST_MAX, ST_MAX) + ST_MAX]
        weights[:, bar, si] = (w + ((x * err[..., None] + half) >> MIX_SH)).clamp_(-W_MAX, W_MAX)
        used.append(w)
        raws.append(raw)
    return torch.stack(used, dim=2), torch.stack(raws, dim=2), weights - w0[:, None]


class Head(nn.Module):
    """Coder-domain head: 8 binary decisions per byte in the stretch domain, mixed with the evidence of the online
    successor tables (trunk-driven gains) and of the bit cells by an online mixer with trained boot weights."""

    def __init__(self, widths):
        super().__init__()
        self.widths = tuple(widths)
        d, k = widths[-1], len(widths)
        self.hw = nn.Parameter(torch.randn(255, d) * (0.5 / math.sqrt(d)))
        self.hb = nn.Parameter(torch.zeros(k, 255))
        self.gw = nn.Parameter(torch.randn(NORD, d) * (0.1 / math.sqrt(d)))
        self.gb = nn.Parameter(torch.ones(k, NORD))
        w0 = torch.full((k, NSET, N_MIX), 0.25)
        w0[:, :, 0] = 1.0
        self.w0 = nn.Parameter(w0)
        self.register_buffer("sq", torch.from_numpy(SQ))
        self.register_buffer("sq32", torch.from_numpy(SQ).to(torch.int32))

    def init_state(self, bsz):
        return torch.zeros(len(self.widths), bsz, NSET, N_MIX, dtype=torch.int32, device=self.hw.device)

    @staticmethod
    def q(x, scale, lim=None):
        x = ste_round(x * scale)
        return x if lim is None else torch.clamp(x, -lim, lim)

    def inputs(self, k, h, s, node):
        """Mixer inputs of tier k, [B, T, 8, N_MIX]: trunk stretch, gained successor evidence, bit-cell stretch."""
        dk = self.widths[k]
        hk = h[:, :, :dk]
        z = hk @ self.q(self.hw[:, :dk], 64.0, ACT).t()
        z = torch.gather(z, 2, node) + self.q(self.hb[k], 2048.0)[node]
        st_net = torch.clamp(ste_floor((z + 4.0) / 8.0), -ST_MAX, ST_MAX)
        g = hk @ self.q(self.gw[:, :dk], 64.0, ACT).t() + self.q(self.gb[k], 2048.0)
        gain = torch.clamp(ste_floor((g + 32.0) / 64.0), 0.0, ACT)
        sp = torch.clamp(ste_floor(s[..., :NORD] * gain[:, :, None, :] / 32.0), -ST_MAX, ST_MAX)
        return torch.cat([st_net[..., None], sp, s[..., NORD:]], dim=-1)

    def macs(self, width):
        return 8 * width + NORD * width + 8 * 2 * N_MIX

    def params(self, width):
        return 255 * width + 255 + NORD * width + NORD + NSET * N_MIX


class Ladder(nn.Module):
    """Trunk + head; the tiers are width prefixes of the same trunk and are trained together."""

    def __init__(self, rung):
        super().__init__()
        self.rung = rung
        self.widths = tuple(RUNGS[rung]["widths"])
        self.trunk = Trunk(self.widths)
        self.head = Head(self.widths)

    def init_state(self, bsz):
        return self.trunk.init_state(bsz), self.head.init_state(bsz)

    def forward(self, xt, s, top, y, state):
        """xt int64 [B, T, 4] (taps), s float [B, T, 8, NORD + NBC], top int64 [B, T, 8], y int64 [B, T].

        Returns the cost in bits per byte position of every tier (list of float64 [B, T]), the new state and the
        integer P(bit = 1) of every tier (int64 [K, B, T, 8])."""
        h, tstate = self.trunk(xt, state[0])
        hd, dev = self.head, y.device
        shifts = torch.arange(7, -1, -1, device=dev)
        ybit = (y[:, :, None] >> shifts) & 1
        node = ((1 << torch.arange(8, device=dev)) | (y[:, :, None] >> (shifts + 1))) - 1      # 0..254
        setidx = torch.arange(8, device=dev)[None, None, :] * (NORD + 1) + top
        inps = [hd.inputs(k, h, s, node) for k in range(len(self.widths))]
        w0q = hd.q(hd.w0, float(W_ONE), W_MAX)
        with torch.no_grad():
            used, raw, delta = mix_loop(torch.stack(inps), setidx, ybit, w0q.to(torch.int32), state[1], hd.sq32)
            p1 = hd.sq[raw.clamp(-ST_MAX, ST_MAX) + ST_MAX]
        sign = (1 - 2 * ybit).double()
        costs = []
        for k, inp in enumerate(inps):
            st = MixDot.apply(inp, w0q[k][setidx], used[k], raw[k])
            bce = fn.softplus(sign * st / 256.0) / LN2
            pb = torch.where(ybit.bool(), p1[k], 65536 - p1[k]).double()
            costs.append((bce + (-torch.log2(pb / 65536.0) - bce).detach()).sum(-1))
        return costs, (tstate, delta), p1

    def budget(self):
        """(width, multiply-accumulates per byte, parameters, MAC cap) of every tier."""
        return [(w, self.trunk.macs(w) + self.head.macs(w), self.trunk.params(w) + self.head.params(w), cap)
                for w, cap in zip(self.widths, RUNGS[self.rung]["caps"])]

    def check_budget(self):
        for width, macs, params, cap in self.budget():
            print(f"budget width {width}: {macs:,d} MAC/byte (cap {cap:,d}), {params:,d} parameters", flush=True)
            if macs > cap or width % GATE_GROUP:
                raise SystemExit(f"width {width} is not a legal tier of rung {self.rung}")

    def exact_bound(self):
        """Largest |pre-activation| any row can reach with int8 activations; the float32 matrix products are exact
        integers (in any summation order, with or without FMA) while it stays below 2^24."""
        tr, hd = self.trunk, self.head
        with torch.no_grad():
            rows = []
            for li in range(LAYERS):
                rows.append(ACT * tr.qmat(tr.w[li], tr.tri).abs().sum(1) + (tr.b[li] * 4096.0).round().abs())
                rows.append(ACT * tr.qmat(tr.wg[li], tr.gtri).abs().sum(1) + (tr.bg[li] * 4096.0).round().abs())
            rows.append(ACT * hd.q(hd.hw, 64.0, ACT).abs().sum(1) + hd.q(hd.hb, 2048.0).abs().amax(0))
            rows.append(ACT * hd.q(hd.gw, 64.0, ACT).abs().sum(1) + hd.q(hd.gb, 2048.0).abs().amax(0))
            return max(float(r.max()) for r in rows if r.numel())

    def check_exact(self):
        bound = self.exact_bound()
        if bound >= 2 ** 24:
            raise SystemExit(f"integer exactness lost: a pre-activation can reach {bound:.0f} >= 2^24")
        return bound


def detach_state(state):
    return [s.detach() for s in state[0]], state[1].detach()


def export_int(model, k):
    """Integer tensors of tier k (numpy int64), in the layout `bakeoff/ref_int.py` reads."""
    tr, hd = model.trunk, model.head
    dk = model.widths[k]
    ng = dk // GATE_GROUP

    def cpu(t):
        return t.detach().round().long().cpu().numpy()

    with torch.no_grad():
        return {
            "width": dk, "layers": LAYERS, "geared": True, "bc": True,
            "emb": cpu(tr.qemb()[:, :dk]), "tap": pattern(TAP_PATTERN, dk).long().numpy(),
            "kappa_in": cpu(tr.kappa_in[:dk]), "k_fast": cpu(tr.k_fast[:dk]), "k_slow": cpu(tr.k_slow[:dk]),
            "w": [cpu(tr.qmat(tr.w[i], tr.tri)[:dk, :dk]) for i in range(LAYERS)],
            "b": [cpu(tr.b[i] * 4096.0)[:dk] for i in range(LAYERS)],
            "wg": [cpu(tr.qmat(tr.wg[i], tr.gtri)[:ng, :dk]) for i in range(LAYERS)],
            "bg": [cpu(tr.bg[i] * 4096.0)[:ng] for i in range(LAYERS)],
            "hw": cpu(hd.q(hd.hw[:, :dk], 64.0, ACT)), "hb": cpu(hd.q(hd.hb[k], 2048.0)),
            "gw": cpu(hd.q(hd.gw[:, :dk], 64.0, ACT)), "gb": cpu(hd.q(hd.gb[k], 2048.0)),
            "w0": cpu(hd.q(hd.w0[k], float(W_ONE), W_MAX)),
        }
