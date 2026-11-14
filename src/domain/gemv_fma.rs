//! FMA GEMV / GEMM kernels — numeric contract 2 (`fma-v1`, R-107).
//!
//! Contract 2 replaces every `acc + w*x` of the contract-1 kernels in
//! [`crate::domain::gemv`] by one fused multiply-add `fma(w, x, acc)` (a single
//! rounding) and keeps their accumulation ORDER exactly:
//! - per output row (F32) or per (row, column group) (Q8): 8 lanes, lane `l`
//!   accumulates the columns `c ≡ l (mod 8)` of the 8-aligned body in column
//!   order, `acc_l = fma(w_c, x_c, acc_l)`, starting from +0;
//! - horizontal sum `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))` (plain adds);
//! - sequential tail `s = fma(w_c, x_c, s)` over the columns past the last chunk;
//! - Q8: int8 -> f32 conversion is exact; the row is `sum = fma(dot_g, scale_g, sum)`
//!   folded over the groups in order from +0.
//!
//! The scalar kernels reproduce that order with [`fma_f32`], a pure-Rust IEEE 754
//! fusedMultiplyAdd (exact product in f64, one correctly rounded sum; no platform
//! `fmaf`, so a CPU or a build without FMA decodes contract 2 bit for bit — R-104 F2).
//! `f32::mul_add` appears only inside the `#[target_feature(enable = "fma")]` kernels,
//! where it is one `vfmadd` instruction. Both tiers are unit-tested bit for bit.
//!
//! Register tiling only changes which (row, stream) pairs run side by side,
//! never the per-pair order, so every result is independent of the thread
//! count, the stream count and the tile shape.
//!
//! Scope: GEMV/GEMM only (in_proj, x_proj, dt_proj, out_proj, embedding/LM
//! head). Conv1d, the SSM scan, softplus, SiLU and the norms keep contract-1 code.

use std::ops::Range;
use std::sync::OnceLock;

use crate::domain::gemv::{hsum8_scalar, Q8Rows};
#[cfg(target_arch = "x86_64")]
use crate::domain::tensor::no_prefetch_override;
#[cfg(target_arch = "x86_64")]
use crate::domain::gemv::{fused_take, hsum8_avx, load_i8x8_ps, Q8_FUSED_MIN};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Streams per F32 register tile (2 rows x 4 streams = 8 FMA chains).
const F32_TILE_STREAMS: usize = 4;

/// AVX2 + FMA available and SIMD not disabled (cached). `REDUKT_NO_SIMD=1`
/// disables it; `REDUKT_NO_FMA=1` disables only this check, simulating an
/// AVX2 CPU without FMA (the contract-2 GEMV then runs the scalar kernels).
pub fn simd_fma() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let no_fma = std::env::var("REDUKT_NO_FMA").map(|v| v == "1").unwrap_or(false);
        #[cfg(target_arch = "x86_64")]
        {
            !no_fma && crate::domain::gemv::simd_avx2() && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = no_fma;
            false
        }
    })
}

// ---- Scalar reference (bit-exact emulation of the SIMD kernels) ----

/// IEEE 754 `fusedMultiplyAdd(a, b, c)` for f32 in pure Rust: `a * b + c` with a single
/// rounding, bit-identical to `vfmadd` / a correct `fmaf` on every input, with no
/// platform libm call (R-104 F2: `f32::mul_add` lowers to `fmaf` when the build has no
/// FMA target feature).
///
/// The product of two f32 is exact in f64 (48 significant bits, exponent in range), so
/// only the sum is rounded, to 53 bits. Rounding that f64 again to f32 can differ from
/// the single rounding only when the f64 lands exactly on an f32 rounding boundary:
/// every such boundary is a 25-bit dyadic, i.e. an f64 whose low 28 significand bits are
/// zero, and round-to-nearest is monotonic, so any other f64 lies strictly on the same
/// side of every boundary as the exact sum. On a possible boundary the rounding error of
/// the sum is recovered exactly (TwoSum) and the f64 is moved one ulp towards the exact
/// value (round to odd) before the final rounding.
#[inline]
pub(crate) fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    let p = a as f64 * b as f64;
    let c = c as f64;
    let s = p + c;
    let bits = s.to_bits();
    if bits & 0x0FFF_FFFF != 0 || !s.is_finite() {
        return s as f32;
    }
    let t = s - p;
    let err = (p - (s - t)) + (c - t);
    if err == 0.0 {
        return s as f32;
    }
    // Inexact sum on a possible boundary: `s` has an even significand; its neighbour
    // towards the exact value is odd and rounds to f32 like the exact value does.
    let away_from_zero = (err > 0.0) == (s > 0.0);
    f64::from_bits(if away_from_zero { bits + 1 } else { bits - 1 }) as f32
}

/// One dot product in contract-2 order: 8-lane fma body, hsum, fma tail.
#[inline]
fn dot_fma_scalar<W: Copy + Into<f32>>(w: &[W], v: &[f32]) -> f32 {
    let chunks = w.len() / 8;
    let mut acc = [0.0f32; 8];
    for ch in 0..chunks {
        let off = ch * 8;
        for l in 0..8 {
            acc[l] = fma_f32(w[off + l].into(), v[off + l], acc[l]);
        }
    }
    let mut s = hsum8_scalar(&acc);
    for c in chunks * 8..w.len() {
        s = fma_f32(w[c].into(), v[c], s);
    }
    s
}

/// Scalar F32 rows: `out[r] = dot(m[r*cols..], v)` in contract-2 order.
pub(crate) fn f32_rows_fma_scalar(out: &mut [f32], m: &[f32], v: &[f32], cols: usize) {
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot_fma_scalar(&m[r * cols..(r + 1) * cols], &v[..cols]);
    }
}

/// Scalar Q8 rows in contract-2 order.
pub(crate) fn q8w_rows_fma_scalar(out: &mut [f32], m: &Q8Rows, v: &[f32]) {
    let (cols, group) = (m.cols, m.group);
    let gpr = cols / group;
    for (r, o) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for g in 0..gpr {
            let c0 = r * cols + g * group;
            let dot = dot_fma_scalar(&m.q[c0..c0 + group], &v[g * group..(g + 1) * group]);
            sum = fma_f32(dot, m.scales[r * gpr + g], sum);
        }
        *o = sum;
    }
}

// ---- Dispatch ----

/// F32 rows, single stream (AVX2+FMA > scalar `fma_f32`).
pub(crate) fn f32_rows_fma(out: &mut [f32], m: &[f32], v: &[f32], cols: usize) {
    #[cfg(target_arch = "x86_64")]
    if simd_fma() {
        assert!(m.len() >= out.len() * cols && v.len() >= cols, "f32_rows_fma: short input");
        // SAFETY: AVX2+FMA detected; bounds asserted above.
        unsafe { f32_rows_fma_avx2(out, m, v, cols) };
        return;
    }
    f32_rows_fma_scalar(out, m, v, cols);
}

/// F32 rows `rr` (weights `w` = exactly those rows) for N streams:
/// `outs[s][rr] = w @ xs[s]`. Streams in tiles of 4, remainder per stream.
pub(crate) fn f32_rows_multi_fma(
    outs: &mut [&mut [f32]], rr: Range<usize>, w: &[f32], xs: &[&[f32]], cols: usize,
) {
    let n = outs.len();
    let mut s0 = 0;
    #[cfg(target_arch = "x86_64")]
    if simd_fma() {
        assert!(w.len() >= rr.len() * cols, "f32_rows_multi_fma: short weights");
        while n - s0 >= F32_TILE_STREAMS {
            let tile = s0..s0 + F32_TILE_STREAMS;
            // SAFETY: AVX2+FMA detected; caller checked xs[s].len() >= cols and
            // every outs[s] covers rr (par_rows_multi); weights asserted above.
            unsafe { f32_tile_fma_avx2(&mut outs[tile.clone()], rr.clone(), w, &xs[tile], cols) };
            s0 += F32_TILE_STREAMS;
        }
    }
    for s in s0..n {
        f32_rows_fma(&mut outs[s][rr.clone()], w, &xs[s][..cols], cols);
    }
}

/// Q8 rows, single stream (AVX2+FMA > scalar `fma_f32`).
pub(crate) fn q8w_rows_fma(out: &mut [f32], m: &Q8Rows, v: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_fma() {
        // SAFETY: AVX2+FMA detected; m.q has out.len()*cols elements, v has cols.
        unsafe { q8w_rows_fma_avx2(out, m, v) };
        return;
    }
    q8w_rows_fma_scalar(out, m, v);
}

/// Q8 rows for N streams: fused kernel on groups of 4..8 streams (same
/// grouping as contract 1), remainder through the single-stream kernel.
pub(crate) fn q8w_rows_multi_fma(outs: &mut [&mut [f32]], m: &Q8Rows, xs: &[&[f32]]) {
    let n = outs.len();
    let mut s0 = 0;
    #[cfg(target_arch = "x86_64")]
    if simd_fma() {
        while n - s0 >= Q8_FUSED_MIN {
            let take = fused_take(n - s0);
            // SAFETY: AVX2+FMA detected; caller ran check_multi_shapes (xs[s] >= cols)
            // and par_rows_multi gives every outs[s] the same row count as `m`.
            unsafe { q8w_rows_fused_fma(&mut outs[s0..s0 + take], m, &xs[s0..s0 + take]) };
            s0 += take;
        }
    }
    for s in s0..n {
        q8w_rows_fma(outs[s], m, &xs[s][..m.cols]);
    }
}

// ---- AVX2 + FMA kernels ----
// Every `mul_add` below sits in a function that enables the `fma` target feature, so it
// is one `vfmadd` instruction and never the platform `fmaf`; the lines carry the
// `libm-ok: vfmadd` marker for tests/libm_lint.rs, which bans `mul_add` anywhere else.

/// hsum of the 8-lane body, then the sequential fma tail over `from..len`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn finish_f32(acc: __m256, w: *const f32, x: *const f32, from: usize, len: usize) -> f32 {
    let mut s = hsum8_avx(acc);
    for c in from..len {
        s = (*w.add(c)).mul_add(*x.add(c), s); // libm-ok: vfmadd
    }
    s
}

/// One F32 row against one vector.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot1_f32_fma(w: *const f32, x: *const f32, len: usize) -> f32 {
    let chunks = len / 8;
    let mut acc = _mm256_setzero_ps();
    for ch in 0..chunks {
        let off = ch * 8;
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(off)), _mm256_loadu_ps(x.add(off)), acc);
    }
    finish_f32(acc, w, x, chunks * 8, len)
}

/// F32 single stream: 4-row ILP (4 independent FMA chains), odd rows one by one.
/// Software prefetch as in the contract-1 kernel (the single-stream GEMV is
/// DRAM-bound); `REDUKT_NO_PREFETCH=1` disables it. Never affects results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn f32_rows_fma_avx2(out: &mut [f32], m: &[f32], v: &[f32], cols: usize) {
    let rows = out.len();
    let (mp, vp) = (m.as_ptr(), v.as_ptr());
    let chunks = cols / 8;
    let rows_4 = rows / 4 * 4;
    let prefetch = !no_prefetch_override();
    for r in (0..rows_4).step_by(4) {
        let w = [mp.add(r * cols), mp.add((r + 1) * cols), mp.add((r + 2) * cols), mp.add((r + 3) * cols)];
        if prefetch {
            // Next 4 rows, staying inside `m` (pointer arithmetic must not leave it).
            for k in 0..4.min(rows.saturating_sub(r + 4)) {
                _mm_prefetch::<_MM_HINT_T0>(w[k].add(4 * cols) as *const i8);
            }
        }
        let mut a = [_mm256_setzero_ps(); 4];
        for ch in 0..chunks {
            let off = ch * 8;
            if prefetch && ch + 4 < chunks {
                for k in 0..4 {
                    _mm_prefetch::<_MM_HINT_T0>(w[k].add(off + 32) as *const i8);
                }
            }
            let vv = _mm256_loadu_ps(vp.add(off));
            for k in 0..4 {
                a[k] = _mm256_fmadd_ps(_mm256_loadu_ps(w[k].add(off)), vv, a[k]);
            }
        }
        for k in 0..4 {
            out[r + k] = finish_f32(a[k], w[k], vp, chunks * 8, cols);
        }
    }
    for r in rows_4..rows {
        out[r] = dot1_f32_fma(mp.add(r * cols), vp, cols);
    }
}

/// F32 register tile: 2 rows x 4 streams (8 FMA chains). Each weight chunk is
/// loaded once for 4 streams, each activation chunk once for 2 rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn f32_tile_fma_avx2(outs: &mut [&mut [f32]], rr: Range<usize>, w: &[f32], xs: &[&[f32]], cols: usize) {
    const S: usize = F32_TILE_STREAMS;
    let rows = rr.len();
    let chunks = cols / 8;
    let wp = w.as_ptr();
    let mut xp = [std::ptr::null::<f32>(); S];
    for s in 0..S {
        xp[s] = xs[s].as_ptr();
    }
    let mut r = 0;
    while r + 2 <= rows {
        let (w0, w1) = (wp.add(r * cols), wp.add((r + 1) * cols));
        let mut a0 = [_mm256_setzero_ps(); S];
        let mut a1 = [_mm256_setzero_ps(); S];
        for ch in 0..chunks {
            let off = ch * 8;
            let wv0 = _mm256_loadu_ps(w0.add(off));
            let wv1 = _mm256_loadu_ps(w1.add(off));
            for s in 0..S {
                let x = _mm256_loadu_ps(xp[s].add(off));
                a0[s] = _mm256_fmadd_ps(wv0, x, a0[s]);
                a1[s] = _mm256_fmadd_ps(wv1, x, a1[s]);
            }
        }
        for s in 0..S {
            outs[s][rr.start + r] = finish_f32(a0[s], w0, xp[s], chunks * 8, cols);
            outs[s][rr.start + r + 1] = finish_f32(a1[s], w1, xp[s], chunks * 8, cols);
        }
        r += 2;
    }
    if r < rows {
        for s in 0..S {
            outs[s][rr.start + r] = dot1_f32_fma(wp.add(r * cols), xp[s], cols);
        }
    }
}

/// Q8 group dot for 4 rows (4 FMA chains). `base[k]` = row k's group start.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn group_dot4_fma(qp: *const i8, base: [usize; 4], vp: *const f32, len: usize) -> [f32; 4] {
    let chunks = len / 8;
    let mut a = [_mm256_setzero_ps(); 4];
    for ch in 0..chunks {
        let off = ch * 8;
        let vv = _mm256_loadu_ps(vp.add(off));
        for k in 0..4 {
            a[k] = _mm256_fmadd_ps(load_i8x8_ps(qp.add(base[k] + off)), vv, a[k]);
        }
    }
    let mut s = [hsum8_avx(a[0]), hsum8_avx(a[1]), hsum8_avx(a[2]), hsum8_avx(a[3])];
    for c in chunks * 8..len {
        let vc = *vp.add(c);
        for k in 0..4 {
            s[k] = (*qp.add(base[k] + c) as f32).mul_add(vc, s[k]); // libm-ok: vfmadd
        }
    }
    s
}

/// Q8 group dot for one row.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn group_dot1_fma(qp: *const i8, vp: *const f32, len: usize) -> f32 {
    let chunks = len / 8;
    let mut acc = _mm256_setzero_ps();
    for ch in 0..chunks {
        let off = ch * 8;
        acc = _mm256_fmadd_ps(load_i8x8_ps(qp.add(off)), _mm256_loadu_ps(vp.add(off)), acc);
    }
    let mut s = hsum8_avx(acc);
    for c in chunks * 8..len {
        s = (*qp.add(c) as f32).mul_add(*vp.add(c), s); // libm-ok: vfmadd
    }
    s
}

/// Q8 single stream: 4-row ILP, rows past the last multiple of 4 one by one.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8w_rows_fma_avx2(out: &mut [f32], m: &Q8Rows, v: &[f32]) {
    let rows = out.len();
    let (cols, group) = (m.cols, m.group);
    let gpr = cols / group;
    let (qp, vp) = (m.q.as_ptr(), v.as_ptr());
    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let mut sum = [0.0f32; 4];
        for g in 0..gpr {
            let c0 = g * group;
            let base = [r * cols + c0, (r + 1) * cols + c0, (r + 2) * cols + c0, (r + 3) * cols + c0];
            let d = group_dot4_fma(qp, base, vp.add(c0), group);
            for k in 0..4 {
                sum[k] = d[k].mul_add(*m.scales.get_unchecked((r + k) * gpr + g), sum[k]); // libm-ok: vfmadd
            }
        }
        out[r..r + 4].copy_from_slice(&sum);
    }
    for r in rows_4..rows {
        let mut sum = 0.0f32;
        for g in 0..gpr {
            let c0 = g * group;
            let d = group_dot1_fma(qp.add(r * cols + c0), vp.add(c0), group);
            sum = d.mul_add(*m.scales.get_unchecked(r * gpr + g), sum); // libm-ok: vfmadd
        }
        out[r] = sum;
    }
}

/// Dispatch the fused Q8 FMA kernel on the stream count (monomorphized 4..=8).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8w_rows_fused_fma(outs: &mut [&mut [f32]], m: &Q8Rows, xs: &[&[f32]]) {
    match outs.len() {
        4 => q8w_rows_fused_fma_n::<4>(outs, m, xs),
        5 => q8w_rows_fused_fma_n::<5>(outs, m, xs),
        6 => q8w_rows_fused_fma_n::<6>(outs, m, xs),
        7 => q8w_rows_fused_fma_n::<7>(outs, m, xs),
        8 => q8w_rows_fused_fma_n::<8>(outs, m, xs),
        _ => for s in 0..outs.len() { q8w_rows_fma_avx2(outs[s], m, &xs[s][..m.cols]) },
    }
}

/// Fused Q8 FMA kernel: each int8 chunk is converted once and fma'd into S
/// register-resident per-stream accumulators (S independent chains).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8w_rows_fused_fma_n<const S: usize>(outs: &mut [&mut [f32]], m: &Q8Rows, xs: &[&[f32]]) {
    let rows = outs[0].len();
    let (cols, group) = (m.cols, m.group);
    let gpr = cols / group;
    let chunks = group / 8;
    let qp = m.q.as_ptr();
    let mut xp = [std::ptr::null::<f32>(); S];
    for s in 0..S {
        debug_assert!(xs[s].len() >= cols && outs[s].len() == rows);
        xp[s] = xs[s].as_ptr();
    }
    for r in 0..rows {
        let mut sum = [0.0f32; S];
        for g in 0..gpr {
            let c0 = g * group;
            let qrow = qp.add(r * cols + c0);
            let mut acc = [_mm256_setzero_ps(); S];
            for ch in 0..chunks {
                let off = ch * 8;
                let w = load_i8x8_ps(qrow.add(off));
                for s in 0..S {
                    acc[s] = _mm256_fmadd_ps(w, _mm256_loadu_ps(xp[s].add(c0 + off)), acc[s]);
                }
            }
            let scale = *m.scales.get_unchecked(r * gpr + g);
            for s in 0..S {
                let mut d = hsum8_avx(acc[s]);
                for c in chunks * 8..group {
                    d = (*qrow.add(c) as f32).mul_add(*xp[s].add(c0 + c), d); // libm-ok: vfmadd
                }
                sum[s] = d.mul_add(scale, sum[s]); // libm-ok: vfmadd
            }
        }
        for s in 0..S {
            *outs[s].get_unchecked_mut(r) = sum[s];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64* — deterministic test stream.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn bits32(&mut self) -> u32 {
            (self.next() >> 32) as u32
        }

        /// Finite f32 from a mix of regimes: uniform, wide exponent, subnormal, ±0.
        fn mixed(&mut self) -> f32 {
            let b = self.bits32();
            match self.next() % 8 {
                0 => f32::from_bits(b & 0x807f_ffff),                    // subnormal / ±0
                1 => f32::from_bits((b & 0x807f_ffff) | (1 << 23)),      // tiny normal
                2 | 3 => {
                    let e = (self.next() % 254 + 1) as u32;              // any finite exponent
                    f32::from_bits((b & 0x807f_ffff) | (e << 23))
                }
                _ => (b as f32 / u32::MAX as f32) * 2.0 - 1.0,
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fmadd8(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
        let (va, vb) = (_mm256_loadu_ps(a.as_ptr()), _mm256_loadu_ps(b.as_ptr()));
        _mm256_storeu_ps(out.as_mut_ptr(), _mm256_fmadd_ps(va, vb, _mm256_loadu_ps(c.as_ptr())));
    }

    /// Same bits, or both NaN (NaN payloads are not part of any contract).
    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// R-107 primitive check: `f32::mul_add` (scalar path) == `vfmadd231ps` on 1e6
    /// triples covering subnormals, ±0, extreme exponents, exact cancellation
    /// (`c = -round(a*b)`, result = rounding error of a*b) and near-cancellation.
    #[test]
    fn mul_add_matches_simd_fmadd_on_1e6_triples() {
        if !simd_fma() {
            eprintln!("SKIP: AVX2+FMA unavailable or REDUKT_NO_SIMD=1");
            return;
        }
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut a, mut b, mut c) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
        let mut simd = [0.0f32; 8];
        let (mut subnormal_out, mut cancel) = (0usize, 0usize);
        for i in 0..125_000 {
            for l in 0..8 {
                a[l] = rng.mixed();
                b[l] = rng.mixed();
                c[l] = match (i + l) % 4 {
                    0 => -(a[l] * b[l]),
                    1 => -(a[l] * b[l]) * (1.0 + f32::EPSILON),
                    _ => rng.mixed(),
                };
            }
            #[cfg(target_arch = "x86_64")]
            unsafe { fmadd8(&a, &b, &c, &mut simd) };
            for l in 0..8 {
                let s = std::hint::black_box(a[l]).mul_add(std::hint::black_box(b[l]), std::hint::black_box(c[l]));
                assert!(same(s, simd[l]), "fma({:e}, {:e}, {:e}): scalar {:e} (0x{:08x}) simd {:e} (0x{:08x})",
                    a[l], b[l], c[l], s, s.to_bits(), simd[l], simd[l].to_bits());
                subnormal_out += s.is_subnormal() as usize;
                cancel += ((i + l) % 4 == 0 && s != 0.0) as usize;
            }
        }
        // The sample really exercises the hard cases.
        assert!(subnormal_out > 1000 && cancel > 10_000, "subnormal={} cancel={}", subnormal_out, cancel);
    }

    // ---- R-104 F2: `fma_f32` (pure Rust) is a correctly rounded fusedMultiplyAdd ----

    /// 2^e as f32 (normal range).
    fn pow2(e: i32) -> f32 {
        f32::from_bits(((e + 127) as u32) << 23)
    }

    /// Exact sums just below / just above an f32 rounding midpoint, the case where
    /// rounding the f64 sum to f32 (double rounding) is wrong. The expected result follows
    /// from the construction, so this runs on every architecture:
    /// (1 + a)(1 - a) = 1 - a^2 with a = i * 2^-23, hence
    ///   below: m + 1/2 - a^2/2      -> m      (the naive result is m + 1 for odd m)
    ///   above: (m + 1) - 1/2 + a^2/2 -> m + 1  (the naive result is m for even m)
    #[test]
    fn soft_fma_rounds_once_next_to_a_midpoint() {
        let mut rng = Rng(0x243f_6a88_85a3_08d3);
        let (mut naive_wrong, mut cases) = (0usize, 0usize);
        for n in 0..40_000 {
            let i = (rng.next() % 255 + 1) as f32;
            let alpha = i * pow2(-23);
            let subnormal = n % 4 == 3;
            // Normal results: m in [2^23, 2^24) scaled by 2^e. Subnormal results: m * 2^-149.
            let e = (rng.next() % 201) as i32 - 100;
            let m = if subnormal { rng.bits32() % (7 << 20) + (1 << 20) } else { rng.bits32() % (1 << 23) + (1 << 23) };
            let value = |k: u32| if subnormal { f32::from_bits(k) } else { k as f32 * pow2(e) };
            let (a, b) = if subnormal {
                ((1.0 + alpha) * pow2(-75), (1.0 - alpha) * pow2(-75))
            } else {
                (1.0 + alpha, (1.0 - alpha) * pow2(e - 1))
            };
            for (a, c, expected) in [(a, value(m), value(m)), (-a, value(m + 1), value(m + 1))] {
                let got = fma_f32(a, b, c);
                assert_eq!(got.to_bits(), expected.to_bits(),
                    "fma({:e}, {:e}, {:e}) = {:e}, expected {:e}", a, b, c, got, expected);
                let naive = (a as f64 * b as f64 + c as f64) as f32;
                naive_wrong += (naive.to_bits() != expected.to_bits()) as usize;
                cases += 1;
            }
        }
        // Half of the cases defeat double rounding (ties-to-even picks the wrong neighbour).
        assert!(naive_wrong * 3 > cases, "fixture too easy: naive wrong in {} of {}", naive_wrong, cases);
    }

    #[test]
    fn soft_fma_special_values_and_exact_cancellation() {
        let bits = |x: f32| x.to_bits();
        assert_eq!(bits(fma_f32(1.0, 1.0, -1.0)), bits(0.0));
        assert_eq!(bits(fma_f32(0.0, 5.0, -0.0)), bits(0.0));
        assert_eq!(bits(fma_f32(-0.0, 5.0, -0.0)), bits(-0.0));
        assert_eq!(bits(fma_f32(3.0, 0.0, 7.5)), bits(7.5));
        assert_eq!(fma_f32(f32::MAX, 2.0, 0.0), f32::INFINITY);
        assert_eq!(fma_f32(f32::MAX, 2.0, -f32::MAX), f32::MAX);
        assert_eq!(fma_f32(1e30, 1e30, f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert_eq!(fma_f32(f32::INFINITY, 2.0, 1.0), f32::INFINITY);
        assert!(fma_f32(f32::INFINITY, 0.0, 1.0).is_nan());
        assert!(fma_f32(f32::INFINITY, 1.0, f32::NEG_INFINITY).is_nan());
        assert!(fma_f32(f32::NAN, 1.0, 1.0).is_nan());
        // fma(a, b, -round(a*b)) is the rounding error of a*b: exact in f64, one rounding to f32.
        let mut rng = Rng(0x1319_8a2e_0370_7344);
        for _ in 0..100_000 {
            let (a, b) = (rng.mixed(), rng.mixed());
            let prod = a * b;
            if !prod.is_finite() { continue; }
            let expected = (a as f64 * b as f64 - prod as f64) as f32;
            assert!(same(fma_f32(a, b, -prod), expected), "fma({:e}, {:e}, -{:e})", a, b, prod);
        }
    }

    /// `fma_f32` against the hardware instruction on 4e6 triples (same hard regimes as the
    /// `mul_add` check above: subnormals, +-0, extreme exponents, exact and near cancellation).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soft_fma_matches_simd_fmadd_on_4e6_triples() {
        if !simd_fma() {
            eprintln!("SKIP: AVX2+FMA unavailable or REDUKT_NO_SIMD=1");
            return;
        }
        let mut rng = Rng(0xa409_3822_299f_31d0);
        let (mut a, mut b, mut c) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
        let mut simd = [0.0f32; 8];
        let mut slow_path = 0usize;
        for i in 0..500_000 {
            for l in 0..8 {
                a[l] = rng.mixed();
                b[l] = rng.mixed();
                c[l] = match (i + l) % 4 {
                    0 => -(a[l] * b[l]),
                    1 => -(a[l] * b[l]) * (1.0 + f32::EPSILON),
                    _ => rng.mixed(),
                };
            }
            unsafe { fmadd8(&a, &b, &c, &mut simd) };
            for l in 0..8 {
                let s = fma_f32(a[l], b[l], c[l]);
                assert!(same(s, simd[l]), "fma({:e}, {:e}, {:e}): soft {:e} (0x{:08x}) simd {:e} (0x{:08x})",
                    a[l], b[l], c[l], s, s.to_bits(), simd[l], simd[l].to_bits());
                let sum = a[l] as f64 * b[l] as f64 + c[l] as f64;
                slow_path += (sum.is_finite() && sum.to_bits() & 0x0FFF_FFFF == 0) as usize;
            }
        }
        assert!(slow_path > 10_000, "the sample must reach the boundary path ({})", slow_path);
    }

    #[test]
    fn dot_order_is_contract_2() {
        // 11 columns: 8-lane body + 3-column tail, all fma.
        let w: Vec<f32> = (0..11).map(|i| 1.0 + i as f32 * 0.37).collect();
        let v: Vec<f32> = (0..11).map(|i| 0.5 - i as f32 * 0.11).collect();
        let mut acc = [0.0f32; 8];
        for l in 0..8 {
            acc[l] = w[l].mul_add(v[l], 0.0);
        }
        let mut s = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
        for c in 8..11 {
            s = w[c].mul_add(v[c], s);
        }
        assert_eq!(dot_fma_scalar(&w, &v).to_bits(), s.to_bits());
    }
}
