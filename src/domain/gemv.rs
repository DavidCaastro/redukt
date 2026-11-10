//! Pooled GEMV kernels for the Mamba backbone.
//!
//! - R-098: row-partitioned multithreaded f32 GEMV on the persistent pool
//!   ([`par_pool`]). Each output row is computed by the exact same
//!   single-thread kernel as before (`mat_vec_mul_avx2_nofma_raw` /
//!   `mat_vec_mul_scalar_raw`), so logits — and therefore AZTH bytes — are
//!   bit-identical to the single-thread path for any thread count.
//! - R-099: Q8 weight-only GEMV (W8A32). Weights are symmetric int8 with one
//!   f32 scale per group of columns (group = whole row, or 64); activations
//!   stay f32 (never quantized: per-tensor W8A8 is what sank R-064).
//!   `out[r] = Σ_g scale[r,g] * Σ_{c∈g} f32(q[r,c]) * x[c]`.
//!
//! Q8 kernel numeric contract (the kernel is SIMD/scalar tier-invariant; a Q8
//! *archive* is still tier-bound through the f32 SSM/x_proj/dt_proj kernels):
//! per (row, group) the AVX2 kernel uses 8 lanes, lane `l` accumulating the
//! group's columns `c ≡ l (mod 8)` with separate `vmulps` + `vaddps` (no FMA),
//! then a fixed horizontal-sum tree, then a sequential tail; the row result is
//! `0 + Σ_g dot_g * scale_g` in group order. The scalar kernel emulates exactly
//! that operation order with plain f32 arithmetic, so SIMD and scalar produce
//! bit-identical results (unit-tested). FMA is deliberately not used: the
//! scalar path would need a correctly-rounded libm `fmaf` to match it.

use std::sync::OnceLock;

use crate::domain::par_pool::{self, Pool, SyncPtr, MAX_BLOCKS};
use crate::domain::tensor::{Tensor, mat_vec_mul_scalar_raw};
#[cfg(target_arch = "x86_64")]
use crate::domain::tensor::mat_vec_mul_avx2_nofma_raw;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Matrices smaller than this run single-threaded (dispatch cost dominates).
const PAR_MIN_BYTES: usize = 128 * 1024;
/// Upper bound on weight bytes per block (keeps a block's stream short).
const BLOCK_MAX_BYTES: usize = 64 * 1024;
/// Target blocks per thread (granularity for P/E-core load balancing).
const BLOCKS_PER_THREAD: usize = 8;

/// AVX2 available and not disabled by `REDUKT_NO_SIMD=1` (cached per process).
pub fn simd_avx2() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let no_simd = std::env::var("REDUKT_NO_SIMD").map(|v| v == "1").unwrap_or(false);
        #[cfg(target_arch = "x86_64")]
        {
            !no_simd && is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = no_simd;
            false
        }
    })
}

/// Total threads used by pooled GEMV (1 = no worker threads).
pub fn pool_threads() -> usize {
    par_pool::global().threads()
}

/// Rows per block: multiple of 4 (4-row ILP), ≤ BLOCK_MAX_BYTES, and small
/// enough to give each thread ~BLOCKS_PER_THREAD blocks.
fn rows_per_block(rows: usize, row_bytes: usize, threads: usize) -> usize {
    let by_balance = rows / (threads * BLOCKS_PER_THREAD).max(1);
    let by_size = BLOCK_MAX_BYTES / row_bytes.max(1);
    let mut rpb = (by_balance.min(by_size).max(4) + 3) / 4 * 4;
    while (rows + rpb - 1) / rpb > MAX_BLOCKS {
        rpb *= 2;
    }
    rpb
}

/// Partition `out` (one entry per row) into row blocks and run `kernel`
/// on the pool. `kernel(r0, out_block)` computes rows `r0..r0+out_block.len()`.
fn par_rows<K>(pool: &Pool, out: &mut [f32], row_bytes: usize, kernel: K)
where
    K: Fn(usize, &mut [f32]) + Sync,
{
    let rows = out.len();
    let threads = pool.threads();
    if threads <= 1 || rows * row_bytes < PAR_MIN_BYTES || rows < 8 {
        kernel(0, out);
        return;
    }
    let rpb = rows_per_block(rows, row_bytes, threads);
    let n_blocks = (rows + rpb - 1) / rpb;
    let ptr = SyncPtr::new(out.as_mut_ptr());
    pool.run(n_blocks, &|b| {
        let r0 = b * rpb;
        let r1 = (r0 + rpb).min(rows);
        // SAFETY: blocks cover disjoint row ranges [r0, r1) within `out`.
        let block = unsafe { ptr.slice_mut(r0, r1 - r0) };
        kernel(r0, block);
    });
}

/// Rows per inner sub-block of the per-stream F32 multi kernel (R-100): a
/// 4-row group of F32 weights at K=1536 is 24 KB, so it stays in L1 while it
/// is applied to every stream. Multiple of 4 (4-row ILP groups stay aligned).
const MULTI_SUB_ROWS: usize = 4;

/// Multi-stream row partition (R-100 lockstep). Every row block of weights is
/// fetched from DRAM once and applied to all `outs.len()` activation vectors
/// while it is hot in cache; `kernel(r0, blocks)` computes rows
/// `r0..r0+blocks[0].len()` of every stream (`blocks[s]` = stream s's output
/// rows). Per (stream, row) the arithmetic must be exactly the single-stream
/// kernel's, so every output is bit-identical to the corresponding
/// single-stream GEMV regardless of the stream count.
fn par_rows_multi<K>(pool: &Pool, outs: &mut [&mut [f32]], rows: usize, row_bytes: usize, kernel: K)
where
    K: Fn(usize, &mut [&mut [f32]]) + Sync,
{
    let n = outs.len();
    if n == 0 {
        return;
    }
    let threads = pool.threads();
    if threads <= 1 || rows * row_bytes * n < PAR_MIN_BYTES || rows < 8 {
        let mut blocks: Vec<&mut [f32]> = outs.iter_mut().map(|o| &mut o[..rows]).collect();
        kernel(0, &mut blocks);
        return;
    }
    let rpb = rows_per_block(rows, row_bytes, threads);
    let n_blocks = (rows + rpb - 1) / rpb;
    let ptrs: Vec<SyncPtr<f32>> = outs.iter_mut().map(|o| SyncPtr::new(o[..rows].as_mut_ptr())).collect();
    pool.run(n_blocks, &|b| {
        let r0 = b * rpb;
        let r1 = (r0 + rpb).min(rows);
        // SAFETY: blocks cover disjoint row ranges of each stream's output.
        let mut blocks: Vec<&mut [f32]> = ptrs.iter().map(|p| unsafe { p.slice_mut(r0, r1 - r0) }).collect();
        kernel(r0, &mut blocks);
    });
}

// ---- F32 (bit-identical to the single-thread kernel) ----

/// Single-thread f32 rows kernel: same dispatch as `mat_vec_mul_into`'s
/// single-thread path (AVX2 noFMA > scalar).
pub fn f32_rows(out: &mut [f32], m: &[f32], v: &[f32], cols: usize) {
    let rows = out.len();
    #[cfg(target_arch = "x86_64")]
    if simd_avx2() {
        // SAFETY: AVX2 detected; m has rows*cols elements, v has cols.
        unsafe { mat_vec_mul_avx2_nofma_raw(out, m, v, rows, cols) };
        return;
    }
    mat_vec_mul_scalar_raw(out, m, v, rows, cols);
}

/// out = mat @ v on the global pool. Bit-identical to the single-thread path.
pub fn gemv_f32_into(out: &mut [f32], mat: &Tensor, v: &[f32]) {
    gemv_f32_with(par_pool::global(), out, mat, v);
}

fn gemv_f32_with(pool: &Pool, out: &mut [f32], mat: &Tensor, v: &[f32]) {
    let rows = mat.shape[0];
    let cols = mat.shape[1];
    let data = &mat.data;
    par_rows(pool, &mut out[..rows], cols * 4, |r0, o| {
        let r1 = r0 + o.len();
        f32_rows(o, &data[r0 * cols..r1 * cols], &v[..cols], cols);
    });
}

/// outs[s] = mat @ xs[s] for every stream, sharing each weight fetch (R-100).
/// Bit-identical to `gemv_f32_into(outs[s], mat, xs[s])` for every s.
pub fn gemv_f32_multi(outs: &mut [&mut [f32]], mat: &Tensor, xs: &[&[f32]]) {
    gemv_f32_multi_with(par_pool::global(), outs, mat, xs);
}

fn gemv_f32_multi_with(pool: &Pool, outs: &mut [&mut [f32]], mat: &Tensor, xs: &[&[f32]]) {
    let rows = mat.shape[0];
    let cols = mat.shape[1];
    let data = &mat.data;
    debug_assert_eq!(outs.len(), xs.len());
    par_rows_multi(pool, outs, rows, cols * 4, |r0, blocks| {
        let len = blocks[0].len();
        for sub0 in (0..len).step_by(MULTI_SUB_ROWS) {
            let sub1 = (sub0 + MULTI_SUB_ROWS).min(len);
            let w = &data[(r0 + sub0) * cols..(r0 + sub1) * cols];
            for (o, x) in blocks.iter_mut().zip(xs) {
                f32_rows(&mut o[sub0..sub1], w, &x[..cols], cols);
            }
        }
    });
}

// ---- Q8 weight-only (W8A32) ----

/// Weight-only int8 matrix: symmetric, one f32 scale per `group` consecutive
/// columns of a row (`group == cols` → per-row scale).
/// `w[r,c] ≈ q[r*cols+c] * scales[r*groups_per_row + c/group]`.
pub struct Q8wMatrix {
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
}

impl Q8wMatrix {
    /// Quantize a (rows, cols) f32 tensor. `group` must equal `cols` or be a
    /// multiple of 8 that divides `cols`; anything else falls back to per-row.
    /// scale = max|w|/127 per group (1.0 for an all-zero group);
    /// q = clamp(round(w * (1/scale)), -127, 127).
    pub fn quantize(t: &Tensor, group: usize) -> Self {
        let rows = t.shape[0];
        let cols = t.shape[1];
        let group = if group > 0 && group % 8 == 0 && cols % group == 0 { group } else { cols };
        let gpr = cols / group.max(1);
        let mut q = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows * gpr];
        for (gi, (w, qg)) in t.data.chunks(group).zip(q.chunks_mut(group)).enumerate() {
            let amax = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = 1.0 / scale;
            for (o, &x) in qg.iter_mut().zip(w) {
                *o = (x * inv).round().clamp(-127.0, 127.0) as i8;
            }
            scales[gi] = scale;
        }
        Self { q, scales, rows, cols, group }
    }

    /// Resident bytes (== weight bytes streamed by one GEMV).
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }

    fn groups_per_row(&self) -> usize {
        self.cols / self.group
    }

    /// Dequantize row `row` into `out[..cols]`: `q * scale`.
    pub fn dequant_row_into(&self, out: &mut [f32], row: usize) {
        let cols = self.cols;
        let gpr = self.groups_per_row();
        let q = &self.q[row * cols..(row + 1) * cols];
        let sc = &self.scales[row * gpr..(row + 1) * gpr];
        for ((o, qg), &s) in out[..cols].chunks_mut(self.group).zip(q.chunks(self.group)).zip(sc) {
            for (x, &w) in o.iter_mut().zip(qg) {
                *x = w as f32 * s;
            }
        }
    }
}

/// out = dequant(mat) @ v on the global pool (Q8 weights, f32 activations).
pub fn gemv_q8w_into(out: &mut [f32], mat: &Q8wMatrix, v: &[f32]) {
    gemv_q8w_with(par_pool::global(), out, mat, v);
}

fn gemv_q8w_with(pool: &Pool, out: &mut [f32], mat: &Q8wMatrix, v: &[f32]) {
    let (rows, cols, group) = (mat.rows, mat.cols, mat.group);
    let gpr = mat.groups_per_row();
    let (q, sc) = (&mat.q, &mat.scales);
    par_rows(pool, &mut out[..rows], cols + gpr * 4, |r0, o| {
        let r1 = r0 + o.len();
        let blk = Q8Rows { q: &q[r0 * cols..r1 * cols], scales: &sc[r0 * gpr..r1 * gpr], cols, group };
        q8w_rows(o, &blk, &v[..cols]);
    });
}

/// outs[s] = dequant(mat) @ xs[s] for every stream, sharing each weight fetch
/// (R-100). Bit-identical to `gemv_q8w_into(outs[s], mat, xs[s])` for every s.
pub fn gemv_q8w_multi(outs: &mut [&mut [f32]], mat: &Q8wMatrix, xs: &[&[f32]]) {
    gemv_q8w_multi_with(par_pool::global(), outs, mat, xs);
}

fn gemv_q8w_multi_with(pool: &Pool, outs: &mut [&mut [f32]], mat: &Q8wMatrix, xs: &[&[f32]]) {
    let (rows, cols, group) = (mat.rows, mat.cols, mat.group);
    let gpr = mat.groups_per_row();
    let (q, sc) = (&mat.q, &mat.scales);
    debug_assert_eq!(outs.len(), xs.len());
    par_rows_multi(pool, outs, rows, cols + gpr * 4, |r0, blocks| {
        let r1 = r0 + blocks[0].len();
        let blk = Q8Rows { q: &q[r0 * cols..r1 * cols], scales: &sc[r0 * gpr..r1 * gpr], cols, group };
        q8w_rows_multi(blocks, &blk, xs);
    });
}

/// Streams per fused Q8 kernel call (register budget: S accumulators + 2).
const Q8_FUSED_MAX: usize = 8;
/// Below this many streams the 4-row single kernel per stream has more
/// independent accumulator chains than the fused one.
const Q8_FUSED_MIN: usize = 4;

/// Q8 rows for N streams. AVX2: streams in groups of 4..8 through the fused
/// kernel (each int8 weight chunk converted once, multiplied into S
/// register-resident accumulators); remainders < 4 and the scalar path use the
/// single-stream kernel per stream. Per (stream, row) the operation order is
/// the single kernel's (see `q8w_row_fused_avx2`), so results are bit-identical.
fn q8w_rows_multi(outs: &mut [&mut [f32]], m: &Q8Rows, xs: &[&[f32]]) {
    let n = outs.len();
    let mut s0 = 0;
    #[cfg(target_arch = "x86_64")]
    if simd_avx2() {
        while n - s0 >= Q8_FUSED_MIN {
            let take = (n - s0).min(Q8_FUSED_MAX);
            // Leave at least Q8_FUSED_MIN for the next fused call when possible.
            let take = if n - s0 - take > 0 && n - s0 - take < Q8_FUSED_MIN && take > Q8_FUSED_MIN {
                take - (Q8_FUSED_MIN - (n - s0 - take))
            } else {
                take
            };
            // SAFETY: AVX2 detected; shapes checked by the fused kernel.
            unsafe { q8w_rows_fused_avx2(&mut outs[s0..s0 + take], m, &xs[s0..s0 + take]) };
            s0 += take;
        }
    }
    for s in s0..n {
        q8w_rows(outs[s], m, &xs[s][..m.cols]);
    }
}

/// Dispatch the fused kernel on the stream count (monomorphized 4..=8).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8w_rows_fused_avx2(outs: &mut [&mut [f32]], m: &Q8Rows, xs: &[&[f32]]) {
    match outs.len() {
        4 => q8w_rows_fused_n::<4>(outs, m, xs),
        5 => q8w_rows_fused_n::<5>(outs, m, xs),
        6 => q8w_rows_fused_n::<6>(outs, m, xs),
        7 => q8w_rows_fused_n::<7>(outs, m, xs),
        8 => q8w_rows_fused_n::<8>(outs, m, xs),
        _ => for s in 0..outs.len() { q8w_rows_avx2(outs[s], m, &xs[s][..m.cols]) },
    }
}

/// Fused Q8 kernel: every row is streamed once; each 8-wide int8 chunk is
/// converted to f32 once and multiplied into S per-stream accumulators.
/// Per stream the arithmetic is exactly `group_dot1_avx` + the row loop of
/// `q8w_rows_avx2`: lane-wise vmulps + vaddps per chunk, `hsum8_avx`,
/// sequential tail, `sum += dot * scale` in group order from 0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8w_rows_fused_n<const S: usize>(outs: &mut [&mut [f32]], m: &Q8Rows, xs: &[&[f32]]) {
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
                    acc[s] = _mm256_add_ps(acc[s], _mm256_mul_ps(w, _mm256_loadu_ps(xp[s].add(c0 + off))));
                }
            }
            let scale = *m.scales.get_unchecked(r * gpr + g);
            for s in 0..S {
                let mut d = hsum8_avx(acc[s]);
                for c in chunks * 8..group {
                    d += *qrow.add(c) as f32 * *xp[s].add(c0 + c);
                }
                sum[s] += d * scale;
            }
        }
        for s in 0..S {
            *outs[s].get_unchecked_mut(r) = sum[s];
        }
    }
}

/// A contiguous run of Q8 rows (kernel input).
struct Q8Rows<'a> {
    q: &'a [i8],
    scales: &'a [f32],
    cols: usize,
    group: usize,
}

fn q8w_rows(out: &mut [f32], m: &Q8Rows, v: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd_avx2() {
        // SAFETY: AVX2 detected; m.q has out.len()*cols elements, v has cols.
        unsafe { q8w_rows_avx2(out, m, v) };
        return;
    }
    q8w_rows_scalar(out, m, v);
}

/// Horizontal sum of 8 lanes in the exact order of `hsum8_avx`:
/// ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)).
#[inline]
fn hsum8_scalar(a: &[f32; 8]) -> f32 {
    let s0 = a[0] + a[4];
    let s1 = a[1] + a[5];
    let s2 = a[2] + a[6];
    let s3 = a[3] + a[7];
    (s0 + s2) + (s1 + s3)
}

/// One group of one row, scalar: 8-lane mul+add, hsum, sequential tail.
#[inline]
fn group_dot_scalar(q: &[i8], v: &[f32]) -> f32 {
    let chunks = q.len() / 8;
    let mut acc = [0.0f32; 8];
    for ch in 0..chunks {
        let off = ch * 8;
        for l in 0..8 {
            acc[l] += q[off + l] as f32 * v[off + l];
        }
    }
    let mut s = hsum8_scalar(&acc);
    for c in chunks * 8..q.len() {
        s += q[c] as f32 * v[c];
    }
    s
}

/// Scalar Q8 weight-only kernel — bit-exact emulation of `q8w_rows_avx2`:
/// row = Σ_groups (group_dot * scale), summed in group order starting at 0.
fn q8w_rows_scalar(out: &mut [f32], m: &Q8Rows, v: &[f32]) {
    let (cols, group) = (m.cols, m.group);
    let gpr = cols / group;
    for (r, o) in out.iter_mut().enumerate() {
        let mut row_sum = 0.0f32;
        for g in 0..gpr {
            let c0 = r * cols + g * group;
            let dot = group_dot_scalar(&m.q[c0..c0 + group], &v[g * group..(g + 1) * group]);
            row_sum += dot * m.scales[r * gpr + g];
        }
        *o = row_sum;
    }
}

/// 8 x i8 → 8 x f32 (vpmovsxbd + vcvtdq2ps; exact for |q| ≤ 127).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_i8x8_ps(p: *const i8) -> __m256 {
    let b = _mm_loadl_epi64(p as *const __m128i);
    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b))
}

/// Horizontal sum ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum8_avx(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    _mm_cvtss_f32(_mm_add_ss(sum2, _mm_shuffle_ps::<1>(sum2, sum2)))
}

/// Dot of one group for 4 rows at once (4 independent 8-lane accumulators).
/// `base[k]` = offset of row k's group start in `qp`; `vp` = group start in v.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn group_dot4_avx(qp: *const i8, base: [usize; 4], vp: *const f32, len: usize) -> [f32; 4] {
    let chunks = len / 8;
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    for ch in 0..chunks {
        let off = ch * 8;
        let vv = _mm256_loadu_ps(vp.add(off));
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(load_i8x8_ps(qp.add(base[0] + off)), vv));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(load_i8x8_ps(qp.add(base[1] + off)), vv));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(load_i8x8_ps(qp.add(base[2] + off)), vv));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(load_i8x8_ps(qp.add(base[3] + off)), vv));
    }
    let mut s = [hsum8_avx(a0), hsum8_avx(a1), hsum8_avx(a2), hsum8_avx(a3)];
    for c in chunks * 8..len {
        let vc = *vp.add(c);
        for k in 0..4 {
            s[k] += *qp.add(base[k] + c) as f32 * vc;
        }
    }
    s
}

/// Dot of one group for a single row.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn group_dot1_avx(qp: *const i8, vp: *const f32, len: usize) -> f32 {
    let chunks = len / 8;
    let mut acc = _mm256_setzero_ps();
    for ch in 0..chunks {
        let off = ch * 8;
        acc = _mm256_add_ps(acc, _mm256_mul_ps(load_i8x8_ps(qp.add(off)), _mm256_loadu_ps(vp.add(off))));
    }
    let mut s = hsum8_avx(acc);
    for c in chunks * 8..len {
        s += *qp.add(c) as f32 * *vp.add(c);
    }
    s
}

/// AVX2 Q8 weight-only kernel: 4-row ILP, one 8-lane f32 accumulator per
/// (row, group), vmulps + vaddps (no FMA). See module docs for the contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8w_rows_avx2(out: &mut [f32], m: &Q8Rows, v: &[f32]) {
    let rows = out.len();
    let (cols, group) = (m.cols, m.group);
    let gpr = cols / group;
    let qp = m.q.as_ptr();
    let vp = v.as_ptr();
    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let mut sum = [0.0f32; 4];
        for g in 0..gpr {
            let c0 = g * group;
            let base = [r * cols + c0, (r + 1) * cols + c0, (r + 2) * cols + c0, (r + 3) * cols + c0];
            let d = group_dot4_avx(qp, base, vp.add(c0), group);
            for k in 0..4 {
                sum[k] += d[k] * *m.scales.get_unchecked((r + k) * gpr + g);
            }
        }
        out[r..r + 4].copy_from_slice(&sum);
    }
    for r in rows_4..rows {
        let mut sum = 0.0f32;
        for g in 0..gpr {
            let c0 = g * group;
            let d = group_dot1_avx(qp.add(r * cols + c0), vp.add(c0), group);
            sum += d * *m.scales.get_unchecked(r * gpr + g);
        }
        out[r] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Deterministic pseudo-random f32 in [-1, 1) (xorshift32).
    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n).map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        }).collect()
    }

    fn rand_tensor(rows: usize, cols: usize, seed: u32) -> Tensor {
        Tensor::from_data(rand_vec(rows * cols, seed), vec![rows, cols])
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn rows_of(m: &Q8wMatrix) -> Q8Rows<'_> {
        Q8Rows { q: &m.q, scales: &m.scales, cols: m.cols, group: m.group }
    }

    // Shapes: Mamba in_proj/out_proj/x_proj/dt_proj-like plus odd remainders.
    const SHAPES: &[(usize, usize)] = &[
        (384, 768), (768, 1536), (80, 1536), (1536, 48), (1003, 771), (4100, 64),
    ];

    #[test]
    fn pooled_f32_gemv_bit_identical_to_single_thread() {
        let pools = [
            Pool::new(1, Duration::from_micros(0)),
            Pool::new(3, Duration::from_micros(50)),
            Pool::new(8, Duration::from_micros(50)),
        ];
        for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
            let m = rand_tensor(rows, cols, 7 + i as u32);
            let v = rand_vec(cols, 99 + i as u32);
            let mut reference = vec![0.0f32; rows];
            f32_rows(&mut reference, &m.data, &v, cols);
            for pool in &pools {
                let mut out = vec![f32::NAN; rows];
                gemv_f32_with(pool, &mut out, &m, &v);
                assert_eq!(bits(&out), bits(&reference),
                    "shape {}x{} threads={}", rows, cols, pool.threads());
            }
        }
    }

    #[test]
    fn pooled_f32_gemv_matches_legacy_mat_vec_mul_into() {
        // Legacy dispatch (single-thread below 4096 rows, scoped threads above).
        for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
            let m = rand_tensor(rows, cols, 31 + i as u32);
            let v = rand_vec(cols, 5 + i as u32);
            let mut legacy = vec![0.0f32; rows];
            crate::domain::tensor::mat_vec_mul_into(&mut legacy, &m, &v);
            let mut pooled = vec![0.0f32; rows];
            gemv_f32_into(&mut pooled, &m, &v);
            assert_eq!(bits(&pooled), bits(&legacy), "shape {}x{}", rows, cols);
        }
    }

    #[test]
    fn q8_quantization_error_bounded_by_half_step() {
        let t = rand_tensor(64, 256, 3);
        for group in [256usize, 64, 32] {
            let q = Q8wMatrix::quantize(&t, group);
            assert_eq!(q.group, group);
            let mut row = vec![0.0f32; 256];
            for r in 0..64 {
                q.dequant_row_into(&mut row, r);
                for (g, (w, d)) in t.data[r * 256..(r + 1) * 256].chunks(group)
                    .zip(row.chunks(group)).enumerate()
                {
                    let step = w.iter().fold(0.0f32, |a, &x| a.max(x.abs())) / 127.0;
                    for (a, b) in w.iter().zip(d) {
                        assert!((a - b).abs() <= step * 0.5 + 1e-7, "group={} r={} g={}", group, r, g);
                    }
                }
            }
        }
    }

    #[test]
    fn q8_invalid_group_falls_back_to_per_row() {
        let t = rand_tensor(4, 100, 9);
        assert_eq!(Q8wMatrix::quantize(&t, 64).group, 100); // 64 ∤ 100
        assert_eq!(Q8wMatrix::quantize(&t, 12).group, 100); // not a multiple of 8
        assert_eq!(Q8wMatrix::quantize(&t, 0).group, 100);
    }

    #[test]
    fn q8_zero_group_is_safe() {
        let mut t = rand_tensor(8, 64, 11);
        for x in &mut t.data[64..128] { *x = 0.0; } // row 1 all zero
        for x in &mut t.data[128..136] { *x = 0.0; } // one zero group in row 2
        let q = Q8wMatrix::quantize(&t, 8);
        let v = rand_vec(64, 2);
        let mut out = vec![1.0f32; 8];
        q8w_rows_scalar(&mut out, &rows_of(&q), &v);
        assert_eq!(out[1], 0.0);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn q8_simd_and_scalar_kernels_bit_identical() {
        if !simd_avx2() {
            eprintln!("SKIP: AVX2 unavailable or REDUKT_NO_SIMD=1");
            return;
        }
        for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
            let t = rand_tensor(rows, cols, 17 + i as u32);
            let v: Vec<f32> = rand_vec(cols, 23 + i as u32).iter().map(|x| x * 40.0).collect();
            for group in [cols, 64, 8] {
                let q = Q8wMatrix::quantize(&t, group);
                let mut simd = vec![0.0f32; rows];
                let mut scalar = vec![0.0f32; rows];
                #[cfg(target_arch = "x86_64")]
                unsafe { q8w_rows_avx2(&mut simd, &rows_of(&q), &v) };
                q8w_rows_scalar(&mut scalar, &rows_of(&q), &v);
                assert_eq!(bits(&simd), bits(&scalar), "shape {}x{} group={}", rows, cols, q.group);
            }
        }
    }

    #[test]
    fn q8_gemv_close_to_f32_reference() {
        let (rows, cols) = (512, 768);
        let t = rand_tensor(rows, cols, 41);
        let v = rand_vec(cols, 43);
        let mut f = vec![0.0f32; rows];
        f32_rows(&mut f, &t.data, &v, cols);
        for group in [cols, 64] {
            let q = Q8wMatrix::quantize(&t, group);
            let mut o = vec![0.0f32; rows];
            gemv_q8w_into(&mut o, &q, &v);
            // Per-element error ≤ step/2 → |Δdot| ≤ Σ|v|·max_step/2 (hard bound).
            let sum_abs_v: f32 = v.iter().map(|x| x.abs()).sum();
            for r in 0..rows {
                let amax = t.data[r * cols..(r + 1) * cols].iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                let bound = sum_abs_v * amax / 127.0 * 0.5 + 1e-3;
                assert!((f[r] - o[r]).abs() <= bound, "row {} f32={} q8={} bound={}", r, f[r], o[r], bound);
            }
        }
    }

    #[test]
    fn pooled_q8_gemv_bit_identical_across_thread_counts() {
        let (rows, cols) = (3072, 768);
        let t = rand_tensor(rows, cols, 51);
        let v = rand_vec(cols, 53);
        for group in [cols, 64] {
            let q = Q8wMatrix::quantize(&t, group);
            let mut reference = vec![0.0f32; rows];
            q8w_rows(&mut reference, &rows_of(&q), &v);
            for n in [1usize, 2, 5, 12] {
                let pool = Pool::new(n, Duration::from_micros(50));
                let mut out = vec![f32::NAN; rows];
                gemv_q8w_with(&pool, &mut out, &q, &v);
                assert_eq!(bits(&out), bits(&reference), "threads={} group={}", n, group);
            }
        }
    }

    /// R-100: every stream of the multi-stream kernels is bit-identical to the
    /// single-stream GEMV of that stream, for any N and any thread count.
    #[test]
    fn multi_stream_f32_bit_identical_to_single_stream() {
        let pools = [
            Pool::new(1, Duration::from_micros(0)),
            Pool::new(3, Duration::from_micros(50)),
            Pool::new(8, Duration::from_micros(50)),
        ];
        for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
            let m = rand_tensor(rows, cols, 61 + i as u32);
            for n in [1usize, 2, 3, 5, 8] {
                let xs: Vec<Vec<f32>> = (0..n).map(|s| rand_vec(cols, 100 + 7 * s as u32 + i as u32)).collect();
                let refs: Vec<Vec<f32>> = xs.iter().map(|x| {
                    let mut r = vec![0.0f32; rows];
                    f32_rows(&mut r, &m.data, x, cols);
                    r
                }).collect();
                for pool in &pools {
                    let mut outs: Vec<Vec<f32>> = vec![vec![f32::NAN; rows]; n];
                    let mut out_refs: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
                    let x_refs: Vec<&[f32]> = xs.iter().map(|x| x.as_slice()).collect();
                    gemv_f32_multi_with(pool, &mut out_refs, &m, &x_refs);
                    for s in 0..n {
                        assert_eq!(bits(&outs[s]), bits(&refs[s]),
                            "shape {}x{} n={} s={} threads={}", rows, cols, n, s, pool.threads());
                    }
                }
            }
        }
    }

    #[test]
    fn multi_stream_q8_bit_identical_to_single_stream() {
        let pools = [Pool::new(1, Duration::from_micros(0)), Pool::new(5, Duration::from_micros(50))];
        for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
            let t = rand_tensor(rows, cols, 71 + i as u32);
            for group in [cols, 64] {
                let q = Q8wMatrix::quantize(&t, group);
                for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17] {
                    let xs: Vec<Vec<f32>> = (0..n).map(|s| rand_vec(cols, 200 + 3 * s as u32 + i as u32)).collect();
                    let refs: Vec<Vec<f32>> = xs.iter().map(|x| {
                        let mut r = vec![0.0f32; rows];
                        q8w_rows(&mut r, &rows_of(&q), x);
                        r
                    }).collect();
                    for pool in &pools {
                        let mut outs: Vec<Vec<f32>> = vec![vec![f32::NAN; rows]; n];
                        let mut out_refs: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
                        let x_refs: Vec<&[f32]> = xs.iter().map(|x| x.as_slice()).collect();
                        gemv_q8w_multi_with(pool, &mut out_refs, &q, &x_refs);
                        for s in 0..n {
                            assert_eq!(bits(&outs[s]), bits(&refs[s]),
                                "shape {}x{} group={} n={} s={}", rows, cols, q.group, n, s);
                        }
                    }
                }
            }
        }
    }

    /// Kernel-level lockstep benchmark: weight GB/s of single GEMV vs N-stream
    /// GEMM on Mamba-130M shapes. cargo test --release bench_multi -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_multi_stream_gemm() {
        use std::time::Instant;
        let pool = par_pool::global();
        for &(rows, cols, name) in &[(3072usize, 768usize, "in_proj"), (768, 1536, "out_proj"), (50280, 768, "head")] {
            let t = rand_tensor(rows, cols, 5);
            let q = Q8wMatrix::quantize(&t, 64);
            for n in [1usize, 2, 4, 8, 16, 32] {
                let xs: Vec<Vec<f32>> = (0..n).map(|s| rand_vec(cols, 9 + s as u32)).collect();
                let x_refs: Vec<&[f32]> = xs.iter().map(|x| x.as_slice()).collect();
                let mut outs: Vec<Vec<f32>> = vec![vec![0.0; rows]; n];
                let reps = (200 / n).max(5);
                let mut best = [f64::MAX; 4];
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..reps { for s in 0..n { gemv_q8w_with(pool, &mut outs[s], &q, &xs[s]); } }
                    best[0] = best[0].min(t0.elapsed().as_secs_f64() / reps as f64);
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        let mut o: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
                        gemv_q8w_multi_with(pool, &mut o, &q, &x_refs);
                    }
                    best[1] = best[1].min(t0.elapsed().as_secs_f64() / reps as f64);
                    let t0 = Instant::now();
                    for _ in 0..reps { for s in 0..n { gemv_f32_with(pool, &mut outs[s], &t, &xs[s]); } }
                    best[2] = best[2].min(t0.elapsed().as_secs_f64() / reps as f64);
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        let mut o: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
                        gemv_f32_multi_with(pool, &mut o, &t, &x_refs);
                    }
                    best[3] = best[3].min(t0.elapsed().as_secs_f64() / reps as f64);
                }
                let (qb, fb) = (q.bytes() as f64, (rows * cols * 4) as f64);
                let gbs = |bytes: f64, secs: f64| bytes * n as f64 / secs / 1e9;
                eprintln!("{:8} n={:2} | Q8: {:7.3} ms ({:5.1} GB/s) vs GEMM {:7.3} ms ({:5.1} GB/s eff) {:4.2}x",
                    name, n, best[0] * 1e3, gbs(qb, best[0]), best[1] * 1e3, gbs(qb, best[1]), best[0] / best[1]);
                eprintln!("{:8} n={:2} | F32: {:7.3} ms ({:5.1} GB/s) vs GEMM {:7.3} ms ({:5.1} GB/s eff) {:4.2}x",
                    name, n, best[2] * 1e3, gbs(fb, best[2]), best[3] * 1e3, gbs(fb, best[3]), best[2] / best[3]);
            }
        }
    }

    #[test]
    fn rows_per_block_is_multiple_of_four_and_bounded() {
        for &(rows, row_bytes, threads) in &[
            (3072usize, 3072usize, 12usize), (768, 6144, 12), (50280, 3072, 12),
            (3072, 768, 12), (50280, 768, 4), (16, 4, 64), (1_000_000, 4, 2),
        ] {
            let rpb = rows_per_block(rows, row_bytes, threads);
            assert_eq!(rpb % 4, 0);
            assert!((rows + rpb - 1) / rpb <= MAX_BLOCKS);
        }
    }
}
