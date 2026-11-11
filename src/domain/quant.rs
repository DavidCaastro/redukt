//! Quantization module — reusable, empirically auto-selecting.
//!
//! Provides a unified `QuantMatrix` enum that wraps f32, Q8, and Q4 weight
//! matrices behind a single interface. At load time, benchmarks ALL available
//! quantization levels on the actual data and picks the best balance of
//! quality, speed, and memory.
//!
//! Supports two quantization policies:
//! - `Uniform(level)`: same level for all matrices (fast, good for small models)
//! - `PerMatrix`: each matrix independently benchmarked and auto-selected
//!
//! Usage:
//!   // Uniform: benchmark once, apply everywhere
//!   let (policy, results) = QuantPolicy::auto_detect(&sample_matrix);
//!   let qm = quantize_matrix(&tensor, &policy);
//!
//!   // Per-matrix: each matrix finds its own optimal level
//!   let policy = QuantPolicy::PerMatrix;
//!   let qm = quantize_matrix(&tensor, &policy);

use crate::domain::tensor::*;

/// All quantization levels to benchmark.
const ALL_LEVELS: &[QuantLevel] = &[
    QuantLevel::F32,
    QuantLevel::Q8,
    QuantLevel::Q4 { group_size: 64 },
];

// GGML type IDs for native quantized matmul dispatch.
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;
const GGML_Q5_K: u32 = 13;
const GGML_Q6_K: u32 = 14;

/// Block size and byte size for supported native GGML quantization types.
/// Returns (block_size, block_bytes) or None if unsupported.
pub fn ggml_block_info(ggml_type: u32) -> Option<(usize, usize)> {
    match ggml_type {
        GGML_Q8_0 => Some((32, 34)),     // f16 scale + 32 i8
        GGML_Q4_K => Some((256, 144)),    // f16 d + f16 dmin + scales[12] + qs[128]
        GGML_Q5_K => Some((256, 176)),    // f16 d + f16 dmin + scales[12] + qh[32] + qs[128]
        GGML_Q6_K => Some((256, 210)),    // ql[128] + qh[64] + scales[16] + f16 d
        _ => None,
    }
}

/// Quantization precision level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantLevel {
    F32,
    Q8,
    Q4 { group_size: usize },
}

impl QuantLevel {
    pub fn name(&self) -> &'static str {
        match self {
            QuantLevel::F32 => "F32",
            QuantLevel::Q8 => "Q8",
            QuantLevel::Q4 { .. } => "Q4",
        }
    }
}

/// Native GGUF quantized tensor — stores raw quantized bytes, no dequantization.
/// Supports mat_vec directly on Q8_0, Q4_K, Q6_K block formats.
pub struct GgufQuantTensor {
    pub raw: Vec<u8>,      // raw quantized bytes (owned, copied from GGUF file)
    pub ggml_type: u32,    // GGML type ID (8=Q8_0, 12=Q4_K, 14=Q6_K, etc.)
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,  // bytes per row in the quantized format
}

impl GgufQuantTensor {
    /// Memory usage in bytes.
    pub fn mem_bytes(&self) -> usize {
        self.raw.len()
    }

    /// Matrix-vector multiply directly on quantized blocks (no dequantization).
    pub fn mat_vec_mul(&self, vec: &[f32], out: &mut [f32]) {
        match self.ggml_type {
            GGML_Q8_0 => self.mat_vec_q8_0(out, vec),
            GGML_Q4_K => self.mat_vec_q4_k(out, vec),
            GGML_Q5_K => self.mat_vec_q5_k(out, vec),
            GGML_Q6_K => self.mat_vec_q6_k(out, vec),
            _ => panic!("[quant] unsupported GGML type {} for native matmul", self.ggml_type),
        }
    }

    /// GGML type name for display.
    pub fn type_name(&self) -> &'static str {
        match self.ggml_type {
            GGML_Q8_0 => "Q8_0",
            GGML_Q4_K => "Q4_K",
            GGML_Q5_K => "Q5_K",
            GGML_Q6_K => "Q6_K",
            _ => "unknown",
        }
    }

    fn mat_vec_q8_0(&self, out: &mut [f32], vec: &[f32]) {
        let n_blocks = self.cols / 32;
        for r in 0..self.rows {
            let row_start = r * self.row_bytes;
            let mut sum = 0.0f32;
            let mut pos = row_start;
            let mut vi = 0;
            for _ in 0..n_blocks {
                let scale = f16_to_f32_q(u16::from_le_bytes(
                    self.raw[pos..pos + 2].try_into().unwrap()
                ));
                pos += 2;
                let mut block_dot = 0.0f32;
                for j in 0..32 {
                    block_dot += self.raw[pos + j] as i8 as f32 * vec[vi + j];
                }
                sum += scale * block_dot;
                pos += 32;
                vi += 32;
            }
            out[r] = sum;
        }
    }

    fn mat_vec_q4_k(&self, out: &mut [f32], vec: &[f32]) {
        let n_blocks = self.cols / 256;
        for r in 0..self.rows {
            let row_start = r * self.row_bytes;
            let mut sum = 0.0f32;
            let mut pos = row_start;
            let mut vi = 0;

            for _ in 0..n_blocks {
                let d = f16_to_f32_q(u16::from_le_bytes(self.raw[pos..pos + 2].try_into().unwrap()));
                let dmin = f16_to_f32_q(u16::from_le_bytes(self.raw[pos + 2..pos + 4].try_into().unwrap()));
                let (sc, mn) = unpack_k4_scales_q(&self.raw[pos + 4..pos + 16]);
                let qs = &self.raw[pos + 16..pos + 144];

                let mut is = 0usize;
                let mut qo = 0usize;

                for _ in 0..4 {
                    let d1 = d * sc[is] as f32;
                    let m1 = dmin * mn[is] as f32;
                    let d2 = d * sc[is + 1] as f32;
                    let m2 = dmin * mn[is + 1] as f32;

                    let mut dot_lo = 0.0f32;
                    let mut sum_lo = 0.0f32;
                    for l in 0..32 {
                        let q = (qs[qo + l] & 0x0F) as f32;
                        dot_lo += q * vec[vi];
                        sum_lo += vec[vi];
                        vi += 1;
                    }
                    sum += d1 * dot_lo - m1 * sum_lo;

                    let mut dot_hi = 0.0f32;
                    let mut sum_hi = 0.0f32;
                    for l in 0..32 {
                        let q = ((qs[qo + l] >> 4) & 0x0F) as f32;
                        dot_hi += q * vec[vi];
                        sum_hi += vec[vi];
                        vi += 1;
                    }
                    sum += d2 * dot_hi - m2 * sum_hi;

                    qo += 32;
                    is += 2;
                }
                pos += 144;
            }
            out[r] = sum;
        }
    }

    fn mat_vec_q5_k(&self, out: &mut [f32], vec: &[f32]) {
        let n_blocks = self.cols / 256;
        for r in 0..self.rows {
            let row_start = r * self.row_bytes;
            let mut sum = 0.0f32;
            let mut pos = row_start;
            let mut vi = 0;

            for _ in 0..n_blocks {
                let d = f16_to_f32_q(u16::from_le_bytes(self.raw[pos..pos + 2].try_into().unwrap()));
                let dmin = f16_to_f32_q(u16::from_le_bytes(self.raw[pos + 2..pos + 4].try_into().unwrap()));
                let (sc, mn) = unpack_k4_scales_q(&self.raw[pos + 4..pos + 16]);
                let qh = &self.raw[pos + 16..pos + 48];
                let qs = &self.raw[pos + 48..pos + 176];

                let mut is = 0usize;
                let mut qo = 0usize;
                let mut elem = 0usize;

                for _ in 0..4 {
                    let d1 = d * sc[is] as f32;
                    let m1 = dmin * mn[is] as f32;
                    let d2 = d * sc[is + 1] as f32;
                    let m2 = dmin * mn[is + 1] as f32;

                    for l in 0..32 {
                        let qh_bit = ((qh[elem / 8] >> (elem % 8)) & 1) as f32;
                        let q = (qs[qo + l] & 0x0F) as f32 + qh_bit * 16.0;
                        sum += (q * d1 - m1) * vec[vi];
                        vi += 1;
                        elem += 1;
                    }
                    for l in 0..32 {
                        let qh_bit = ((qh[elem / 8] >> (elem % 8)) & 1) as f32;
                        let q = ((qs[qo + l] >> 4) & 0x0F) as f32 + qh_bit * 16.0;
                        sum += (q * d2 - m2) * vec[vi];
                        vi += 1;
                        elem += 1;
                    }

                    qo += 32;
                    is += 2;
                }
                pos += 176;
            }
            out[r] = sum;
        }
    }

    fn mat_vec_q6_k(&self, out: &mut [f32], vec: &[f32]) {
        let n_blocks = self.cols / 256;
        for r in 0..self.rows {
            let row_start = r * self.row_bytes;
            let mut sum = 0.0f32;
            let mut pos = row_start;
            let mut vi = 0;

            for _ in 0..n_blocks {
                let ql = &self.raw[pos..pos + 128];
                let qh = &self.raw[pos + 128..pos + 192];
                let scales = &self.raw[pos + 192..pos + 208];
                let d = f16_to_f32_q(u16::from_le_bytes(
                    self.raw[pos + 208..pos + 210].try_into().unwrap()
                ));

                for half in 0..2usize {
                    let ql_base = half * 64;
                    let qh_base = half * 32;
                    let sc_base = half * 8;

                    for l in 0..32 {
                        let sc_val = scales[sc_base + l / 16] as i8 as f32;
                        let lo4 = (ql[ql_base + l] & 0x0F) as i32;
                        let hi2 = ((qh[qh_base + l] >> 0) & 3) as i32;
                        let q = (lo4 | (hi2 << 4)) - 32;
                        sum += q as f32 * d * sc_val * vec[vi];
                        vi += 1;
                    }
                    for l in 0..32 {
                        let sc_val = scales[sc_base + 2 + l / 16] as i8 as f32;
                        let lo4 = (ql[ql_base + 32 + l] & 0x0F) as i32;
                        let hi2 = ((qh[qh_base + l] >> 2) & 3) as i32;
                        let q = (lo4 | (hi2 << 4)) - 32;
                        sum += q as f32 * d * sc_val * vec[vi];
                        vi += 1;
                    }
                    for l in 0..32 {
                        let sc_val = scales[sc_base + 4 + l / 16] as i8 as f32;
                        let lo4 = ((ql[ql_base + l] >> 4) & 0x0F) as i32;
                        let hi2 = ((qh[qh_base + l] >> 4) & 3) as i32;
                        let q = (lo4 | (hi2 << 4)) - 32;
                        sum += q as f32 * d * sc_val * vec[vi];
                        vi += 1;
                    }
                    for l in 0..32 {
                        let sc_val = scales[sc_base + 6 + l / 16] as i8 as f32;
                        let lo4 = ((ql[ql_base + 32 + l] >> 4) & 0x0F) as i32;
                        let hi2 = ((qh[qh_base + l] >> 6) & 3) as i32;
                        let q = (lo4 | (hi2 << 4)) - 32;
                        sum += q as f32 * d * sc_val * vec[vi];
                        vi += 1;
                    }
                }
                pos += 210;
            }
            out[r] = sum;
        }
    }
}

/// Unified quantized matrix — dispatches to the right kernel.
/// Enum-based (not trait object) to avoid dynamic dispatch in hot loops.
pub enum QuantMatrix {
    F32(Tensor),
    Q8(Q8Tensor),
    Q4(Q4Tensor),
    /// Native GGUF quantized — no dequantization, mat_vec on raw blocks.
    Gguf(GgufQuantTensor),
}

impl QuantMatrix {
    /// Quantize an f32 (rows, cols) tensor to the specified level.
    pub fn from_f32(t: &Tensor, level: QuantLevel) -> Self {
        match level {
            QuantLevel::F32 => QuantMatrix::F32(t.clone()),
            QuantLevel::Q8 => QuantMatrix::Q8(Q8Tensor::from_f32(t)),
            QuantLevel::Q4 { group_size } => QuantMatrix::Q4(Q4Tensor::from_f32(t, group_size)),
        }
    }

    /// Matrix-vector multiply: dispatches to the optimal kernel.
    pub fn mat_vec_mul(&self, vec: &Tensor) -> Tensor {
        match self {
            QuantMatrix::F32(t) => mat_vec_mul(t, vec),
            QuantMatrix::Q8(t) => q8_mat_vec_mul(t, vec),
            QuantMatrix::Q4(t) => q4_mat_vec_mul(t, vec),
            QuantMatrix::Gguf(t) => {
                let mut out = vec![0.0f32; t.rows];
                t.mat_vec_mul(&vec.data, &mut out);
                Tensor::from_data(out, std::vec![t.rows])
            }
        }
    }

    /// Matrix-vector multiply into pre-allocated output buffer (zero allocation).
    pub fn mat_vec_mul_into(&self, out: &mut [f32], vec: &[f32], v_q: &mut [i8]) {
        match self {
            QuantMatrix::F32(t) => mat_vec_mul_into(out, t, vec),
            QuantMatrix::Q8(t) => q8_mat_vec_mul_into(out, t, vec, v_q),
            QuantMatrix::Q4(t) => q4_mat_vec_mul_into(out, t, vec, v_q),
            QuantMatrix::Gguf(t) => {
                t.mat_vec_mul(vec, out);
            }
        }
    }

    /// Memory usage in bytes.
    pub fn mem_bytes(&self) -> usize {
        match self {
            QuantMatrix::F32(t) => t.data.len() * 4,
            QuantMatrix::Q8(t) => t.mem_bytes(),
            QuantMatrix::Q4(t) => t.mem_bytes(),
            QuantMatrix::Gguf(t) => t.mem_bytes(),
        }
    }

    /// Rows and cols of the underlying matrix.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            QuantMatrix::F32(t) => (t.shape[0], t.shape[1]),
            QuantMatrix::Q8(t) => (t.rows, t.cols),
            QuantMatrix::Q4(t) => (t.rows, t.cols),
            QuantMatrix::Gguf(t) => (t.rows, t.cols),
        }
    }

    /// Dequantize to f64 for spectral analysis. Returns (data, rows, cols).
    pub fn dequantize_f64(&self) -> (Vec<f64>, usize, usize) {
        match self {
            QuantMatrix::F32(t) => {
                let data: Vec<f64> = t.data.iter().map(|&x| x as f64).collect();
                (data, t.shape[0], t.shape[1])
            }
            QuantMatrix::Q8(q) => {
                let mut data = Vec::with_capacity(q.rows * q.cols);
                for r in 0..q.rows {
                    let scale = q.scales[r] as f64;
                    let row_start = r * q.cols;
                    for c in 0..q.cols {
                        data.push(q.q_data[row_start + c] as f64 * scale);
                    }
                }
                (data, q.rows, q.cols)
            }
            QuantMatrix::Q4(q) => {
                let mut data = Vec::with_capacity(q.rows * q.cols);
                for r in 0..q.rows {
                    for c in 0..q.cols {
                        let byte_idx = (r * q.cols + c) / 2;
                        let packed = q.packed[byte_idx];
                        let val = if (r * q.cols + c) % 2 == 0 {
                            (packed & 0x0F) as i8 - 8
                        } else {
                            ((packed >> 4) & 0x0F) as i8 - 8
                        };
                        let group = (r * q.cols + c) / q.group_size;
                        let scale = if group < q.scales.len() { q.scales[group] as f64 } else { 1.0 };
                        data.push(val as f64 * scale);
                    }
                }
                (data, q.rows, q.cols)
            }
            QuantMatrix::Gguf(_) => {
                panic!("[quant] dequantize_f64 not supported for native GGUF tensors — use F32 loading path for spectral analysis")
            }
        }
    }

    /// Display name for the quantization format.
    pub fn format_name(&self) -> &'static str {
        match self {
            QuantMatrix::F32(_) => "F32",
            QuantMatrix::Q8(_) => "Q8",
            QuantMatrix::Q4(_) => "Q4",
            QuantMatrix::Gguf(t) => t.type_name(),
        }
    }
}

/// Quantization policy — determines how matrices select their precision.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum QuantPolicy {
    /// Same quantization level for all matrices (fast load, good for small models).
    Uniform(QuantLevel),
    /// Each matrix independently benchmarks all levels and picks its optimal one.
    /// Adds load-time overhead (~1s per 72 matrices) but finds per-matrix optima.
    PerMatrix,
}

impl QuantPolicy {
    /// Auto-detect the best uniform policy by benchmarking a sample matrix.
    /// Returns the policy and benchmark results for reporting.
    pub fn auto_detect(sample_matrix: &Tensor) -> (Self, Vec<BenchResult>) {
        let (level, results) = select_best(sample_matrix);
        (QuantPolicy::Uniform(level), results)
    }
}

/// Quantize a matrix according to the given policy.
/// Returns the quantized matrix and the level that was selected.
pub fn quantize_matrix(tensor: &Tensor, policy: &QuantPolicy) -> (QuantMatrix, QuantLevel) {
    match policy {
        QuantPolicy::Uniform(level) => {
            (QuantMatrix::from_f32(tensor, *level), *level)
        }
        QuantPolicy::PerMatrix => {
            let (level, _) = select_best(tensor);
            (QuantMatrix::from_f32(tensor, level), level)
        }
    }
}

/// Result of benchmarking a single quantization level.
pub struct BenchResult {
    pub level: QuantLevel,
    pub max_abs_error: f32,
    pub mean_rel_error: f32,
    pub ns_per_mul: u64,
    pub mem_bytes: usize,
    pub quality_pass: bool,
    pub score: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_kb = self.mem_bytes as f64 / 1024.0;
        write!(f, "{:<3}  err={:.6}  rel={:.3}%  {:.1}us/mul  {:.0}KB  score={:.2}  {}",
               self.level.name(),
               self.max_abs_error,
               self.mean_rel_error * 100.0,
               self.ns_per_mul as f64 / 1000.0,
               mem_kb,
               self.score,
               if self.quality_pass { "PASS" } else { "FAIL" })
    }
}

/// Benchmark all quantization levels on a sample matrix and select the best.
///
/// Runs each level through:
/// 1. Quality test: quantized matmul vs f32 reference (error thresholds)
/// 2. Speed test: time N matmuls, measure ns/mul
/// 3. Memory: measure quantized size
///
/// Score = speed_factor * memory_factor (higher is better, only among passing levels).
/// Returns the best level and all benchmark results for reporting.
pub fn select_best(sample_matrix: &Tensor) -> (QuantLevel, Vec<BenchResult>) {
    let cols = sample_matrix.shape[1];
    let test_vec = make_test_vector(sample_matrix, cols);

    // F32 reference result (for quality comparison)
    let ref_result = mat_vec_mul(sample_matrix, &test_vec);

    // F32 speed baseline (for relative scoring)
    let f32_ns = bench_matmul_ns(sample_matrix, &test_vec);

    let f32_mem = sample_matrix.data.len() * 4;

    let mut results = Vec::with_capacity(ALL_LEVELS.len());

    for &level in ALL_LEVELS {
        // Quantize
        let qm = QuantMatrix::from_f32(sample_matrix, level);
        let q_result = qm.mat_vec_mul(&test_vec);

        // Quality: error vs f32
        let (max_abs, mean_rel) = compute_errors(&ref_result.data, &q_result.data);
        let quality_pass = max_abs < 0.5 && mean_rel < 0.05;

        // Speed: benchmark
        let ns = bench_qm_ns(&qm, &test_vec);

        // Memory
        let mem = qm.mem_bytes();

        // Score: speed_factor * memory_factor (both relative to f32)
        // Higher = better. Only meaningful for passing levels.
        let speed_factor = f32_ns as f64 / ns.max(1) as f64;
        let memory_factor = f32_mem as f64 / mem.max(1) as f64;
        let score = if quality_pass { speed_factor * memory_factor.sqrt() } else { 0.0 };

        results.push(BenchResult {
            level,
            max_abs_error: max_abs,
            mean_rel_error: mean_rel,
            ns_per_mul: ns,
            mem_bytes: mem,
            quality_pass,
            score,
        });
    }

    // Pick the passing level with highest score
    let best = results.iter()
        .filter(|r| r.quality_pass)
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
        .map(|r| r.level)
        .unwrap_or(QuantLevel::F32);

    (best, results)
}

/// Compute max absolute and mean relative error between two vectors.
fn compute_errors(reference: &[f32], quantized: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut sum_rel = 0.0f32;
    let n = reference.len();
    for i in 0..n {
        let abs_err = (reference[i] - quantized[i]).abs();
        if abs_err > max_abs { max_abs = abs_err; }
        let denom = reference[i].abs().max(1e-8);
        sum_rel += abs_err / denom;
    }
    (max_abs, sum_rel / n as f32)
}

/// Benchmark f32 mat_vec_mul: run multiple iterations, return median ns per call.
fn bench_matmul_ns(matrix: &Tensor, vec: &Tensor) -> u64 {
    let n_iters = 20;
    let mut times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = std::time::Instant::now();
        let _ = mat_vec_mul(matrix, vec);
        times.push(t0.elapsed().as_nanos() as u64);
    }
    times.sort();
    times[n_iters / 2] // median
}

/// Benchmark QuantMatrix mat_vec_mul: run multiple iterations, return median ns.
fn bench_qm_ns(qm: &QuantMatrix, vec: &Tensor) -> u64 {
    let n_iters = 20;
    let mut times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = std::time::Instant::now();
        let _ = qm.mat_vec_mul(vec);
        times.push(t0.elapsed().as_nanos() as u64);
    }
    times.sort();
    times[n_iters / 2] // median
}

// --- Helper functions for native GGUF quantized matmul ---

/// f16 → f32 conversion (local copy to avoid domain→infrastructure dependency).
fn f16_to_f32_q(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut f = frac as f32 / 1024.0;
        f *= 1.0 / 16384.0; // 2^-14, exact (no powi: R-104 F3 lint)
        if sign == 1 { -f } else { f }
    } else if exp == 31 {
        if frac == 0 { f32::from_bits((sign << 31) | 0x7F800000) } else { f32::NAN }
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
    }
}

/// Unpack 6-bit K-quant scales (local copy for Q4_K/Q5_K kernels).
fn unpack_k4_scales_q(s: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for i in 0..4 {
        sc[i] = s[i] & 63;
        mn[i] = s[i + 4] & 63;
    }
    for i in 0..4 {
        sc[i + 4] = (s[i + 8] & 0x0F) | ((s[i] >> 6) << 4);
        mn[i + 4] = (s[i + 8] >> 4)    | ((s[i + 4] >> 6) << 4);
    }
    (sc, mn)
}

/// Generate a deterministic test vector from matrix content.
fn make_test_vector(matrix: &Tensor, cols: usize) -> Tensor {
    let mut data = vec![0.0f32; cols];
    let rows = matrix.shape[0];
    for c in 0..cols {
        let r = c % rows;
        data[c] = matrix.data[r * cols + c];
    }
    let max_val = data.iter().fold(0.0f32, |a, &x| a.max(x.abs())).max(1e-8);
    for v in &mut data {
        *v /= max_val;
    }
    Tensor::from_data(data, vec![cols])
}
