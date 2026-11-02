#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Dense tensor backed by a contiguous f32 buffer.
/// Shapes are row-major: a (R, C) matrix stores R*C floats,
/// row 0 first, then row 1, etc.
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// View as 1D
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Create from raw f32 data + shape
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "from_data: data len {} != shape product {}", data.len(), n);
        Self { data, shape }
    }
}

/// Transpose a 2D tensor: (R, C) → (C, R).
pub fn transpose_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2, "transpose_2d: expected 2D tensor");
    let r = t.shape[0];
    let c = t.shape[1];
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = t.data[i * c + j];
        }
    }
    Tensor::from_data(out, vec![c, r])
}

// ---- Core math operations ----

/// y = mat @ vec, where mat is (rows, cols) and vec is (cols,)
/// Result is (rows,). Optimized with 4-way ILP and reslicing.
pub fn mat_vec_mul(mat: &Tensor, vec: &Tensor) -> Tensor {
    assert_eq!(mat.shape.len(), 2);
    assert_eq!(vec.shape.len(), 1);
    let rows = mat.shape[0];
    let cols = mat.shape[1];
    assert_eq!(vec.shape[0], cols);

    let mut out = vec![0.0f32; rows];
    let m = mat.as_slice();
    let v = &vec.as_slice()[..cols];

    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let row0 = &m[r * cols..(r + 1) * cols];
        let row1 = &m[(r + 1) * cols..(r + 2) * cols];
        let row2 = &m[(r + 2) * cols..(r + 3) * cols];
        let row3 = &m[(r + 3) * cols..(r + 4) * cols];
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for c in 0..cols {
            let vc = v[c];
            s0 += row0[c] * vc;
            s1 += row1[c] * vc;
            s2 += row2[c] * vc;
            s3 += row3[c] * vc;
        }
        out[r] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
    }
    for r in rows_4..rows {
        let row = &m[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += row[c] * v[c];
        }
        out[r] = sum;
    }
    Tensor::from_data(out, vec![rows])
}

// ---- Q8 quantized tensor ----

/// Row-quantized int8 weight matrix.
/// Each row has one f32 scale factor: weight[r][c] ≈ q_data[r*cols+c] * scale[r]
/// Memory: rows*cols bytes + rows*4 bytes ≈ rows*cols + 0.5% overhead
///
/// Note: block-32 (llama.cpp Q8_0 style) was tested and found WORSE for RWKV-7:
/// +0.0378 BPB and -34% speed on 100KB. RWKV has uniform weight distributions
/// (RWKVQuant, ICML 2025) so per-row scaling is already near-optimal, and the
/// coarser quantization acts as beneficial implicit regularization. See R14.
pub struct Q8Tensor {
    pub q_data: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Q8Tensor {
    /// Quantize an f32 (rows, cols) tensor to Q8 per-row.
    pub fn from_f32(t: &Tensor) -> Self {
        assert_eq!(t.shape.len(), 2);
        let rows = t.shape[0];
        let cols = t.shape[1];
        let mut q_data = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows];

        for r in 0..rows {
            let row_start = r * cols;
            let row = &t.data[row_start..row_start + cols];

            let abs_max = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            scales[r] = scale;
            for c in 0..cols {
                let quantized = (row[c] * inv_scale).round();
                q_data[row_start + c] = quantized.clamp(-127.0, 127.0) as i8;
            }
        }

        Self { q_data, scales, rows, cols }
    }

    /// Memory usage in bytes
    pub fn mem_bytes(&self) -> usize {
        self.q_data.len() + self.scales.len() * 4
    }
}

/// y = Q8_mat @ f32_vec using integer accumulation (allocating, for benchmarks).
/// Hot path uses q8_mat_vec_mul_into instead.
pub fn q8_mat_vec_mul(mat: &Q8Tensor, vec: &Tensor) -> Tensor {
    q8_mat_vec_mul_scalar(mat, vec)
}

/// Scalar fallback: i8*i8 → i32 with 4-row ILP.
fn q8_mat_vec_mul_scalar(mat: &Q8Tensor, vec: &Tensor) -> Tensor {
    assert_eq!(vec.shape.len(), 1);
    let rows = mat.rows;
    let cols = mat.cols;
    assert_eq!(vec.shape[0], cols);

    let v = &vec.as_slice()[..cols];
    let (v_q, v_scale) = quantize_vec_i8(v);

    let mut out = vec![0.0f32; rows];
    let q = &mat.q_data;
    let scales = &mat.scales;

    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let q0 = &q[r * cols..(r + 1) * cols];
        let q1 = &q[(r + 1) * cols..(r + 2) * cols];
        let q2 = &q[(r + 2) * cols..(r + 3) * cols];
        let q3 = &q[(r + 3) * cols..(r + 4) * cols];
        let (mut s0, mut s1, mut s2, mut s3) = (0i32, 0i32, 0i32, 0i32);
        for c in 0..cols {
            let vc = v_q[c] as i32;
            s0 += q0[c] as i32 * vc;
            s1 += q1[c] as i32 * vc;
            s2 += q2[c] as i32 * vc;
            s3 += q3[c] as i32 * vc;
        }
        out[r] = s0 as f32 * (scales[r] * v_scale);
        out[r + 1] = s1 as f32 * (scales[r + 1] * v_scale);
        out[r + 2] = s2 as f32 * (scales[r + 2] * v_scale);
        out[r + 3] = s3 as f32 * (scales[r + 3] * v_scale);
    }
    for r in rows_4..rows {
        let qr = &q[r * cols..(r + 1) * cols];
        let mut sum = 0i32;
        for c in 0..cols {
            sum += qr[c] as i32 * v_q[c] as i32;
        }
        out[r] = sum as f32 * (scales[r] * v_scale);
    }
    Tensor::from_data(out, vec![rows])
}

/// Quantize f32 vector to i8 with single global scale. Returns (quantized, scale).
fn quantize_vec_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let mut abs_max = 0.0f32;
    for &x in v { let a = x.abs(); if a > abs_max { abs_max = a; } }
    let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    let mut q = vec![0i8; v.len()];
    for i in 0..v.len() {
        q[i] = (v[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    (q, scale)
}

// ---- AVX-VNNI kernel ----

/// Horizontal sum of 8 x i32 lanes in __m256i.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_avx2(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256::<1>(v);
    let lo = _mm256_castsi256_si128(v);
    let sum4 = _mm_add_epi32(lo, hi);
    let hi64 = _mm_unpackhi_epi64(sum4, sum4);
    let sum2 = _mm_add_epi32(sum4, hi64);
    let hi32 = _mm_shuffle_epi32::<0x01>(sum2);
    _mm_cvtsi128_si32(_mm_add_epi32(sum2, hi32))
}

/// Horizontal sum of 8 x f32 lanes in __m256.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_f32_avx(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    _mm_cvtss_f32(_mm_add_ss(sum2, _mm_shuffle_ps::<1>(sum2, sum2)))
}

/// AVX-VNNI kernel writing into pre-allocated buffers (zero allocation).
/// v_q must already be quantized via quantize_vec_i8_into before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
unsafe fn q8_mat_vec_mul_vnni_into(out: &mut [f32], mat: &Q8Tensor, v_q: &[i8], v_scale: f32) {
    let rows = mat.rows;
    let cols = mat.cols;
    assert_eq!(cols % 32, 0, "VNNI requires cols divisible by 32");

    // Precompute correction: 128 * sum(v_q) — accounts for i8→u8 bias on weights
    let sum_v: i32 = v_q[..cols].iter().map(|&x| x as i32).sum();
    let correction = 128i32 * sum_v;

    let bias = _mm256_set1_epi8(-128i8); // 0x80 for XOR trick
    let chunks = cols / 32;
    let prefetch = !no_prefetch_override();

    let q = mat.q_data.as_ptr();
    let v_ptr = v_q.as_ptr();
    let scales = &mat.scales;

    // 4-row ILP
    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        let base0 = r * cols;
        let base1 = base0 + cols;
        let base2 = base1 + cols;
        let base3 = base2 + cols;

        // Prefetch first cache lines of next 4-row block
        if prefetch && r + 4 < rows {
            let nb = (r + 4) * cols;
            _mm_prefetch(q.add(nb) as *const i8, _MM_HINT_T0);
            _mm_prefetch(q.add(nb + cols) as *const i8, _MM_HINT_T0);
            _mm_prefetch(q.add(nb + 2 * cols) as *const i8, _MM_HINT_T0);
            _mm_prefetch(q.add(nb + 3 * cols) as *const i8, _MM_HINT_T0);
        }

        for ch in 0..chunks {
            let off = ch * 32;

            // Prefetch 2 chunks ahead to hide L2 latency
            if prefetch && ch + 2 < chunks {
                let poff = (ch + 2) * 32;
                _mm_prefetch(q.add(base0 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(q.add(base1 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(q.add(base2 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(q.add(base3 + poff) as *const i8, _MM_HINT_T0);
            }

            let vb = _mm256_loadu_si256(v_ptr.add(off) as *const __m256i);

            let w0 = _mm256_loadu_si256(q.add(base0 + off) as *const __m256i);
            let w1 = _mm256_loadu_si256(q.add(base1 + off) as *const __m256i);
            let w2 = _mm256_loadu_si256(q.add(base2 + off) as *const __m256i);
            let w3 = _mm256_loadu_si256(q.add(base3 + off) as *const __m256i);

            acc0 = _mm256_dpbusd_avx_epi32(acc0, _mm256_xor_si256(w0, bias), vb);
            acc1 = _mm256_dpbusd_avx_epi32(acc1, _mm256_xor_si256(w1, bias), vb);
            acc2 = _mm256_dpbusd_avx_epi32(acc2, _mm256_xor_si256(w2, bias), vb);
            acc3 = _mm256_dpbusd_avx_epi32(acc3, _mm256_xor_si256(w3, bias), vb);
        }

        let d0 = hsum_i32_avx2(acc0) - correction;
        let d1 = hsum_i32_avx2(acc1) - correction;
        let d2 = hsum_i32_avx2(acc2) - correction;
        let d3 = hsum_i32_avx2(acc3) - correction;

        out[r]     = d0 as f32 * (scales[r]     * v_scale);
        out[r + 1] = d1 as f32 * (scales[r + 1] * v_scale);
        out[r + 2] = d2 as f32 * (scales[r + 2] * v_scale);
        out[r + 3] = d3 as f32 * (scales[r + 3] * v_scale);
    }
    // Remainder rows
    for r in rows_4..rows {
        let mut acc = _mm256_setzero_si256();
        let base = r * cols;
        for ch in 0..chunks {
            let off = ch * 32;
            let vb = _mm256_loadu_si256(v_ptr.add(off) as *const __m256i);
            let w = _mm256_loadu_si256(q.add(base + off) as *const __m256i);
            acc = _mm256_dpbusd_avx_epi32(acc, _mm256_xor_si256(w, bias), vb);
        }
        let d = hsum_i32_avx2(acc) - correction;
        out[r] = d as f32 * (scales[r] * v_scale);
    }
}

/// AVX2 kernel for Q8 mat-vec: i8*i8 → i32 using sign trick + maddubs.
/// Available on all AVX2 CPUs (Haswell 2013+). Falls back to scalar for
/// remainder elements when cols is not divisible by 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8_mat_vec_mul_avx2_into(
    out: &mut [f32], mat: &Q8Tensor, v_q: &[i8], v_scale: f32,
) {
    let rows = mat.rows;
    let cols = mat.cols;
    let chunks = cols / 32;
    let rem_start = chunks * 32;
    let prefetch = !no_prefetch_override();

    let q = mat.q_data.as_ptr();
    let v_ptr = v_q.as_ptr();
    let scales = &mat.scales;
    let ones = _mm256_set1_epi16(1);

    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        let b0 = r * cols;
        let b1 = b0 + cols;
        let b2 = b1 + cols;
        let b3 = b2 + cols;

        // Prefetch first cache lines of next 4-row block
        if prefetch && r + 4 < rows {
            let nb = (r + 4) * cols;
            _mm_prefetch(q.add(nb) as *const i8, _MM_HINT_T0);
            _mm_prefetch(q.add(nb + cols) as *const i8, _MM_HINT_T0);
            _mm_prefetch(q.add(nb + 2 * cols) as *const i8, _MM_HINT_T0);
            _mm_prefetch(q.add(nb + 3 * cols) as *const i8, _MM_HINT_T0);
        }

        for ch in 0..chunks {
            let off = ch * 32;
            let vb = _mm256_loadu_si256(v_ptr.add(off) as *const __m256i);

            // Prefetch 2 chunks ahead to hide L2 latency (~14 cycles)
            if prefetch && ch + 2 < chunks {
                let poff = (ch + 2) * 32;
                _mm_prefetch(q.add(b0 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(q.add(b1 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(q.add(b2 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(q.add(b3 + poff) as *const i8, _MM_HINT_T0);
            }

            macro_rules! dot {
                ($base:expr, $acc:expr) => {{
                    let w = _mm256_loadu_si256(q.add($base + off) as *const __m256i);
                    let w_abs = _mm256_abs_epi8(w);
                    let v_signed = _mm256_sign_epi8(vb, w);
                    let p16 = _mm256_maddubs_epi16(w_abs, v_signed);
                    $acc = _mm256_add_epi32($acc, _mm256_madd_epi16(p16, ones));
                }};
            }
            dot!(b0, acc0);
            dot!(b1, acc1);
            dot!(b2, acc2);
            dot!(b3, acc3);
        }

        let (mut d0, mut d1, mut d2, mut d3) = (
            hsum_i32_avx2(acc0), hsum_i32_avx2(acc1),
            hsum_i32_avx2(acc2), hsum_i32_avx2(acc3),
        );

        for c in rem_start..cols {
            let vc = *v_q.get_unchecked(c) as i32;
            d0 += *mat.q_data.get_unchecked(b0 + c) as i32 * vc;
            d1 += *mat.q_data.get_unchecked(b1 + c) as i32 * vc;
            d2 += *mat.q_data.get_unchecked(b2 + c) as i32 * vc;
            d3 += *mat.q_data.get_unchecked(b3 + c) as i32 * vc;
        }

        out[r]     = d0 as f32 * (scales[r]     * v_scale);
        out[r + 1] = d1 as f32 * (scales[r + 1] * v_scale);
        out[r + 2] = d2 as f32 * (scales[r + 2] * v_scale);
        out[r + 3] = d3 as f32 * (scales[r + 3] * v_scale);
    }

    for r in rows_4..rows {
        let mut acc = _mm256_setzero_si256();
        let base = r * cols;
        for ch in 0..chunks {
            let off = ch * 32;
            let vb = _mm256_loadu_si256(v_ptr.add(off) as *const __m256i);
            let w = _mm256_loadu_si256(q.add(base + off) as *const __m256i);
            let w_abs = _mm256_abs_epi8(w);
            let v_signed = _mm256_sign_epi8(vb, w);
            let p16 = _mm256_maddubs_epi16(w_abs, v_signed);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
        }
        let mut d = hsum_i32_avx2(acc);
        for c in rem_start..cols {
            d += *mat.q_data.get_unchecked(base + c) as i32 * *v_q.get_unchecked(c) as i32;
        }
        out[r] = d as f32 * (scales[r] * v_scale);
    }
}

// ---- Q4 group-quantized tensor ----

/// Group-quantized 4-bit weight matrix.
/// Each group of `group_size` elements shares one f32 scale.
/// Two weights packed per byte (low nibble = even index, high nibble = odd index).
/// Values stored as signed: mapped from i4 range [-8, 7].
/// Memory: rows*cols/2 bytes + (rows*cols/group_size)*4 bytes
pub struct Q4Tensor {
    pub packed: Vec<u8>,       // two i4 values per byte
    pub scales: Vec<f32>,      // one scale per group
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

impl Q4Tensor {
    /// Quantize an f32 (rows, cols) tensor to Q4 with group quantization.
    pub fn from_f32(t: &Tensor, group_size: usize) -> Self {
        assert_eq!(t.shape.len(), 2);
        let rows = t.shape[0];
        let cols = t.shape[1];
        let total = rows * cols;
        assert_eq!(total % 2, 0, "Q4 requires even number of elements");

        let n_groups = (total + group_size - 1) / group_size;
        let mut packed = vec![0u8; total / 2];
        let mut scales = vec![0.0f32; n_groups];

        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(total);
            let group = &t.data[start..end];

            let abs_max = group.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if abs_max > 0.0 { abs_max / 7.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales[g] = scale;

            for i in start..end {
                let q = (t.data[i] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                let qu = (q & 0x0F) as u8; // keep low 4 bits
                let byte_idx = i / 2;
                if i % 2 == 0 {
                    packed[byte_idx] = (packed[byte_idx] & 0xF0) | qu;
                } else {
                    packed[byte_idx] = (packed[byte_idx] & 0x0F) | (qu << 4);
                }
            }
        }

        Self { packed, scales, rows, cols, group_size }
    }

    /// Memory usage in bytes
    pub fn mem_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }

}

/// y = Q4_mat @ f32_vec.
/// Dequantizes weights to i8 per group, quantizes input vec to i8 once,
/// then accumulates as i32.
pub fn q4_mat_vec_mul(mat: &Q4Tensor, vec: &Tensor) -> Tensor {
    assert_eq!(vec.shape.len(), 1);
    let rows = mat.rows;
    let cols = mat.cols;
    assert_eq!(vec.shape[0], cols);

    let v = &vec.as_slice()[..cols];

    // Quantize input vector to i8 (amortized over all rows)
    let mut v_abs_max = 0.0f32;
    for &x in v { let a = x.abs(); if a > v_abs_max { v_abs_max = a; } }
    let v_scale = if v_abs_max > 0.0 { v_abs_max / 127.0 } else { 1.0 };
    let v_inv = 1.0 / v_scale;
    let mut v_q = vec![0i8; cols];
    for c in 0..cols {
        v_q[c] = (v[c] * v_inv).round().clamp(-127.0, 127.0) as i8;
    }

    let mut out = vec![0.0f32; rows];
    let gs = mat.group_size;
    let groups_per_row = (cols + gs - 1) / gs;

    for r in 0..rows {
        let row_start = r * cols;
        let mut row_sum = 0.0f32;

        for g in 0..groups_per_row {
            let g_start = g * gs;
            let g_end = (g_start + gs).min(cols);
            let global_group = (row_start + g_start) / gs;
            let w_scale = mat.scales[global_group];

            let mut acc = 0i32;
            for c in g_start..g_end {
                let idx = row_start + c;
                let byte = mat.packed[idx / 2];
                let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let val = if nibble & 0x08 != 0 {
                    (nibble | 0xF0) as i8
                } else {
                    nibble as i8
                };
                acc += val as i32 * v_q[c] as i32;
            }
            row_sum += acc as f32 * (w_scale * v_scale);
        }
        out[r] = row_sum;
    }
    Tensor::from_data(out, vec![rows])
}


/// Q4 mat-vec into pre-allocated output, using SIMD Q8-style dot product per group.
/// Unpacks Q4→i8 on-the-fly, then uses sign-trick + maddubs for each group.
/// Set REDUKT_NO_SIMD=1 to force scalar path (benchmarking).
pub fn q4_mat_vec_mul_into(out: &mut [f32], mat: &Q4Tensor, vec: &[f32], v_q: &mut [i8]) {
    let cols = mat.cols;
    let v_scale = quantize_vec_i8_into(&vec[..cols], &mut v_q[..cols]);

    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") && mat.group_size == 32 {
        unsafe { q4_mat_vec_mul_avx2_into(out, mat, &v_q[..cols], v_scale) };
        return;
    }

    q4_mat_vec_mul_scalar_into(out, mat, &v_q[..cols], v_scale);
}

/// Scalar fallback for Q4 mat-vec with pre-quantized vector.
fn q4_mat_vec_mul_scalar_into(
    out: &mut [f32], mat: &Q4Tensor, v_q: &[i8], v_scale: f32,
) {
    let rows = mat.rows;
    let cols = mat.cols;
    let gs = mat.group_size;
    let groups_per_row = (cols + gs - 1) / gs;

    for r in 0..rows {
        let row_start = r * cols;
        let mut row_sum = 0.0f32;

        for g in 0..groups_per_row {
            let g_start = g * gs;
            let g_end = (g_start + gs).min(cols);
            let global_group = (row_start + g_start) / gs;
            let w_scale = mat.scales[global_group];

            let mut acc = 0i32;
            for c in g_start..g_end {
                let idx = row_start + c;
                let byte = mat.packed[idx / 2];
                let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let val = if nibble & 0x08 != 0 {
                    (nibble | 0xF0) as i8
                } else {
                    nibble as i8
                };
                acc += val as i32 * v_q[c] as i32;
            }
            row_sum += acc as f32 * (w_scale * v_scale);
        }
        out[r] = row_sum;
    }
}

/// AVX2 kernel for Q4 mat-vec: unpack Q4→i8, then i8*i8→i32 sign-trick per group.
/// Requires group_size == 32 (one AVX2 register per group).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4_mat_vec_mul_avx2_into(
    out: &mut [f32], mat: &Q4Tensor, v_q: &[i8], v_scale: f32,
) {
    let rows = mat.rows;
    let cols = mat.cols;
    let gs = mat.group_size; // must be 32
    let groups_per_row = cols / gs;
    let packed = mat.packed.as_ptr();
    let scales = mat.scales.as_ptr();
    let v_ptr = v_q.as_ptr();
    let ones = _mm256_set1_epi16(1);
    let lo_mask = _mm256_set1_epi8(0x0F);
    let sign_thresh = _mm256_set1_epi8(7);
    let sign_bits = _mm256_set1_epi8(-16i8); // 0xF0

    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        for g in 0..groups_per_row {
            let g_start = g * gs;
            let vb = _mm256_loadu_si256(v_ptr.add(g_start) as *const __m256i);

            // For each of the 4 rows, unpack Q4 and compute dot product
            macro_rules! q4_dot {
                ($row:expr) => {{
                    let row_start = $row * cols;
                    let pack_off = (row_start + g_start) / 2;
                    // Load 16 packed bytes = 32 Q4 values
                    let p = _mm_loadu_si128(packed.add(pack_off) as *const __m128i);
                    // Low nibbles (even indices)
                    let lo = _mm_and_si128(p, _mm256_castsi256_si128(lo_mask));
                    // High nibbles (odd indices)
                    let hi = _mm_and_si128(
                        _mm_srli_epi16(p, 4),
                        _mm256_castsi256_si128(lo_mask),
                    );
                    // Interleave: [val0, val1, val2, val3, ...]
                    let interleaved = _mm256_set_m128i(
                        _mm_unpackhi_epi8(lo, hi),
                        _mm_unpacklo_epi8(lo, hi),
                    );
                    // Sign extend: values > 7 are negative in 4-bit signed
                    let need_sign = _mm256_cmpgt_epi8(interleaved, sign_thresh);
                    let w = _mm256_or_si256(
                        interleaved,
                        _mm256_and_si256(need_sign, sign_bits),
                    );
                    // Sign-trick dot product (same as Q8 kernel)
                    let w_abs = _mm256_abs_epi8(w);
                    let v_signed = _mm256_sign_epi8(vb, w);
                    let p16 = _mm256_maddubs_epi16(w_abs, v_signed);
                    let p32 = _mm256_madd_epi16(p16, ones);
                    hsum_i32_avx2(p32)
                }};
            }

            let d0 = q4_dot!(r);
            let d1 = q4_dot!(r + 1);
            let d2 = q4_dot!(r + 2);
            let d3 = q4_dot!(r + 3);

            // Apply per-group weight scale
            let global_g0 = (r * cols + g_start) / gs;
            let global_g1 = ((r + 1) * cols + g_start) / gs;
            let global_g2 = ((r + 2) * cols + g_start) / gs;
            let global_g3 = ((r + 3) * cols + g_start) / gs;

            sum0 += d0 as f32 * (*scales.add(global_g0) * v_scale);
            sum1 += d1 as f32 * (*scales.add(global_g1) * v_scale);
            sum2 += d2 as f32 * (*scales.add(global_g2) * v_scale);
            sum3 += d3 as f32 * (*scales.add(global_g3) * v_scale);
        }
        out[r] = sum0;
        out[r + 1] = sum1;
        out[r + 2] = sum2;
        out[r + 3] = sum3;
    }

    // Remainder rows
    for r in rows_4..rows {
        let mut row_sum = 0.0f32;
        for g in 0..groups_per_row {
            let g_start = g * gs;
            let vb = _mm256_loadu_si256(v_ptr.add(g_start) as *const __m256i);
            let row_start = r * cols;
            let pack_off = (row_start + g_start) / 2;
            let p = _mm_loadu_si128(packed.add(pack_off) as *const __m128i);
            let lo = _mm_and_si128(p, _mm256_castsi256_si128(lo_mask));
            let hi = _mm_and_si128(
                _mm_srli_epi16(p, 4),
                _mm256_castsi256_si128(lo_mask),
            );
            let interleaved = _mm256_set_m128i(
                _mm_unpackhi_epi8(lo, hi),
                _mm_unpacklo_epi8(lo, hi),
            );
            let need_sign = _mm256_cmpgt_epi8(interleaved, sign_thresh);
            let w = _mm256_or_si256(interleaved, _mm256_and_si256(need_sign, sign_bits));
            let w_abs = _mm256_abs_epi8(w);
            let v_signed = _mm256_sign_epi8(vb, w);
            let p16 = _mm256_maddubs_epi16(w_abs, v_signed);
            let p32 = _mm256_madd_epi16(p16, ones);
            let d = hsum_i32_avx2(p32);
            let global_g = (row_start + g_start) / gs;
            row_sum += d as f32 * (*scales.add(global_g) * v_scale);
        }
        out[r] = row_sum;
    }
}

/// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let n = x.numel();
    let mean: f32 = x.data.iter().sum::<f32>() / n as f32;
    let var: f32 = x.data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let data: Vec<f32> = x.data.iter()
        .zip(weight.data.iter())
        .zip(bias.data.iter())
        .map(|((&x, &w), &b)| (x - mean) * inv_std * w + b)
        .collect();
    Tensor::from_data(data, x.shape.clone())
}


// ---- Zero-allocation buffer-reuse variants ----
// These write into pre-existing &mut [f32] buffers instead of allocating.
// Used by Scratch-based forward pass to eliminate ~1400 allocs/token.

pub fn sub_into(out: &mut [f32], a: &[f32], b: &[f32]) {
    for (o, (&a, &b)) in out.iter_mut().zip(a.iter().zip(b.iter())) { *o = a - b; }
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (a, &b) in a.iter_mut().zip(b.iter()) { *a += b; }
}

pub fn add_scaled_into(out: &mut [f32], a: &[f32], b: &[f32], scale: &[f32]) {
    for (((o, &a), &b), &s) in out.iter_mut().zip(a.iter()).zip(b.iter()).zip(scale.iter()) {
        *o = a + b * s;
    }
}

pub fn mul_into(out: &mut [f32], a: &[f32], b: &[f32]) {
    for (o, (&a, &b)) in out.iter_mut().zip(a.iter().zip(b.iter())) { *o = a * b; }
}

// ---- Portable transcendentals (R-013b) ----
// Pure-Rust implementations using ONLY IEEE 754 basic operations (+, -, *, /, bitwise).
// These produce BIT-IDENTICAL results on all platforms (x86, ARM64, WASM).
// Coefficients match fast_exp_ps (Cephes) for SIMD/scalar cross-tier determinism.

/// Portable exp(f32) — Cephes degree-5 polynomial + range reduction.
/// Same coefficients as fast_exp_ps: SIMD lane == scalar result.
#[inline]
pub fn portable_expf(x: f32) -> f32 {
    const LOG2E: f32 = 1.44269504088896341;
    const LN2_HI: f32 = 6.93145751953125E-1;
    const LN2_LO: f32 = 1.42860682030941723212E-6;
    const EXP_LO: f32 = -87.3365447505531;
    const EXP_HI: f32 = 88.3762626647949;
    const P0: f32 = 1.9875691500E-4;
    const P1: f32 = 1.3981999507E-3;
    const P2: f32 = 8.3334519073E-3;
    const P3: f32 = 4.1665795894E-2;
    const P4: f32 = 1.6666665459E-1;
    const P5: f32 = 5.0000001201E-1;

    let x = if x < EXP_LO { EXP_LO } else if x > EXP_HI { EXP_HI } else { x };
    let fx = x * LOG2E;
    // Round-to-nearest-even (matches _mm256_cvtps_epi32 default rounding)
    let nf = portable_roundeven(fx);
    let n = nf as i32;
    let r = (x - nf * LN2_HI) - nf * LN2_LO;
    let r2 = r * r;
    let mut y = P0;
    y = y * r + P1;
    y = y * r + P2;
    y = y * r + P3;
    y = y * r + P4;
    y = y * r + P5;
    y = (y * r2 + r) + 1.0;
    let pow2n = f32::from_bits(((n + 127) as u32) << 23);
    y * pow2n
}

/// Portable natural log (f32) — musl/FreeBSD algorithm.
/// Uses only IEEE 754 basic ops. Bit-exact across all platforms.
#[inline]
pub fn portable_lnf(x: f32) -> f32 {
    const LN2_HI: f32 = 6.9313812256e-01;
    const LN2_LO: f32 = 9.0580006145e-06;
    const LG1: f32 = 0.66666662693;
    const LG2: f32 = 0.40000972152;
    const LG3: f32 = 0.28498786688;
    const LG4: f32 = 0.24279078841;

    let mut ix = x.to_bits();
    let mut k: i32 = 0;

    if ix < 0x00800000 || (ix >> 31) != 0 {
        if ix << 1 == 0 { return f32::NEG_INFINITY; } // log(+-0)
        if (ix >> 31) != 0 { return f32::NAN; } // log(negative)
        k -= 25;
        ix = (x * f32::from_bits(0x4c000000)).to_bits(); // x * 2^25
    } else if ix >= 0x7f800000 {
        return x; // +inf or NaN passthrough
    } else if ix == 0x3f800000 {
        return 0.0; // log(1) = 0 exact
    }

    // Normalize to [sqrt(2)/2, sqrt(2)]
    ix = ix.wrapping_add(0x3f800000u32.wrapping_sub(0x3f3504f3));
    k += ((ix >> 23) as i32) - 0x7f;
    ix = (ix & 0x007fffff) + 0x3f3504f3;
    let xn = f32::from_bits(ix);

    let f = xn - 1.0;
    let s = f / (2.0 + f);
    let dk = k as f32;
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * LG4);
    let t2 = z * (LG1 + w * LG3);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    s * (hfsq + r) + dk * LN2_LO - hfsq + f + dk * LN2_HI
}

/// Portable sigmoid(f32) = 1/(1+exp(-x)).
#[inline]
pub fn portable_sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + portable_expf(-x))
}

/// Portable tanh(f32) = 2*sigmoid(2x) - 1.
#[inline]
pub fn portable_tanhf(x: f32) -> f32 {
    2.0 * portable_sigmoidf(2.0 * x) - 1.0
}

/// SiLU (Swish) activation: x * sigmoid(x). Used by Mamba SSM.
#[inline]
pub fn portable_silu(x: f32) -> f32 {
    x * portable_sigmoidf(x)
}

/// Softplus: ln(1 + exp(x)). Used by Mamba dt discretization.
#[inline]
pub fn portable_softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { portable_lnf(1.0 + portable_expf(x)) }
}

/// RMSNorm in-place: x[i] = x[i] / sqrt(mean(x²) + eps) * weight[i].
pub fn rms_norm_into(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(n, weight.len());
    let mut sum_sq = 0.0f32;
    for &v in x.iter() { sum_sq += v * v; }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n { x[i] = x[i] * inv_rms * weight[i]; }
}

/// Round-to-nearest-even for f32 (matches IEEE 754 default / _mm256_cvtps_epi32).
#[inline]
fn portable_roundeven(x: f32) -> f32 {
    const MAGIC: f32 = 8388608.0; // 2^23
    if x >= 0.0 { (x + MAGIC) - MAGIC } else { (x - MAGIC) + MAGIC }
}

// ---- AVX2 transcendental approximations (R-022) ----

/// Fast exp(x) for 8 packed f32 — Cephes-style polynomial, ~23-bit accuracy.
/// Clamps to [-87.33, 88.72] to avoid overflow/underflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fast_exp_ps(x: __m256) -> __m256 {
    let log2e  = _mm256_set1_ps(1.44269504088896341f32);
    let ln2_hi = _mm256_set1_ps(6.93145751953125E-1f32);
    let ln2_lo = _mm256_set1_ps(1.42860682030941723212E-6f32);
    let exp_lo = _mm256_set1_ps(-87.3365447505531f32);
    let exp_hi = _mm256_set1_ps(88.3762626647949f32);
    let p0 = _mm256_set1_ps(1.9875691500E-4f32);
    let p1 = _mm256_set1_ps(1.3981999507E-3f32);
    let p2 = _mm256_set1_ps(8.3334519073E-3f32);
    let p3 = _mm256_set1_ps(4.1665795894E-2f32);
    let p4 = _mm256_set1_ps(1.6666665459E-1f32);
    let p5 = _mm256_set1_ps(5.0000001201E-1f32);
    let one = _mm256_set1_ps(1.0f32);
    let x = _mm256_min_ps(_mm256_max_ps(x, exp_lo), exp_hi);
    let fx = _mm256_mul_ps(x, log2e);
    let n_i = _mm256_cvtps_epi32(fx);
    let n_f = _mm256_cvtepi32_ps(n_i);
    let r = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)),
        _mm256_mul_ps(n_f, ln2_lo),
    );
    let r2 = _mm256_mul_ps(r, r);
    let mut y = p0;
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p1);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p2);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p3);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p4);
    y = _mm256_add_ps(_mm256_mul_ps(y, r), p5);
    y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y, r2), r), one);
    let pow2n = _mm256_castsi256_ps(
        _mm256_slli_epi32::<23>(_mm256_add_epi32(n_i, _mm256_set1_epi32(127))),
    );
    _mm256_mul_ps(y, pow2n)
}

/// Fast sigmoid(x) = 1/(1+exp(-x)) for 8 packed f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fast_sigmoid_ps(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0f32);
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_ps(neg_x)))
}

/// Fast tanh(x) = 2*sigmoid(2x) - 1 for 8 packed f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fast_tanh_ps(x: __m256) -> __m256 {
    let two = _mm256_set1_ps(2.0f32);
    let one = _mm256_set1_ps(1.0f32);
    _mm256_sub_ps(_mm256_mul_ps(fast_sigmoid_ps(_mm256_mul_ps(x, two)), two), one)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tanh_inplace_avx2(x: &mut [f32]) {
    let chunks = x.len() / 8;
    let p = x.as_mut_ptr();
    for i in 0..chunks {
        let off = i * 8;
        let v = _mm256_loadu_ps(p.add(off));
        _mm256_storeu_ps(p.add(off), fast_tanh_ps(v));
    }
    for i in (chunks * 8)..x.len() {
        *x.get_unchecked_mut(i) = portable_tanhf(*x.get_unchecked(i));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sigmoid_inplace_avx2(x: &mut [f32]) {
    let chunks = x.len() / 8;
    let p = x.as_mut_ptr();
    for i in 0..chunks {
        let off = i * 8;
        let v = _mm256_loadu_ps(p.add(off));
        _mm256_storeu_ps(p.add(off), fast_sigmoid_ps(v));
    }
    for i in (chunks * 8)..x.len() {
        let v = *x.get_unchecked(i);
        *x.get_unchecked_mut(i) = portable_sigmoidf(v);
    }
}

pub fn tanh_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe { tanh_inplace_avx2(x) };
        return;
    }
    for v in x.iter_mut() { *v = portable_tanhf(*v); }
}

pub fn sigmoid_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe { sigmoid_inplace_avx2(x) };
        return;
    }
    for v in x.iter_mut() { *v = portable_sigmoidf(*v); }
}

/// out[i] = sigmoid(base[i] + offset[i]). Used by model Steps 4, 8.
pub fn sigmoid_add_into(out: &mut [f32], base: &[f32], offset: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe { sigmoid_add_into_avx2(out, base, offset) };
        return;
    }
    for i in 0..out.len() {
        out[i] = portable_sigmoidf(base[i] + offset[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sigmoid_add_into_avx2(out: &mut [f32], base: &[f32], offset: &[f32]) {
    let n = out.len();
    let chunks = n / 8;
    let bp = base.as_ptr();
    let op = offset.as_ptr();
    let dp = out.as_mut_ptr();
    for i in 0..chunks {
        let off = i * 8;
        let sum = _mm256_add_ps(
            _mm256_loadu_ps(bp.add(off)),
            _mm256_loadu_ps(op.add(off)),
        );
        _mm256_storeu_ps(dp.add(off), fast_sigmoid_ps(sum));
    }
    for i in (chunks * 8)..n {
        *out.get_unchecked_mut(i) = portable_sigmoidf(base[i] + offset[i]);
    }
}

/// out[i] = exp(-0.606531 * sigmoid(base[i] + offset[i])). Used by model Step 3.
pub fn decay_sigmoid_into(out: &mut [f32], base: &[f32], offset: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe { decay_sigmoid_into_avx2(out, base, offset) };
        return;
    }
    for i in 0..out.len() {
        let val = base[i] + offset[i];
        out[i] = portable_expf(-0.606531f32 * portable_sigmoidf(val));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decay_sigmoid_into_avx2(out: &mut [f32], base: &[f32], offset: &[f32]) {
    let decay_k = _mm256_set1_ps(-0.606531f32);
    let n = out.len();
    let chunks = n / 8;
    let bp = base.as_ptr();
    let op = offset.as_ptr();
    let dp = out.as_mut_ptr();
    for i in 0..chunks {
        let off = i * 8;
        let sum = _mm256_add_ps(
            _mm256_loadu_ps(bp.add(off)),
            _mm256_loadu_ps(op.add(off)),
        );
        let sig = fast_sigmoid_ps(sum);
        _mm256_storeu_ps(dp.add(off), fast_exp_ps(_mm256_mul_ps(decay_k, sig)));
    }
    for i in (chunks * 8)..n {
        let val = base[i] + offset[i];
        *out.get_unchecked_mut(i) = portable_expf(-0.606531f32 * portable_sigmoidf(val));
    }
}

/// v_out[i] += (v_first[i] - v_out[i]) * sigmoid(base[i] + offset[i]). Step 8.
pub fn sigmoid_mix_inplace(
    v_out: &mut [f32], v_first: &[f32], base: &[f32], offset: &[f32],
) {
    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe { sigmoid_mix_inplace_avx2(v_out, v_first, base, offset) };
        return;
    }
    for i in 0..v_out.len() {
        let mix = portable_sigmoidf(base[i] + offset[i]);
        v_out[i] += (v_first[i] - v_out[i]) * mix;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sigmoid_mix_inplace_avx2(
    v_out: &mut [f32], v_first: &[f32], base: &[f32], offset: &[f32],
) {
    let n = v_out.len();
    let chunks = n / 8;
    let vp = v_out.as_mut_ptr();
    let fp = v_first.as_ptr();
    let bp = base.as_ptr();
    let op = offset.as_ptr();
    for i in 0..chunks {
        let off = i * 8;
        let sum = _mm256_add_ps(
            _mm256_loadu_ps(bp.add(off)),
            _mm256_loadu_ps(op.add(off)),
        );
        let mix = fast_sigmoid_ps(sum);
        let vo = _mm256_loadu_ps(vp.add(off));
        let vf = _mm256_loadu_ps(fp.add(off));
        let result = _mm256_add_ps(vo, _mm256_mul_ps(_mm256_sub_ps(vf, vo), mix));
        _mm256_storeu_ps(vp.add(off), result);
    }
    for i in (chunks * 8)..n {
        let mix = portable_sigmoidf(base[i] + offset[i]);
        *v_out.get_unchecked_mut(i) += (v_first[i] - v_out[i]) * mix;
    }
}

// ---- Mamba SIMD kernels (R-038) ----

/// Horizontal sum of 8 packed f32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let hi2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, hi2))
}

/// In-place SiLU (Swish): buf[i] = buf[i] * sigmoid(buf[i]).
pub fn silu_inplace(buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe { silu_inplace_avx2(buf) };
        return;
    }
    for v in buf.iter_mut() { *v = portable_silu(*v); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn silu_inplace_avx2(buf: &mut [f32]) {
    let chunks = buf.len() / 8;
    let p = buf.as_mut_ptr();
    for i in 0..chunks {
        let off = i * 8;
        let x = _mm256_loadu_ps(p.add(off));
        _mm256_storeu_ps(p.add(off), _mm256_mul_ps(x, fast_sigmoid_ps(x)));
    }
    for i in (chunks * 8)..buf.len() {
        *buf.get_unchecked_mut(i) = portable_silu(*buf.get_unchecked(i));
    }
}

/// Gated SiLU multiply: out[i] = y[i] * silu(z[i]).
pub fn gated_silu_mul(out: &mut [f32], y_buf: &[f32], z: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe { gated_silu_mul_avx2(out, y_buf, z) };
        return;
    }
    for i in 0..out.len() { out[i] = y_buf[i] * portable_silu(z[i]); }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gated_silu_mul_avx2(out: &mut [f32], y_buf: &[f32], z: &[f32]) {
    let chunks = out.len() / 8;
    let op = out.as_mut_ptr();
    let yp = y_buf.as_ptr();
    let zp = z.as_ptr();
    for i in 0..chunks {
        let off = i * 8;
        let yv = _mm256_loadu_ps(yp.add(off));
        let zv = _mm256_loadu_ps(zp.add(off));
        _mm256_storeu_ps(op.add(off), _mm256_mul_ps(yv, _mm256_mul_ps(zv, fast_sigmoid_ps(zv))));
    }
    for i in (chunks * 8)..out.len() {
        *out.get_unchecked_mut(i) = y_buf[i] * portable_silu(z[i]);
    }
}

/// SSM recurrence step for d_state=16, vectorized over state dimension.
/// 16 = 2×8 AVX2 lanes. Replaces 32 scalar expf with 4 SIMD fast_exp_ps per j.
pub fn ssm_recurrence_d16(
    y_out: &mut [f32], ss: &mut [f32], dt: &[f32], conv_out: &[f32],
    a_log: &[f32], b: &[f32], c: &[f32], d_vec: &[f32], d_inner: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe {
            ssm_recurrence_d16_avx2(
                y_out, ss, dt, conv_out, a_log, b, c, d_vec, d_inner,
            )
        };
        return;
    }
    ssm_recurrence_d16_scalar(y_out, ss, dt, conv_out, a_log, b, c, d_vec, d_inner);
}

/// dt-sparse threshold (R-057): skip SSM state update when |dt| < ε.
/// State is unchanged (da≈1, dt*x*b≈0), saving exp() calls.
const DT_SPARSE_EPS: f32 = 1e-6;

fn ssm_recurrence_d16_scalar(
    y_out: &mut [f32], ss: &mut [f32], dt: &[f32], conv_out: &[f32],
    a_log: &[f32], b: &[f32], c: &[f32], d_vec: &[f32], d_inner: usize,
) {
    for j in 0..d_inner {
        let dt_j = dt[j];
        let x_j = conv_out[j];

        if dt_j.abs() < DT_SPARSE_EPS {
            // R-057: dt ≈ 0 → state unchanged, just compute output from existing state
            let base = j * 16;
            let mut y_val = 0.0f32;
            for n in 0..16 {
                y_val += ss[base + n] * c[n];
            }
            y_out[j] = y_val + d_vec[j] * x_j;
        } else {
            let mut y_val = 0.0f32;
            for n in 0..16 {
                let a_val = -portable_expf(a_log[j * 16 + n]);
                let da = portable_expf(dt_j * a_val);
                let idx = j * 16 + n;
                ss[idx] = da * ss[idx] + dt_j * x_j * b[n];
                y_val += ss[idx] * c[n];
            }
            y_out[j] = y_val + d_vec[j] * x_j;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ssm_recurrence_d16_avx2(
    y_out: &mut [f32], ss: &mut [f32], dt: &[f32], conv_out: &[f32],
    a_log: &[f32], b: &[f32], c: &[f32], d_vec: &[f32], d_inner: usize,
) {
    let b_lo = _mm256_loadu_ps(b.as_ptr());
    let b_hi = _mm256_loadu_ps(b.as_ptr().add(8));
    let c_lo = _mm256_loadu_ps(c.as_ptr());
    let c_hi = _mm256_loadu_ps(c.as_ptr().add(8));
    let zero = _mm256_setzero_ps();
    let ss_ptr = ss.as_mut_ptr();
    let alog_ptr = a_log.as_ptr();

    for j in 0..d_inner {
        let base = j * 16;

        // R-057: dt-sparse skip — avoid 4 exp() calls when dt ≈ 0
        if dt[j].abs() < DT_SPARSE_EPS {
            let ss_lo = _mm256_loadu_ps(ss_ptr.add(base));
            let ss_hi = _mm256_loadu_ps(ss_ptr.add(base + 8));
            let sum = _mm256_add_ps(
                _mm256_mul_ps(ss_lo, c_lo), _mm256_mul_ps(ss_hi, c_hi),
            );
            y_out[j] = hsum_avx2(sum) + d_vec[j] * conv_out[j];
            continue;
        }

        let dt_v = _mm256_set1_ps(dt[j]);
        let dtx_v = _mm256_set1_ps(dt[j] * conv_out[j]);

        // exp(a_log) — positive decay constants
        let a_lo = fast_exp_ps(_mm256_loadu_ps(alog_ptr.add(base)));
        let a_hi = fast_exp_ps(_mm256_loadu_ps(alog_ptr.add(base + 8)));

        // da = exp(-dt * exp(a_log))
        let da_lo = fast_exp_ps(_mm256_sub_ps(zero, _mm256_mul_ps(dt_v, a_lo)));
        let da_hi = fast_exp_ps(_mm256_sub_ps(zero, _mm256_mul_ps(dt_v, a_hi)));

        // ss = da * ss + dt*x * b
        let ns_lo = _mm256_add_ps(
            _mm256_mul_ps(da_lo, _mm256_loadu_ps(ss_ptr.add(base))),
            _mm256_mul_ps(dtx_v, b_lo),
        );
        let ns_hi = _mm256_add_ps(
            _mm256_mul_ps(da_hi, _mm256_loadu_ps(ss_ptr.add(base + 8))),
            _mm256_mul_ps(dtx_v, b_hi),
        );
        _mm256_storeu_ps(ss_ptr.add(base), ns_lo);
        _mm256_storeu_ps(ss_ptr.add(base + 8), ns_hi);

        // y[j] = dot(new_ss, c) + d_vec[j] * conv_out[j]
        let sum = _mm256_add_ps(
            _mm256_mul_ps(ns_lo, c_lo), _mm256_mul_ps(ns_hi, c_hi),
        );
        y_out[j] = hsum_avx2(sum) + d_vec[j] * conv_out[j];
    }
}

pub fn squared_relu_into(out: &mut [f32], x: &[f32]) {
    for (o, &x) in out.iter_mut().zip(x.iter()) { let r = x.max(0.0); *o = r * r; }
}

pub fn layer_norm_into(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

pub fn group_norm_into(out: &mut [f32], x: &[f32], num_groups: usize, weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len();
    let gs = n / num_groups;
    for g in 0..num_groups {
        let start = g * gs;
        let end = start + gs;
        let slice = &x[start..end];
        let mean: f32 = slice.iter().sum::<f32>() / gs as f32;
        let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / gs as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in start..end {
            out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

pub fn l2_normalize_per_head_inplace(x: &mut [f32], num_heads: usize, head_size: usize) {
    for h in 0..num_heads {
        let start = h * head_size;
        let end = start + head_size;
        let norm: f32 = x[start..end].iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
        for i in start..end { x[i] *= inv; }
    }
}

/// AVX2 kernel for f32 mat-vec WITHOUT FMA: vmulps + vaddps, 8-wide, 4-row ILP.
/// Separate mul+add matches scalar per-element rounding, but accumulation order
/// differs (8 parallel lanes vs sequential) — NOT bit-exact with scalar but
/// much closer than FMA. Deterministic across runs on same SIMD tier.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mat_vec_mul_avx2_nofma_into(out: &mut [f32], mat: &Tensor, vec: &[f32]) {
    let rows = mat.shape[0];
    let cols = mat.shape[1];
    let m = mat.data.as_ptr();
    let v = vec.as_ptr();
    let chunks = cols / 8;
    let rem_start = chunks * 8;
    let prefetch = !no_prefetch_override();

    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        let b0 = r * cols;
        let b1 = b0 + cols;
        let b2 = b1 + cols;
        let b3 = b2 + cols;

        // Prefetch first cache lines of next 4-row block
        if prefetch && r + 4 < rows {
            let nb = (r + 4) * cols;
            _mm_prefetch(m.add(nb) as *const i8, _MM_HINT_T0);
            _mm_prefetch(m.add(nb + cols) as *const i8, _MM_HINT_T0);
            _mm_prefetch(m.add(nb + 2 * cols) as *const i8, _MM_HINT_T0);
            _mm_prefetch(m.add(nb + 3 * cols) as *const i8, _MM_HINT_T0);
        }

        for ch in 0..chunks {
            let off = ch * 8;

            // Prefetch 4 chunks ahead (32 floats = 128 bytes = 2 cache lines)
            if prefetch && ch + 4 < chunks {
                let poff = (ch + 4) * 8;
                _mm_prefetch(m.add(b0 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(m.add(b1 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(m.add(b2 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(m.add(b3 + poff) as *const i8, _MM_HINT_T0);
            }

            let vv = _mm256_loadu_ps(v.add(off));
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(m.add(b0 + off)), vv));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(m.add(b1 + off)), vv));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(m.add(b2 + off)), vv));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(m.add(b3 + off)), vv));
        }

        let (mut s0, mut s1, mut s2, mut s3) = (
            hsum_f32_avx(a0), hsum_f32_avx(a1),
            hsum_f32_avx(a2), hsum_f32_avx(a3),
        );

        for c in rem_start..cols {
            let vc = *vec.get_unchecked(c);
            s0 += *mat.data.get_unchecked(b0 + c) * vc;
            s1 += *mat.data.get_unchecked(b1 + c) * vc;
            s2 += *mat.data.get_unchecked(b2 + c) * vc;
            s3 += *mat.data.get_unchecked(b3 + c) * vc;
        }
        out[r] = s0; out[r + 1] = s1; out[r + 2] = s2; out[r + 3] = s3;
    }

    for r in rows_4..rows {
        let mut acc = _mm256_setzero_ps();
        let base = r * cols;
        for ch in 0..chunks {
            let off = ch * 8;
            let vv = _mm256_loadu_ps(v.add(off));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(m.add(base + off)), vv));
        }
        let mut s = hsum_f32_avx(acc);
        for c in rem_start..cols {
            s += *mat.data.get_unchecked(base + c) * *vec.get_unchecked(c);
        }
        out[r] = s;
    }
}

/// AVX2+FMA kernel for f32 mat-vec: 8-wide FMA with 4-row ILP.
/// NOT dispatched in production — FMA rounding breaks cross-path determinism.
/// Kept for benchmarks and future opt-in use (e.g., --fast-math flag).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn mat_vec_mul_avx2_fma_into(out: &mut [f32], mat: &Tensor, vec: &[f32]) {
    let rows = mat.shape[0];
    let cols = mat.shape[1];
    let m = mat.data.as_ptr();
    let v = vec.as_ptr();
    let chunks = cols / 8;
    let rem_start = chunks * 8;

    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        let b0 = r * cols;
        let b1 = b0 + cols;
        let b2 = b1 + cols;
        let b3 = b2 + cols;

        for ch in 0..chunks {
            let off = ch * 8;
            let vv = _mm256_loadu_ps(v.add(off));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(m.add(b0 + off)), vv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(m.add(b1 + off)), vv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(m.add(b2 + off)), vv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(m.add(b3 + off)), vv, a3);
        }

        let (mut s0, mut s1, mut s2, mut s3) = (
            hsum_f32_avx(a0), hsum_f32_avx(a1),
            hsum_f32_avx(a2), hsum_f32_avx(a3),
        );

        for c in rem_start..cols {
            let vc = *vec.get_unchecked(c);
            s0 += *mat.data.get_unchecked(b0 + c) * vc;
            s1 += *mat.data.get_unchecked(b1 + c) * vc;
            s2 += *mat.data.get_unchecked(b2 + c) * vc;
            s3 += *mat.data.get_unchecked(b3 + c) * vc;
        }
        out[r] = s0; out[r + 1] = s1; out[r + 2] = s2; out[r + 3] = s3;
    }

    for r in rows_4..rows {
        let mut acc = _mm256_setzero_ps();
        let base = r * cols;
        for ch in 0..chunks {
            let off = ch * 8;
            let vv = _mm256_loadu_ps(v.add(off));
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(m.add(base + off)), vv, acc);
        }
        let mut s = hsum_f32_avx(acc);
        for c in rem_start..cols {
            s += *mat.data.get_unchecked(base + c) * *vec.get_unchecked(c);
        }
        out[r] = s;
    }
}

/// Minimum rows to activate multi-threaded dispatch.
/// Below this threshold, thread spawn overhead dominates.
const PARALLEL_MATMUL_MIN_ROWS: usize = 4096;

/// Number of threads for parallel matmul (R-054).
/// Cached at first call. Returns 1 if parallelism is unavailable.
fn parallel_matmul_threads() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CACHED: AtomicUsize = AtomicUsize::new(0);
    let v = CACHED.load(Ordering::Relaxed);
    if v != 0 { return v; }
    let n = std::thread::available_parallelism()
        .map(|p| p.get().min(4)) // Cap at 4 — diminishing returns beyond
        .unwrap_or(1);
    CACHED.store(n, Ordering::Relaxed);
    n
}

/// f32 mat-vec into pre-allocated output: out = mat @ vec.
/// Dispatches: AVX2 (no FMA) > scalar.
/// For large matrices (rows > 4096), splits rows across threads (R-054).
/// AVX2 uses vmulps+vaddps (no fmadd) — per-element rounding matches scalar,
/// but accumulation order differs (8-lane parallel vs sequential).
/// Deterministic on same SIMD tier; cross-tier may differ slightly.
/// Set REDUKT_NO_SIMD=1 to force scalar (cross-path roundtrip testing).
pub fn mat_vec_mul_into(out: &mut [f32], mat: &Tensor, vec: &[f32]) {
    let rows = mat.shape[0];
    let num_threads = parallel_matmul_threads();

    // Parallel dispatch for large matmuls (R-054)
    if rows >= PARALLEL_MATMUL_MIN_ROWS && num_threads > 1 {
        mat_vec_mul_parallel_into(out, mat, vec, num_threads);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() && is_x86_feature_detected!("avx2") {
        unsafe { mat_vec_mul_avx2_nofma_into(out, mat, vec) };
        return;
    }
    mat_vec_mul_scalar_into(out, mat, vec);
}

/// Multi-threaded f32 mat-vec: partitions rows across threads.
/// Each thread runs the full SIMD/scalar dispatch on its chunk.
/// Deterministic: same rows always assigned to same partition.
fn mat_vec_mul_parallel_into(
    out: &mut [f32], mat: &Tensor, vec: &[f32], num_threads: usize,
) {
    let rows = mat.shape[0];
    let cols = mat.shape[1];
    // Align chunks to 4-row boundary for ILP
    let chunk_raw = rows / num_threads;
    let chunk = (chunk_raw / 4) * 4;

    std::thread::scope(|s| {
        let mat_data: &[f32] = &mat.data;
        let mut remaining_out = out;
        let mut start_row = 0;

        for t in 0..num_threads {
            let end_row = if t == num_threads - 1 { rows } else { start_row + chunk };
            let nrows = end_row - start_row;
            let (chunk_out, rest) = remaining_out.split_at_mut(nrows);
            remaining_out = rest;

            let mat_offset = start_row * cols;
            let mat_slice = &mat_data[mat_offset..mat_offset + nrows * cols];
            let v = &vec[..cols];

            s.spawn(move || {
                #[cfg(target_arch = "x86_64")]
                if !no_simd_override() && is_x86_feature_detected!("avx2") {
                    unsafe {
                        mat_vec_mul_avx2_nofma_raw(chunk_out, mat_slice, v, nrows, cols);
                    }
                    return;
                }
                mat_vec_mul_scalar_raw(chunk_out, mat_slice, v, nrows, cols);
            });

            start_row = end_row;
        }
    });
}

/// Scalar f32 mat-vec on raw slice: 4-row ILP. Used by parallel dispatch.
fn mat_vec_mul_scalar_raw(
    out: &mut [f32], m: &[f32], v: &[f32], rows: usize, cols: usize,
) {
    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let row0 = &m[r * cols..(r + 1) * cols];
        let row1 = &m[(r + 1) * cols..(r + 2) * cols];
        let row2 = &m[(r + 2) * cols..(r + 3) * cols];
        let row3 = &m[(r + 3) * cols..(r + 4) * cols];
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for c in 0..cols {
            let vc = v[c];
            s0 += row0[c] * vc; s1 += row1[c] * vc;
            s2 += row2[c] * vc; s3 += row3[c] * vc;
        }
        out[r] = s0; out[r + 1] = s1; out[r + 2] = s2; out[r + 3] = s3;
    }
    for r in rows_4..rows {
        let row = &m[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for c in 0..cols { sum += row[c] * v[c]; }
        out[r] = sum;
    }
}

/// AVX2 noFMA f32 mat-vec on raw slice. Used by parallel dispatch.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mat_vec_mul_avx2_nofma_raw(
    out: &mut [f32], m: &[f32], v: &[f32], rows: usize, cols: usize,
) {
    let mp = m.as_ptr();
    let vp = v.as_ptr();
    let chunks = cols / 8;
    let rem_start = chunks * 8;
    let prefetch = !no_prefetch_override();

    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        let b0 = r * cols;
        let b1 = b0 + cols;
        let b2 = b1 + cols;
        let b3 = b2 + cols;

        if prefetch && r + 4 < rows {
            let nb = (r + 4) * cols;
            _mm_prefetch(mp.add(nb) as *const i8, _MM_HINT_T0);
            _mm_prefetch(mp.add(nb + cols) as *const i8, _MM_HINT_T0);
            _mm_prefetch(mp.add(nb + 2 * cols) as *const i8, _MM_HINT_T0);
            _mm_prefetch(mp.add(nb + 3 * cols) as *const i8, _MM_HINT_T0);
        }

        for ch in 0..chunks {
            let off = ch * 8;
            if prefetch && ch + 4 < chunks {
                let poff = (ch + 4) * 8;
                _mm_prefetch(mp.add(b0 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(mp.add(b1 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(mp.add(b2 + poff) as *const i8, _MM_HINT_T0);
                _mm_prefetch(mp.add(b3 + poff) as *const i8, _MM_HINT_T0);
            }
            let vv = _mm256_loadu_ps(vp.add(off));
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(mp.add(b0 + off)), vv));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(mp.add(b1 + off)), vv));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(mp.add(b2 + off)), vv));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(mp.add(b3 + off)), vv));
        }

        let (mut s0, mut s1, mut s2, mut s3) = (
            hsum_f32_avx(a0), hsum_f32_avx(a1),
            hsum_f32_avx(a2), hsum_f32_avx(a3),
        );
        for c in rem_start..cols {
            let vc = *v.get_unchecked(c);
            s0 += *m.get_unchecked(b0 + c) * vc;
            s1 += *m.get_unchecked(b1 + c) * vc;
            s2 += *m.get_unchecked(b2 + c) * vc;
            s3 += *m.get_unchecked(b3 + c) * vc;
        }
        out[r] = s0; out[r + 1] = s1; out[r + 2] = s2; out[r + 3] = s3;
    }

    for r in rows_4..rows {
        let mut acc = _mm256_setzero_ps();
        let base = r * cols;
        for ch in 0..chunks {
            let off = ch * 8;
            let vv = _mm256_loadu_ps(vp.add(off));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(mp.add(base + off)), vv));
        }
        let mut s = hsum_f32_avx(acc);
        for c in rem_start..cols {
            s += *m.get_unchecked(base + c) * *v.get_unchecked(c);
        }
        out[r] = s;
    }
}

/// Scalar fallback for f32 mat-vec: 4-row ILP. Delegates to raw version.
fn mat_vec_mul_scalar_into(out: &mut [f32], mat: &Tensor, vec: &[f32]) {
    mat_vec_mul_scalar_raw(out, &mat.data, vec, mat.shape[0], mat.shape[1]);
}

/// Quantize f32 vec into reusable i8 buffer, returns scale.
pub fn quantize_vec_i8_into(v: &[f32], q: &mut [i8]) -> f32 {
    let mut abs_max = 0.0f32;
    for &x in v { let a = x.abs(); if a > abs_max { abs_max = a; } }
    let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for i in 0..v.len() {
        q[i] = (v[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Returns true if REDUKT_NO_SIMD env var is set (forces scalar fallback).
fn no_simd_override() -> bool {
    std::env::var("REDUKT_NO_SIMD").map(|v| v == "1").unwrap_or(false)
}

/// Returns true if REDUKT_NO_PREFETCH env var is set (disables software prefetch).
fn no_prefetch_override() -> bool {
    std::env::var("REDUKT_NO_PREFETCH").map(|v| v == "1").unwrap_or(false)
}

/// Q8 mat-vec into pre-allocated output + reusable v_q buffer.
/// Dispatches: AVX-VNNI > AVX2 > scalar (4-row ILP).
/// Set REDUKT_NO_SIMD=1 to force scalar path (benchmarking).
pub fn q8_mat_vec_mul_into(out: &mut [f32], mat: &Q8Tensor, vec: &[f32], v_q: &mut [i8]) {
    let cols = mat.cols;
    let v_scale = quantize_vec_i8_into(&vec[..cols], &mut v_q[..cols]);

    #[cfg(target_arch = "x86_64")]
    if !no_simd_override() {
        if is_x86_feature_detected!("avxvnni") && cols % 32 == 0 {
            unsafe { q8_mat_vec_mul_vnni_into(out, mat, &v_q[..cols], v_scale) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { q8_mat_vec_mul_avx2_into(out, mat, &v_q[..cols], v_scale) };
            return;
        }
    }

    q8_mat_vec_mul_scalar_into(out, mat, v_q, v_scale);
}

/// Scalar fallback for q8_mat_vec_mul_into: i8*i8→i32 with 4-row ILP.
fn q8_mat_vec_mul_scalar_into(out: &mut [f32], mat: &Q8Tensor, v_q: &[i8], v_scale: f32) {
    let rows = mat.rows;
    let cols = mat.cols;
    let q = &mat.q_data;
    let scales = &mat.scales;
    let rows_4 = rows / 4 * 4;
    for r in (0..rows_4).step_by(4) {
        let q0 = &q[r * cols..(r + 1) * cols];
        let q1 = &q[(r + 1) * cols..(r + 2) * cols];
        let q2 = &q[(r + 2) * cols..(r + 3) * cols];
        let q3 = &q[(r + 3) * cols..(r + 4) * cols];
        let (mut s0, mut s1, mut s2, mut s3) = (0i32, 0i32, 0i32, 0i32);
        for c in 0..cols {
            let vc = v_q[c] as i32;
            s0 += q0[c] as i32 * vc; s1 += q1[c] as i32 * vc;
            s2 += q2[c] as i32 * vc; s3 += q3[c] as i32 * vc;
        }
        out[r] = s0 as f32 * (scales[r] * v_scale);
        out[r + 1] = s1 as f32 * (scales[r + 1] * v_scale);
        out[r + 2] = s2 as f32 * (scales[r + 2] * v_scale);
        out[r + 3] = s3 as f32 * (scales[r + 3] * v_scale);
    }
    for r in rows_4..rows {
        let qr = &q[r * cols..(r + 1) * cols];
        let mut sum = 0i32;
        for c in 0..cols { sum += qr[c] as i32 * v_q[c] as i32; }
        out[r] = sum as f32 * (scales[r] * v_scale);
    }
}

pub fn outer_product_into(out: &mut [f32], a: &[f32], b: &[f32]) {
    let n = a.len();
    let m = b.len();
    for i in 0..n {
        for j in 0..m {
            out[i * m + j] = a[i] * b[j];
        }
    }
}

#[cfg(test)]
pub fn mat_mat_mul_small_into(c: &mut [f32], a: &[f32], b: &[f32], n: usize) {
    for v in c[..n * n].iter_mut() { *v = 0.0; }
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
}

pub fn mat_vec_mul_small_into(out: &mut [f32], a: &[f32], v: &[f32], n: usize) {
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..n { sum += a[i * n + j] * v[j]; }
        out[i] = sum;
    }
}

/// Softmax over a 1D tensor
pub fn softmax(x: &Tensor) -> Tensor {
    let max_val = x.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        // All inputs are -infinity: return uniform distribution instead of NaN
        let n = x.data.len();
        let uniform = 1.0 / n as f32;
        return Tensor::from_data(vec![uniform; n], x.shape.clone());
    }
    let exps: Vec<f32> = x.data.iter().map(|&v| portable_expf(v - max_val)).collect();
    let sum: f32 = exps.iter().sum();
    let data: Vec<f32> = exps.iter().map(|&v| v / sum.max(f32::EPSILON)).collect();
    Tensor::from_data(data, x.shape.clone())
}

/// Zero-alloc softmax: write probabilities into caller-provided buffer.
/// Bit-exact with `softmax()` — uses division (not reciprocal multiplication).
pub fn softmax_into(logits: &[f32], out: &mut [f32]) {
    debug_assert_eq!(logits.len(), out.len());
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        let uniform = 1.0 / logits.len() as f32;
        for v in out.iter_mut() { *v = uniform; }
        return;
    }
    let mut sum = 0.0f32;
    for i in 0..logits.len() {
        let e = portable_expf(logits[i] - max_val);
        out[i] = e;
        sum += e;
    }
    let denom = sum.max(f32::EPSILON);
    for v in out.iter_mut() { *v /= denom; }
}

/// Compute entropy H = -sum(p * ln(p)) from logits (in nats).
/// Uses numerically stable softmax internally.
pub fn entropy_from_logits(logits: &[f32]) -> f32 {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| portable_expf(v - max_val)).collect();
    let sum: f32 = exps.iter().sum();
    let mut h = 0.0f32;
    for &e in &exps {
        let p = e / sum;
        if p > 1e-30 {
            h -= p * portable_lnf(p);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_matrix(rows: usize, cols: usize) -> Tensor {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows * cols {
            data.push(((i as f32 * 0.01).sin()) * 2.0);
        }
        Tensor::from_data(data, vec![rows, cols])
    }

    fn make_test_vec(cols: usize) -> Vec<f32> {
        (0..cols).map(|i| ((i as f32 * 0.03).cos()) * 1.5).collect()
    }

    #[test]
    fn test_mat_vec_mul_into_matches_allocating() {
        let mat = make_test_matrix(64, 256);
        let v = make_test_vec(256);
        let v_tensor = Tensor::from_data(v.clone(), vec![256]);

        let ref_out = mat_vec_mul(&mat, &v_tensor);
        let mut out = vec![0.0f32; 64];
        mat_vec_mul_into(&mut out, &mat, &v);

        for i in 0..64 {
            assert!((ref_out.data[i] - out[i]).abs() < 1e-2,
                "row {}: alloc={} into={}", i, ref_out.data[i], out[i]);
        }
    }

    #[test]
    fn test_f32_scalar_matches_dispatch() {
        let mat = make_test_matrix(64, 256);
        let v = make_test_vec(256);

        let mut out_scalar = vec![0.0f32; 64];
        mat_vec_mul_scalar_into(&mut out_scalar, &mat, &v);

        let mut out_dispatch = vec![0.0f32; 64];
        mat_vec_mul_into(&mut out_dispatch, &mat, &v);

        for i in 0..64 {
            assert!((out_scalar[i] - out_dispatch[i]).abs() < 1e-2,
                "row {}: scalar={} dispatch={}", i, out_scalar[i], out_dispatch[i]);
        }
    }

    #[test]
    fn test_f32_non_aligned_cols() {
        let mat = make_test_matrix(16, 100);
        let v = make_test_vec(100);
        let v_tensor = Tensor::from_data(v.clone(), vec![100]);

        let ref_out = mat_vec_mul(&mat, &v_tensor);
        let mut out = vec![0.0f32; 16];
        mat_vec_mul_into(&mut out, &mat, &v);

        for i in 0..16 {
            assert!((ref_out.data[i] - out[i]).abs() < 1e-2,
                "row {}: ref={} into={}", i, ref_out.data[i], out[i]);
        }
    }

    #[test]
    fn test_q8_dispatch_matches_scalar() {
        let mat_f32 = make_test_matrix(64, 256);
        let mat_q8 = Q8Tensor::from_f32(&mat_f32);
        let v = make_test_vec(256);
        let mut v_q = vec![0i8; 256];
        let v_scale = quantize_vec_i8_into(&v, &mut v_q);

        let mut out_scalar = vec![0.0f32; 64];
        q8_mat_vec_mul_scalar_into(&mut out_scalar, &mat_q8, &v_q, v_scale);

        let mut out_dispatch = vec![0.0f32; 64];
        let mut v_q2 = vec![0i8; 256];
        q8_mat_vec_mul_into(&mut out_dispatch, &mat_q8, &v, &mut v_q2);

        for i in 0..64 {
            assert!((out_scalar[i] - out_dispatch[i]).abs() < 1e-2,
                "row {}: scalar={} dispatch={}", i, out_scalar[i], out_dispatch[i]);
        }
    }

    #[test]
    fn test_q8_non_aligned_cols() {
        let mat_f32 = make_test_matrix(16, 100);
        let mat_q8 = Q8Tensor::from_f32(&mat_f32);
        let v = make_test_vec(100);

        let mut out = vec![0.0f32; 16];
        let mut v_q = vec![0i8; 100];
        q8_mat_vec_mul_into(&mut out, &mat_q8, &v, &mut v_q);

        let v_tensor = Tensor::from_data(v.clone(), vec![100]);
        let ref_out = q8_mat_vec_mul(&mat_q8, &v_tensor);

        for i in 0..16 {
            assert!((ref_out.data[i] - out[i]).abs() < 1e-2,
                "row {}: alloc={} into={}", i, ref_out.data[i], out[i]);
        }
    }

    #[test]
    fn test_q8_single_row() {
        let mat_f32 = make_test_matrix(1, 256);
        let mat_q8 = Q8Tensor::from_f32(&mat_f32);
        let v = make_test_vec(256);

        let mut out = vec![0.0f32; 1];
        let mut v_q = vec![0i8; 256];
        q8_mat_vec_mul_into(&mut out, &mat_q8, &v, &mut v_q);

        let v_tensor = Tensor::from_data(v, vec![256]);
        let ref_out = q8_mat_vec_mul(&mat_q8, &v_tensor);

        assert!((ref_out.data[0] - out[0]).abs() < 1e-2,
            "single row: ref={} got={}", ref_out.data[0], out[0]);
    }

    #[test]
    fn test_f32_single_row() {
        let mat = make_test_matrix(1, 256);
        let v = make_test_vec(256);
        let v_tensor = Tensor::from_data(v.clone(), vec![256]);

        let ref_out = mat_vec_mul(&mat, &v_tensor);
        let mut out = vec![0.0f32; 1];
        mat_vec_mul_into(&mut out, &mat, &v);

        assert!((ref_out.data[0] - out[0]).abs() < 1e-2,
            "single row: ref={} got={}", ref_out.data[0], out[0]);
    }

    #[test]
    fn test_q4_simd_matches_scalar() {
        // Q4 SIMD kernel must match scalar Q4 output (both use same precision)
        let mat_f32 = make_test_matrix(64, 768);
        let mat_q4 = Q4Tensor::from_f32(&mat_f32, 32);
        let v = make_test_vec(768);
        let mut v_q = vec![0i8; 768];
        let v_scale = quantize_vec_i8_into(&v, &mut v_q);

        let mut out_scalar = vec![0.0f32; 64];
        q4_mat_vec_mul_scalar_into(&mut out_scalar, &mat_q4, &v_q, v_scale);

        let mut out_dispatch = vec![0.0f32; 64];
        let mut v_q2 = vec![0i8; 768];
        q4_mat_vec_mul_into(&mut out_dispatch, &mat_q4, &v, &mut v_q2);

        for i in 0..64 {
            assert!((out_scalar[i] - out_dispatch[i]).abs() < 1e-1,
                "row {}: scalar={} dispatch={}", i, out_scalar[i], out_dispatch[i]);
        }
    }

    #[test]
    fn test_q4_simd_matches_allocating() {
        // Q4 SIMD into-buffer must match allocating Q4 mat_vec_mul
        let mat_f32 = make_test_matrix(32, 768);
        let mat_q4 = Q4Tensor::from_f32(&mat_f32, 32);
        let v = make_test_vec(768);

        let v_tensor = Tensor::from_data(v.clone(), vec![768]);
        let ref_out = q4_mat_vec_mul(&mat_q4, &v_tensor);

        let mut out = vec![0.0f32; 32];
        let mut v_q = vec![0i8; 768];
        q4_mat_vec_mul_into(&mut out, &mat_q4, &v, &mut v_q);

        for i in 0..32 {
            assert!((ref_out.data[i] - out[i]).abs() < 1e-1,
                "row {}: ref={} got={}", i, ref_out.data[i], out[i]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_q8_avx2_directly() {
        if !is_x86_feature_detected!("avx2") { return; }

        let mat_f32 = make_test_matrix(64, 256);
        let mat_q8 = Q8Tensor::from_f32(&mat_f32);
        let v = make_test_vec(256);
        let mut v_q = vec![0i8; 256];
        let v_scale = quantize_vec_i8_into(&v, &mut v_q);

        let mut out_scalar = vec![0.0f32; 64];
        q8_mat_vec_mul_scalar_into(&mut out_scalar, &mat_q8, &v_q, v_scale);

        let mut out_avx2 = vec![0.0f32; 64];
        unsafe { q8_mat_vec_mul_avx2_into(&mut out_avx2, &mat_q8, &v_q, v_scale) };

        for i in 0..64 {
            assert!((out_scalar[i] - out_avx2[i]).abs() < 1e-2,
                "row {}: scalar={} avx2={}", i, out_scalar[i], out_avx2[i]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f32_avx2_fma_directly() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let mat = make_test_matrix(64, 256);
        let v = make_test_vec(256);

        let mut out_scalar = vec![0.0f32; 64];
        mat_vec_mul_scalar_into(&mut out_scalar, &mat, &v);

        let mut out_avx2 = vec![0.0f32; 64];
        unsafe { mat_vec_mul_avx2_fma_into(&mut out_avx2, &mat, &v) };

        for i in 0..64 {
            assert!((out_scalar[i] - out_avx2[i]).abs() < 1e-2,
                "row {}: scalar={} avx2={}", i, out_scalar[i], out_avx2[i]);
        }
    }

    // ---- Bit-exact tests for cross-path determinism ----

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_q8_avx2_bit_exact() {
        if !is_x86_feature_detected!("avx2") { return; }

        let mat_f32 = make_test_matrix(768, 768);
        let mat_q8 = Q8Tensor::from_f32(&mat_f32);
        let v = make_test_vec(768);
        let mut v_q = vec![0i8; 768];
        let v_scale = quantize_vec_i8_into(&v, &mut v_q);

        let mut out_scalar = vec![0.0f32; 768];
        q8_mat_vec_mul_scalar_into(&mut out_scalar, &mat_q8, &v_q, v_scale);

        let mut out_avx2 = vec![0.0f32; 768];
        unsafe { q8_mat_vec_mul_avx2_into(&mut out_avx2, &mat_q8, &v_q, v_scale) };

        let mut mismatches = 0;
        for i in 0..768 {
            if out_scalar[i].to_bits() != out_avx2[i].to_bits() {
                if mismatches < 5 {
                    eprintln!("  row {}: scalar={:e} (0x{:08x}) avx2={:e} (0x{:08x}) delta={:e}",
                        i, out_scalar[i], out_scalar[i].to_bits(),
                        out_avx2[i], out_avx2[i].to_bits(),
                        (out_scalar[i] - out_avx2[i]).abs());
                }
                mismatches += 1;
            }
        }
        eprintln!("Q8 AVX2 bit-exact: {}/768 mismatches", mismatches);
        assert_eq!(mismatches, 0, "Q8 AVX2 must be bit-exact with scalar");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_q8_vnni_bit_exact() {
        if !is_x86_feature_detected!("avxvnni") { return; }

        let mat_f32 = make_test_matrix(768, 768);
        let mat_q8 = Q8Tensor::from_f32(&mat_f32);
        let v = make_test_vec(768);
        let mut v_q = vec![0i8; 768];
        let v_scale = quantize_vec_i8_into(&v, &mut v_q);

        let mut out_scalar = vec![0.0f32; 768];
        q8_mat_vec_mul_scalar_into(&mut out_scalar, &mat_q8, &v_q, v_scale);

        let mut out_vnni = vec![0.0f32; 768];
        unsafe { q8_mat_vec_mul_vnni_into(&mut out_vnni, &mat_q8, &v_q, v_scale) };

        let mut mismatches = 0;
        for i in 0..768 {
            if out_scalar[i].to_bits() != out_vnni[i].to_bits() {
                if mismatches < 5 {
                    eprintln!("  row {}: scalar={:e} (0x{:08x}) vnni={:e} (0x{:08x}) delta={:e}",
                        i, out_scalar[i], out_scalar[i].to_bits(),
                        out_vnni[i], out_vnni[i].to_bits(),
                        (out_scalar[i] - out_vnni[i]).abs());
                }
                mismatches += 1;
            }
        }
        eprintln!("Q8 VNNI bit-exact: {}/768 mismatches", mismatches);
        assert_eq!(mismatches, 0, "Q8 VNNI must be bit-exact with scalar");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f32_avx2_bit_exact() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let mat = make_test_matrix(768, 768);
        let v = make_test_vec(768);

        let mut out_scalar = vec![0.0f32; 768];
        mat_vec_mul_scalar_into(&mut out_scalar, &mat, &v);

        let mut out_avx2 = vec![0.0f32; 768];
        unsafe { mat_vec_mul_avx2_fma_into(&mut out_avx2, &mat, &v) };

        let mut mismatches = 0;
        for i in 0..768 {
            if out_scalar[i].to_bits() != out_avx2[i].to_bits() {
                if mismatches < 5 {
                    eprintln!("  row {}: scalar={:e} (0x{:08x}) avx2={:e} (0x{:08x}) delta={:e}",
                        i, out_scalar[i], out_scalar[i].to_bits(),
                        out_avx2[i], out_avx2[i].to_bits(),
                        (out_scalar[i] - out_avx2[i]).abs());
                }
                mismatches += 1;
            }
        }
        eprintln!("F32 AVX2+FMA bit-exact: {}/768 mismatches", mismatches);
        // F32 FMA is NOT expected to be bit-exact (FMA rounding + float non-associativity).
        // This is WHY mat_vec_mul_into does NOT dispatch to FMA — it would break
        // cross-path roundtrip. The kernel exists for future opt-in use only.
        assert!(mismatches > 0, "F32 FMA should NOT be bit-exact (confirms design decision)");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f32_avx2_nofma_directly() {
        if !is_x86_feature_detected!("avx2") { return; }

        let mat = make_test_matrix(64, 256);
        let v = make_test_vec(256);

        let mut out_scalar = vec![0.0f32; 64];
        mat_vec_mul_scalar_into(&mut out_scalar, &mat, &v);

        let mut out_avx2 = vec![0.0f32; 64];
        unsafe { mat_vec_mul_avx2_nofma_into(&mut out_avx2, &mat, &v) };

        for i in 0..64 {
            assert!((out_scalar[i] - out_avx2[i]).abs() < 1e-3,
                "row {}: scalar={} avx2_nofma={}", i, out_scalar[i], out_avx2[i]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f32_avx2_nofma_non_aligned() {
        if !is_x86_feature_detected!("avx2") { return; }

        let mat = make_test_matrix(16, 100); // 100 not divisible by 8
        let v = make_test_vec(100);

        let mut out_scalar = vec![0.0f32; 16];
        mat_vec_mul_scalar_into(&mut out_scalar, &mat, &v);

        let mut out_avx2 = vec![0.0f32; 16];
        unsafe { mat_vec_mul_avx2_nofma_into(&mut out_avx2, &mat, &v) };

        for i in 0..16 {
            assert!((out_scalar[i] - out_avx2[i]).abs() < 1e-3,
                "row {}: scalar={} avx2_nofma={}", i, out_scalar[i], out_avx2[i]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f32_avx2_nofma_single_row() {
        if !is_x86_feature_detected!("avx2") { return; }

        let mat = make_test_matrix(1, 768);
        let v = make_test_vec(768);

        let mut out_scalar = vec![0.0f32; 1];
        mat_vec_mul_scalar_into(&mut out_scalar, &mat, &v);

        let mut out_avx2 = vec![0.0f32; 1];
        unsafe { mat_vec_mul_avx2_nofma_into(&mut out_avx2, &mat, &v) };

        assert!((out_scalar[0] - out_avx2[0]).abs() < 1e-3,
            "single row: scalar={} avx2_nofma={}", out_scalar[0], out_avx2[0]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f32_avx2_nofma_bit_analysis() {
        if !is_x86_feature_detected!("avx2") { return; }

        let mat = make_test_matrix(768, 768);
        let v = make_test_vec(768);

        let mut out_scalar = vec![0.0f32; 768];
        mat_vec_mul_scalar_into(&mut out_scalar, &mat, &v);

        let mut out_avx2 = vec![0.0f32; 768];
        unsafe { mat_vec_mul_avx2_nofma_into(&mut out_avx2, &mat, &v) };

        let mut mismatches = 0;
        let mut max_delta = 0.0f32;
        for i in 0..768 {
            let delta = (out_scalar[i] - out_avx2[i]).abs();
            if delta > max_delta { max_delta = delta; }
            if out_scalar[i].to_bits() != out_avx2[i].to_bits() {
                if mismatches < 3 {
                    eprintln!("  row {}: scalar={:e} (0x{:08x}) nofma={:e} (0x{:08x}) d={:e}",
                        i, out_scalar[i], out_scalar[i].to_bits(),
                        out_avx2[i], out_avx2[i].to_bits(), delta);
                }
                mismatches += 1;
            }
        }
        eprintln!("F32 AVX2 noFMA: {}/768 mismatches, max_delta={:e}", mismatches, max_delta);
        // AVX2 noFMA accumulation order differs from scalar (8-lane parallel vs sequential).
        // This is expected and acceptable — deterministic on same SIMD tier.
        // Much closer than FMA (which had 733/768 mismatches).
    }

    // ---- SIMD transcendental accuracy tests (R-022) ----

    #[test]
    fn test_sigmoid_inplace_accuracy() {
        let inputs: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect();
        let mut simd_out = inputs.clone();
        let mut scalar_out = inputs.clone();

        sigmoid_inplace(&mut simd_out);
        for v in scalar_out.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }

        let mut max_err = 0.0f32;
        for i in 0..inputs.len() {
            let err = (simd_out[i] - scalar_out[i]).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("sigmoid_inplace max error: {:e} (over {} values)", max_err, inputs.len());
        assert!(max_err < 2e-7, "sigmoid max error {:e} exceeds 2e-7", max_err);
    }

    #[test]
    fn test_tanh_inplace_accuracy() {
        let inputs: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect();
        let mut simd_out = inputs.clone();
        let mut scalar_out = inputs.clone();

        tanh_inplace(&mut simd_out);
        for v in scalar_out.iter_mut() {
            *v = v.tanh();
        }

        let mut max_err = 0.0f32;
        for i in 0..inputs.len() {
            let err = (simd_out[i] - scalar_out[i]).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("tanh_inplace max error: {:e} (over {} values)", max_err, inputs.len());
        assert!(max_err < 5e-7, "tanh max error {:e} exceeds 5e-7", max_err);
    }

    #[test]
    fn test_sigmoid_add_into_accuracy() {
        let n = 768;
        let base: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 4.0).collect();
        let offset: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).cos() * 3.0).collect();

        let mut out_simd = vec![0.0f32; n];
        sigmoid_add_into(&mut out_simd, &base, &offset);

        let mut max_err = 0.0f32;
        for i in 0..n {
            let expected = 1.0 / (1.0 + (-(base[i] + offset[i])).exp());
            let err = (out_simd[i] - expected).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("sigmoid_add_into max error: {:e} (D={})", max_err, n);
        assert!(max_err < 2e-7, "sigmoid_add_into max error {:e} exceeds 2e-7", max_err);
    }

    #[test]
    fn test_decay_sigmoid_into_accuracy() {
        let n = 768;
        let base: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 4.0).collect();
        let offset: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).cos() * 3.0).collect();

        let mut out_simd = vec![0.0f32; n];
        decay_sigmoid_into(&mut out_simd, &base, &offset);

        let mut max_err = 0.0f32;
        for i in 0..n {
            let val = base[i] + offset[i];
            let expected = (-0.606531 * (1.0 / (1.0 + (-val).exp()))).exp();
            let err = (out_simd[i] - expected).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("decay_sigmoid_into max error: {:e} (D={})", max_err, n);
        assert!(max_err < 5e-7, "decay_sigmoid_into max error {:e} exceeds 5e-7", max_err);
    }

    #[test]
    fn test_sigmoid_mix_inplace_accuracy() {
        let n = 768;
        let base: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 4.0).collect();
        let offset: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).cos() * 3.0).collect();
        let v_first: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.019).sin() * 2.0).collect();
        let v_init: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).cos() * 1.5).collect();

        let mut out_simd = v_init.clone();
        sigmoid_mix_inplace(&mut out_simd, &v_first, &base, &offset);

        let mut max_err = 0.0f32;
        for i in 0..n {
            let mix = 1.0 / (1.0 + (-(base[i] + offset[i])).exp());
            let expected = v_init[i] + (v_first[i] - v_init[i]) * mix;
            let err = (out_simd[i] - expected).abs();
            if err > max_err { max_err = err; }
        }
        eprintln!("sigmoid_mix_inplace max error: {:e} (D={})", max_err, n);
        assert!(max_err < 5e-7, "sigmoid_mix_inplace max error {:e} exceeds 5e-7", max_err);
    }

    #[test]
    fn test_sigmoid_boundary_values() {
        // Test extreme inputs: very negative, zero, very positive
        let mut vals = vec![-100.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 100.0];
        let expected: Vec<f32> = vals.iter().map(|&x: &f32| 1.0 / (1.0 + (-x).exp())).collect();
        sigmoid_inplace(&mut vals);

        for i in 0..vals.len() {
            let err = (vals[i] - expected[i]).abs();
            assert!(err < 2e-7, "sigmoid({}) = {} expected {} err {:e}",
                expected[i], vals[i], expected[i], err);
        }
    }

    #[test]
    fn test_tanh_boundary_values() {
        let mut vals = vec![-100.0, -50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0, 100.0];
        let expected: Vec<f32> = vals.iter().map(|&x: &f32| x.tanh()).collect();
        tanh_inplace(&mut vals);

        for i in 0..vals.len() {
            let err = (vals[i] - expected[i]).abs();
            assert!(err < 5e-7, "tanh: got {} expected {} err {:e}",
                vals[i], expected[i], err);
        }
    }

    // ---- R-015: Rank-1 factorization correctness ----

    #[test]
    fn test_rank1_factorization_equivalence() {
        // Verify: state @ outer_product(u, v) ≈ outer_product(state @ u, v)
        let n = 64;
        let nn = n * n;

        // Create synthetic state matrix and vectors
        let state: Vec<f32> = (0..nn).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();
        let u: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos() * 1.5).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.019).sin() * 1.0).collect();

        // Method 1 (original): ab = outer_product(u, v), result = state @ ab
        let mut ab = vec![0.0f32; nn];
        outer_product_into(&mut ab, &u, &v);
        let mut result_original = vec![0.0f32; nn];
        mat_mat_mul_small_into(&mut result_original, &state, &ab, n);

        // Method 2 (factored): temp = state @ u, result = outer_product(temp, v)
        let mut temp = vec![0.0f32; n];
        mat_vec_mul_small_into(&mut temp, &state, &u, n);
        let mut result_factored = vec![0.0f32; nn];
        outer_product_into(&mut result_factored, &temp, &v);

        let mut max_err = 0.0f32;
        let mut mismatches = 0;
        for i in 0..nn {
            let err = (result_original[i] - result_factored[i]).abs();
            if err > max_err { max_err = err; }
            if result_original[i].to_bits() != result_factored[i].to_bits() {
                mismatches += 1;
            }
        }
        eprintln!("rank-1 factorization: {}/{} bit mismatches, max_err={:e}", mismatches, nn, max_err);
        assert!(max_err < 1e-4, "rank-1 max error {:e} too large", max_err);
    }

    // ---- Throughput benchmarks (run with --ignored --nocapture) ----

    fn bench_median_ns<F: FnMut()>(mut f: F, iters: usize) -> u64 {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            f();
            times.push(t0.elapsed().as_nanos() as u64);
        }
        times.sort();
        times[iters / 2]
    }

    #[test]
    #[ignore] // benchmark — run with: cargo test bench_simd -- --ignored --nocapture
    fn bench_simd_q8_768x768() {
        let mat_f32 = make_test_matrix(768, 768);
        let mat_q8 = Q8Tensor::from_f32(&mat_f32);
        let v = make_test_vec(768);
        let mut out = vec![0.0f32; 768];
        let mut v_q = vec![0i8; 768];

        let iters = 200;

        // Scalar baseline
        let v_scale = quantize_vec_i8_into(&v, &mut v_q);
        let ns_scalar = bench_median_ns(|| {
            q8_mat_vec_mul_scalar_into(&mut out, &mat_q8, &v_q, v_scale);
        }, iters);

        // Dispatch (VNNI > AVX2 > scalar)
        let ns_dispatch = bench_median_ns(|| {
            q8_mat_vec_mul_into(&mut out, &mat_q8, &v, &mut v_q);
        }, iters);

        // AVX2 directly (if available)
        #[cfg(target_arch = "x86_64")]
        let ns_avx2 = if is_x86_feature_detected!("avx2") {
            let v_scale = quantize_vec_i8_into(&v, &mut v_q);
            bench_median_ns(|| {
                unsafe { q8_mat_vec_mul_avx2_into(&mut out, &mat_q8, &v_q, v_scale) };
            }, iters)
        } else { 0 };

        #[cfg(target_arch = "x86_64")]
        let ns_vnni = if is_x86_feature_detected!("avxvnni") && 768 % 32 == 0 {
            let v_scale = quantize_vec_i8_into(&v, &mut v_q);
            bench_median_ns(|| {
                unsafe { q8_mat_vec_mul_vnni_into(&mut out, &mat_q8, &v_q, v_scale) };
            }, iters)
        } else { 0 };

        eprintln!("\n=== Q8 matmul benchmark: 768x768 ({} iters, median) ===", iters);
        eprintln!("  scalar:   {:>8} ns  ({:.1} μs)", ns_scalar, ns_scalar as f64 / 1000.0);
        eprintln!("  dispatch: {:>8} ns  ({:.1} μs)", ns_dispatch, ns_dispatch as f64 / 1000.0);
        #[cfg(target_arch = "x86_64")]
        {
            if ns_avx2 > 0 {
                eprintln!("  AVX2:     {:>8} ns  ({:.1} μs)  {:.2}x vs scalar",
                    ns_avx2, ns_avx2 as f64 / 1000.0,
                    ns_scalar as f64 / ns_avx2.max(1) as f64);
            }
            if ns_vnni > 0 {
                eprintln!("  VNNI:     {:>8} ns  ({:.1} μs)  {:.2}x vs scalar",
                    ns_vnni, ns_vnni as f64 / 1000.0,
                    ns_scalar as f64 / ns_vnni.max(1) as f64);
            }
        }
        eprintln!("  speedup:  {:.2}x (dispatch vs scalar)",
            ns_scalar as f64 / ns_dispatch.max(1) as f64);
    }

    #[test]
    #[ignore]
    fn bench_simd_f32_768x768() {
        let mat = make_test_matrix(768, 768);
        let v = make_test_vec(768);
        let mut out = vec![0.0f32; 768];

        let iters = 200;

        // Scalar baseline
        let ns_scalar = bench_median_ns(|| {
            mat_vec_mul_scalar_into(&mut out, &mat, &v);
        }, iters);

        // Dispatch (AVX2 noFMA > scalar)
        let ns_dispatch = bench_median_ns(|| {
            mat_vec_mul_into(&mut out, &mat, &v);
        }, iters);

        // AVX2 noFMA directly (if available)
        #[cfg(target_arch = "x86_64")]
        let ns_nofma = if is_x86_feature_detected!("avx2") {
            bench_median_ns(|| {
                unsafe { mat_vec_mul_avx2_nofma_into(&mut out, &mat, &v) };
            }, iters)
        } else { 0 };

        // AVX2+FMA directly (reference, not dispatched)
        #[cfg(target_arch = "x86_64")]
        let ns_fma = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            bench_median_ns(|| {
                unsafe { mat_vec_mul_avx2_fma_into(&mut out, &mat, &v) };
            }, iters)
        } else { 0 };

        eprintln!("\n=== F32 matmul benchmark: 768x768 ({} iters, median) ===", iters);
        eprintln!("  scalar:   {:>8} ns  ({:.1} μs)", ns_scalar, ns_scalar as f64 / 1000.0);
        eprintln!("  dispatch: {:>8} ns  ({:.1} μs)", ns_dispatch, ns_dispatch as f64 / 1000.0);
        #[cfg(target_arch = "x86_64")]
        {
            if ns_nofma > 0 {
                eprintln!("  noFMA:    {:>8} ns  ({:.1} μs)  {:.2}x vs scalar",
                    ns_nofma, ns_nofma as f64 / 1000.0,
                    ns_scalar as f64 / ns_nofma.max(1) as f64);
            }
            if ns_fma > 0 {
                eprintln!("  FMA:      {:>8} ns  ({:.1} μs)  {:.2}x vs scalar",
                    ns_fma, ns_fma as f64 / 1000.0,
                    ns_scalar as f64 / ns_fma.max(1) as f64);
            }
        }
        eprintln!("  speedup:  {:.2}x (dispatch vs scalar)",
            ns_scalar as f64 / ns_dispatch.max(1) as f64);
    }

    #[test]
    #[ignore]
    fn bench_simd_q8_65536x768() {
        // Head matrix (vocab_size × dim) — largest matmul in RWKV-7
        let mat_f32 = make_test_matrix(65536, 768);
        let mat_q8 = Q8Tensor::from_f32(&mat_f32);
        let v = make_test_vec(768);
        let mut out = vec![0.0f32; 65536];
        let mut v_q = vec![0i8; 768];

        let iters = 20;

        let v_scale = quantize_vec_i8_into(&v, &mut v_q);
        let ns_scalar = bench_median_ns(|| {
            q8_mat_vec_mul_scalar_into(&mut out, &mat_q8, &v_q, v_scale);
        }, iters);

        let ns_dispatch = bench_median_ns(|| {
            q8_mat_vec_mul_into(&mut out, &mat_q8, &v, &mut v_q);
        }, iters);

        eprintln!("\n=== Q8 matmul benchmark: 65536x768 (head) ({} iters, median) ===", iters);
        eprintln!("  scalar:   {:>10} ns  ({:.1} ms)", ns_scalar, ns_scalar as f64 / 1e6);
        eprintln!("  dispatch: {:>10} ns  ({:.1} ms)", ns_dispatch, ns_dispatch as f64 / 1e6);
        eprintln!("  speedup:  {:.2}x (dispatch vs scalar)",
            ns_scalar as f64 / ns_dispatch.max(1) as f64);
    }

    #[test]
    #[ignore]
    fn bench_simd_transcendentals_768() {
        let n = 768;
        let base: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 4.0).collect();
        let offset: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).cos() * 3.0).collect();
        let v_first: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.019).sin() * 2.0).collect();
        let mut out = vec![0.0f32; n];
        let mut buf = base.clone();

        let iters = 5000;

        // sigmoid_inplace
        let ns_sigmoid = bench_median_ns(|| {
            buf.copy_from_slice(&base);
            sigmoid_inplace(&mut buf);
        }, iters);

        // tanh_inplace
        let ns_tanh = bench_median_ns(|| {
            buf.copy_from_slice(&base);
            tanh_inplace(&mut buf);
        }, iters);

        // decay_sigmoid_into
        let ns_decay = bench_median_ns(|| {
            decay_sigmoid_into(&mut out, &base, &offset);
        }, iters);

        // sigmoid_add_into
        let ns_sigadd = bench_median_ns(|| {
            sigmoid_add_into(&mut out, &base, &offset);
        }, iters);

        // sigmoid_mix_inplace
        let ns_sigmix = bench_median_ns(|| {
            buf.copy_from_slice(&base);
            sigmoid_mix_inplace(&mut buf, &v_first, &base, &offset);
        }, iters);

        // Scalar baselines
        let ns_sigmoid_scalar = bench_median_ns(|| {
            buf.copy_from_slice(&base);
            for v in buf.iter_mut() { *v = 1.0 / (1.0 + (-*v).exp()); }
        }, iters);

        let ns_decay_scalar = bench_median_ns(|| {
            for i in 0..n {
                let val = base[i] + offset[i];
                out[i] = (-0.606531 * (1.0 / (1.0 + (-val).exp()))).exp();
            }
        }, iters);

        eprintln!("\n=== Transcendental benchmark: D={} ({} iters, median) ===", n, iters);
        eprintln!("  sigmoid_inplace:    {:>6} ns  (scalar: {} ns, {:.2}x)",
            ns_sigmoid, ns_sigmoid_scalar,
            ns_sigmoid_scalar as f64 / ns_sigmoid.max(1) as f64);
        eprintln!("  tanh_inplace:       {:>6} ns", ns_tanh);
        eprintln!("  decay_sigmoid_into: {:>6} ns  (scalar: {} ns, {:.2}x)",
            ns_decay, ns_decay_scalar,
            ns_decay_scalar as f64 / ns_decay.max(1) as f64);
        eprintln!("  sigmoid_add_into:   {:>6} ns", ns_sigadd);
        eprintln!("  sigmoid_mix_inplace:{:>6} ns", ns_sigmix);
    }

    // ---- R-013b: Portable transcendental bit-exactness ----

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn portable_expf_matches_simd() {
        // Verify portable_expf produces bit-identical results to fast_exp_ps
        let test_vals: Vec<f32> = (-870..=880).map(|i| i as f32 * 0.1).collect();
        let mut max_ulp = 0u32;
        let mut mismatches = 0;
        for chunk in test_vals.chunks(8) {
            if chunk.len() < 8 { continue; }
            let mut simd_out = [0.0f32; 8];
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let v = _mm256_loadu_ps(chunk.as_ptr());
                    _mm256_storeu_ps(simd_out.as_mut_ptr(), fast_exp_ps(v));
                }
                for i in 0..8 {
                    let scalar = portable_expf(chunk[i]);
                    let s_bits = scalar.to_bits();
                    let v_bits = simd_out[i].to_bits();
                    if s_bits != v_bits {
                        let ulp = (s_bits as i64 - v_bits as i64).unsigned_abs() as u32;
                        if ulp > max_ulp { max_ulp = ulp; }
                        mismatches += 1;
                    }
                }
            }
        }
        eprintln!("portable_expf vs fast_exp_ps: {} mismatches, max {} ULP", mismatches, max_ulp);
        assert_eq!(mismatches, 0, "portable_expf must be bit-exact with fast_exp_ps");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn portable_sigmoidf_matches_simd() {
        let test_vals: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let mut mismatches = 0;
        for chunk in test_vals.chunks(8) {
            if chunk.len() < 8 { continue; }
            let mut simd_out = [0.0f32; 8];
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let v = _mm256_loadu_ps(chunk.as_ptr());
                    _mm256_storeu_ps(simd_out.as_mut_ptr(), fast_sigmoid_ps(v));
                }
                for i in 0..8 {
                    let scalar = portable_sigmoidf(chunk[i]);
                    if scalar.to_bits() != simd_out[i].to_bits() {
                        if mismatches < 5 {
                            eprintln!("  sigmoid({:.4}): scalar={:e} (0x{:08x}) simd={:e} (0x{:08x})",
                                chunk[i], scalar, scalar.to_bits(), simd_out[i], simd_out[i].to_bits());
                        }
                        mismatches += 1;
                    }
                }
            }
        }
        eprintln!("portable_sigmoidf vs fast_sigmoid_ps: {} mismatches out of {}",
            mismatches, test_vals.len());
        assert_eq!(mismatches, 0, "portable_sigmoidf must be bit-exact with fast_sigmoid_ps");
    }

    // ---- R-038: Mamba SIMD kernel tests ----

    #[test]
    fn silu_inplace_matches_scalar() {
        let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        let mut simd_buf = input.clone();
        let mut scalar_buf = input.clone();
        silu_inplace(&mut simd_buf);
        for v in scalar_buf.iter_mut() { *v = portable_silu(*v); }
        for i in 0..input.len() {
            let diff = (simd_buf[i] - scalar_buf[i]).abs();
            assert!(diff < 1e-5, "silu_inplace mismatch at {}: {} vs {}",
                i, simd_buf[i], scalar_buf[i]);
        }
    }

    #[test]
    fn gated_silu_mul_matches_scalar() {
        let y_buf: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let z: Vec<f32> = (0..32).map(|i| (i as f32 - 8.0) * 0.2).collect();
        let mut out_simd = vec![0.0f32; 32];
        let mut out_scalar = vec![0.0f32; 32];
        gated_silu_mul(&mut out_simd, &y_buf, &z);
        for i in 0..32 { out_scalar[i] = y_buf[i] * portable_silu(z[i]); }
        for i in 0..32 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            assert!(diff < 1e-5, "gated_silu_mul mismatch at {}: {} vs {}",
                i, out_simd[i], out_scalar[i]);
        }
    }

    #[test]
    fn ssm_recurrence_d16_matches_scalar() {
        let d_inner = 4;
        let ds = 16;
        let n = d_inner * ds;
        // Deterministic test data
        let dt: Vec<f32> = (0..d_inner).map(|i| 0.5 + i as f32 * 0.1).collect();
        let conv_out: Vec<f32> = (0..d_inner).map(|i| 1.0 - i as f32 * 0.3).collect();
        let a_log: Vec<f32> = (0..n).map(|i| -1.0 - (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..ds).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let c: Vec<f32> = (0..ds).map(|i| 0.05 * (ds as f32 - i as f32)).collect();
        let d_vec: Vec<f32> = (0..d_inner).map(|i| 0.2 + i as f32 * 0.05).collect();
        let ss_init: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();

        // Scalar reference
        let mut y_scalar = vec![0.0f32; d_inner];
        let mut ss_scalar = ss_init.clone();
        ssm_recurrence_d16_scalar(
            &mut y_scalar, &mut ss_scalar, &dt, &conv_out,
            &a_log, &b, &c, &d_vec, d_inner,
        );

        // SIMD path
        let mut y_simd = vec![0.0f32; d_inner];
        let mut ss_simd = ss_init.clone();
        ssm_recurrence_d16(
            &mut y_simd, &mut ss_simd, &dt, &conv_out,
            &a_log, &b, &c, &d_vec, d_inner,
        );

        for j in 0..d_inner {
            let diff = (y_simd[j] - y_scalar[j]).abs();
            let tol = y_scalar[j].abs().max(1e-6) * 1e-4;
            assert!(diff < tol, "ssm y[{}]: simd={} scalar={} diff={}",
                j, y_simd[j], y_scalar[j], diff);
        }
        for i in 0..n {
            let diff = (ss_simd[i] - ss_scalar[i]).abs();
            let tol = ss_scalar[i].abs().max(1e-6) * 1e-4;
            assert!(diff < tol, "ssm ss[{}]: simd={} scalar={} diff={}",
                i, ss_simd[i], ss_scalar[i], diff);
        }
    }

    #[test]
    fn portable_expf_known_values() {
        // Verify known values to catch polynomial bugs
        let cases: [(f32, f32); 5] = [
            (0.0, 1.0),
            (1.0, std::f32::consts::E),
            (-1.0, 1.0 / std::f32::consts::E),
            (-87.0, (-87.0f32).exp()),  // near underflow
            (88.0, (88.0f32).exp()),    // near overflow
        ];
        for &(x, expected) in &cases {
            let got = portable_expf(x);
            let rel_err = ((got - expected) / expected).abs();
            assert!(rel_err < 1e-6, "portable_expf({}) = {}, expected {}, rel_err {:e}",
                x, got, expected, rel_err);
        }
    }

    #[test]
    fn portable_lnf_known_values() {
        let cases: [(f32, f32); 5] = [
            (1.0, 0.0),
            (std::f32::consts::E, 1.0),
            (0.5, (0.5f32).ln()),
            (10.0, (10.0f32).ln()),
            (1e-10, (1e-10f32).ln()),
        ];
        for &(x, expected) in &cases {
            let got = portable_lnf(x);
            let abs_err = (got - expected).abs();
            let tol = if expected.abs() > 1.0 { expected.abs() * 1e-6 } else { 1e-6 };
            assert!(abs_err < tol, "portable_lnf({}) = {}, expected {}, err {:e}",
                x, got, expected, abs_err);
        }
    }
}
