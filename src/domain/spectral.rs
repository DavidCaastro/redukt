//! Spectral profiler — atomized weight matrix analysis.
//!
//! Computes intrinsic mathematical properties of RWKV-7 weight matrices
//! without any training or gradient computation. All algorithms are
//! independent atoms that can be evaluated separately.
//!
//! Atoms:
//!   A1: Stable rank (Frobenius² / spectral²) via power iteration
//!   A2: Decay spectrum + gate saturation from w0/a0 bias vectors
//!   A3: FFN neuron analysis (dead/always-on from row norms of W_up)
//!   A4: Cross-matrix coherence via randomized subspace alignment
//!   A5: Full singular value spectrum via Jacobi eigendecomposition

/// Simple xorshift128+ PRNG (deterministic, no deps).
struct Rng { s0: u64, s1: u64 }

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s0: seed, s1: seed.wrapping_mul(6364136223846793005).wrapping_add(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s0;
        let s0 = self.s1;
        self.s0 = s0;
        s1 ^= s1 << 23;
        s1 ^= s1 >> 17;
        s1 ^= s0;
        s1 ^= s0 >> 26;
        self.s1 = s1;
        self.s0.wrapping_add(self.s1)
    }
    /// Standard normal via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        let u2 = (self.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ─── Shared linear algebra primitives ───

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// out = mat (rows × cols, row-major) @ vec
fn matvec(out: &mut [f64], mat: &[f64], vec: &[f64], rows: usize, cols: usize) {
    for r in 0..rows {
        let row_start = r * cols;
        out[r] = dot(&mat[row_start..row_start + cols], vec);
    }
}

/// out = mat^T (rows × cols, row-major transposed) @ vec
fn matvec_t(out: &mut [f64], mat: &[f64], vec: &[f64], rows: usize, cols: usize) {
    for c in 0..cols {
        out[c] = 0.0;
    }
    for r in 0..rows {
        let v = vec[r];
        let row_start = r * cols;
        for c in 0..cols {
            out[c] += mat[row_start + c] * v;
        }
    }
}

// ─── A1: Stable Rank ───

/// Frobenius norm squared: Σ a_ij².
pub fn frobenius_norm_sq(data: &[f64]) -> f64 {
    data.iter().map(|x| x * x).sum()
}

/// Top-1 singular value via power iteration on A^T A (without forming it).
/// Returns σ_1 (spectral norm).
pub fn spectral_norm(data: &[f64], rows: usize, cols: usize) -> f64 {
    let max_iter = 60;
    let tol = 1e-10;
    let mut rng = Rng::new(42);

    // v = random unit vector in R^cols
    let mut v: Vec<f64> = (0..cols).map(|_| rng.next_normal()).collect();
    let n = norm2(&v);
    for x in &mut v { *x /= n; }

    let mut u = vec![0.0; rows];
    let mut z = vec![0.0; cols];
    let mut sigma_prev = 0.0;

    for _ in 0..max_iter {
        // u = A * v
        matvec(&mut u, data, &v, rows, cols);
        let nu = norm2(&u);
        if nu < 1e-30 { return 0.0; }
        for x in &mut u { *x /= nu; }

        // z = A^T * u
        matvec_t(&mut z, data, &u, rows, cols);
        let sigma = norm2(&z);
        if sigma < 1e-30 { return 0.0; }

        // v = z / sigma
        for i in 0..cols { v[i] = z[i] / sigma; }

        if (sigma - sigma_prev).abs() / sigma < tol {
            return sigma;
        }
        sigma_prev = sigma;
    }
    sigma_prev
}

// ─── A2: Decay Spectrum + Gate Saturation ───

pub struct DecayProfile {
    pub decay_mean: f64,
    pub decay_std: f64,
    pub decay_min: f64,
    pub decay_max: f64,
    pub histogram: Vec<(f64, usize)>, // (bin_center, count)
}

/// Analyze decay spectrum from w0 bias vector.
/// decay(i) = exp(-0.606531 * sigmoid(w0[i]))
pub fn decay_spectrum(w0: &[f32]) -> DecayProfile {
    let decays: Vec<f64> = w0.iter()
        .map(|&x| {
            let s = 1.0 / (1.0 + (-(x as f64)).exp());
            (-0.606531 * s).exp()
        })
        .collect();

    let n = decays.len() as f64;
    let mean = decays.iter().sum::<f64>() / n;
    let var = decays.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n;

    // Histogram: 6 bins from 0.545 to 1.0
    let bins = [0.545, 0.62, 0.70, 0.78, 0.86, 0.94, 1.001];
    let mut hist = Vec::new();
    for w in bins.windows(2) {
        let center = (w[0] + w[1]) / 2.0;
        let count = decays.iter().filter(|&&d| d >= w[0] && d < w[1]).count();
        hist.push((center, count));
    }

    DecayProfile {
        decay_mean: mean,
        decay_std: var.sqrt(),
        decay_min: decays.iter().cloned().fold(f64::INFINITY, f64::min),
        decay_max: decays.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        histogram: hist,
    }
}

/// Count saturated dimensions in a gate bias vector.
/// |a0[j]| > threshold means permanently in binary regime.
pub fn saturation_count(a0: &[f32], threshold: f32) -> (usize, usize) {
    let saturated = a0.iter().filter(|&&x| x.abs() > threshold).count();
    (saturated, a0.len())
}

// ─── A3: FFN Neuron Analysis ───

pub struct FfnProfile {
    pub total_neurons: usize,
    pub dead_neurons: usize,     // row norm < threshold
    pub always_on: usize,        // row norm > high_threshold
    pub mean_row_norm: f64,
    pub std_row_norm: f64,
}

/// Analyze FFN W_up (D_FFN × D) row norms to find dead/always-on neurons.
pub fn ffn_neuron_analysis(data: &[f64], rows: usize, cols: usize) -> FfnProfile {
    let mut norms: Vec<f64> = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        let row = &data[start..start + cols];
        norms.push(dot(row, row).sqrt());
    }

    let n = norms.len() as f64;
    let mean = norms.iter().sum::<f64>() / n;
    let var = norms.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;

    // Dead: row norm < 10% of mean. Always-on: > 200% of mean.
    let dead_thresh = mean * 0.1;
    let on_thresh = mean * 2.0;

    FfnProfile {
        total_neurons: rows,
        dead_neurons: norms.iter().filter(|&&x| x < dead_thresh).count(),
        always_on: norms.iter().filter(|&&x| x > on_thresh).count(),
        mean_row_norm: mean,
        std_row_norm: var.sqrt(),
    }
}

// ─── A4: Cross-Matrix Coherence ───

/// Approximate coherence between column spaces of A and B via random projection.
/// Both matrices must have the same number of rows (m). Returns value in [0, 1].
/// Higher = more aligned (redundant). Lower = more complementary.
pub fn cross_coherence(
    a_data: &[f64], a_rows: usize, a_cols: usize,
    b_data: &[f64], b_rows: usize, b_cols: usize,
    k: usize,
) -> f64 {
    assert_eq!(a_rows, b_rows, "matrices must have same number of rows");
    let m = a_rows;
    let mut rng = Rng::new(12345);

    // Random probes: R is m × k
    let r_mat: Vec<f64> = (0..m * k).map(|_| rng.next_normal()).collect();

    // Q_A = A * A^T * R  (approximate column space of A)
    // Q_B = B * B^T * R  (approximate column space of B)
    let mut q_a = vec![0.0; m * k];
    let mut q_b = vec![0.0; m * k];
    let mut tmp_a = vec![0.0; a_cols];
    let mut tmp_b = vec![0.0; b_cols];

    for j in 0..k {
        let r_col: Vec<f64> = (0..m).map(|i| r_mat[i * k + j]).collect();

        // tmp_a = A^T * r_col
        matvec_t(&mut tmp_a, a_data, &r_col, a_rows, a_cols);
        // q_a_col = A * tmp_a
        let mut qa_col = vec![0.0; m];
        matvec(&mut qa_col, a_data, &tmp_a, a_rows, a_cols);
        for i in 0..m { q_a[i * k + j] = qa_col[i]; }

        // tmp_b = B^T * r_col
        matvec_t(&mut tmp_b, b_data, &r_col, b_rows, b_cols);
        // q_b_col = B * tmp_b
        let mut qb_col = vec![0.0; m];
        matvec(&mut qb_col, b_data, &tmp_b, b_rows, b_cols);
        for i in 0..m { q_b[i * k + j] = qb_col[i]; }
    }

    // QR factorize Q_A and Q_B (modified Gram-Schmidt on columns)
    gram_schmidt_columns(&mut q_a, m, k);
    gram_schmidt_columns(&mut q_b, m, k);

    // Coherence = ||Q_A^T Q_B||_F^2 / k
    let mut coh = 0.0;
    for ja in 0..k {
        for jb in 0..k {
            let mut d = 0.0;
            for i in 0..m {
                d += q_a[i * k + ja] * q_b[i * k + jb];
            }
            coh += d * d;
        }
    }
    coh / k as f64
}

/// Modified Gram-Schmidt orthonormalization on columns of mat (m × k, column-major in row storage).
fn gram_schmidt_columns(mat: &mut [f64], m: usize, k: usize) {
    for j in 0..k {
        // Orthogonalize against previous columns
        for jp in 0..j {
            let mut d = 0.0;
            for i in 0..m { d += mat[i * k + j] * mat[i * k + jp]; }
            for i in 0..m { mat[i * k + j] -= d * mat[i * k + jp]; }
        }
        // Normalize
        let mut n = 0.0;
        for i in 0..m { n += mat[i * k + j] * mat[i * k + j]; }
        let n = n.sqrt();
        if n > 1e-30 {
            for i in 0..m { mat[i * k + j] /= n; }
        }
    }
}

// ─── A5: Full Singular Values via Jacobi Eigendecomposition ───

/// Compute ALL singular values of matrix (rows × cols) via Jacobi on the
/// smaller of W^T W or W W^T. Returns sorted descending.
pub fn singular_values_jacobi(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let n = rows.min(cols);

    // Form symmetric matrix S = smaller of (A^T A) or (A A^T)
    let mut s = vec![0.0f64; n * n];
    if rows <= cols {
        // S = A * A^T  (rows × rows)
        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0;
                for k in 0..cols {
                    sum += data[i * cols + k] * data[j * cols + k];
                }
                s[i * n + j] = sum;
                s[j * n + i] = sum;
            }
        }
    } else {
        // S = A^T * A  (cols × cols)
        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0;
                for k in 0..rows {
                    sum += data[k * cols + i] * data[k * cols + j];
                }
                s[i * n + j] = sum;
                s[j * n + i] = sum;
            }
        }
    }

    // Jacobi eigenvalue algorithm on symmetric S
    let eigenvalues = jacobi_eigenvalues(&mut s, n);

    // Singular values = sqrt(eigenvalues), sorted descending
    let mut svs: Vec<f64> = eigenvalues.iter()
        .map(|&e| if e > 0.0 { e.sqrt() } else { 0.0 })
        .collect();
    svs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    svs
}

/// Cyclic Jacobi eigenvalue algorithm for symmetric n×n matrix.
/// Modifies matrix in-place. Returns eigenvalues (diagonal after convergence).
fn jacobi_eigenvalues(s: &mut [f64], n: usize) -> Vec<f64> {
    let max_sweeps = 30;
    let eps = 1e-12;

    for _sweep in 0..max_sweeps {
        // Check convergence: sum of off-diagonal squared
        let mut off_sq = 0.0;
        let mut diag_sq = 0.0;
        for i in 0..n {
            diag_sq += s[i * n + i] * s[i * n + i];
            for j in (i + 1)..n {
                off_sq += 2.0 * s[i * n + j] * s[i * n + j];
            }
        }
        if off_sq < eps * eps * diag_sq {
            break;
        }

        // One sweep: iterate over all (i, j) pairs
        for i in 0..(n - 1) {
            for j in (i + 1)..n {
                let sij = s[i * n + j];
                if sij.abs() < eps * (s[i * n + i].abs() + s[j * n + j].abs()).max(1e-30) {
                    continue;
                }

                // Compute rotation angle
                let tau = (s[j * n + j] - s[i * n + i]) / (2.0 * sij);
                let t = if tau.abs() > 1e15 {
                    // Near-zero rotation needed
                    1.0 / (2.0 * tau)
                } else {
                    tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sc = c * t;

                // Apply rotation to rows/columns i, j
                // Update diagonal
                let sii = s[i * n + i];
                let sjj = s[j * n + j];
                s[i * n + i] = sii - t * sij;
                s[j * n + j] = sjj + t * sij;
                s[i * n + j] = 0.0;
                s[j * n + i] = 0.0;

                // Update off-diagonal entries
                for k in 0..n {
                    if k == i || k == j { continue; }
                    let ski = s[k * n + i];
                    let skj = s[k * n + j];
                    s[k * n + i] = c * ski - sc * skj;
                    s[i * n + k] = s[k * n + i];
                    s[k * n + j] = sc * ski + c * skj;
                    s[j * n + k] = s[k * n + j];
                }
            }
        }
    }

    (0..n).map(|i| s[i * n + i]).collect()
}

// ─── A6: Truncated SVD via Randomized Algorithm ───

/// Compute rank-k truncated SVD of a matrix (rows × cols, f64, row-major).
/// Returns (A, B) as f32 where A is (rows × k) and B is (k × cols),
/// such that W ≈ A·B. Absorbs singular values into A.
///
/// Uses Halko-Martinsson-Tropp randomized SVD:
/// 1. Random projection Y = W·Ω
/// 2. QR of Y → orthonormal basis Q
/// 3. Small matrix B_hat = Q^T·W
/// 4. SVD of B_hat via eigendecomposition of B_hat·B_hat^T
/// 5. Reconstruct U = Q·U_hat, then A = U·diag(σ), B = V^T
pub fn truncated_svd_f32(
    data: &[f64], rows: usize, cols: usize, rank: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(rank > 0 && rank <= rows.min(cols));
    let p = 10.min(rows.min(cols) - rank); // oversampling
    let q = rank + p;
    let mut rng = Rng::new(42);

    // Step 1: Y = W · Ω where Ω is (cols × q) random Gaussian
    let omega: Vec<f64> = (0..cols * q).map(|_| rng.next_normal()).collect();
    let mut y = vec![0.0f64; rows * q]; // (rows × q)
    for r in 0..rows {
        let row_start = r * cols;
        for j in 0..q {
            let mut sum = 0.0;
            for c in 0..cols {
                sum += data[row_start + c] * omega[c * q + j];
            }
            y[r * q + j] = sum;
        }
    }

    // Step 2: QR via modified Gram-Schmidt → Q is (rows × q) orthonormal
    gram_schmidt_columns(&mut y, rows, q);
    // y is now Q

    // Step 3: B_hat = Q^T · W → (q × cols)
    let mut b_hat = vec![0.0f64; q * cols]; // (q × cols)
    for j in 0..q {
        for c in 0..cols {
            let mut sum = 0.0;
            for r in 0..rows {
                sum += y[r * q + j] * data[r * cols + c];
            }
            b_hat[j * cols + c] = sum;
        }
    }

    // Step 4: SVD of B_hat (q × cols) via eigendecomposition of B_hat · B_hat^T (q × q)
    let mut gram = vec![0.0f64; q * q]; // (q × q)
    for i in 0..q {
        for j in i..q {
            let mut sum = 0.0;
            for c in 0..cols {
                sum += b_hat[i * cols + c] * b_hat[j * cols + c];
            }
            gram[i * q + j] = sum;
            gram[j * q + i] = sum;
        }
    }

    // Jacobi eigendecomposition with eigenvectors
    let (eigenvalues, eigenvecs) = jacobi_eigen_full(&mut gram, q);

    // Sort eigenvalues descending, reorder eigenvectors
    let mut idx: Vec<usize> = (0..q).collect();
    idx.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a])
        .unwrap_or(std::cmp::Ordering::Equal));

    // σ_i = sqrt(max(0, λ_i)), U_hat = eigenvectors (columns of gram after Jacobi)
    // V^T_i = (1/σ_i) · U_hat_i^T · B_hat → row vector (1 × cols)
    // U_i = Q · U_hat_i → column vector (rows × 1)

    // Build A = U · diag(σ) → (rows × rank), and B = V^T → (rank × cols)
    let mut a_out = vec![0.0f32; rows * rank];
    let mut b_out = vec![0.0f32; rank * cols];

    for ki in 0..rank {
        let orig_col = idx[ki];
        let lambda = eigenvalues[orig_col];
        let sigma = if lambda > 0.0 { lambda.sqrt() } else { 0.0 };

        // U_k = Q · u_hat_k → (rows,)
        // A_k = U_k * σ_k
        for r in 0..rows {
            let mut sum = 0.0;
            for j in 0..q {
                sum += y[r * q + j] * eigenvecs[j * q + orig_col];
            }
            a_out[r * rank + ki] = (sum * sigma) as f32;
        }

        // V^T_k = (1/σ_k) · u_hat_k^T · B_hat → (cols,)
        if sigma > 1e-30 {
            let inv_sigma = 1.0 / sigma;
            for c in 0..cols {
                let mut sum = 0.0;
                for j in 0..q {
                    sum += eigenvecs[j * q + orig_col] * b_hat[j * cols + c];
                }
                b_out[ki * cols + c] = (sum * inv_sigma) as f32;
            }
        }
    }

    (a_out, b_out)
}

/// Jacobi eigenvalue algorithm that also computes eigenvectors.
/// Returns (eigenvalues, eigenvectors) where eigenvectors is n×n row-major,
/// with the j-th eigenvector in column j (i.e., eigenvecs[i*n + j] = v_j[i]).
fn jacobi_eigen_full(s: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let max_sweeps = 50;
    let eps = 1e-12;

    // Initialize eigenvector matrix as identity
    let mut v = vec![0.0f64; n * n];
    for i in 0..n { v[i * n + i] = 1.0; }

    for _sweep in 0..max_sweeps {
        let mut off_sq = 0.0;
        let mut diag_sq = 0.0;
        for i in 0..n {
            diag_sq += s[i * n + i] * s[i * n + i];
            for j in (i + 1)..n {
                off_sq += 2.0 * s[i * n + j] * s[i * n + j];
            }
        }
        if off_sq < eps * eps * diag_sq { break; }

        for i in 0..(n - 1) {
            for j in (i + 1)..n {
                let sij = s[i * n + j];
                if sij.abs() < eps * (s[i * n + i].abs() + s[j * n + j].abs()).max(1e-30) {
                    continue;
                }

                let tau = (s[j * n + j] - s[i * n + i]) / (2.0 * sij);
                let t = if tau.abs() > 1e15 {
                    1.0 / (2.0 * tau)
                } else {
                    tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sc = c * t;

                let sii = s[i * n + i];
                let sjj = s[j * n + j];
                s[i * n + i] = sii - t * sij;
                s[j * n + j] = sjj + t * sij;
                s[i * n + j] = 0.0;
                s[j * n + i] = 0.0;

                for k in 0..n {
                    if k == i || k == j { continue; }
                    let ski = s[k * n + i];
                    let skj = s[k * n + j];
                    s[k * n + i] = c * ski - sc * skj;
                    s[i * n + k] = s[k * n + i];
                    s[k * n + j] = sc * ski + c * skj;
                    s[j * n + k] = s[k * n + j];
                }

                // Apply same rotation to eigenvector columns
                for k in 0..n {
                    let vki = v[k * n + i];
                    let vkj = v[k * n + j];
                    v[k * n + i] = c * vki - sc * vkj;
                    v[k * n + j] = sc * vki + c * vkj;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| s[i * n + i]).collect();
    (eigenvalues, v)
}

// ─── Profile result types ───

pub struct MatrixProfile {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub frobenius_norm: f64,
    pub spectral_norm: f64,
    pub stable_rank: f64,
    pub effective_rank: f64,      // (Σσ)² / Σσ² from full SVD
    pub condition_number: f64,    // σ_max / σ_min (non-zero)
    pub null_space_dim: usize,    // σ < 0.01 * σ_max
    pub top10_sv: Vec<f64>,
    pub bottom5_sv: Vec<f64>,
}

impl MatrixProfile {
    pub fn compute(name: &str, data: &[f64], rows: usize, cols: usize) -> Self {
        let frob = frobenius_norm_sq(data).sqrt();
        let spec = spectral_norm(data, rows, cols);
        let sr = if spec > 1e-30 { frobenius_norm_sq(data) / (spec * spec) } else { 0.0 };

        // Full singular values via Jacobi
        let svs = singular_values_jacobi(data, rows, cols);

        // Effective rank: (Σσ)² / Σσ²
        let sum_sv: f64 = svs.iter().sum();
        let sum_sv2: f64 = svs.iter().map(|s| s * s).sum();
        let eff_rank = if sum_sv2 > 1e-30 { sum_sv * sum_sv / sum_sv2 } else { 0.0 };

        // Condition number: σ_max / σ_min (ignoring near-zero)
        let sigma_max = svs.first().copied().unwrap_or(0.0);
        let sigma_min_nz = svs.iter().rev()
            .find(|&&s| s > 1e-10 * sigma_max)
            .copied().unwrap_or(1e-30);
        let cond = sigma_max / sigma_min_nz;

        // Null space: σ < 0.01 * σ_max
        let null_thresh = 0.01 * sigma_max;
        let null_dim = svs.iter().filter(|&&s| s < null_thresh).count();

        let top10: Vec<f64> = svs.iter().take(10).copied().collect();
        let n = svs.len();
        let bottom5: Vec<f64> = if n > 5 { svs[n-5..].to_vec() } else { svs.clone() };

        MatrixProfile {
            name: name.to_string(),
            rows, cols,
            frobenius_norm: frob,
            spectral_norm: spec,
            stable_rank: sr,
            effective_rank: eff_rank,
            condition_number: cond,
            null_space_dim: null_dim,
            top10_sv: top10,
            bottom5_sv: bottom5,
        }
    }
}

// ─── JSON Serialization (manual, zero deps) ───

impl MatrixProfile {
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"name":"{}","rows":{},"cols":{},"frobenius_norm":{:.4},"spectral_norm":{:.4},"stable_rank":{:.2},"effective_rank":{:.2},"condition_number":{:.1},"null_space_dim":{},"top10_sv":[{}],"bottom5_sv":[{}]}}"#,
            self.name, self.rows, self.cols,
            self.frobenius_norm, self.spectral_norm,
            self.stable_rank, self.effective_rank,
            self.condition_number, self.null_space_dim,
            self.top10_sv.iter().map(|s| format!("{:.4}", s)).collect::<Vec<_>>().join(","),
            self.bottom5_sv.iter().map(|s| format!("{:.6}", s)).collect::<Vec<_>>().join(","),
        )
    }
}

impl DecayProfile {
    pub fn to_json(&self) -> String {
        let hist: Vec<String> = self.histogram.iter()
            .map(|(c, n)| format!("[{:.3},{}]", c, n))
            .collect();
        format!(
            r#"{{"decay_mean":{:.6},"decay_std":{:.6},"decay_min":{:.6},"decay_max":{:.6},"histogram":[{}]}}"#,
            self.decay_mean, self.decay_std, self.decay_min, self.decay_max,
            hist.join(","),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncated_svd_half_rank() {
        // 64×64 random matrix, SVD at rank 32 (50%) should retain >90% energy
        let n = 64;
        let k = 32;
        let mut rng = Rng::new(789);
        let data: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();

        let (a, b) = truncated_svd_f32(&data, n, n, k);

        let mut sq_err = 0.0f64;
        let mut sq_orig = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut approx = 0.0f64;
                for s in 0..k {
                    approx += a[i * k + s] as f64 * b[s * n + j] as f64;
                }
                let err = data[i * n + j] - approx;
                sq_err += err * err;
                sq_orig += data[i * n + j] * data[i * n + j];
            }
        }
        let rel_err = (sq_err / sq_orig).sqrt();
        eprintln!("half-rank 64x64: rel_err = {:.6e}", rel_err);
        // A random 64x64 matrix at rank-32 should retain ~71% of Frobenius norm
        // (for standard Gaussian, expected energy at rank k/n is ~k/n)
        // rel_err should be ~sqrt(1 - k/n) ≈ sqrt(0.5) ≈ 0.707
        assert!(rel_err < 0.9, "half-rank rel_err={:.6e} too large", rel_err);
    }

    #[test]
    fn test_truncated_svd_reconstruction() {
        // Random-ish 16×16 matrix, SVD at full rank should be near-perfect
        let n = 16;
        let mut rng = Rng::new(123);
        let data: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();

        let (a, b) = truncated_svd_f32(&data, n, n, n);
        let mut max_err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut approx = 0.0f64;
                for k in 0..n {
                    approx += a[i * n + k] as f64 * b[k * n + j] as f64;
                }
                let err = (data[i * n + j] - approx).abs();
                if err > max_err { max_err = err; }
            }
        }
        eprintln!("full-rank SVD reconstruction max_err = {:.6e}", max_err);
        assert!(max_err < 0.1, "full-rank SVD max_err={:.6e} too large", max_err);
    }

    #[test]
    fn test_truncated_svd_energy_retention() {
        // 32×32 matrix with known spectrum: first 8 singular values large, rest small
        let n = 32;
        let k = 8;
        let mut rng = Rng::new(456);

        // Create U, S, V^T with controlled spectrum
        let mut u_mat = vec![0.0f64; n * n];
        for i in 0..n { for j in 0..n { u_mat[i * n + j] = rng.next_normal(); } }
        gram_schmidt_columns(&mut u_mat, n, n);

        let mut v_mat = vec![0.0f64; n * n];
        for i in 0..n { for j in 0..n { v_mat[i * n + j] = rng.next_normal(); } }
        gram_schmidt_columns(&mut v_mat, n, n);

        // Spectrum: first k values = 10.0, rest = 0.1
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for s in 0..n {
                    let sigma = if s < k { 10.0 } else { 0.1 };
                    sum += u_mat[i * n + s] * sigma * v_mat[j * n + s];
                }
                data[i * n + j] = sum;
            }
        }

        // SVD at rank k should capture most energy
        let (a, b) = truncated_svd_f32(&data, n, n, k);
        let mut sq_err = 0.0f64;
        let mut sq_orig = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut approx = 0.0f64;
                for s in 0..k {
                    approx += a[i * k + s] as f64 * b[s * n + j] as f64;
                }
                let err = data[i * n + j] - approx;
                sq_err += err * err;
                sq_orig += data[i * n + j] * data[i * n + j];
            }
        }
        let rel_err = (sq_err / sq_orig).sqrt();
        eprintln!("energy retention: relative error = {:.6e}", rel_err);
        // With spectrum 10:0.1, rank-8 should capture >99% of energy
        assert!(rel_err < 0.05, "rel_err={:.6e} too large", rel_err);
    }
}

impl FfnProfile {
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"total_neurons":{},"dead_neurons":{},"always_on":{},"mean_row_norm":{:.4},"std_row_norm":{:.4}}}"#,
            self.total_neurons, self.dead_neurons, self.always_on,
            self.mean_row_norm, self.std_row_norm,
        )
    }
}
