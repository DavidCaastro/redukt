//! Mamba SSM inference — single-step RNN mode, zero dependencies.
//!
//! Reference: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//! Model: state-spaces/mamba-130m-hf (d_model=768, n_layer=24, vocab=50280)

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use crate::domain::tensor::{
    Tensor, portable_softplus, rms_norm_into, layer_norm_into,
    silu_inplace, gated_silu_mul, ssm_recurrence_d16,
};
use crate::domain::gemv::{f32_rows, gemv_f32_into, gemv_q8w_into, Q8wMatrix};
use crate::domain::par_pool;
use crate::domain::backbone::WeightProfile;
use crate::domain::ram_monitor::check_ram_or_abort;
use crate::infrastructure::rwkv7::safetensors::SafeTensorsFile;

// --- Config ---

pub struct MambaConfig {
    pub d_model: usize,   // 768
    pub d_inner: usize,   // 1536 (2 * d_model)
    pub d_state: usize,   // 16
    pub d_conv: usize,    // 4
    pub dt_rank: usize,   // 48
    pub n_layer: usize,   // 24
    pub vocab_size: usize, // 50280
}

impl MambaConfig {
    fn from_d_model(d_model: usize, n_layer: usize, vocab_size: usize) -> Self {
        Self {
            d_model,
            d_inner: 2 * d_model,
            d_state: 16,
            d_conv: 4,
            dt_rank: (d_model + 15) / 16, // ceil(d_model / 16)
            n_layer,
            vocab_size,
        }
    }
}

// --- Weight matrices (R-099) ---

/// Q8 profile recipe (AZTH v5 profile code 1) — part of the format contract:
/// changing it changes the logits, so it would need a new profile code.
/// in_proj, out_proj and the (tied) embedding/LM head are int8 with one f32
/// scale per 64 columns. Measured at 10 KB (Mamba-130M): a per-row scale costs
/// +6.6-7.0% BPB on enwik8/facturae (outlier columns of the tied embedding
/// dominate each row's max|w|); 64-column groups keep every file < +1.3%.
const Q8_GROUP: usize = 64;

/// A large weight matrix in the active numeric profile.
/// F32: dense f32. Q8: symmetric int8, f32 scale per group, weight-only.
enum MambaMatrix {
    F32(Tensor),
    Q8(Q8wMatrix),
}

impl MambaMatrix {
    fn new(t: Tensor, profile: WeightProfile) -> Self {
        match profile {
            WeightProfile::F32 => Self::F32(t),
            WeightProfile::Q8 => Self::Q8(Q8wMatrix::quantize(&t, Q8_GROUP)),
        }
    }

    /// out = W @ v (pooled, row-partitioned, bit-identical for any thread count).
    #[inline]
    fn gemv(&self, out: &mut [f32], v: &[f32]) {
        match self {
            Self::F32(t) => gemv_f32_into(out, t, v),
            Self::Q8(q) => gemv_q8w_into(out, q, v),
        }
    }

    /// Copy (dequantize) row `row` into `out`.
    fn row_into(&self, out: &mut [f32], row: usize) {
        match self {
            Self::F32(t) => {
                let d = t.shape[1];
                out[..d].copy_from_slice(&t.data[row * d..(row + 1) * d]);
            }
            Self::Q8(q) => q.dequant_row_into(out, row),
        }
    }

    /// Resident bytes (== bytes streamed by one GEMV).
    fn bytes(&self) -> usize {
        match self {
            Self::F32(t) => t.numel() * 4,
            Self::Q8(q) => q.bytes(),
        }
    }

    fn as_f32(&self) -> Option<&Tensor> {
        match self {
            Self::F32(t) => Some(t),
            Self::Q8(_) => None,
        }
    }
}

// --- Per-channel phases on the pool (R-098) ---

/// Channels per pooled block for conv1d+SiLU. Block starts are multiples of 8,
/// so every element takes the same SIMD-body/scalar-tail path as in a
/// full-length call → bit-identical to the serial loop.
const CONV_BLOCK: usize = 256;
/// Channels per pooled block for dt_proj+softplus+SSM+gate (multiple of 8).
const SSM_BLOCK: usize = 64;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Causal conv1d (shift register) + SiLU for channels `j0..j0+out.len()`.
/// `cs` = this block's conv state, `x` = this block's x-branch inputs.
fn conv_silu_block(lw: &MambaLayerWeights, dc: usize, j0: usize, cs: &mut [f32], x: &[f32], out: &mut [f32]) {
    for (jj, o) in out.iter_mut().enumerate() {
        let j = j0 + jj;
        let st = &mut cs[jj * (dc - 1)..(jj + 1) * (dc - 1)];
        let w = &lw.conv1d_w[j * dc..(j + 1) * dc];
        let mut val = 0.0f32;
        for k in 0..(dc - 1) {
            val += st[k] * w[k];
        }
        val += x[jj] * w[dc - 1];
        val += lw.conv1d_b[j];
        // Shift state left, insert new value
        for k in 0..(dc - 2) {
            st[k] = st[k + 1];
        }
        st[dc - 2] = x[jj];
        *o = val;
    }
    silu_inplace(out);
}

/// Read-only per-layer inputs of the selective-scan phase.
struct ScanInputs<'a> {
    dt_raw: &'a [f32],
    b: &'a [f32],
    c: &'a [f32],
    conv_out: &'a [f32],
    z: &'a [f32],
}

/// dt = softplus(dt_proj·dt_raw + bias) → SSM recurrence → y·silu(z), for
/// channels `j0..j0+dt.len()`. Per-channel independent → partition-invariant.
fn scan_block(
    lw: &MambaLayerWeights, dr: usize, ds: usize, j0: usize, inp: &ScanInputs,
    dt: &mut [f32], ss: &mut [f32], y: &mut [f32], gated: &mut [f32],
) {
    let n = dt.len();
    let j1 = j0 + n;
    f32_rows(dt, &lw.dt_proj_w.data[j0 * dr..j1 * dr], inp.dt_raw, dr);
    for (jj, d) in dt.iter_mut().enumerate() {
        *d = portable_softplus(*d + lw.dt_proj_b[j0 + jj]);
    }
    ssm_recurrence_d16(
        y, ss, dt, &inp.conv_out[j0..j1], &lw.a_log[j0 * ds..j1 * ds],
        inp.b, inp.c, &lw.d_vec[j0..j1], n,
    );
    gated_silu_mul(gated, y, &inp.z[j0..j1]);
}

// --- Layer weights ---

struct MambaLayerWeights {
    norm_w: Vec<f32>,          // (d_model,)
    norm_b: Option<Vec<f32>>,  // (d_model,) — LayerNorm bias (R-088, None = RMSNorm)
    in_proj_w: MambaMatrix,    // (2*d_inner, d_model) — F32 or Q8 (R-099)
    conv1d_w: Vec<f32>,        // (d_inner, d_conv) — flat, used element-wise
    conv1d_b: Vec<f32>,        // (d_inner,)
    x_proj_w: Tensor,          // (dt_rank + 2*d_state, d_inner)
    dt_proj_w: Tensor,         // (d_inner, dt_rank)
    dt_proj_b: Vec<f32>,       // (d_inner,)
    a_log: Vec<f32>,           // (d_inner, d_state) — flat, used element-wise
    d_vec: Vec<f32>,           // (d_inner,)
    out_proj_w: MambaMatrix,   // (d_model, d_inner) — F32 or Q8 (R-099)
}

impl MambaLayerWeights {
    /// Weight bytes read per token by this layer.
    fn bytes(&self) -> usize {
        self.in_proj_w.bytes() + self.out_proj_w.bytes()
            + (self.norm_w.len() + self.norm_b.as_ref().map_or(0, |b| b.len())
                + self.conv1d_w.len() + self.conv1d_b.len() + self.x_proj_w.numel()
                + self.dt_proj_w.numel() + self.dt_proj_b.len() + self.a_log.len()
                + self.d_vec.len()) * 4
    }
}

// --- Model ---

pub struct MambaModel {
    pub config: MambaConfig,
    emb_w: MambaMatrix,             // (vocab_size, d_model) — embedding (tied head)
    lm_head_w: Option<MambaMatrix>, // (vocab_size, d_model) — separate LM head (untied)
    profile: WeightProfile,         // R-099: numeric weight profile
    layers: Vec<MambaLayerWeights>,
    norm_f_w: Vec<f32>,        // (d_model,)
    norm_f_b: Option<Vec<f32>>, // (d_model,) — LayerNorm bias (R-088)
    // R-062: highway layer skip configuration
    pub highway_start: usize,   // first layer to check (0 = disabled)
    pub highway_thr: f32,       // ||result||/||x|| threshold
    pub resync_interval: usize, // force full forward every N tokens
}

// --- State ---

#[derive(Clone)]
pub struct MambaState {
    // Per layer: conv_state (d_inner, d_conv-1) + ssm_state (d_inner, d_state)
    conv_states: Vec<Vec<f32>>,  // n_layer × (d_inner * (d_conv-1))
    ssm_states: Vec<Vec<f32>>,   // n_layer × (d_inner * d_state)
}

impl MambaState {
    pub fn new(config: &MambaConfig) -> Self {
        let conv_len = config.d_inner * (config.d_conv - 1);
        let ssm_len = config.d_inner * config.d_state;
        Self {
            conv_states: (0..config.n_layer).map(|_| vec![0.0; conv_len]).collect(),
            ssm_states: (0..config.n_layer).map(|_| vec![0.0; ssm_len]).collect(),
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.conv_states { s.fill(0.0); }
        for s in &mut self.ssm_states { s.fill(0.0); }
    }
}

// --- Scratch ---

pub struct MambaScratch {
    pub x: Vec<f32>,           // (d_model,) — normed hidden state after forward_layers_normed
    xz: Vec<f32>,              // (2*d_inner,)
    conv_out: Vec<f32>,        // (d_inner,)
    z_proj: Vec<f32>,          // (dt_rank + 2*d_state,)
    dt: Vec<f32>,              // (d_inner,)
    y: Vec<f32>,               // (d_inner,)
    gated: Vec<f32>,           // (d_inner,)
    result: Vec<f32>,          // (d_model,)
    dt_raw: Vec<f32>,          // (dt_rank,)
    normed: Vec<f32>,          // (d_model,) — avoids per-step clone
    // R-059: activation sparsity measurement
    pub sparsity_total_channels: u64,
    pub sparsity_near_zero: u64,
    // R-062: highway layer skip measurement
    pub highway_token_count: u64,
    pub highway_layers_run: u64,
    pub highway_layers_skipped: u64,
}

impl MambaScratch {
    fn new(config: &MambaConfig) -> Self {
        Self {
            x: vec![0.0; config.d_model],
            xz: vec![0.0; 2 * config.d_inner],
            conv_out: vec![0.0; config.d_inner],
            z_proj: vec![0.0; config.dt_rank + 2 * config.d_state],
            dt: vec![0.0; config.d_inner],
            y: vec![0.0; config.d_inner],
            gated: vec![0.0; config.d_inner],
            result: vec![0.0; config.d_model],
            dt_raw: vec![0.0; config.dt_rank],
            normed: vec![0.0; config.d_model],
            sparsity_total_channels: 0,
            sparsity_near_zero: 0,
            highway_token_count: 0,
            highway_layers_run: 0,
            highway_layers_skipped: 0,
        }
    }
}

// --- Model loading ---

impl MambaModel {
    /// Load with the default F32 profile.
    pub fn load(path: &Path) -> Self {
        Self::load_with_profile(path, WeightProfile::F32)
    }

    /// Load and convert in_proj/out_proj/embedding(/head) to `profile` (R-099).
    /// A_log, D, dt_proj, x_proj, conv1d and norms always stay f32.
    pub fn load_with_profile(path: &Path, profile: WeightProfile) -> Self {
        let st_path = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_path_buf()
        };
        eprintln!("[mamba] loading from {} ...", st_path.display());
        check_ram_or_abort(90.0, "mamba model load");

        let st = SafeTensorsFile::open(&st_path)
            .unwrap_or_else(|e| panic!("[mamba] {}", e));

        // Detect config from embedding shape
        let emb = st.get_tensor("backbone.embeddings.weight");
        let vocab_size = emb.shape[0];
        let d_model = emb.shape[1];

        // Count layers
        let mut n_layer = 0;
        while st.has_tensor(&format!("backbone.layers.{}.norm.weight", n_layer)) {
            n_layer += 1;
        }
        eprintln!(
            "[mamba] detected: d_model={}, n_layer={}, vocab={}",
            d_model, n_layer, vocab_size
        );

        let config = MambaConfig::from_d_model(d_model, n_layer, vocab_size);
        let di = config.d_inner;
        let ds = config.d_state;
        let dr = config.dt_rank;

        let norm_f = st.get_tensor("backbone.norm_f.weight");
        // R-088: detect LayerNorm (has bias) vs RMSNorm (no bias)
        let norm_f_b = if st.has_tensor("backbone.norm_f.bias") {
            let b = st.get_tensor("backbone.norm_f.bias");
            eprintln!("[mamba] LayerNorm detected (norm has bias) — using layer_norm");
            Some(b.data)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("backbone.layers.{}", i);
            let norm = st.get_tensor(&format!("{}.norm.weight", p));
            let norm_b = if st.has_tensor(&format!("{}.norm.bias", p)) {
                Some(st.get_tensor(&format!("{}.norm.bias", p)).data)
            } else {
                None
            };
            let in_proj = st.get_tensor(&format!("{}.mixer.in_proj.weight", p));
            let conv_w = st.get_tensor(&format!("{}.mixer.conv1d.weight", p));
            let conv_b = st.get_tensor(&format!("{}.mixer.conv1d.bias", p));
            let x_proj = st.get_tensor(&format!("{}.mixer.x_proj.weight", p));
            let dt_proj = st.get_tensor(&format!("{}.mixer.dt_proj.weight", p));
            let dt_bias = st.get_tensor(&format!("{}.mixer.dt_proj.bias", p));
            let a_log = st.get_tensor(&format!("{}.mixer.A_log", p));
            let d_vec = st.get_tensor(&format!("{}.mixer.D", p));
            let out_proj = st.get_tensor(&format!("{}.mixer.out_proj.weight", p));

            layers.push(MambaLayerWeights {
                norm_w: norm.data,
                norm_b,
                in_proj_w: MambaMatrix::new(
                    Tensor::from_data(in_proj.data, vec![2 * di, d_model]), profile,
                ),
                conv1d_w: conv_w.data, // flat (d_inner * d_conv)
                conv1d_b: conv_b.data,
                x_proj_w: Tensor::from_data(x_proj.data, vec![dr + 2 * ds, di]),
                dt_proj_w: Tensor::from_data(dt_proj.data, vec![di, dr]),
                dt_proj_b: dt_bias.data,
                a_log: a_log.data, // flat (d_inner * d_state)
                d_vec: d_vec.data,
                out_proj_w: MambaMatrix::new(
                    Tensor::from_data(out_proj.data, vec![d_model, di]), profile,
                ),
            });

            if (i + 1) % 8 == 0 || i == n_layer - 1 {
                eprintln!("[mamba] loaded layer {}/{}", i + 1, n_layer);
            }
        }

        let emb_t = Tensor::from_data(emb.data, vec![vocab_size, d_model]);
        let emb_w = MambaMatrix::new(emb_t, profile);

        // R-088: detect separate lm_head.weight (untied embeddings, e.g. MambaByte)
        let lm_head_w = if st.has_tensor("lm_head.weight") {
            let lm = st.get_tensor("lm_head.weight");
            eprintln!("[mamba] separate lm_head.weight detected (untied embeddings)");
            let lm_t = Tensor::from_data(lm.data, vec![vocab_size, d_model]);
            Some(MambaMatrix::new(lm_t, profile))
        } else {
            None
        };

        let layer_bytes: usize = layers.iter().map(|l| l.bytes()).sum();
        let norm_f_bytes = (norm_f.data.len() + norm_f_b.as_ref().map_or(0, |b| b.len())) * 4;
        let head_bytes = lm_head_w.as_ref().unwrap_or(&emb_w).bytes();
        let mut total_bytes = emb_w.bytes() + layer_bytes + norm_f_bytes;
        if let Some(ref lm) = lm_head_w {
            total_bytes += lm.bytes();
        }
        let mb = |b: usize| b as f64 / 1024.0 / 1024.0;
        eprintln!("[mamba] model loaded: {:.1} MB (profile {})", mb(total_bytes), profile);
        eprintln!(
            "[mamba] weight bytes/token: {:.1} MB (layers {:.1} + head {:.1}), {} GEMV threads",
            mb(layer_bytes + norm_f_bytes + head_bytes), mb(layer_bytes), mb(head_bytes),
            crate::domain::gemv::pool_threads(),
        );

        Self {
            config,
            emb_w,
            lm_head_w,
            profile,
            layers,
            norm_f_w: norm_f.data,
            norm_f_b,
            highway_start: 0,     // disabled by default
            highway_thr: 0.01,
            resync_interval: 32,
        }
    }

    pub fn create_state(&self) -> MambaState {
        MambaState::new(&self.config)
    }

    pub fn create_scratch(&self) -> MambaScratch {
        MambaScratch::new(&self.config)
    }

    /// Wanda-inspired column pruning: remove least-important d_inner channels (R-061).
    /// NOTE: Marked NO VIABLE — included for experimental CLI only.
    /// Operates on Q8 data by dequantizing columns for importance scoring,
    /// then rebuilds pruned Q8 tensors.
    pub fn apply_column_pruning(&mut self, prune_frac: f32) {
        let di = self.config.d_inner;
        let d = self.config.d_model;
        let ds = self.config.d_state;
        let dc = self.config.d_conv;
        let dr = self.config.dt_rank;
        let n_prune = (di as f32 * prune_frac) as usize;
        if n_prune == 0 { return; }
        if self.profile != WeightProfile::F32 {
            eprintln!("[mamba] column pruning requires the F32 profile — skipped");
            return;
        }
        let new_di = di - n_prune;

        for layer in &mut self.layers {
            let (in_w, out_w) = match (layer.in_proj_w.as_f32(), layer.out_proj_w.as_f32()) {
                (Some(a), Some(b)) => (a.data.clone(), b.data.clone()),
                _ => continue,
            };
            let mut importance = vec![0.0f32; di];
            for j in 0..di {
                let mut in_sq = 0.0f32;
                for k in 0..d {
                    let xv = in_w[j * d + k];
                    let zv = in_w[(j + di) * d + k];
                    in_sq += xv * xv + zv * zv;
                }
                let mut out_sq = 0.0f32;
                for i in 0..d {
                    let v = out_w[i * di + j];
                    out_sq += v * v;
                }
                importance[j] = in_sq.sqrt() * out_sq.sqrt();
            }

            let mut idx: Vec<usize> = (0..di).collect();
            idx.sort_by(|&a, &b| {
                importance[a].partial_cmp(&importance[b]).unwrap()
            });
            let mut keep: Vec<usize> = idx[n_prune..].to_vec();
            keep.sort();

            // Rebuild in_proj_w: (2*di, d) → (2*new_di, d)
            let mut new_in = vec![0.0f32; 2 * new_di * d];
            for (ni, &oj) in keep.iter().enumerate() {
                let xs = oj * d;
                let xd = ni * d;
                new_in[xd..xd + d].copy_from_slice(&in_w[xs..xs + d]);
                let zs = (oj + di) * d;
                let zd = (ni + new_di) * d;
                new_in[zd..zd + d].copy_from_slice(&in_w[zs..zs + d]);
            }
            layer.in_proj_w = MambaMatrix::F32(Tensor::from_data(new_in, vec![2 * new_di, d]));

            // conv1d_w, conv1d_b
            let mut new_cw = vec![0.0f32; new_di * dc];
            for (ni, &oj) in keep.iter().enumerate() {
                new_cw[ni * dc..(ni + 1) * dc]
                    .copy_from_slice(&layer.conv1d_w[oj * dc..(oj + 1) * dc]);
            }
            layer.conv1d_w = new_cw;
            layer.conv1d_b = keep.iter().map(|&j| layer.conv1d_b[j]).collect();

            // x_proj_w: (dr+2*ds, di) → (dr+2*ds, new_di)
            let xr = dr + 2 * ds;
            let mut new_xp = vec![0.0f32; xr * new_di];
            for row in 0..xr {
                for (ni, &oj) in keep.iter().enumerate() {
                    new_xp[row * new_di + ni] =
                        layer.x_proj_w.data[row * di + oj];
                }
            }
            layer.x_proj_w = Tensor::from_data(new_xp, vec![xr, new_di]);

            // dt_proj_w: (di, dr) → (new_di, dr)
            let mut new_dt = vec![0.0f32; new_di * dr];
            for (ni, &oj) in keep.iter().enumerate() {
                new_dt[ni * dr..(ni + 1) * dr].copy_from_slice(
                    &layer.dt_proj_w.data[oj * dr..(oj + 1) * dr],
                );
            }
            layer.dt_proj_w = Tensor::from_data(new_dt, vec![new_di, dr]);
            layer.dt_proj_b = keep.iter().map(|&j| layer.dt_proj_b[j]).collect();

            // a_log, d_vec
            let mut new_al = vec![0.0f32; new_di * ds];
            for (ni, &oj) in keep.iter().enumerate() {
                new_al[ni * ds..(ni + 1) * ds]
                    .copy_from_slice(&layer.a_log[oj * ds..(oj + 1) * ds]);
            }
            layer.a_log = new_al;
            layer.d_vec = keep.iter().map(|&j| layer.d_vec[j]).collect();

            // out_proj_w: (d, di) → (d, new_di)
            let mut new_op = vec![0.0f32; d * new_di];
            for row in 0..d {
                for (ni, &oj) in keep.iter().enumerate() {
                    new_op[row * new_di + ni] = out_w[row * di + oj];
                }
            }
            layer.out_proj_w = MambaMatrix::F32(Tensor::from_data(new_op, vec![d, new_di]));
        }

        eprintln!(
            "[mamba] column pruning: d_inner {} → {} ({:.0}% pruned)",
            di, new_di, prune_frac * 100.0,
        );
        self.config.d_inner = new_di;
    }

    /// Enable highway layer skip (R-062).
    pub fn enable_highway(&mut self, start_layer: usize, thr: f32, resync: usize) {
        self.highway_start = start_layer;
        self.highway_thr = thr;
        self.resync_interval = resync;
    }

    /// Run embedding + all layers + final RMSNorm; result in `scratch.x` (R-060).
    /// Does NOT compute the LM head. Use `compute_head_into` for that.
    pub fn forward_layers_normed(
        &self, token: usize, state: &mut MambaState, scratch: &mut MambaScratch,
    ) {
        let d = self.config.d_model;
        let di = self.config.d_inner;
        let ds = self.config.d_state;
        let dc = self.config.d_conv;
        let dr = self.config.dt_rank;

        // Embedding lookup (Q8 profile: dequantize the row)
        self.emb_w.row_into(&mut scratch.x[..d], token);
        let pool = par_pool::global();

        for layer_idx in 0..self.config.n_layer {
            let lw = &self.layers[layer_idx];

            // Norm: LayerNorm (with bias) or RMSNorm (without)
            if let Some(ref bias) = lw.norm_b {
                layer_norm_into(&mut scratch.normed, &scratch.x, &lw.norm_w, bias, 1e-5);
            } else {
                scratch.normed.copy_from_slice(&scratch.x);
                rms_norm_into(&mut scratch.normed, &lw.norm_w, 1e-5);
            }

            // in_proj: (2*d_inner, d_model) @ normed → xz
            lw.in_proj_w.gemv(&mut scratch.xz, &scratch.normed);

            // Split into x_branch and z_branch
            let (x_branch, z_branch) = scratch.xz.split_at(di);

            // Conv1d as shift register + SiLU, channel blocks on the pool (R-098)
            {
                let cs = &mut state.conv_states[layer_idx];
                let parts: Vec<Mutex<(&mut [f32], &mut [f32])>> = cs
                    .chunks_mut(CONV_BLOCK * (dc - 1))
                    .zip(scratch.conv_out[..di].chunks_mut(CONV_BLOCK))
                    .map(Mutex::new)
                    .collect();
                pool.run(parts.len(), &|bi| {
                    let mut g = lock(&parts[bi]);
                    let (cs_b, out_b) = &mut *g;
                    let j0 = bi * CONV_BLOCK;
                    conv_silu_block(lw, dc, j0, cs_b, &x_branch[j0..j0 + out_b.len()], out_b);
                });
            }

            // x_proj: (dt_rank + 2*d_state, d_inner) @ conv_out → z_proj [f32, narrow]
            gemv_f32_into(&mut scratch.z_proj, &lw.x_proj_w, &scratch.conv_out);

            // Split z_proj → dt_raw, B, C
            scratch.dt_raw[..dr].copy_from_slice(&scratch.z_proj[..dr]);

            // dt = softplus(dt_proj @ dt_raw + bias); SSM recurrence (d_state=16);
            // gated = y * silu(z). Per-channel → channel blocks on the pool (R-098).
            {
                let inp = ScanInputs {
                    dt_raw: &scratch.dt_raw[..dr],
                    b: &scratch.z_proj[dr..dr + ds],
                    c: &scratch.z_proj[dr + ds..dr + 2 * ds],
                    conv_out: &scratch.conv_out[..di],
                    z: z_branch,
                };
                let ss = &mut state.ssm_states[layer_idx];
                type ScanPart<'a> = (&'a mut [f32], &'a mut [f32], &'a mut [f32], &'a mut [f32]);
                let parts: Vec<Mutex<ScanPart>> = scratch.dt[..di].chunks_mut(SSM_BLOCK)
                    .zip(ss.chunks_mut(SSM_BLOCK * ds))
                    .zip(scratch.y[..di].chunks_mut(SSM_BLOCK))
                    .zip(scratch.gated[..di].chunks_mut(SSM_BLOCK))
                    .map(|(((dt, ss), y), g)| Mutex::new((dt, ss, y, g)))
                    .collect();
                pool.run(parts.len(), &|bi| {
                    let mut g = lock(&parts[bi]);
                    let (dt_b, ss_b, y_b, g_b) = &mut *g;
                    scan_block(lw, dr, ds, bi * SSM_BLOCK, &inp, dt_b, ss_b, y_b, g_b);
                });
            }

            // R-059: measure gated[] activation sparsity
            {
                const GATED_EPS: f32 = 1e-4;
                for j in 0..di {
                    scratch.sparsity_total_channels += 1;
                    if scratch.gated[j].abs() < GATED_EPS {
                        scratch.sparsity_near_zero += 1;
                    }
                }
            }

            // out_proj: (d_model, d_inner) @ gated → result
            lw.out_proj_w.gemv(&mut scratch.result, &scratch.gated);

            // Residual add
            for j in 0..d {
                scratch.x[j] += scratch.result[j];
            }

            scratch.highway_layers_run += 1;

            // R-062: highway skip — if upper-layer contribution is tiny, skip rest
            if self.highway_start > 0 && layer_idx >= self.highway_start
                && scratch.highway_token_count % self.resync_interval as u64 != 0
            {
                let delta_sq: f32 = scratch.result[..d].iter()
                    .map(|v| v * v).sum();
                let x_sq: f32 = scratch.x[..d].iter()
                    .map(|v| v * v).sum();
                let thr_sq = self.highway_thr * self.highway_thr;
                if delta_sq < thr_sq * x_sq {
                    let remaining = self.config.n_layer - layer_idx - 1;
                    scratch.highway_layers_skipped += remaining as u64;
                    break;
                }
            }
        }

        scratch.highway_token_count += 1;

        // Final norm: LayerNorm or RMSNorm
        if let Some(ref bias) = self.norm_f_b {
            let tmp: Vec<f32> = scratch.x.clone();
            layer_norm_into(&mut scratch.x, &tmp, &self.norm_f_w, bias, 1e-5);
        } else {
            rms_norm_into(&mut scratch.x, &self.norm_f_w, 1e-5);
        }
    }

    /// Compute LM head from a pre-normed hidden vector (R-060).
    /// `hidden` must be a slice of length `d_model` (e.g. a copy of `scratch.x`).
    /// Uses separate lm_head weights if untied (R-088), otherwise embedding weights.
    pub fn compute_head_into(&self, hidden: &[f32], out_logits: &mut [f32]) {
        let head = self.lm_head_w.as_ref().unwrap_or(&self.emb_w);
        head.gemv(out_logits, hidden);
    }

    /// Numeric weight profile this model was loaded with (R-099).
    pub fn profile(&self) -> WeightProfile {
        self.profile
    }

    /// Single-step forward pass (RNN mode).
    pub fn forward_into(
        &self, token: usize, state: &mut MambaState,
        scratch: &mut MambaScratch, out_logits: &mut [f32],
    ) {
        self.forward_layers_normed(token, state, scratch);
        let head = self.lm_head_w.as_ref().unwrap_or(&self.emb_w);
        head.gemv(out_logits, &scratch.x);
    }
}

// --- Backbone trait implementation ---

use crate::domain::backbone::{Backbone, BackboneState, BackboneScratch};

impl BackboneState for MambaState {
    fn reset(&mut self) { self.reset(); }
}

impl BackboneScratch for MambaScratch {
    fn reset_perf(&mut self) {
        self.sparsity_total_channels = 0;
        self.sparsity_near_zero = 0;
        self.highway_token_count = 0;
        self.highway_layers_run = 0;
        self.highway_layers_skipped = 0;
    }
}

impl Backbone for MambaModel {
    type State = MambaState;
    type Scratch = MambaScratch;

    fn vocab_size(&self) -> usize { self.config.vocab_size }
    fn create_state(&self) -> MambaState { MambaState::new(&self.config) }
    fn create_scratch(&self) -> MambaScratch { MambaScratch::new(&self.config) }

    fn forward_into(
        &self, token: usize, state: &mut MambaState,
        scratch: &mut MambaScratch, out_logits: &mut [f32],
    ) {
        self.forward_into(token, state, scratch, out_logits);
    }
}
