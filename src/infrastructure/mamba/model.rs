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
use crate::domain::gemv::{
    f32_rows, gemv_f32_into, gemv_f32_multi, gemv_q8w_into, gemv_q8w_multi, Kernel, Q8wMatrix,
};
use crate::domain::gemv_fma::simd_fma;
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
    fn gemv(&self, out: &mut [f32], v: &[f32], k: Kernel) {
        match self {
            Self::F32(t) => gemv_f32_into(out, t, v, k),
            Self::Q8(q) => gemv_q8w_into(out, q, v, k),
        }
    }

    /// outs[s] = W @ xs[s] for N lockstep streams, one weight fetch per row
    /// block (R-100). Bit-identical to `gemv` per stream.
    #[inline]
    fn gemv_multi(&self, outs: &mut [&mut [f32]], xs: &[&[f32]], k: Kernel) {
        match self {
            Self::F32(t) => gemv_f32_multi(outs, t, xs, k),
            Self::Q8(q) => gemv_q8w_multi(outs, q, xs, k),
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

/// Shape and kernel parameters of the scan phase (shared by every block).
#[derive(Clone, Copy)]
struct ScanDims {
    dr: usize,
    ds: usize,
    k: Kernel,
}

/// dt = softplus(dt_proj·dt_raw + bias) → SSM recurrence → y·silu(z), for
/// channels `j0..j0+dt.len()`. Per-channel independent → partition-invariant.
fn scan_block(
    lw: &MambaLayerWeights, dims: ScanDims, j0: usize, inp: &ScanInputs,
    dt: &mut [f32], ss: &mut [f32], y: &mut [f32], gated: &mut [f32],
) {
    let ScanDims { dr, ds, k } = dims;
    let n = dt.len();
    let j1 = j0 + n;
    f32_rows(dt, &lw.dt_proj_w.data[j0 * dr..j1 * dr], inp.dt_raw, dr, k);
    for (jj, d) in dt.iter_mut().enumerate() {
        *d = portable_softplus(*d + lw.dt_proj_b[j0 + jj]);
    }
    ssm_recurrence_d16(
        y, ss, dt, &inp.conv_out[j0..j1], &lw.a_log[j0 * ds..j1 * ds],
        inp.b, inp.c, &lw.d_vec[j0..j1], n,
    );
    gated_silu_mul(gated, y, &inp.z[j0..j1]);
}

/// Pre-norm of one layer: LayerNorm (with bias) or RMSNorm into `scratch.normed`.
fn layer_norm_step(lw: &MambaLayerWeights, scratch: &mut MambaScratch) {
    if let Some(ref bias) = lw.norm_b {
        layer_norm_into(&mut scratch.normed, &scratch.x, &lw.norm_w, bias, 1e-5);
    } else {
        scratch.normed.copy_from_slice(&scratch.x);
        rms_norm_into(&mut scratch.normed, &lw.norm_w, 1e-5);
    }
}

/// R-059: count near-zero gated channels of this layer.
fn count_gated_sparsity(scratch: &mut MambaScratch, di: usize) {
    const GATED_EPS: f32 = 1e-4;
    for j in 0..di {
        scratch.sparsity_total_channels += 1;
        if scratch.gated[j].abs() < GATED_EPS {
            scratch.sparsity_near_zero += 1;
        }
    }
}

/// One pooled unit of the lockstep conv phase: a channel block of one stream.
struct ConvPart<'a> {
    j0: usize,
    cs: &'a mut [f32],
    x: &'a [f32],
    out: &'a mut [f32],
}

/// One pooled unit of the lockstep scan phase: a channel block of one stream.
struct ScanPart<'a> {
    stream: usize,
    j0: usize,
    dt: &'a mut [f32],
    ss: &'a mut [f32],
    y: &'a mut [f32],
    gated: &'a mut [f32],
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
    kernel: Kernel,                 // R-107: GEMV kernel family (numeric contract)
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
    tokens_since_reset: u64,     // for the experimental periodic reset (see `reset_every`)
}

impl MambaState {
    pub fn new(config: &MambaConfig) -> Self {
        let conv_len = config.d_inner * (config.d_conv - 1);
        let ssm_len = config.d_inner * config.d_state;
        Self {
            conv_states: (0..config.n_layer).map(|_| vec![0.0; conv_len]).collect(),
            ssm_states: (0..config.n_layer).map(|_| vec![0.0; ssm_len]).collect(),
            tokens_since_reset: 0,
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.conv_states { s.fill(0.0); }
        for s in &mut self.ssm_states { s.fill(0.0); }
        self.tokens_since_reset = 0;
    }

    /// Experimental length-collapse mitigation (2026-11-10 diagnosis): zero the
    /// recurrent state every `REDUKT_MAMBA_RESET_EVERY` tokens. Mamba-130M degenerates
    /// to near-uniform logits beyond ~16-18K tokens on some inputs (reproduced with the
    /// HF reference implementation). Encoder and decoder must use the same value; this
    /// is NOT part of the AZTH format yet, so the variable is for measurement only.
    fn maybe_periodic_reset(&mut self) {
        let every = reset_every();
        if every > 0 && self.tokens_since_reset >= every {
            self.reset();
        }
        self.tokens_since_reset += 1;
    }
}

/// `REDUKT_MAMBA_RESET_EVERY=N` (0 or unset = never), read once.
fn reset_every() -> u64 {
    static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let v = std::env::var("REDUKT_MAMBA_RESET_EVERY").ok().and_then(|v| v.trim().parse::<u64>().ok());
        if let Some(n) = v.filter(|n| *n > 0) {
            eprintln!("[mamba] EXPERIMENTAL: recurrent state reset every {} tokens (REDUKT_MAMBA_RESET_EVERY)", n);
        }
        v.unwrap_or(0)
    })
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
            kernel: Kernel::Plain,
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
        state.maybe_periodic_reset();

        // Embedding lookup (Q8 profile: dequantize the row)
        self.emb_w.row_into(&mut scratch.x[..d], token);
        let pool = par_pool::global();
        let k = self.kernel;

        for layer_idx in 0..self.config.n_layer {
            let lw = &self.layers[layer_idx];

            // Norm: LayerNorm (with bias) or RMSNorm (without)
            layer_norm_step(lw, scratch);

            // in_proj: (2*d_inner, d_model) @ normed → xz
            lw.in_proj_w.gemv(&mut scratch.xz, &scratch.normed, k);

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
            gemv_f32_into(&mut scratch.z_proj, &lw.x_proj_w, &scratch.conv_out, k);

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
                let dims = ScanDims { dr, ds, k };
                pool.run(parts.len(), &|bi| {
                    let mut g = lock(&parts[bi]);
                    let (dt_b, ss_b, y_b, g_b) = &mut *g;
                    scan_block(lw, dims, bi * SSM_BLOCK, &inp, dt_b, ss_b, y_b, g_b);
                });
            }

            // R-059: measure gated[] activation sparsity
            count_gated_sparsity(scratch, di);

            // out_proj: (d_model, d_inner) @ gated → result
            lw.out_proj_w.gemv(&mut scratch.result, &scratch.gated, k);

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
        self.final_norm(scratch);
    }

    /// Final norm (LayerNorm or RMSNorm) of `scratch.x` in place.
    fn final_norm(&self, scratch: &mut MambaScratch) {
        if let Some(ref bias) = self.norm_f_b {
            let tmp: Vec<f32> = scratch.x.clone();
            layer_norm_into(&mut scratch.x, &tmp, &self.norm_f_w, bias, 1e-5);
        } else {
            rms_norm_into(&mut scratch.x, &self.norm_f_w, 1e-5);
        }
    }

    // --- Lockstep multi-stream forward (R-100) ---

    /// Bytes of one stream's recurrent state (conv + SSM state, all layers).
    pub fn state_bytes(&self) -> usize {
        let c = &self.config;
        c.n_layer * c.d_inner * (c.d_conv - 1 + c.d_state) * 4
    }

    /// Bytes of one stream's scratch workspace.
    pub fn scratch_bytes(&self) -> usize {
        let c = &self.config;
        (3 * c.d_model + 6 * c.d_inner + 2 * c.dt_rank + 2 * c.d_state) * 4
    }

    /// Embedding + all layers + final norm for N independent streams advancing
    /// one token each (lockstep). Every large matrix is streamed from DRAM once
    /// per step and applied to all N activations; per-channel phases run on the
    /// pool over (stream, channel block). Each stream's result is bit-identical
    /// to `forward_layers_normed` on that stream alone. Results in `scratch.x`.
    pub fn forward_layers_normed_multi(
        &self, tokens: &[usize], states: &mut [&mut MambaState], scratches: &mut [&mut MambaScratch],
    ) {
        let n = tokens.len();
        debug_assert!(states.len() == n && scratches.len() == n);
        if n == 1 || self.highway_start > 0 {
            for i in 0..n {
                self.forward_layers_normed(tokens[i], states[i], scratches[i]);
            }
            return;
        }
        for st in states.iter_mut() {
            st.maybe_periodic_reset();
        }
        let d = self.config.d_model;
        let di = self.config.d_inner;
        let dr = self.config.dt_rank;
        let k = self.kernel;

        for (sc, &tok) in scratches.iter_mut().zip(tokens) {
            self.emb_w.row_into(&mut sc.x[..d], tok);
        }

        for layer_idx in 0..self.config.n_layer {
            let lw = &self.layers[layer_idx];
            for sc in scratches.iter_mut() {
                layer_norm_step(lw, sc);
            }
            Self::in_proj_multi(lw, scratches, k);
            Self::conv_phase_multi(lw, &self.config, layer_idx, states, scratches);
            Self::x_proj_multi(lw, scratches, di, k);
            for sc in scratches.iter_mut() {
                sc.dt_raw[..dr].copy_from_slice(&sc.z_proj[..dr]);
            }
            Self::scan_phase_multi(lw, &self.config, layer_idx, states, scratches, k);
            for sc in scratches.iter_mut() {
                count_gated_sparsity(sc, di);
            }
            Self::out_proj_multi(lw, scratches, di, k);
            for sc in scratches.iter_mut() {
                for j in 0..d {
                    sc.x[j] += sc.result[j];
                }
                sc.highway_layers_run += 1;
            }
        }

        for sc in scratches.iter_mut() {
            sc.highway_token_count += 1;
            self.final_norm(sc);
        }
    }

    /// in_proj as an N-stream GEMM: normed -> xz.
    fn in_proj_multi(lw: &MambaLayerWeights, scratches: &mut [&mut MambaScratch], k: Kernel) {
        let mut xs: Vec<&[f32]> = Vec::with_capacity(scratches.len());
        let mut outs: Vec<&mut [f32]> = Vec::with_capacity(scratches.len());
        for sc in scratches.iter_mut() {
            let MambaScratch { normed, xz, .. } = &mut **sc;
            xs.push(normed.as_slice());
            outs.push(xz.as_mut_slice());
        }
        lw.in_proj_w.gemv_multi(&mut outs, &xs, k);
    }

    /// x_proj as an N-stream GEMM: conv_out -> z_proj.
    fn x_proj_multi(lw: &MambaLayerWeights, scratches: &mut [&mut MambaScratch], di: usize, k: Kernel) {
        let mut xs: Vec<&[f32]> = Vec::with_capacity(scratches.len());
        let mut outs: Vec<&mut [f32]> = Vec::with_capacity(scratches.len());
        for sc in scratches.iter_mut() {
            let MambaScratch { conv_out, z_proj, .. } = &mut **sc;
            xs.push(&conv_out[..di]);
            outs.push(z_proj.as_mut_slice());
        }
        gemv_f32_multi(&mut outs, &lw.x_proj_w, &xs, k);
    }

    /// out_proj as an N-stream GEMM: gated -> result.
    fn out_proj_multi(lw: &MambaLayerWeights, scratches: &mut [&mut MambaScratch], di: usize, k: Kernel) {
        let mut xs: Vec<&[f32]> = Vec::with_capacity(scratches.len());
        let mut outs: Vec<&mut [f32]> = Vec::with_capacity(scratches.len());
        for sc in scratches.iter_mut() {
            let MambaScratch { gated, result, .. } = &mut **sc;
            xs.push(&gated[..di]);
            outs.push(result.as_mut_slice());
        }
        lw.out_proj_w.gemv_multi(&mut outs, &xs, k);
    }

    /// Conv1d + SiLU for all streams: pool units = (stream, channel block).
    fn conv_phase_multi(
        lw: &MambaLayerWeights, cfg: &MambaConfig, layer_idx: usize,
        states: &mut [&mut MambaState], scratches: &mut [&mut MambaScratch],
    ) {
        let (di, dc) = (cfg.d_inner, cfg.d_conv);
        let mut parts: Vec<Mutex<ConvPart>> = Vec::new();
        for (st, sc) in states.iter_mut().zip(scratches.iter_mut()) {
            let MambaScratch { xz, conv_out, .. } = &mut **sc;
            let x_branch = &xz[..di];
            let blocks = st.conv_states[layer_idx].chunks_mut(CONV_BLOCK * (dc - 1))
                .zip(conv_out[..di].chunks_mut(CONV_BLOCK));
            for (bi, (cs, out)) in blocks.enumerate() {
                let j0 = bi * CONV_BLOCK;
                let x = &x_branch[j0..j0 + out.len()];
                parts.push(Mutex::new(ConvPart { j0, cs, x, out }));
            }
        }
        par_pool::global().run(parts.len(), &|bi| {
            let mut g = lock(&parts[bi]);
            let p = &mut *g;
            conv_silu_block(lw, dc, p.j0, p.cs, p.x, p.out);
        });
    }

    /// dt_proj + softplus + SSM + gate for all streams: units = (stream, channel block).
    fn scan_phase_multi(
        lw: &MambaLayerWeights, cfg: &MambaConfig, layer_idx: usize,
        states: &mut [&mut MambaState], scratches: &mut [&mut MambaScratch], k: Kernel,
    ) {
        let (di, ds, dr) = (cfg.d_inner, cfg.d_state, cfg.dt_rank);
        let mut inputs: Vec<ScanInputs> = Vec::with_capacity(scratches.len());
        let mut parts: Vec<Mutex<ScanPart>> = Vec::new();
        for (stream, (st, sc)) in states.iter_mut().zip(scratches.iter_mut()).enumerate() {
            let MambaScratch { xz, conv_out, z_proj, dt, y, gated, dt_raw, .. } = &mut **sc;
            inputs.push(ScanInputs {
                dt_raw: &dt_raw[..dr],
                b: &z_proj[dr..dr + ds],
                c: &z_proj[dr + ds..dr + 2 * ds],
                conv_out: &conv_out[..di],
                z: &xz[di..],
            });
            let blocks = dt[..di].chunks_mut(SSM_BLOCK)
                .zip(st.ssm_states[layer_idx].chunks_mut(SSM_BLOCK * ds))
                .zip(y[..di].chunks_mut(SSM_BLOCK))
                .zip(gated[..di].chunks_mut(SSM_BLOCK));
            for (bi, (((dt, ss), y), gated)) in blocks.enumerate() {
                parts.push(Mutex::new(ScanPart { stream, j0: bi * SSM_BLOCK, dt, ss, y, gated }));
            }
        }
        let dims = ScanDims { dr, ds, k };
        par_pool::global().run(parts.len(), &|bi| {
            let mut g = lock(&parts[bi]);
            let p = &mut *g;
            scan_block(lw, dims, p.j0, &inputs[p.stream], p.dt, p.ss, p.y, p.gated);
        });
    }

    /// LM head for N streams from their normed hidden vectors (R-100).
    pub fn compute_head_multi(&self, hiddens: &[&[f32]], out_logits: &mut [&mut [f32]]) {
        let head = self.lm_head_w.as_ref().unwrap_or(&self.emb_w);
        head.gemv_multi(out_logits, hiddens, self.kernel);
    }

    /// Lockstep forward: N tokens in (one per stream), N logit vectors out.
    /// Bit-identical per stream to `forward_into`.
    pub fn forward_multi(
        &self, tokens: &[usize], states: &mut [&mut MambaState],
        scratches: &mut [&mut MambaScratch], out_logits: &mut [&mut [f32]],
    ) {
        self.forward_layers_normed_multi(tokens, states, scratches);
        let hiddens: Vec<&[f32]> = scratches.iter().map(|sc| sc.x.as_slice()).collect();
        self.compute_head_multi(&hiddens, out_logits);
    }

    /// Compute LM head from a pre-normed hidden vector (R-060).
    /// `hidden` must be a slice of length `d_model` (e.g. a copy of `scratch.x`).
    /// Uses separate lm_head weights if untied (R-088), otherwise embedding weights.
    pub fn compute_head_into(&self, hidden: &[f32], out_logits: &mut [f32]) {
        let head = self.lm_head_w.as_ref().unwrap_or(&self.emb_w);
        head.gemv(out_logits, hidden, self.kernel);
    }

    /// Numeric weight profile this model was loaded with (R-099).
    pub fn profile(&self) -> WeightProfile {
        self.profile
    }

    /// GEMV/GEMM kernel family in use (R-107).
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// Select the GEMV/GEMM kernel family (R-107): `Kernel::Fma` implements
    /// numeric contract 2. Without AVX2+FMA (or with `REDUKT_NO_SIMD=1`) the
    /// FMA family runs its scalar `f32::mul_add` kernels: bit-identical, slower.
    pub fn set_kernel(&mut self, k: Kernel) {
        if k == Kernel::Fma && !simd_fma() {
            eprintln!(
                "[mamba] WARNING: numeric contract fma-v1 without AVX2+FMA (or REDUKT_NO_SIMD=1): \
                 using the scalar mul_add kernels (bit-exact, much slower)"
            );
        }
        self.kernel = k;
    }

    /// Single-step forward pass (RNN mode).
    pub fn forward_into(
        &self, token: usize, state: &mut MambaState,
        scratch: &mut MambaScratch, out_logits: &mut [f32],
    ) {
        self.forward_layers_normed(token, state, scratch);
        let head = self.lm_head_w.as_ref().unwrap_or(&self.emb_w);
        head.gemv(out_logits, &scratch.x, self.kernel);
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

    fn forward_multi(
        &self, tokens: &[usize], states: &mut [&mut MambaState],
        scratches: &mut [&mut MambaScratch], out_logits: &mut [&mut [f32]],
    ) {
        self.forward_multi(tokens, states, scratches, out_logits);
    }

    fn stream_bytes(&self) -> usize {
        self.state_bytes() + self.scratch_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in [-amp, amp) (xorshift32).
    fn rand_vec(n: usize, seed: u32, amp: f32) -> Vec<f32> {
        let mut st = seed.max(1);
        (0..n).map(|_| {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            ((st as f32 / u32::MAX as f32) * 2.0 - 1.0) * amp
        }).collect()
    }

    fn rand_tensor(rows: usize, cols: usize, seed: u32, amp: f32) -> Tensor {
        Tensor::from_data(rand_vec(rows * cols, seed, amp), vec![rows, cols])
    }

    /// Small random Mamba with the real layer structure (no weights file).
    fn synthetic_model(d_model: usize, n_layer: usize, vocab: usize, profile: WeightProfile) -> MambaModel {
        let config = MambaConfig::from_d_model(d_model, n_layer, vocab);
        let (di, ds, dr, dc) = (config.d_inner, config.d_state, config.dt_rank, config.d_conv);
        let mut seed = 1000u32;
        let mut next = || { seed += 17; seed };
        let layers = (0..n_layer).map(|_| MambaLayerWeights {
            norm_w: rand_vec(d_model, next(), 0.5).iter().map(|x| 1.0 + x).collect(),
            norm_b: None,
            in_proj_w: MambaMatrix::new(rand_tensor(2 * di, d_model, next(), 0.3), profile),
            conv1d_w: rand_vec(di * dc, next(), 0.5),
            conv1d_b: rand_vec(di, next(), 0.1),
            x_proj_w: rand_tensor(dr + 2 * ds, di, next(), 0.2),
            dt_proj_w: rand_tensor(di, dr, next(), 0.5),
            dt_proj_b: rand_vec(di, next(), 1.0).iter().map(|x| x - 2.0).collect(),
            a_log: rand_vec(di * ds, next(), 1.0).iter().map(|x| x.abs()).collect(),
            d_vec: rand_vec(di, next(), 1.0),
            out_proj_w: MambaMatrix::new(rand_tensor(d_model, di, next(), 0.2), profile),
        }).collect();
        MambaModel {
            emb_w: MambaMatrix::new(rand_tensor(vocab, d_model, next(), 1.0), profile),
            lm_head_w: None,
            profile,
            kernel: Kernel::Plain,
            layers,
            norm_f_w: rand_vec(d_model, next(), 0.5).iter().map(|x| 1.0 + x).collect(),
            norm_f_b: None,
            highway_start: 0,
            highway_thr: 0.01,
            resync_interval: 32,
            config,
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// R-100: lockstep forward of N streams == N independent single forwards
    /// (logits and recurrent state), for both weight profiles and (R-107) both
    /// kernel families.
    #[test]
    fn forward_multi_bit_identical_to_single_stream() {
        let cases = [WeightProfile::F32, WeightProfile::Q8]
            .into_iter().flat_map(|p| [(p, Kernel::Plain), (p, Kernel::Fma)]);
        for (profile, kernel) in cases {
            // vocab*d_model*4 > PAR_MIN_BYTES so the head GEMM takes the pooled path.
            let mut model = synthetic_model(64, 2, 1200, profile);
            model.set_kernel(kernel);
            let v = model.config.vocab_size;
            for n in [2usize, 3, 5, 9] {
                let steps = 6;
                let toks: Vec<Vec<usize>> = (0..n)
                    .map(|s| (0..steps).map(|t| (s * 131 + t * 977 + 7) % v).collect())
                    .collect();
                // Reference: each stream alone.
                let mut ref_states: Vec<MambaState> = (0..n).map(|_| model.create_state()).collect();
                let mut ref_logits = vec![vec![0.0f32; v]; n];
                for s in 0..n {
                    let mut sc = model.create_scratch();
                    for t in 0..steps {
                        model.forward_into(toks[s][t], &mut ref_states[s], &mut sc, &mut ref_logits[s]);
                    }
                }
                // Lockstep.
                let mut states: Vec<MambaState> = (0..n).map(|_| model.create_state()).collect();
                let mut scratches: Vec<MambaScratch> = (0..n).map(|_| model.create_scratch()).collect();
                let mut logits = vec![vec![f32::NAN; v]; n];
                for t in 0..steps {
                    let step: Vec<usize> = (0..n).map(|s| toks[s][t]).collect();
                    let mut st: Vec<&mut MambaState> = states.iter_mut().collect();
                    let mut sc: Vec<&mut MambaScratch> = scratches.iter_mut().collect();
                    let mut lo: Vec<&mut [f32]> = logits.iter_mut().map(|l| l.as_mut_slice()).collect();
                    model.forward_multi(&step, &mut st, &mut sc, &mut lo);
                }
                for s in 0..n {
                    assert_eq!(bits(&logits[s]), bits(&ref_logits[s]), "{} {:?} n={} stream {}", profile, kernel, n, s);
                    for l in 0..model.config.n_layer {
                        assert_eq!(bits(&states[s].ssm_states[l]), bits(&ref_states[s].ssm_states[l]));
                        assert_eq!(bits(&states[s].conv_states[l]), bits(&ref_states[s].conv_states[l]));
                    }
                }
            }
        }
    }

    /// Real-weights lockstep benchmark: ms/token of N single forwards vs one
    /// forward_multi(N). cargo test --release bench_forward_multi -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_forward_multi_real_weights() {
        use std::time::Instant;
        let dir = Path::new("weights/mamba-130m");
        if !dir.join("model.safetensors").exists() {
            eprintln!("SKIP: no mamba weights");
            return;
        }
        let fma = std::env::var("REDUKT_BENCH_FMA").map(|v| v == "1").unwrap_or(false);
        for profile in [WeightProfile::Q8, WeightProfile::F32] {
            let mut model = MambaModel::load_with_profile(dir, profile);
            model.set_kernel(if fma { Kernel::Fma } else { Kernel::Plain });
            let v = model.config.vocab_size;
            for n in [1usize, 2, 4, 8, 16, 32] {
                let mut states: Vec<MambaState> = (0..n).map(|_| model.create_state()).collect();
                let mut scratches: Vec<MambaScratch> = (0..n).map(|_| model.create_scratch()).collect();
                let mut logits = vec![vec![0.0f32; v]; n];
                let steps = (64 / n).max(4);
                let (mut best_single, mut best_multi) = (f64::MAX, f64::MAX);
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for t in 0..steps {
                        for s in 0..n {
                            model.forward_into((t * 31 + s * 7) % v, &mut states[s], &mut scratches[s], &mut logits[s]);
                        }
                    }
                    best_single = best_single.min(t0.elapsed().as_secs_f64() / (steps * n) as f64);
                    let t0 = Instant::now();
                    for t in 0..steps {
                        let toks: Vec<usize> = (0..n).map(|s| (t * 31 + s * 7) % v).collect();
                        let mut st: Vec<&mut MambaState> = states.iter_mut().collect();
                        let mut sc: Vec<&mut MambaScratch> = scratches.iter_mut().collect();
                        let mut lo: Vec<&mut [f32]> = logits.iter_mut().map(|l| l.as_mut_slice()).collect();
                        model.forward_multi(&toks, &mut st, &mut sc, &mut lo);
                    }
                    best_multi = best_multi.min(t0.elapsed().as_secs_f64() / (steps * n) as f64);
                }
                eprintln!("{} n={:2}: N x forward_into {:6.2} ms/token ({:6.1} tok/s) | forward_multi {:6.2} ms/token",
                    profile, n, best_single * 1e3, 1.0 / best_single, best_multi * 1e3);
                eprintln!("        ({:6.1} tok/s aggregate, {:6.2} ms/step) | {:4.2}x",
                    1.0 / best_multi, best_multi * 1e3 * n as f64, best_single / best_multi);
            }
        }
    }

    #[test]
    fn stream_bytes_matches_allocations() {
        let model = synthetic_model(32, 3, 100, WeightProfile::F32);
        let st = model.create_state();
        let state_bytes: usize = st.conv_states.iter().chain(st.ssm_states.iter()).map(|v| v.len() * 4).sum();
        assert_eq!(model.state_bytes(), state_bytes);
        let sc = model.create_scratch();
        let scratch_bytes = (sc.x.len() + sc.xz.len() + sc.conv_out.len() + sc.z_proj.len() + sc.dt.len()
            + sc.y.len() + sc.gated.len() + sc.result.len() + sc.dt_raw.len() + sc.normed.len()) * 4;
        assert_eq!(model.scratch_bytes(), scratch_bytes);
    }
}
