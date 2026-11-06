//! Mamba SSM inference — single-step RNN mode, zero dependencies.
//!
//! Reference: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//! Model: state-spaces/mamba-130m-hf (d_model=768, n_layer=24, vocab=50280)

use std::path::Path;

use crate::domain::tensor::{
    Tensor, mat_vec_mul_into, portable_softplus, rms_norm_into,
    silu_inplace, gated_silu_mul, ssm_recurrence_d16,
};
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

// --- Layer weights ---

struct MambaLayerWeights {
    norm_w: Vec<f32>,          // (d_model,)
    in_proj_w: Tensor,         // (2*d_inner, d_model)
    conv1d_w: Vec<f32>,        // (d_inner, d_conv) — flat, used element-wise
    conv1d_b: Vec<f32>,        // (d_inner,)
    x_proj_w: Tensor,          // (dt_rank + 2*d_state, d_inner)
    dt_proj_w: Tensor,         // (d_inner, dt_rank)
    dt_proj_b: Vec<f32>,       // (d_inner,)
    a_log: Vec<f32>,           // (d_inner, d_state) — flat, used element-wise
    d_vec: Vec<f32>,           // (d_inner,)
    out_proj_w: Tensor,        // (d_model, d_inner)
}

// --- Model ---

pub struct MambaModel {
    pub config: MambaConfig,
    emb_w: Tensor,             // (vocab_size, d_model) — embedding + lm_head (tied)
    layers: Vec<MambaLayerWeights>,
    norm_f_w: Vec<f32>,        // (d_model,)
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
    pub fn load(path: &Path) -> Self {
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

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("backbone.layers.{}", i);
            let norm = st.get_tensor(&format!("{}.norm.weight", p));
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
                in_proj_w: Tensor::from_data(in_proj.data, vec![2 * di, d_model]),
                conv1d_w: conv_w.data, // flat (d_inner * d_conv)
                conv1d_b: conv_b.data,
                x_proj_w: Tensor::from_data(x_proj.data, vec![dr + 2 * ds, di]),
                dt_proj_w: Tensor::from_data(dt_proj.data, vec![di, dr]),
                dt_proj_b: dt_bias.data,
                a_log: a_log.data, // flat (d_inner * d_state)
                d_vec: d_vec.data,
                out_proj_w: Tensor::from_data(out_proj.data, vec![d_model, di]),
            });

            if (i + 1) % 8 == 0 || i == n_layer - 1 {
                eprintln!("[mamba] loaded layer {}/{}", i + 1, n_layer);
            }
        }

        let emb_tensor = Tensor::from_data(emb.data, vec![vocab_size, d_model]);

        let mem_mb = (emb_tensor.numel() * 4
            + norm_f.data.len() * 4
            + layers.iter().map(|l| {
                (l.norm_w.len() + l.in_proj_w.numel() + l.conv1d_w.len()
                    + l.conv1d_b.len() + l.x_proj_w.numel() + l.dt_proj_w.numel()
                    + l.dt_proj_b.len() + l.a_log.len() + l.d_vec.len()
                    + l.out_proj_w.numel()) * 4
            }).sum::<usize>()) as f64 / 1024.0 / 1024.0;
        eprintln!("[mamba] model loaded: {:.1} MB", mem_mb);

        Self {
            config,
            emb_w: emb_tensor,
            layers,
            norm_f_w: norm_f.data,
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
        let new_di = di - n_prune;

        for layer in &mut self.layers {
            let mut importance = vec![0.0f32; di];
            for j in 0..di {
                let mut in_sq = 0.0f32;
                for k in 0..d {
                    let xv = layer.in_proj_w.data[j * d + k];
                    let zv = layer.in_proj_w.data[(j + di) * d + k];
                    in_sq += xv * xv + zv * zv;
                }
                let mut out_sq = 0.0f32;
                for i in 0..d {
                    let v = layer.out_proj_w.data[i * di + j];
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
                new_in[xd..xd + d].copy_from_slice(
                    &layer.in_proj_w.data[xs..xs + d],
                );
                let zs = (oj + di) * d;
                let zd = (ni + new_di) * d;
                new_in[zd..zd + d].copy_from_slice(
                    &layer.in_proj_w.data[zs..zs + d],
                );
            }
            layer.in_proj_w = Tensor::from_data(new_in, vec![2 * new_di, d]);

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
                    new_op[row * new_di + ni] =
                        layer.out_proj_w.data[row * di + oj];
                }
            }
            layer.out_proj_w = Tensor::from_data(new_op, vec![d, new_di]);
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

        // Embedding lookup
        let emb_start = token * d;
        scratch.x.copy_from_slice(&self.emb_w.data[emb_start..emb_start + d]);

        for layer_idx in 0..self.config.n_layer {
            let lw = &self.layers[layer_idx];

            // RMSNorm
            scratch.normed.copy_from_slice(&scratch.x);
            rms_norm_into(&mut scratch.normed, &lw.norm_w, 1e-5);

            // in_proj: (2*d_inner, d_model) @ normed → xz
            mat_vec_mul_into(&mut scratch.xz, &lw.in_proj_w, &scratch.normed);

            // Split into x_branch and z_branch
            let (x_branch, z_branch) = scratch.xz.split_at(di);

            // Conv1d as shift register (compute + shift, activation deferred)
            let cs = &mut state.conv_states[layer_idx];
            for j in 0..di {
                let mut val = 0.0f32;
                for k in 0..(dc - 1) {
                    val += cs[j * (dc - 1) + k] * lw.conv1d_w[j * dc + k];
                }
                val += x_branch[j] * lw.conv1d_w[j * dc + (dc - 1)];
                val += lw.conv1d_b[j];

                // Shift state left, insert new value
                for k in 0..(dc - 2) {
                    cs[j * (dc - 1) + k] = cs[j * (dc - 1) + k + 1];
                }
                cs[j * (dc - 1) + (dc - 2)] = x_branch[j];

                scratch.conv_out[j] = val;
            }
            // Bulk SiLU via SIMD
            silu_inplace(&mut scratch.conv_out[..di]);

            // x_proj: (dt_rank + 2*d_state, d_inner) @ conv_out → z_proj [f32, narrow]
            mat_vec_mul_into(&mut scratch.z_proj, &lw.x_proj_w, &scratch.conv_out);

            // Split z_proj → dt_raw, B, C
            scratch.dt_raw[..dr].copy_from_slice(&scratch.z_proj[..dr]);
            let b_slice = &scratch.z_proj[dr..dr + ds];
            let c_slice = &scratch.z_proj[dr + ds..dr + 2 * ds];

            // dt = softplus(dt_proj @ dt_raw + dt_proj_bias) [f32, narrow]
            mat_vec_mul_into(&mut scratch.dt, &lw.dt_proj_w, &scratch.dt_raw);
            for j in 0..di {
                scratch.dt[j] = portable_softplus(scratch.dt[j] + lw.dt_proj_b[j]);
            }

            // SSM recurrence (SIMD-accelerated for d_state=16)
            let ss = &mut state.ssm_states[layer_idx];
            ssm_recurrence_d16(
                &mut scratch.y[..di], ss, &scratch.dt[..di],
                &scratch.conv_out[..di], &lw.a_log, b_slice, c_slice,
                &lw.d_vec, di,
            );

            // Gate: gated[j] = y[j] * silu(z_branch[j])
            gated_silu_mul(&mut scratch.gated[..di], &scratch.y[..di], z_branch);

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
            mat_vec_mul_into(&mut scratch.result, &lw.out_proj_w, &scratch.gated);

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

        // Final RMSNorm
        rms_norm_into(&mut scratch.x, &self.norm_f_w, 1e-5);
    }

    /// Compute LM head from a pre-normed hidden vector (R-060).
    /// `hidden` must be a slice of length `d_model` (e.g. a copy of `scratch.x`).
    pub fn compute_head_into(&self, hidden: &[f32], out_logits: &mut [f32]) {
        mat_vec_mul_into(out_logits, &self.emb_w, hidden);
    }

    /// Single-step forward pass (RNN mode).
    pub fn forward_into(
        &self, token: usize, state: &mut MambaState,
        scratch: &mut MambaScratch, out_logits: &mut [f32],
    ) {
        self.forward_layers_normed(token, state, scratch);
        mat_vec_mul_into(out_logits, &self.emb_w, &scratch.x);
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
