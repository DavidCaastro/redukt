//! RWKV-7 "Goose" inference — single-step RNN mode.
//!
//! Reference: BlinkDL/RWKV-LM rwkv_v7_demo_rnn.py
//! Spec: docs/research/r02-rwkv7-forward-pass.md

use std::path::Path;

use crate::domain::tensor::*;
use crate::domain::quant::{QuantLevel, QuantMatrix, QuantPolicy, quantize_matrix, ggml_block_info};
use crate::domain::ram_monitor::{check_ram_or_abort, log_ram};
use crate::infrastructure::rwkv7::safetensors::SafeTensorsFile;
use crate::infrastructure::rwkv7::gguf::GgufFile;

pub struct Rwkv7Config {
    pub n_layer: usize,
    pub n_embd: usize,    // D = 768
    pub n_head: usize,     // H = 12
    pub head_size: usize,  // N = 64
    pub vocab_size: usize, // V = 65536
    pub g1k: bool,         // G1k variant: uses k_mod in bonus term
}

impl Rwkv7Config {
    pub fn default_0_1b() -> Self {
        Self { n_layer: 12, n_embd: 768, n_head: 12, head_size: 64, vocab_size: 65536, g1k: false }
    }

    pub fn default_0_4b() -> Self {
        Self { n_layer: 24, n_embd: 1024, n_head: 16, head_size: 64, vocab_size: 65536, g1k: false }
    }

    pub fn default_1_5b() -> Self {
        Self { n_layer: 24, n_embd: 2048, n_head: 32, head_size: 64, vocab_size: 65536, g1k: true }
    }

    pub fn default_2_9b() -> Self {
        Self { n_layer: 32, n_embd: 2560, n_head: 40, head_size: 64, vocab_size: 65536, g1k: true }
    }

    /// Auto-detect config from weights path/directory name.
    pub fn from_weights_dir(dir: &str) -> Self {
        if dir.contains("2.9b") || dir.contains("2.9B") {
            Self::default_2_9b()
        } else if dir.contains("1.5b") || dir.contains("1.5B") {
            Self::default_1_5b()
        } else if dir.contains("0.4b") || dir.contains("0.4B") {
            Self::default_0_4b()
        } else {
            Self::default_0_1b()
        }
    }
}

/// Per-layer weights for time mixing (attention).
struct TimeMixWeights {
    // Token shift mixing vectors (D,)
    x_r: Tensor, x_w: Tensor, x_k: Tensor, x_v: Tensor, x_a: Tensor, x_g: Tensor,
    // Decay
    w0: Tensor,  // (D,)
    w1: Tensor,  // (D, decay_lora)
    w2: Tensor,  // (decay_lora, D)
    // 'a' gate
    a0: Tensor,  // (D,)
    a1: Tensor,  // (D, a_lora)
    a2: Tensor,  // (a_lora, D)
    // v_first mixing
    v0: Tensor,  // (D,)
    v1: Tensor,  // (D, v_lora)
    v2: Tensor,  // (v_lora, D)
    // Output gate
    g1: Tensor,  // (D, g_lora)
    g2: Tensor,  // (g_lora, D)
    // Key/receptance modifiers
    k_k: Tensor, // (D,)
    k_a: Tensor, // (D,)
    r_k: Tensor, // (D,) flattened
    // Linear projections (D, D) — auto-quantized
    key_w: QuantMatrix,
    value_w: QuantMatrix,
    receptance_w: QuantMatrix,
    output_w: QuantMatrix,
    // GroupNorm
    ln_x_w: Tensor, // (D,)
    ln_x_b: Tensor, // (D,)
}

/// Per-layer weights for channel mixing (FFN).
struct ChannelMixWeights {
    x_k: Tensor,          // (D,)
    key_w: QuantMatrix,    // (D_FFN, D) — auto-quantized
    value_w: QuantMatrix,  // (D, D_FFN) — auto-quantized
}

/// Per-layer weights including LayerNorm.
struct LayerWeights {
    ln1_w: Tensor, ln1_b: Tensor,
    ln2_w: Tensor, ln2_b: Tensor,
    time_mix: TimeMixWeights,
    channel_mix: ChannelMixWeights,
}

pub struct Rwkv7Model {
    pub config: Rwkv7Config,
    emb: Tensor,           // (V, D) — pre-normalized, kept f32
    layers: Vec<LayerWeights>,
    ln_out_w: Tensor,      // (D,)
    ln_out_b: Tensor,      // (D,)
    head_w: QuantMatrix,    // (V, D) — auto-quantized (biggest matrix)
}

/// Per-layer activation steering for generation (B.11.2 / B.11.3).
/// Supports multiplicative scaling (B.11.2, KILLED) and additive offsets (B.11.3).
pub struct SteeringConfig {
    /// Per-layer scale factors for tm_out. Length must equal n_layer.
    /// 1.0 = no change, >1.0 amplifies state readout.
    pub tm_out_scale: Vec<f32>,
    /// Per-layer additive offset vectors for tm_out (B.11.3 state-offset tuning).
    /// None = no offset. Some(vec) must have length n_layer, each inner vec D-sized.
    pub tm_out_offset: Option<Vec<Vec<f32>>>,
    /// Global scale for offsets. Applied as: tm_out[j] += offset_scale * offset[layer][j]
    pub offset_scale: f32,
}

impl SteeringConfig {
    /// Scale tm_out in layers [start..end) by alpha, rest unchanged.
    pub fn late_layers(n_layer: usize, start: usize, end: usize, alpha: f32) -> Self {
        let mut scales = vec![1.0f32; n_layer];
        for i in start..end.min(n_layer) {
            scales[i] = alpha;
        }
        Self { tm_out_scale: scales, tm_out_offset: None, offset_scale: 0.0 }
    }

    /// State-offset tuning: additive offset on tm_out per layer (B.11.3).
    /// offsets[layer] is a D-sized vector added to tm_out at every timestep.
    pub fn with_offset(offsets: Vec<Vec<f32>>, scale: f32) -> Self {
        let n = offsets.len();
        Self {
            tm_out_scale: vec![1.0f32; n],
            tm_out_offset: Some(offsets),
            offset_scale: scale,
        }
    }
}

/// Recurrent state for inference.
#[derive(Clone)]
pub struct Rwkv7State {
    // Per layer: [x_prev_att (D,), state_matrix (H*N*N,), x_prev_ffn (D,)]
    x_prev_att: Vec<Tensor>,   // n_layer tensors of shape (D,)
    state_mat: Vec<Vec<f32>>,  // n_layer * H matrices of N*N, stored flat as H*N*N
    x_prev_ffn: Vec<Tensor>,   // n_layer tensors of shape (D,)
    v_first: Tensor,           // (D,)
}

impl Rwkv7State {
    pub fn new(config: &Rwkv7Config) -> Self {
        let d = config.n_embd;
        let h = config.n_head;
        let n = config.head_size;
        Self {
            x_prev_att: (0..config.n_layer).map(|_| Tensor::zeros(&[d])).collect(),
            state_mat: (0..config.n_layer).map(|_| vec![0.0f32; h * n * n]).collect(),
            x_prev_ffn: (0..config.n_layer).map(|_| Tensor::zeros(&[d])).collect(),
            v_first: Tensor::zeros(&[d]),
        }
    }

    /// Number of layers in this state.
    pub fn n_layers(&self) -> usize { self.x_prev_att.len() }

    /// Read-only access to per-layer attention residual (D-sized).
    pub fn att_residual(&self, layer: usize) -> &[f32] { &self.x_prev_att[layer].data }

    /// Read-only access to per-layer WKV state matrix (H*N*N flat).
    pub fn wkv_state(&self, layer: usize) -> &[f32] { &self.state_mat[layer] }

    /// Read-only access to per-layer FFN residual (D-sized).
    pub fn ffn_residual(&self, layer: usize) -> &[f32] { &self.x_prev_ffn[layer].data }

    /// Apply additive state delta: state_mat[l] += scale * delta_state[l],
    /// x_prev_att[l] += scale * delta_att[l]. Used by B.11.4 state-delta tuning.
    pub fn apply_state_delta(&mut self, delta_state: &[Vec<f32>], delta_att: &[Vec<f32>], scale: f32) {
        let n_layer = self.state_mat.len();
        for l in 0..n_layer.min(delta_state.len()) {
            for j in 0..self.state_mat[l].len().min(delta_state[l].len()) {
                self.state_mat[l][j] += scale * delta_state[l][j];
            }
        }
        for l in 0..n_layer.min(delta_att.len()) {
            for j in 0..self.x_prev_att[l].data.len().min(delta_att[l].len()) {
                self.x_prev_att[l].data[j] += scale * delta_att[l][j];
            }
        }
    }
}

/// Pre-allocated workspace for zero-allocation forward pass.
/// Allocated once at model load, reused every token (~1400 allocs/token eliminated).
pub struct Scratch {
    // D-sized buffers
    x: Vec<f32>,
    ln_buf: Vec<f32>,
    xx: Vec<f32>,
    xr: Vec<f32>,
    xw: Vec<f32>,
    xk: Vec<f32>,
    xv: Vec<f32>,
    xa: Vec<f32>,
    xg: Vec<f32>,
    r: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    w_full: Vec<f32>,
    a: Vec<f32>,
    kk: Vec<f32>,
    k_mod: Vec<f32>,
    v_out: Vec<f32>,
    out_vec: Vec<f32>,
    out_normed: Vec<f32>,
    gated: Vec<f32>,
    g: Vec<f32>,
    lora_d: Vec<f32>,
    tm_out: Vec<f32>,
    cm_out: Vec<f32>,
    // LoRA intermediate
    lora_tmp: Vec<f32>,
    // FFN (d_ffn sized)
    ffn_proj: Vec<f32>,
    ffn_act: Vec<f32>,
    // Per-head (N*N)
    vk: Vec<f32>,
    ab: Vec<f32>,
    s_ab: Vec<f32>,
    // Per-head (N)
    kk_a: Vec<f32>,
    neg_kk: Vec<f32>,
    out_h: Vec<f32>,
    // Q8 quantization buffer
    v_q: Vec<i8>,
}

impl Scratch {
    fn new(d: usize, n: usize, d_ffn: usize, max_lora: usize) -> Self {
        let nn = n * n;
        Self {
            x: vec![0.0; d], ln_buf: vec![0.0; d],
            xx: vec![0.0; d], xr: vec![0.0; d], xw: vec![0.0; d],
            xk: vec![0.0; d], xv: vec![0.0; d], xa: vec![0.0; d], xg: vec![0.0; d],
            r: vec![0.0; d], k: vec![0.0; d], v: vec![0.0; d],
            w_full: vec![0.0; d], a: vec![0.0; d], kk: vec![0.0; d],
            k_mod: vec![0.0; d], v_out: vec![0.0; d],
            out_vec: vec![0.0; d], out_normed: vec![0.0; d],
            gated: vec![0.0; d], g: vec![0.0; d],
            lora_d: vec![0.0; d], tm_out: vec![0.0; d], cm_out: vec![0.0; d],
            lora_tmp: vec![0.0; max_lora],
            ffn_proj: vec![0.0; d_ffn], ffn_act: vec![0.0; d_ffn],
            vk: vec![0.0; nn], ab: vec![0.0; nn], s_ab: vec![0.0; nn],
            kk_a: vec![0.0; n], neg_kk: vec![0.0; n], out_h: vec![0.0; n],
            v_q: vec![0; d_ffn.max(d)],
        }
    }
}

impl Rwkv7Model {
    /// Load model from a SafeTensors file.
    /// Expects HuggingFace-format keys (model.layers.N.att.xxx).
    pub fn load(path: &Path, config: Rwkv7Config) -> Self {
        eprintln!("[rwkv7] loading weights from {} ...", path.display());
        let st = SafeTensorsFile::open(path)
            .unwrap_or_else(|e| panic!("[rwkv7] failed to open weights: {}", e));

        // Detect key format: HF (RWKV/RWKV7-Goose-*-HF) vs BlinkDL
        let names = st.tensor_names();
        let hf_format = names.iter().any(|n| n.starts_with("model.embeddings"));

        let d = config.n_embd;
        let v = config.vocab_size;

        let (emb, head_w_f32, ln_out_w, ln_out_b);

        if hf_format {
            let emb_raw = st.get_tensor("model.embeddings.weight");
            let ln0_w = st.get_tensor("model.layers.0.pre_norm.weight");
            let ln0_b = st.get_tensor("model.layers.0.pre_norm.bias");

            let mut emb_data = Vec::with_capacity(v * d);
            for row in 0..v {
                let start = row * d;
                let row_t = Tensor::from_data(emb_raw.data[start..start + d].to_vec(), vec![d]);
                let normed = layer_norm(&row_t, &ln0_w, &ln0_b, 1e-5);
                emb_data.extend_from_slice(&normed.data);
            }
            emb = Tensor::from_data(emb_data, vec![v, d]);
            head_w_f32 = st.get_tensor("lm_head.weight");
            ln_out_w = st.get_tensor("model.norm.weight");
            ln_out_b = st.get_tensor("model.norm.bias");
        } else {
            let emb_raw = st.get_tensor("emb.weight");
            let ln0_w = st.get_tensor("blocks.0.ln0.weight");
            let ln0_b = st.get_tensor("blocks.0.ln0.bias");

            let mut emb_data = Vec::with_capacity(v * d);
            for row in 0..v {
                let start = row * d;
                let row_t = Tensor::from_data(emb_raw.data[start..start + d].to_vec(), vec![d]);
                let normed = layer_norm(&row_t, &ln0_w, &ln0_b, 1e-5);
                emb_data.extend_from_slice(&normed.data);
            }
            emb = Tensor::from_data(emb_data, vec![v, d]);
            head_w_f32 = st.get_tensor("head.weight");
            ln_out_w = st.get_tensor("ln_out.weight");
            ln_out_b = st.get_tensor("ln_out.bias");
        }

        // Determine quantization policy via empirical benchmark on head matrix
        eprintln!("[rwkv7] benchmarking quantization levels ...");
        let (policy, bench_results) = QuantPolicy::auto_detect(&head_w_f32);
        for r in &bench_results {
            eprintln!("[rwkv7]   {}", r);
        }
        let policy_name = match &policy {
            QuantPolicy::Uniform(l) => format!("Uniform({})", l.name()),
            QuantPolicy::PerMatrix => "PerMatrix".to_string(),
        };
        eprintln!("[rwkv7] policy: {}", policy_name);

        // Head goes through the same policy
        let (head_w, _) = quantize_matrix(&head_w_f32, &policy);
        eprintln!("[rwkv7] head: {} ({:.1} MB)",
                  head_w.format_name(), head_w.mem_bytes() as f64 / 1_048_576.0);

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let layer = if hf_format {
                load_layer_hf(&st, i, &policy)
            } else {
                load_layer_blink(&st, i, &policy)
            };
            eprintln!("[rwkv7]   layer {} loaded", i);
            layers.push(layer);
        }

        report_quantization(&layers, &head_w, v, d);
        eprintln!("[rwkv7] loaded {} layers, D={}, H={}, N={}, V={}",
                  config.n_layer, d, config.n_head, config.head_size, v);

        Self { config, emb, layers, ln_out_w, ln_out_b, head_w }
    }

    /// Load model from a GGUF file.
    /// Auto-detects tensor key format (llama.cpp RWKV converter naming).
    pub fn load_gguf(path: &Path, config: Rwkv7Config) -> Self {
        log_ram("before GGUF load");
        eprintln!("[rwkv7] loading weights from {} (GGUF) ...", path.display());
        let gf = GgufFile::open(path)
            .unwrap_or_else(|e| panic!("[rwkv7] failed to open GGUF: {}", e));
        check_ram_or_abort(95.0, "after GGUF file read");

        // Try to refine config from GGUF metadata
        let config = if let Some((n_layer, n_embd, n_head, head_size, vocab_size)) = gf.detect_config() {
            eprintln!("[rwkv7] GGUF metadata: layers={}, embd={}, heads={}, head_size={}, vocab={}",
                      n_layer, n_embd, n_head, head_size, vocab_size);
            Rwkv7Config { n_layer, n_embd, n_head, head_size, vocab_size, g1k: config.g1k }
        } else {
            eprintln!("[rwkv7] GGUF metadata incomplete, using provided config");
            config
        };

        let d = config.n_embd;
        let v = config.vocab_size;

        // Detect key format from tensor names
        let names = gf.tensor_names();
        let llama_format = names.iter().any(|n| n.starts_with("token_embd"));
        let blink_format = names.iter().any(|n| n.starts_with("blocks.") || n.starts_with("emb."));
        let hf_format = names.iter().any(|n| n.starts_with("model.embeddings"));

        eprintln!("[rwkv7] GGUF key format: {}",
                  if llama_format { "llama.cpp" }
                  else if hf_format { "HuggingFace" }
                  else if blink_format { "BlinkDL" }
                  else { "unknown" });

        // Detect sub-variant of llama format (old: attn_w1, new: time_mix_w1)
        let llama_v2 = llama_format && names.iter().any(|n| n.contains("time_mix_w1"));

        // Load global tensors (small, always dequantized to F32).
        // Head matrix loaded separately via load_gguf_matrix (native quant when possible).
        let (emb_raw, ln0_w, ln0_b, ln_out_w, ln_out_b) = if llama_format {
            let ln0_key = if gf.has_tensor("token_embd_norm.weight") {
                ("token_embd_norm.weight", "token_embd_norm.bias")
            } else {
                ("blk.0.attn_norm.weight", "blk.0.attn_norm.bias")
            };
            (
                gf.get_tensor("token_embd.weight"),
                gf.get_tensor(ln0_key.0),
                gf.get_tensor(ln0_key.1),
                gf.get_tensor("output_norm.weight"),
                gf.get_tensor("output_norm.bias"),
            )
        } else if hf_format {
            (
                gf.get_tensor("model.embeddings.weight"),
                gf.get_tensor("model.layers.0.pre_norm.weight"),
                gf.get_tensor("model.layers.0.pre_norm.bias"),
                gf.get_tensor("model.norm.weight"),
                gf.get_tensor("model.norm.bias"),
            )
        } else {
            (
                gf.get_tensor("emb.weight"),
                gf.get_tensor("blocks.0.ln0.weight"),
                gf.get_tensor("blocks.0.ln0.bias"),
                gf.get_tensor("ln_out.weight"),
                gf.get_tensor("ln_out.bias"),
            )
        };

        // Embedding surgery: apply ln0 normalization
        let mut emb_data = Vec::with_capacity(v * d);
        for row in 0..v {
            let start = row * d;
            let row_t = Tensor::from_data(emb_raw.data[start..start + d].to_vec(), vec![d]);
            let normed = layer_norm(&row_t, &ln0_w, &ln0_b, 1e-5);
            emb_data.extend_from_slice(&normed.data);
        }
        let emb = Tensor::from_data(emb_data, vec![v, d]);

        // Detect if GGUF file has quantized tensors → use native quant (no F32 dequant).
        // This saves ~10x RAM: Q4_K stays as 1.8 GB instead of expanding to 11.6 GB in F32.
        let sample_type = gf.tensor_ggml_type(
            if llama_format { "output.weight" }
            else if hf_format { "lm_head.weight" }
            else { "head.weight" }
        ).unwrap_or(0);
        let native_quant = ggml_block_info(sample_type).is_some();

        // Quantization policy: native quant bypasses this (kept as fallback for F32/F16 tensors)
        let policy = if native_quant {
            eprintln!("[rwkv7] native quantized loading — weights stay in GGUF format (no F32 expansion)");
            QuantPolicy::Uniform(QuantLevel::F32) // fallback for any non-k-quant tensors
        } else {
            eprintln!("[rwkv7] GGUF F32/F16 source — forcing F32 policy");
            QuantPolicy::Uniform(QuantLevel::F32)
        };

        // Head matrix: native quant or F32
        let head_key = if llama_format { "output.weight" }
            else if hf_format { "lm_head.weight" }
            else { "head.weight" };
        let head_w = load_gguf_matrix(&gf, head_key, &policy);
        eprintln!("[rwkv7] head: {} ({:.1} MB)",
                  head_w.format_name(), head_w.mem_bytes() as f64 / 1_048_576.0);

        check_ram_or_abort(95.0, "after head loaded");

        // Load layers
        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let layer = if llama_v2 {
                load_layer_gguf_llama_v2(&gf, i, &policy)
            } else if llama_format {
                load_layer_gguf_llama(&gf, i, &policy)
            } else if hf_format {
                load_layer_gguf_hf(&gf, i, &policy)
            } else {
                load_layer_gguf_blink(&gf, i, &policy)
            };
            eprintln!("[rwkv7]   layer {} loaded", i);
            layers.push(layer);
            if (i + 1) % 4 == 0 {
                check_ram_or_abort(95.0, &format!("after layer {}", i));
            }
        }

        report_quantization(&layers, &head_w, v, d);
        log_ram("model fully loaded");
        eprintln!("[rwkv7] loaded {} layers, D={}, H={}, N={}, V={}",
                  config.n_layer, d, config.n_head, config.head_size, v);

        Self { config, emb, layers, ln_out_w, ln_out_b, head_w }
    }

    /// Auto-detect format from path and load model.
    /// - `.gguf` extension → GGUF loader
    /// - directory with `model.safetensors` → SafeTensors loader
    /// - `.safetensors` file → SafeTensors loader
    pub fn load_auto(path: &Path) -> Self {
        let path_str = path.display().to_string();

        if path_str.ends_with(".gguf") {
            let config = Rwkv7Config::from_weights_dir(&path_str);
            Self::load_gguf(path, config)
        } else if path.is_dir() {
            let st_path = path.join("model.safetensors");
            let config = Rwkv7Config::from_weights_dir(&path_str);
            Self::load(&st_path, config)
        } else {
            // Assume SafeTensors
            let config = Rwkv7Config::from_weights_dir(&path_str);
            Self::load(path, config)
        }
    }

    /// Analyze and optionally modify byte-token embeddings (rows 0-255).
    ///
    /// Methods:
    /// - "analyze": print stats only, no modification
    /// - "norm": equalize byte embedding norms to match text-token median
    /// - "center": blend byte embeddings toward global centroid (alpha=0.3)
    /// - "spread": decorrelate byte embeddings via mean-shift + variance scaling
    pub fn embedding_surgery(&mut self, method: &str) {
        let d = self.config.n_embd;
        let v = self.config.vocab_size;

        // Compute norms for byte tokens (0-255) and text tokens (256+)
        let mut byte_norms = Vec::with_capacity(256);
        let mut text_norms = Vec::with_capacity(v - 256);
        for row in 0..v {
            let start = row * d;
            let norm: f32 = self.emb.data[start..start + d].iter()
                .map(|x| x * x).sum::<f32>().sqrt();
            if row < 256 {
                byte_norms.push(norm);
            } else {
                text_norms.push(norm);
            }
        }

        // Compute centroids
        let mut byte_centroid = vec![0.0f32; d];
        for row in 0..256 {
            let start = row * d;
            for j in 0..d {
                byte_centroid[j] += self.emb.data[start + j];
            }
        }
        for j in 0..d { byte_centroid[j] /= 256.0; }

        let mut global_centroid = vec![0.0f32; d];
        for row in 0..v {
            let start = row * d;
            for j in 0..d {
                global_centroid[j] += self.emb.data[start + j];
            }
        }
        for j in 0..d { global_centroid[j] /= v as f32; }

        // Stats
        let byte_mean_norm = byte_norms.iter().sum::<f32>() / 256.0;
        let text_mean_norm = text_norms.iter().sum::<f32>() / text_norms.len() as f32;
        let mut sorted_text = text_norms.clone();
        sorted_text.sort_by(|a, b| crate::domain::error::safe_f32_cmp(a, b));
        let text_median_norm = sorted_text[sorted_text.len() / 2];
        let mut sorted_byte = byte_norms.clone();
        sorted_byte.sort_by(|a, b| crate::domain::error::safe_f32_cmp(a, b));
        let byte_median_norm = sorted_byte[128];

        // Byte embedding variance (how spread out are they?)
        let byte_var: f32 = (0..256).map(|row| {
            let start = row * d;
            self.emb.data[start..start + d].iter()
                .zip(byte_centroid.iter())
                .map(|(x, c)| (x - c) * (x - c))
                .sum::<f32>()
        }).sum::<f32>() / 256.0;

        // Centroid distance
        let centroid_dist: f32 = byte_centroid.iter().zip(global_centroid.iter())
            .map(|(b, g)| (b - g) * (b - g)).sum::<f32>().sqrt();

        // ASCII (32-126) vs non-ASCII (128-255) norms
        let ascii_mean: f32 = byte_norms[32..127].iter().sum::<f32>() / 95.0;
        let nonascii_mean: f32 = byte_norms[128..256].iter().sum::<f32>() / 128.0;

        eprintln!("[emb-surgery] analysis:");
        eprintln!("  byte norms:   mean={:.3}, median={:.3}, min={:.3}, max={:.3}",
                  byte_mean_norm, byte_median_norm, sorted_byte[0], sorted_byte[255]);
        eprintln!("  text norms:   mean={:.3}, median={:.3}",
                  text_mean_norm, text_median_norm);
        eprintln!("  ascii mean:   {:.3} (32-126), non-ascii: {:.3} (128-255)",
                  ascii_mean, nonascii_mean);
        eprintln!("  byte var:     {:.4}", byte_var);
        eprintln!("  centroid dist: {:.4} (byte centroid vs global centroid)", centroid_dist);
        eprintln!("  norm ratio:   {:.3} (byte/text)", byte_mean_norm / text_mean_norm);

        match method {
            "analyze" => {
                eprintln!("[emb-surgery] analyze only — no modifications");
            }
            "norm" => {
                // Equalize byte embedding norms to text median
                let target = text_median_norm;
                eprintln!("[emb-surgery] norm equalization: target={:.3}", target);
                for row in 0..256 {
                    let norm = byte_norms[row];
                    if norm > 1e-8 {
                        let scale = target / norm;
                        let start = row * d;
                        for j in 0..d {
                            self.emb.data[start + j] *= scale;
                        }
                    }
                }
            }
            s if s.starts_with("center") => {
                // Blend byte embeddings toward global centroid
                // Format: "center" (default alpha=0.3) or "center0.5" (custom alpha)
                let alpha: f32 = if s.len() > 6 {
                    s[6..].parse().unwrap_or(0.3)
                } else {
                    0.3
                };
                eprintln!("[emb-surgery] center blend: alpha={}", alpha);
                for row in 0..256 {
                    let start = row * d;
                    for j in 0..d {
                        self.emb.data[start + j] = (1.0 - alpha) * self.emb.data[start + j]
                            + alpha * global_centroid[j];
                    }
                }
            }
            "spread" => {
                // Increase variance of byte embeddings around their centroid
                let scale = 1.5f32;
                eprintln!("[emb-surgery] spread: scale={}", scale);
                for row in 0..256 {
                    let start = row * d;
                    for j in 0..d {
                        let delta = self.emb.data[start + j] - byte_centroid[j];
                        self.emb.data[start + j] = byte_centroid[j] + scale * delta;
                    }
                }
            }
            _ => {
                eprintln!("[emb-surgery] unknown method '{}' — no modification", method);
            }
        }
    }

    /// Create a pre-allocated scratch workspace for zero-alloc forward passes.
    pub fn create_scratch(&self) -> Scratch {
        let d = self.config.n_embd;
        let n = self.config.head_size;
        let d_ffn = self.layers[0].channel_mix.key_w.shape().0;
        let max_lora = self.layers.iter().map(|lw| {
            let tm = &lw.time_mix;
            // SAFETY: array is always 4 elements, never empty
            *[tm.w1.shape[0], tm.a1.shape[0], tm.g1.shape[0], tm.v1.shape[0]]
                .iter().max().unwrap()
        // SAFETY: self.layers is non-empty (validated at load)
        }).max().unwrap();
        let s = Scratch::new(d, n, d_ffn, max_lora);
        let mem_kb = std::mem::size_of_val(&s) as f64 / 1024.0
            + (d * 24 + max_lora + d_ffn * 2 + n * n * 3 + n * 3) as f64 * 4.0 / 1024.0
            + d_ffn as f64 / 1024.0;
        eprintln!("[rwkv7] scratch: {:.1} KB pre-allocated (zero-alloc forward)", mem_kb);
        s
    }

    /// Single-step forward pass (RNN mode), zero allocation.
    /// Writes logits into out_logits (V-sized slice).
    pub fn forward_into(&self, token: usize, state: &mut Rwkv7State, scratch: &mut Scratch, out_logits: &mut [f32]) {
        assert!(token < self.config.vocab_size,
            "[rwkv7] token {} out of bounds (vocab_size={})", token, self.config.vocab_size);

        let d = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;

        // Embedding lookup — zero alloc
        let emb_start = token * d;
        scratch.x.copy_from_slice(&self.emb.data[emb_start..emb_start + d]);

        for i in 0..self.config.n_layer {
            let lw = &self.layers[i];

            // Time mixing
            layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                            &lw.ln1_w.data, &lw.ln1_b.data, 1e-5);
            time_mixing_scratch(scratch, i, h, n,
                                &state.x_prev_att[i].data,
                                &state.v_first.data,
                                &mut state.state_mat[i],
                                &lw.time_mix,
                                self.config.g1k);
            state.x_prev_att[i].data.copy_from_slice(&scratch.ln_buf);
            if i == 0 {
                state.v_first.data.copy_from_slice(&scratch.v);
            }
            add_inplace(&mut scratch.x, &scratch.tm_out);

            // Channel mixing
            layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                            &lw.ln2_w.data, &lw.ln2_b.data, 1e-5);
            channel_mixing_scratch(scratch,
                                   &state.x_prev_ffn[i].data,
                                   &lw.channel_mix);
            state.x_prev_ffn[i].data.copy_from_slice(&scratch.ln_buf);
            add_inplace(&mut scratch.x, &scratch.cm_out);
        }

        // Final LayerNorm + head projection
        layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                        &self.ln_out_w.data, &self.ln_out_b.data, 1e-5);
        self.head_w.mat_vec_mul_into(out_logits, &scratch.ln_buf, &mut scratch.v_q);
    }

    /// Steered forward pass — identical to forward_into but scales tm_out
    /// per-layer according to SteeringConfig (B.11.2 activation steering).
    /// Use during generation only; prefill should use forward_into.
    pub fn forward_steered(
        &self, token: usize, state: &mut Rwkv7State, scratch: &mut Scratch,
        out_logits: &mut [f32], steering: &SteeringConfig,
    ) {
        assert!(token < self.config.vocab_size,
            "[rwkv7] token {} out of bounds (vocab_size={})", token, self.config.vocab_size);

        let d = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;

        let emb_start = token * d;
        scratch.x.copy_from_slice(&self.emb.data[emb_start..emb_start + d]);

        for i in 0..self.config.n_layer {
            let lw = &self.layers[i];

            layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                            &lw.ln1_w.data, &lw.ln1_b.data, 1e-5);
            time_mixing_scratch(scratch, i, h, n,
                                &state.x_prev_att[i].data,
                                &state.v_first.data,
                                &mut state.state_mat[i],
                                &lw.time_mix,
                                self.config.g1k);
            state.x_prev_att[i].data.copy_from_slice(&scratch.ln_buf);
            if i == 0 {
                state.v_first.data.copy_from_slice(&scratch.v);
            }

            // Activation steering: scale and/or offset tm_out before residual addition
            let alpha = steering.tm_out_scale[i];
            if alpha != 1.0 {
                for j in 0..d { scratch.tm_out[j] *= alpha; }
            }
            if let Some(ref offsets) = steering.tm_out_offset {
                let s = steering.offset_scale;
                if s != 0.0 {
                    for j in 0..d { scratch.tm_out[j] += s * offsets[i][j]; }
                }
            }
            add_inplace(&mut scratch.x, &scratch.tm_out);

            layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                            &lw.ln2_w.data, &lw.ln2_b.data, 1e-5);
            channel_mixing_scratch(scratch,
                                   &state.x_prev_ffn[i].data,
                                   &lw.channel_mix);
            state.x_prev_ffn[i].data.copy_from_slice(&scratch.ln_buf);
            add_inplace(&mut scratch.x, &scratch.cm_out);
        }

        layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                        &self.ln_out_w.data, &self.ln_out_b.data, 1e-5);
        self.head_w.mat_vec_mul_into(out_logits, &scratch.ln_buf, &mut scratch.v_q);
    }

    /// Forward pass that captures per-layer tm_out vectors (B.11.3 activation capture).
    /// Returns n_layer vectors of dimension D. Used to compute mean activation differences
    /// between completion and question formats for state-offset tuning.
    pub fn forward_capture_tm_out(
        &self, token: usize, state: &mut Rwkv7State, scratch: &mut Scratch,
        out_logits: &mut [f32],
    ) -> Vec<Vec<f32>> {
        assert!(token < self.config.vocab_size);
        let d = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;
        let n_layer = self.config.n_layer;

        let emb_start = token * d;
        scratch.x.copy_from_slice(&self.emb.data[emb_start..emb_start + d]);

        let mut captured = Vec::with_capacity(n_layer);

        for i in 0..n_layer {
            let lw = &self.layers[i];
            layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                            &lw.ln1_w.data, &lw.ln1_b.data, 1e-5);
            time_mixing_scratch(scratch, i, h, n,
                                &state.x_prev_att[i].data,
                                &state.v_first.data,
                                &mut state.state_mat[i],
                                &lw.time_mix,
                                self.config.g1k);
            state.x_prev_att[i].data.copy_from_slice(&scratch.ln_buf);
            if i == 0 {
                state.v_first.data.copy_from_slice(&scratch.v);
            }
            captured.push(scratch.tm_out.clone());
            add_inplace(&mut scratch.x, &scratch.tm_out);

            layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                            &lw.ln2_w.data, &lw.ln2_b.data, 1e-5);
            channel_mixing_scratch(scratch,
                                   &state.x_prev_ffn[i].data,
                                   &lw.channel_mix);
            state.x_prev_ffn[i].data.copy_from_slice(&scratch.ln_buf);
            add_inplace(&mut scratch.x, &scratch.cm_out);
        }

        layer_norm_into(&mut scratch.ln_buf, &scratch.x,
                        &self.ln_out_w.data, &self.ln_out_b.data, 1e-5);
        self.head_w.mat_vec_mul_into(out_logits, &scratch.ln_buf, &mut scratch.v_q);
        captured
    }

    /// Compute spectral profile of all weight matrices.
    /// Writes results as JSON to output_path and prints summary to stderr.
    pub fn spectral_profile(&self, output_path: &str) {
        use crate::domain::spectral::*;
        let n_layers = self.config.n_layer;
        let t_start = std::time::Instant::now();

        eprintln!("[spectral] profiling {} layers, d={}, h={} ...",
                  n_layers, self.config.n_embd, self.config.n_head);

        let mut json_layers = Vec::new();

        for li in 0..n_layers {
            let layer = &self.layers[li];
            let tm = &layer.time_mix;
            let cm = &layer.channel_mix;
            let t_layer = std::time::Instant::now();

            // A1+A5: Matrix profiles (dequantize → f64 → analyze → discard)
            let profile_qm = |name: &str, qm: &crate::domain::quant::QuantMatrix| -> MatrixProfile {
                let (data, rows, cols) = qm.dequantize_f64();
                MatrixProfile::compute(name, &data, rows, cols)
            };

            let w_r = profile_qm("W_r", &tm.receptance_w);
            let w_k = profile_qm("W_k", &tm.key_w);
            let w_v = profile_qm("W_v", &tm.value_w);
            let w_o = profile_qm("W_o", &tm.output_w);
            let ffn_up = profile_qm("FFN_up", &cm.key_w);
            let ffn_down = profile_qm("FFN_down", &cm.value_w);

            // A2: Decay spectrum + saturation
            let decay = decay_spectrum(&tm.w0.data);
            let (a0_sat, a0_total) = saturation_count(&tm.a0.data, 5.0);

            // A3: FFN neuron analysis (reuse dequantized FFN_up)
            let (ffn_data, ffn_rows, ffn_cols) = cm.key_w.dequantize_f64();
            let ffn = ffn_neuron_analysis(&ffn_data, ffn_rows, ffn_cols);

            // A4: Cross-coherence (W_r vs W_k, W_r vs W_v, W_k vs W_v)
            let (dr, rr, rc) = tm.receptance_w.dequantize_f64();
            let (dk, kr, kc) = tm.key_w.dequantize_f64();
            let (dv, vr, vc) = tm.value_w.dequantize_f64();
            let coh_rk = cross_coherence(&dr, rr, rc, &dk, kr, kc, 8);
            let coh_rv = cross_coherence(&dr, rr, rc, &dv, vr, vc, 8);
            let coh_kv = cross_coherence(&dk, kr, kc, &dv, vr, vc, 8);

            let elapsed = t_layer.elapsed().as_secs_f64();
            eprintln!("[spectral]   layer {}: W_r rank={:.0}/{} null={}, W_k rank={:.0} null={}, decay={:.4}±{:.4}, sat={}/{}, coh_rk={:.3} ({:.1}s)",
                      li, w_r.effective_rank, w_r.cols, w_r.null_space_dim,
                      w_k.effective_rank, w_k.null_space_dim,
                      decay.decay_mean, decay.decay_std,
                      a0_sat, a0_total, coh_rk, elapsed);

            // Build JSON for this layer
            let layer_json = format!(
                r#"{{"layer":{},"W_r":{},"W_k":{},"W_v":{},"W_o":{},"FFN_up":{},"FFN_down":{},"decay":{},"gate_saturation":{{"saturated":{},"total":{},"fraction":{:.4}}},"ffn_neurons":{},"coherence":{{"r_k":{:.4},"r_v":{:.4},"k_v":{:.4}}}}}"#,
                li,
                w_r.to_json(), w_k.to_json(), w_v.to_json(), w_o.to_json(),
                ffn_up.to_json(), ffn_down.to_json(),
                decay.to_json(),
                a0_sat, a0_total, a0_sat as f64 / a0_total as f64,
                ffn.to_json(),
                coh_rk, coh_rv, coh_kv,
            );
            json_layers.push(layer_json);
        }

        // Embedding profile (f32, not quantized)
        let emb_data: Vec<f64> = self.emb.data.iter().map(|&x| x as f64).collect();
        let emb_profile = MatrixProfile::compute("embedding",
            &emb_data, self.emb.shape[0], self.emb.shape[1]);
        eprintln!("[spectral]   embedding: rank={:.0}/{}, null={}",
                  emb_profile.effective_rank, emb_profile.cols, emb_profile.null_space_dim);

        // Head profile
        let (head_data, head_rows, head_cols) = self.head_w.dequantize_f64();
        let head_profile = MatrixProfile::compute("head", &head_data, head_rows, head_cols);
        eprintln!("[spectral]   head: rank={:.0}/{}, null={}",
                  head_profile.effective_rank, head_profile.cols, head_profile.null_space_dim);

        let total_elapsed = t_start.elapsed().as_secs_f64();
        eprintln!("[spectral] complete in {:.1}s", total_elapsed);

        // Write JSON
        let json = format!(
            r#"{{"model":"rwkv7-{}-world","d_model":{},"n_head":{},"n_layer":{},"head_size":{},"computed_s":{:.1},"embedding":{},"head":{},"layers":[{}]}}"#,
            if self.config.n_embd == 768 { "0.1b" } else { "unknown" },
            self.config.n_embd, self.config.n_head, self.config.n_layer,
            self.config.head_size, total_elapsed,
            emb_profile.to_json(), head_profile.to_json(),
            json_layers.join(","),
        );

        std::fs::write(output_path, &json).unwrap_or_else(|e| {
            eprintln!("[spectral] ERROR writing {}: {}", output_path, e);
        });
        eprintln!("[spectral] profile saved to {}", output_path);
    }
}

/// Zero-allocation time mixing. Input x in s.ln_buf, output in s.tm_out.
fn time_mixing_scratch(
    s: &mut Scratch,
    layer_id: usize, h: usize, n: usize,
    x_prev: &[f32],
    v_first: &[f32],
    state: &mut [f32],
    w: &TimeMixWeights,
    g1k: bool,
) {
    let d = s.ln_buf.len();

    // Step 1: Token shift (x is in s.ln_buf)
    sub_into(&mut s.xx, x_prev, &s.ln_buf);
    add_scaled_into(&mut s.xr, &s.ln_buf, &s.xx, &w.x_r.data);
    add_scaled_into(&mut s.xw, &s.ln_buf, &s.xx, &w.x_w.data);
    add_scaled_into(&mut s.xk, &s.ln_buf, &s.xx, &w.x_k.data);
    add_scaled_into(&mut s.xv, &s.ln_buf, &s.xx, &w.x_v.data);
    add_scaled_into(&mut s.xa, &s.ln_buf, &s.xx, &w.x_a.data);
    add_scaled_into(&mut s.xg, &s.ln_buf, &s.xx, &w.x_g.data);

    // Step 2: Linear projections (auto-quantized)
    w.receptance_w.mat_vec_mul_into(&mut s.r, &s.xr, &mut s.v_q);
    w.key_w.mat_vec_mul_into(&mut s.k, &s.xk, &mut s.v_q);
    w.value_w.mat_vec_mul_into(&mut s.v, &s.xv, &mut s.v_q);

    // Step 3: Data-dependent decay
    mat_vec_mul_into(&mut s.lora_tmp, &w.w1, &s.xw);
    tanh_inplace(&mut s.lora_tmp);
    mat_vec_mul_into(&mut s.lora_d, &w.w2, &s.lora_tmp);
    // Decay: exp(-0.606531 * sigmoid(w0 + lora)) for all variants.
    // The 0.606531 constant comes from llama.cpp's WKV7 kernel.
    // Both SafeTensors (0.1B) and GGUF (1.5B/2.9B) weights are stored
    // in a format calibrated for this formula.
    for i in 0..d {
        let val = w.w0.data[i] + s.lora_d[i];
        s.w_full[i] = (-0.606531 * (1.0 / (1.0 + (-val).exp()))).exp();
    }

    // Step 4: 'a' gate (no tanh — verified empirically: both 0.1B and 1.5B worse with tanh)
    mat_vec_mul_into(&mut s.lora_tmp, &w.a1, &s.xa);
    mat_vec_mul_into(&mut s.lora_d, &w.a2, &s.lora_tmp);
    for i in 0..d {
        s.a[i] = 1.0 / (1.0 + (-(w.a0.data[i] + s.lora_d[i])).exp());
    }

    // Step 5: Output gate
    mat_vec_mul_into(&mut s.lora_tmp, &w.g1, &s.xg);
    sigmoid_inplace(&mut s.lora_tmp);
    mat_vec_mul_into(&mut s.g, &w.g2, &s.lora_tmp);

    // Step 6: Key normalization
    mul_into(&mut s.kk, &s.k, &w.k_k.data);
    l2_normalize_per_head_inplace(&mut s.kk, h, n);

    // Step 7: Key modification with 'a'
    for i in 0..d {
        s.k_mod[i] = s.k[i] * (1.0 + (s.a[i] - 1.0) * w.k_a.data[i]);
    }

    // Step 8: Value mixing with v_first
    s.v_out.copy_from_slice(&s.v);
    if layer_id != 0 {
        mat_vec_mul_into(&mut s.lora_tmp, &w.v1, &s.xv);
        mat_vec_mul_into(&mut s.lora_d, &w.v2, &s.lora_tmp);
        for i in 0..d {
            let mix = 1.0 / (1.0 + (-(w.v0.data[i] + s.lora_d[i])).exp());
            s.v_out[i] += (v_first[i] - s.v_out[i]) * mix;
        }
    }

    // Step 9: State update (per head)
    for head in 0..h {
        let ho = head * n * n;
        let hd = head * n;

        outer_product_into(&mut s.vk, &s.v_out[hd..hd + n], &s.k_mod[hd..hd + n]);

        for j in 0..n { s.kk_a[j] = s.kk[hd + j] * s.a[hd + j]; }
        for j in 0..n { s.neg_kk[j] = -s.kk[hd + j]; }
        outer_product_into(&mut s.ab, &s.neg_kk, &s.kk_a);

        mat_mat_mul_small_into(&mut s.s_ab, &state[ho..ho + n * n], &s.ab, n);
        for i in 0..n {
            for j in 0..n {
                let idx = ho + i * n + j;
                state[idx] = state[idx] * s.w_full[hd + j] + s.s_ab[i * n + j] + s.vk[i * n + j];
            }
        }

        mat_vec_mul_small_into(&mut s.out_h, &state[ho..ho + n * n], &s.r[hd..hd + n], n);
        s.out_vec[hd..hd + n].copy_from_slice(&s.out_h);
    }

    // Step 10: GroupNorm
    group_norm_into(&mut s.out_normed, &s.out_vec, h,
                    &w.ln_x_w.data, &w.ln_x_b.data, 64e-5);

    // Step 11: Bonus (r * k * r_k per head, summed, scaled by v)
    // G1k uses k_mod (modified key after a-gate), non-G1k uses original k.
    for head in 0..h {
        let hd = head * n;
        let mut bonus_sum = 0.0f32;
        let k_src = if g1k { &s.k_mod } else { &s.k };
        for j in 0..n {
            bonus_sum += s.r[hd + j] * k_src[hd + j] * w.r_k.data[hd + j];
        }
        for j in 0..n {
            s.out_normed[hd + j] += bonus_sum * s.v_out[hd + j];
        }
    }

    // Step 12: Gate + output projection
    for i in 0..d {
        s.gated[i] = s.out_normed[i] * s.g[i];
    }
    w.output_w.mat_vec_mul_into(&mut s.tm_out, &s.gated, &mut s.v_q);
}

/// Zero-allocation channel mixing. Input x in s.ln_buf, output in s.cm_out.
fn channel_mixing_scratch(s: &mut Scratch, x_prev: &[f32], w: &ChannelMixWeights) {
    sub_into(&mut s.xx, x_prev, &s.ln_buf);
    add_scaled_into(&mut s.xr, &s.ln_buf, &s.xx, &w.x_k.data);
    w.key_w.mat_vec_mul_into(&mut s.ffn_proj, &s.xr, &mut s.v_q);
    squared_relu_into(&mut s.ffn_act, &s.ffn_proj);
    w.value_w.mat_vec_mul_into(&mut s.cm_out, &s.ffn_act, &mut s.v_q);
}

// ---- Weight loading helpers ----

fn load_layer_hf(st: &SafeTensorsFile, i: usize, policy: &QuantPolicy) -> LayerWeights {
    // HF key mapping (RWKV/RWKV7-Goose-*-HF SafeTensors format):
    //   attn_norm → ln1, ffn_norm → ln2
    //   attn.x_r [1,1,D] → squeeze to [D]
    //   attn.w_lora.lora.{0,2} → w1/w2, .2.bias → w0
    //   attn.a_lora.lora.{0,2} → a1/a2, .2.bias → a0
    //   attn.v_lora (layers 1+) → v1/v2, .2.bias → v0
    //   attn.g_lora.lora.{0,2} → g1/g2
    //   attn.g_norm → ln_x (GroupNorm)
    //   attn.{k,v,r,o}_proj → key/value/receptance/output
    let att = format!("model.layers.{}.attn", i);
    let blk = format!("model.layers.{}", i);

    let has_v_lora = st.has_tensor(&format!("{}.v_lora.lora.0.weight", att));

    // a_lora: .0.weight = down (a_lora_dim, D), .2.weight = up (D, a_lora_dim), .2.bias = a0 (D)
    let a0 = squeeze(st.get_tensor(&format!("{}.a_lora.lora.2.bias", att)));
    let a1 = st.get_tensor(&format!("{}.a_lora.lora.0.weight", att));
    let a2 = st.get_tensor(&format!("{}.a_lora.lora.2.weight", att));

    let (v0, v1, v2) = if has_v_lora {
        (
            squeeze(st.get_tensor(&format!("{}.v_lora.lora.2.bias", att))),
            st.get_tensor(&format!("{}.v_lora.lora.0.weight", att)),
            st.get_tensor(&format!("{}.v_lora.lora.2.weight", att)),
        )
    } else {
        (a0.clone(), a1.clone(), a2.clone())
    };

    LayerWeights {
        ln1_w: st.get_tensor(&format!("{}.attn_norm.weight", blk)),
        ln1_b: st.get_tensor(&format!("{}.attn_norm.bias", blk)),
        ln2_w: st.get_tensor(&format!("{}.ffn_norm.weight", blk)),
        ln2_b: st.get_tensor(&format!("{}.ffn_norm.bias", blk)),
        time_mix: TimeMixWeights {
            x_r: squeeze(st.get_tensor(&format!("{}.x_r", att))),
            x_w: squeeze(st.get_tensor(&format!("{}.x_w", att))),
            x_k: squeeze(st.get_tensor(&format!("{}.x_k", att))),
            x_v: squeeze(st.get_tensor(&format!("{}.x_v", att))),
            x_a: squeeze(st.get_tensor(&format!("{}.x_a", att))),
            x_g: squeeze(st.get_tensor(&format!("{}.x_g", att))),
            w0: squeeze(st.get_tensor(&format!("{}.w_lora.lora.2.bias", att))),
            w1: st.get_tensor(&format!("{}.w_lora.lora.0.weight", att)),
            w2: st.get_tensor(&format!("{}.w_lora.lora.2.weight", att)),
            a0, a1, a2,
            v0, v1, v2,
            g1: st.get_tensor(&format!("{}.g_lora.lora.0.weight", att)),
            g2: st.get_tensor(&format!("{}.g_lora.lora.2.weight", att)),
            k_k: st.get_tensor(&format!("{}.k_k", att)),
            k_a: st.get_tensor(&format!("{}.k_a", att)),
            r_k: flatten_tensor(st.get_tensor(&format!("{}.r_k", att))),
            key_w: quantize_matrix(&st.get_tensor(&format!("{}.k_proj.weight", att)), policy).0,
            value_w: quantize_matrix(&st.get_tensor(&format!("{}.v_proj.weight", att)), policy).0,
            receptance_w: quantize_matrix(&st.get_tensor(&format!("{}.r_proj.weight", att)), policy).0,
            output_w: quantize_matrix(&st.get_tensor(&format!("{}.o_proj.weight", att)), policy).0,
            ln_x_w: st.get_tensor(&format!("{}.g_norm.weight", att)),
            ln_x_b: st.get_tensor(&format!("{}.g_norm.bias", att)),
        },
        channel_mix: ChannelMixWeights {
            x_k: st.get_tensor(&format!("{}.ffn.x_k", blk)),
            key_w: quantize_matrix(&st.get_tensor(&format!("{}.ffn.key.weight", blk)), policy).0,
            value_w: quantize_matrix(&st.get_tensor(&format!("{}.ffn.value.weight", blk)), policy).0,
        },
    }
}

fn load_layer_blink(st: &SafeTensorsFile, i: usize, policy: &QuantPolicy) -> LayerWeights {
    let att = format!("blocks.{}.att", i);
    let ffn = format!("blocks.{}.ffn", i);
    let blk = format!("blocks.{}", i);

    let has_v0 = st.has_tensor(&format!("{}.v0", att));

    // BlinkDL LoRA convention: code uses `vec @ matrix` (vector on left)
    // Our mat_vec_mul uses `matrix @ vec` (matrix on left)
    // Must transpose all LoRA weights: [D, lora] → [lora, D]
    let w1 = transpose_2d(&st.get_tensor(&format!("{}.w1", att)));
    let w2 = transpose_2d(&st.get_tensor(&format!("{}.w2", att)));
    let a1 = transpose_2d(&st.get_tensor(&format!("{}.a1", att)));
    let a2 = transpose_2d(&st.get_tensor(&format!("{}.a2", att)));
    let g1 = transpose_2d(&st.get_tensor(&format!("{}.g1", att)));
    let g2 = transpose_2d(&st.get_tensor(&format!("{}.g2", att)));
    let (v1, v2) = if has_v0 {
        (
            transpose_2d(&st.get_tensor(&format!("{}.v1", att))),
            transpose_2d(&st.get_tensor(&format!("{}.v2", att))),
        )
    } else {
        (a1.clone(), a2.clone())
    };

    LayerWeights {
        ln1_w: st.get_tensor(&format!("{}.ln1.weight", blk)),
        ln1_b: st.get_tensor(&format!("{}.ln1.bias", blk)),
        ln2_w: st.get_tensor(&format!("{}.ln2.weight", blk)),
        ln2_b: st.get_tensor(&format!("{}.ln2.bias", blk)),
        time_mix: TimeMixWeights {
            x_r: squeeze(st.get_tensor(&format!("{}.x_r", att))),
            x_w: squeeze(st.get_tensor(&format!("{}.x_w", att))),
            x_k: squeeze(st.get_tensor(&format!("{}.x_k", att))),
            x_v: squeeze(st.get_tensor(&format!("{}.x_v", att))),
            x_a: squeeze(st.get_tensor(&format!("{}.x_a", att))),
            x_g: squeeze(st.get_tensor(&format!("{}.x_g", att))),
            w0: squeeze(st.get_tensor(&format!("{}.w0", att))),
            w1, w2,
            a0: squeeze(st.get_tensor(&format!("{}.a0", att))),
            a1, a2,
            v0: if has_v0 { squeeze(st.get_tensor(&format!("{}.v0", att))) }
                else { squeeze(st.get_tensor(&format!("{}.a0", att))) },
            v1, v2,
            g1, g2,
            k_k: squeeze(st.get_tensor(&format!("{}.k_k", att))),
            k_a: squeeze(st.get_tensor(&format!("{}.k_a", att))),
            r_k: flatten_tensor(st.get_tensor(&format!("{}.r_k", att))),
            key_w: quantize_matrix(&st.get_tensor(&format!("{}.key.weight", att)), policy).0,
            value_w: quantize_matrix(&st.get_tensor(&format!("{}.value.weight", att)), policy).0,
            receptance_w: quantize_matrix(&st.get_tensor(&format!("{}.receptance.weight", att)), policy).0,
            output_w: quantize_matrix(&st.get_tensor(&format!("{}.output.weight", att)), policy).0,
            ln_x_w: st.get_tensor(&format!("{}.ln_x.weight", att)),
            ln_x_b: st.get_tensor(&format!("{}.ln_x.bias", att)),
        },
        channel_mix: ChannelMixWeights {
            x_k: squeeze(st.get_tensor(&format!("{}.x_k", ffn))),
            key_w: quantize_matrix(&st.get_tensor(&format!("{}.key.weight", ffn)), policy).0,
            value_w: quantize_matrix(&st.get_tensor(&format!("{}.value.weight", ffn)), policy).0,
        },
    }
}

// ---- GGUF weight loading helpers ----

/// Load a layer from GGUF using llama.cpp RWKV converter key names.
/// Keys: blk.N.attn_xxx, blk.N.ffn_xxx
fn load_layer_gguf_llama(gf: &GgufFile, i: usize, policy: &QuantPolicy) -> LayerWeights {
    let blk = format!("blk.{}", i);

    let has_v0 = gf.has_tensor(&format!("{}.attn_v0", blk));

    // GGUF ne[] reversal gives correct shapes — no transpose needed
    let w1 = gf.get_tensor(&format!("{}.attn_w1", blk));
    let w2 = gf.get_tensor(&format!("{}.attn_w2", blk));
    let a1 = gf.get_tensor(&format!("{}.attn_a1", blk));
    let a2 = gf.get_tensor(&format!("{}.attn_a2", blk));
    let g1 = gf.get_tensor(&format!("{}.attn_g1", blk));
    let g2 = gf.get_tensor(&format!("{}.attn_g2", blk));
    let (v1, v2) = if has_v0 {
        (
            gf.get_tensor(&format!("{}.attn_v1", blk)),
            gf.get_tensor(&format!("{}.attn_v2", blk)),
        )
    } else {
        (a1.clone(), a2.clone())
    };

    LayerWeights {
        ln1_w: gf.get_tensor(&format!("{}.attn_norm.weight", blk)),
        ln1_b: gf.get_tensor(&format!("{}.attn_norm.bias", blk)),
        ln2_w: gf.get_tensor(&format!("{}.ffn_norm.weight", blk)),
        ln2_b: gf.get_tensor(&format!("{}.ffn_norm.bias", blk)),
        time_mix: TimeMixWeights {
            x_r: squeeze(gf.get_tensor(&format!("{}.attn_x_r", blk))),
            x_w: squeeze(gf.get_tensor(&format!("{}.attn_x_w", blk))),
            x_k: squeeze(gf.get_tensor(&format!("{}.attn_x_k", blk))),
            x_v: squeeze(gf.get_tensor(&format!("{}.attn_x_v", blk))),
            x_a: squeeze(gf.get_tensor(&format!("{}.attn_x_a", blk))),
            x_g: squeeze(gf.get_tensor(&format!("{}.attn_x_g", blk))),
            w0: squeeze(gf.get_tensor(&format!("{}.attn_w0", blk))),
            w1, w2,
            a0: squeeze(gf.get_tensor(&format!("{}.attn_a0", blk))),
            a1, a2,
            v0: if has_v0 { squeeze(gf.get_tensor(&format!("{}.attn_v0", blk))) }
                else { squeeze(gf.get_tensor(&format!("{}.attn_a0", blk))) },
            v1, v2,
            g1, g2,
            k_k: squeeze(gf.get_tensor(&format!("{}.attn_k_k", blk))),
            k_a: squeeze(gf.get_tensor(&format!("{}.attn_k_a", blk))),
            r_k: flatten_tensor(gf.get_tensor(&format!("{}.attn_r_k", blk))),
            key_w: load_gguf_matrix(gf, &format!("{}.attn_key.weight", blk), policy),
            value_w: load_gguf_matrix(gf, &format!("{}.attn_value.weight", blk), policy),
            receptance_w: load_gguf_matrix(gf, &format!("{}.attn_receptance.weight", blk), policy),
            output_w: load_gguf_matrix(gf, &format!("{}.attn_output.weight", blk), policy),
            ln_x_w: gf.get_tensor(&format!("{}.attn_gate_norm.weight", blk)),
            ln_x_b: gf.get_tensor(&format!("{}.attn_gate_norm.bias", blk)),
        },
        channel_mix: ChannelMixWeights {
            x_k: squeeze(gf.get_tensor(&format!("{}.ffn_x_k", blk))),
            key_w: load_gguf_matrix(gf, &format!("{}.ffn_key.weight", blk), policy),
            value_w: load_gguf_matrix(gf, &format!("{}.ffn_value.weight", blk), policy),
        },
    }
}

/// Split a fused lerp tensor into 6 individual vectors of [D].
/// Shape [6,1,1,D] (row-major) = 6 contiguous groups of D floats.
/// Order: x_r, x_w, x_k, x_v, x_a, x_g
fn split_lerp_fused(t: &Tensor, d: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    assert!(t.data.len() == d * 6, "lerp_fused size mismatch: {} vs {}*6", t.data.len(), d);
    let mk = |offset: usize| -> Tensor {
        Tensor::from_data(t.data[offset * d..(offset + 1) * d].to_vec(), vec![d])
    };
    (mk(0), mk(1), mk(2), mk(3), mk(4), mk(5))
}

/// Load a layer from GGUF using llama.cpp v2 key names (time_mix_* / channel_mix_*).
/// Used by newer llama.cpp converters (shoumenchougou GGUF, etc.)
fn load_layer_gguf_llama_v2(gf: &GgufFile, i: usize, policy: &QuantPolicy) -> LayerWeights {
    let blk = format!("blk.{}", i);

    // Fused lerp: [D,1,1,6] → 6 vectors
    let fused = gf.get_tensor(&format!("{}.time_mix_lerp_fused.weight", blk));
    let d = fused.data.len() / 6;
    let (x_r, x_w, x_k, x_v, x_a, x_g) = split_lerp_fused(&fused, d);

    let has_v0 = gf.has_tensor(&format!("{}.time_mix_v0.weight", blk));

    // GGUF ne[] reversal gives correct shapes for mat@vec — no transpose needed
    let w1 = gf.get_tensor(&format!("{}.time_mix_w1.weight", blk));
    let w2 = gf.get_tensor(&format!("{}.time_mix_w2.weight", blk));
    let a1 = gf.get_tensor(&format!("{}.time_mix_a1.weight", blk));
    let a2 = gf.get_tensor(&format!("{}.time_mix_a2.weight", blk));
    let g1 = gf.get_tensor(&format!("{}.time_mix_g1.weight", blk));
    let g2 = gf.get_tensor(&format!("{}.time_mix_g2.weight", blk));
    let (v1, v2) = if has_v0 {
        (
            gf.get_tensor(&format!("{}.time_mix_v1.weight", blk)),
            gf.get_tensor(&format!("{}.time_mix_v2.weight", blk)),
        )
    } else {
        (a1.clone(), a2.clone())
    };

    // ln2 is attn_norm_2 in v2 format
    let ln2_key = format!("{}.attn_norm_2", blk);
    let (ln2_w, ln2_b) = if gf.has_tensor(&format!("{}.weight", ln2_key)) {
        (gf.get_tensor(&format!("{}.weight", ln2_key)),
         gf.get_tensor(&format!("{}.bias", ln2_key)))
    } else {
        // Fallback to ffn_norm
        (gf.get_tensor(&format!("{}.ffn_norm.weight", blk)),
         gf.get_tensor(&format!("{}.ffn_norm.bias", blk)))
    };

    LayerWeights {
        ln1_w: gf.get_tensor(&format!("{}.attn_norm.weight", blk)),
        ln1_b: gf.get_tensor(&format!("{}.attn_norm.bias", blk)),
        ln2_w, ln2_b,
        time_mix: TimeMixWeights {
            x_r, x_w, x_k, x_v, x_a, x_g,
            w0: squeeze(gf.get_tensor(&format!("{}.time_mix_w0.weight", blk))),
            w1, w2,
            a0: squeeze(gf.get_tensor(&format!("{}.time_mix_a0.weight", blk))),
            a1, a2,
            v0: if has_v0 { squeeze(gf.get_tensor(&format!("{}.time_mix_v0.weight", blk))) }
                else { squeeze(gf.get_tensor(&format!("{}.time_mix_a0.weight", blk))) },
            v1, v2,
            g1, g2,
            k_k: squeeze(gf.get_tensor(&format!("{}.time_mix_k_k.weight", blk))),
            k_a: squeeze(gf.get_tensor(&format!("{}.time_mix_k_a.weight", blk))),
            r_k: flatten_tensor(gf.get_tensor(&format!("{}.time_mix_r_k.weight", blk))),
            key_w: load_gguf_matrix(gf, &format!("{}.time_mix_key.weight", blk), policy),
            value_w: load_gguf_matrix(gf, &format!("{}.time_mix_value.weight", blk), policy),
            receptance_w: load_gguf_matrix(gf, &format!("{}.time_mix_receptance.weight", blk), policy),
            output_w: load_gguf_matrix(gf, &format!("{}.time_mix_output.weight", blk), policy),
            ln_x_w: gf.get_tensor(&format!("{}.time_mix_ln.weight", blk)),
            ln_x_b: gf.get_tensor(&format!("{}.time_mix_ln.bias", blk)),
        },
        channel_mix: ChannelMixWeights {
            x_k: squeeze(gf.get_tensor(&format!("{}.channel_mix_lerp_k.weight", blk))),
            key_w: load_gguf_matrix(gf, &format!("{}.channel_mix_key.weight", blk), policy),
            value_w: load_gguf_matrix(gf, &format!("{}.channel_mix_value.weight", blk), policy),
        },
    }
}

/// Load a layer from GGUF using HuggingFace key names.
fn load_layer_gguf_hf(gf: &GgufFile, i: usize, policy: &QuantPolicy) -> LayerWeights {
    let att = format!("model.layers.{}.attn", i);
    let blk = format!("model.layers.{}", i);

    let has_v_lora = gf.has_tensor(&format!("{}.v_lora.lora.0.weight", att));

    let a0 = squeeze(gf.get_tensor(&format!("{}.a_lora.lora.2.bias", att)));
    let a1 = gf.get_tensor(&format!("{}.a_lora.lora.0.weight", att));
    let a2 = gf.get_tensor(&format!("{}.a_lora.lora.2.weight", att));

    let (v0, v1, v2) = if has_v_lora {
        (
            squeeze(gf.get_tensor(&format!("{}.v_lora.lora.2.bias", att))),
            gf.get_tensor(&format!("{}.v_lora.lora.0.weight", att)),
            gf.get_tensor(&format!("{}.v_lora.lora.2.weight", att)),
        )
    } else {
        (a0.clone(), a1.clone(), a2.clone())
    };

    LayerWeights {
        ln1_w: gf.get_tensor(&format!("{}.attn_norm.weight", blk)),
        ln1_b: gf.get_tensor(&format!("{}.attn_norm.bias", blk)),
        ln2_w: gf.get_tensor(&format!("{}.ffn_norm.weight", blk)),
        ln2_b: gf.get_tensor(&format!("{}.ffn_norm.bias", blk)),
        time_mix: TimeMixWeights {
            x_r: squeeze(gf.get_tensor(&format!("{}.x_r", att))),
            x_w: squeeze(gf.get_tensor(&format!("{}.x_w", att))),
            x_k: squeeze(gf.get_tensor(&format!("{}.x_k", att))),
            x_v: squeeze(gf.get_tensor(&format!("{}.x_v", att))),
            x_a: squeeze(gf.get_tensor(&format!("{}.x_a", att))),
            x_g: squeeze(gf.get_tensor(&format!("{}.x_g", att))),
            w0: squeeze(gf.get_tensor(&format!("{}.w_lora.lora.2.bias", att))),
            w1: gf.get_tensor(&format!("{}.w_lora.lora.0.weight", att)),
            w2: gf.get_tensor(&format!("{}.w_lora.lora.2.weight", att)),
            a0, a1, a2,
            v0, v1, v2,
            g1: gf.get_tensor(&format!("{}.g_lora.lora.0.weight", att)),
            g2: gf.get_tensor(&format!("{}.g_lora.lora.2.weight", att)),
            k_k: gf.get_tensor(&format!("{}.k_k", att)),
            k_a: gf.get_tensor(&format!("{}.k_a", att)),
            r_k: flatten_tensor(gf.get_tensor(&format!("{}.r_k", att))),
            key_w: load_gguf_matrix(gf, &format!("{}.k_proj.weight", att), policy),
            value_w: load_gguf_matrix(gf, &format!("{}.v_proj.weight", att), policy),
            receptance_w: load_gguf_matrix(gf, &format!("{}.r_proj.weight", att), policy),
            output_w: load_gguf_matrix(gf, &format!("{}.o_proj.weight", att), policy),
            ln_x_w: gf.get_tensor(&format!("{}.g_norm.weight", att)),
            ln_x_b: gf.get_tensor(&format!("{}.g_norm.bias", att)),
        },
        channel_mix: ChannelMixWeights {
            x_k: gf.get_tensor(&format!("{}.ffn.x_k", blk)),
            key_w: load_gguf_matrix(gf, &format!("{}.ffn.key.weight", blk), policy),
            value_w: load_gguf_matrix(gf, &format!("{}.ffn.value.weight", blk), policy),
        },
    }
}

/// Load a layer from GGUF using BlinkDL key names.
fn load_layer_gguf_blink(gf: &GgufFile, i: usize, policy: &QuantPolicy) -> LayerWeights {
    let att = format!("blocks.{}.att", i);
    let ffn = format!("blocks.{}.ffn", i);
    let blk = format!("blocks.{}", i);

    let has_v0 = gf.has_tensor(&format!("{}.v0", att));

    // GGUF ne[] reversal gives correct shapes — no transpose needed
    let w1 = gf.get_tensor(&format!("{}.w1", att));
    let w2 = gf.get_tensor(&format!("{}.w2", att));
    let a1 = gf.get_tensor(&format!("{}.a1", att));
    let a2 = gf.get_tensor(&format!("{}.a2", att));
    let g1 = gf.get_tensor(&format!("{}.g1", att));
    let g2 = gf.get_tensor(&format!("{}.g2", att));
    let (v1, v2) = if has_v0 {
        (
            gf.get_tensor(&format!("{}.v1", att)),
            gf.get_tensor(&format!("{}.v2", att)),
        )
    } else {
        (a1.clone(), a2.clone())
    };

    LayerWeights {
        ln1_w: gf.get_tensor(&format!("{}.ln1.weight", blk)),
        ln1_b: gf.get_tensor(&format!("{}.ln1.bias", blk)),
        ln2_w: gf.get_tensor(&format!("{}.ln2.weight", blk)),
        ln2_b: gf.get_tensor(&format!("{}.ln2.bias", blk)),
        time_mix: TimeMixWeights {
            x_r: squeeze(gf.get_tensor(&format!("{}.x_r", att))),
            x_w: squeeze(gf.get_tensor(&format!("{}.x_w", att))),
            x_k: squeeze(gf.get_tensor(&format!("{}.x_k", att))),
            x_v: squeeze(gf.get_tensor(&format!("{}.x_v", att))),
            x_a: squeeze(gf.get_tensor(&format!("{}.x_a", att))),
            x_g: squeeze(gf.get_tensor(&format!("{}.x_g", att))),
            w0: squeeze(gf.get_tensor(&format!("{}.w0", att))),
            w1, w2,
            a0: squeeze(gf.get_tensor(&format!("{}.a0", att))),
            a1, a2,
            v0: if has_v0 { squeeze(gf.get_tensor(&format!("{}.v0", att))) }
                else { squeeze(gf.get_tensor(&format!("{}.a0", att))) },
            v1, v2,
            g1, g2,
            k_k: squeeze(gf.get_tensor(&format!("{}.k_k", att))),
            k_a: squeeze(gf.get_tensor(&format!("{}.k_a", att))),
            r_k: flatten_tensor(gf.get_tensor(&format!("{}.r_k", att))),
            key_w: load_gguf_matrix(gf, &format!("{}.key.weight", att), policy),
            value_w: load_gguf_matrix(gf, &format!("{}.value.weight", att), policy),
            receptance_w: load_gguf_matrix(gf, &format!("{}.receptance.weight", att), policy),
            output_w: load_gguf_matrix(gf, &format!("{}.output.weight", att), policy),
            ln_x_w: gf.get_tensor(&format!("{}.ln_x.weight", att)),
            ln_x_b: gf.get_tensor(&format!("{}.ln_x.bias", att)),
        },
        channel_mix: ChannelMixWeights {
            x_k: squeeze(gf.get_tensor(&format!("{}.x_k", ffn))),
            key_w: load_gguf_matrix(gf, &format!("{}.key.weight", ffn), policy),
            value_w: load_gguf_matrix(gf, &format!("{}.value.weight", ffn), policy),
        },
    }
}

/// Load a big matrix from GGUF: tries native quantized (no dequant) first,
/// falls back to dequantize→F32 for types without native matmul support.
/// Saves ~10x RAM for Q4_K/Q6_K/Q8_0 tensors.
fn load_gguf_matrix(gf: &GgufFile, name: &str, policy: &QuantPolicy) -> QuantMatrix {
    if let Ok(qt) = gf.load_tensor_raw(name) {
        return QuantMatrix::Gguf(qt);
    }
    // Fallback: dequantize to F32 and apply quantization policy
    quantize_matrix(&gf.get_tensor(name), policy).0
}

fn report_quantization(
    layers: &[LayerWeights],
    head_w: &QuantMatrix,
    v: usize, d: usize,
) {
    let mut q_bytes: usize = head_w.mem_bytes();
    let mut f32_equiv: usize = v * d * 4;
    let mut level_counts = std::collections::HashMap::new();
    *level_counts.entry(head_w.format_name()).or_insert(0usize) += 1;
    for lw in layers {
        let tm = &lw.time_mix;
        for q in [&tm.key_w, &tm.value_w, &tm.receptance_w, &tm.output_w] {
            q_bytes += q.mem_bytes();
            let (r, c) = q.shape();
            f32_equiv += r * c * 4;
            *level_counts.entry(q.format_name()).or_insert(0) += 1;
        }
        let cm = &lw.channel_mix;
        for q in [&cm.key_w, &cm.value_w] {
            q_bytes += q.mem_bytes();
            let (r, c) = q.shape();
            f32_equiv += r * c * 4;
            *level_counts.entry(q.format_name()).or_insert(0) += 1;
        }
    }
    let q_mb = q_bytes as f64 / 1_048_576.0;
    let f32_mb = f32_equiv as f64 / 1_048_576.0;
    let saved = f32_mb - q_mb;
    let mut summary_parts: Vec<String> = level_counts.iter()
        .map(|(name, count)| format!("{}×{}", count, name))
        .collect();
    summary_parts.sort();
    eprintln!("[rwkv7] quantization: {} — {:.1} MB (saved {:.1} MB from {:.1} MB f32)",
              summary_parts.join(", "), q_mb, saved, f32_mb);
}

fn flatten_tensor(t: Tensor) -> Tensor {
    let n = t.numel();
    Tensor::from_data(t.data, vec![n])
}

/// Squeeze: remove all dimensions of size 1. E.g. [1,1,768] → [768]
fn squeeze(t: Tensor) -> Tensor {
    let new_shape: Vec<usize> = t.shape.iter().copied().filter(|&s| s > 1).collect();
    if new_shape.is_empty() {
        Tensor::from_data(t.data, vec![1])
    } else {
        Tensor::from_data(t.data, new_shape)
    }
}
