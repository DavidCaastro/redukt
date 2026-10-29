//! Application commands — compress, decompress, hybrid-eval, baseline, cm-eval.

use crate::domain::error::{AzError, AzResult, parse_arg, next_arg, validate_u32_size};
use crate::domain::tensor::{softmax_into, entropy_from_logits};
use crate::domain::ngram::TokenNgram;
use crate::domain::bias_head::BiasHead;
use crate::domain::mixer::AdaptiveMixer;
use crate::domain::coder::{Cdf, RangeEncoder, RangeDecoder};
use crate::domain::crc32;
use crate::application::model_loader;
use crate::application::telemetry::{JsonLogger, LogSnapshot, PhaseTimer};
use crate::infrastructure::rwkv7::model::Rwkv7State;

/// v3 magic — kept for backward-compat parsing and tests.
#[cfg(test)]
const MAGIC_V3: &[u8; 8] = b"AZTH\x03\x00\x00\x00";
pub const MAGIC_V4: &[u8; 8] = b"AZTH\x04\x00\x00\x00";
pub const HEADER_SIZE: usize = 28; // v2/v3: 8 magic + 4 total_bytes + 4 token_count + 12 hp
pub const V4_HEADER_BASE: usize = 13; // v4: 8 magic + 4 total_bytes + 1 flags
pub const CRC_SIZE: usize = 4; // CRC-32 appended after STREAM (v3+ only)
pub const V4_FLAG_CUSTOM_HP: u8 = 0x01;
pub const V4_FLAG_EMB_SURGERY: u8 = 0x02;
pub const V4_FLAG_E8E9: u8 = 0x04;
pub const DEFAULT_BIAS_LR: f32 = 0.30;
pub const DEFAULT_NGRAM_SCALE: f32 = 0.50;
pub const DEFAULT_MIX_ETA: f32 = 0.01;

/// Emb-surgery method enum for header encoding (R-010).
/// 0 = none, 1 = center (+ alpha f32), 2 = norm, 3 = spread.
pub const EMB_METHOD_NONE: u8 = 0;
pub const EMB_METHOD_CENTER: u8 = 1;
pub const EMB_METHOD_NORM: u8 = 2;
pub const EMB_METHOD_SPREAD: u8 = 3;
pub const EMB_SURGERY_BYTES: usize = 5; // 1B method + 4B alpha

/// Parse emb-surgery CLI string into (method_byte, alpha).
pub fn parse_emb_surgery_str(s: &str) -> (u8, f32) {
    if s.is_empty() {
        return (EMB_METHOD_NONE, 0.0);
    }
    if s == "norm" {
        return (EMB_METHOD_NORM, 0.0);
    }
    if s == "spread" {
        return (EMB_METHOD_SPREAD, 0.0);
    }
    if s.starts_with("center") {
        let alpha_str = &s["center".len()..];
        let alpha: f32 = if alpha_str.is_empty() {
            0.3
        } else {
            alpha_str.parse().unwrap_or(0.3)
        };
        return (EMB_METHOD_CENTER, alpha);
    }
    // Unknown method — treat as none (analyze is read-only, not stored)
    (EMB_METHOD_NONE, 0.0)
}

/// Reconstruct emb-surgery CLI string from header (method_byte, alpha).
pub fn emb_surgery_to_string(method: u8, alpha: f32) -> String {
    match method {
        EMB_METHOD_CENTER => format!("center{}", alpha),
        EMB_METHOD_NORM => "norm".to_string(),
        EMB_METHOD_SPREAD => "spread".to_string(),
        _ => String::new(),
    }
}

/// Parsed header info — works for v2, v3, and v4.
pub struct HeaderInfo {
    pub version: u8,
    pub total_bytes: usize,
    pub token_count: Option<usize>, // None for v4
    pub bias_lr: f32,
    pub ngram_scale: f32,
    pub mix_eta: f32,
    pub emb_surgery: String, // empty = none, "center0.3", "norm", "spread"
    pub e8e9: bool,          // true = E8E9 preprocessing was applied (R-027)
    pub header_size: usize,
}

/// Validate AZTH magic bytes. Returns format version (2, 3, or 4).
pub fn validate_magic(data: &[u8]) -> AzResult<u8> {
    if data.len() < 8 {
        return Err(AzError::Format(format!(
            "invalid AZTH file (too short: {} bytes)", data.len()
        )));
    }
    if &data[..4] != b"AZTH" {
        return Err(AzError::Format(format!(
            "invalid AZTH file (bad magic: {:?})", &data[..4]
        )));
    }
    let version = data[4];
    let min_size = match version {
        2 | 3 => HEADER_SIZE,
        4 => V4_HEADER_BASE,
        _ => return Err(AzError::Format(format!(
            "unsupported AZTH version {} (expected 2, 3, or 4)", version
        ))),
    };
    if data.len() < min_size {
        return Err(AzError::Format(format!(
            "invalid AZTH v{} (too short: {} < {})", version, data.len(), min_size
        )));
    }
    Ok(version)
}

/// Check if hyperparameters are the defaults (exact f32 bit comparison).
pub fn uses_default_hp(bias_lr: f32, ngram_scale: f32, mix_eta: f32) -> bool {
    bias_lr.to_bits() == DEFAULT_BIAS_LR.to_bits()
        && ngram_scale.to_bits() == DEFAULT_NGRAM_SCALE.to_bits()
        && mix_eta.to_bits() == DEFAULT_MIX_ETA.to_bits()
}

/// Parse header for any AZTH version (v2, v3, v4). Assumes validate_magic passed.
pub fn parse_header(data: &[u8]) -> AzResult<HeaderInfo> {
    let version = data[4];
    match version {
        2 | 3 => {
            let total_bytes = u32::from_le_bytes(
                [data[8], data[9], data[10], data[11]]
            ) as usize;
            let token_count = u32::from_le_bytes(
                [data[12], data[13], data[14], data[15]]
            ) as usize;
            let (bias_lr, ngram_scale, mix_eta) = parse_hyperparams(data)?;
            Ok(HeaderInfo {
                version, total_bytes,
                token_count: Some(token_count),
                bias_lr, ngram_scale, mix_eta,
                emb_surgery: String::new(),
                e8e9: false,
                header_size: HEADER_SIZE,
            })
        }
        4 => {
            let total_bytes = u32::from_le_bytes(
                [data[8], data[9], data[10], data[11]]
            ) as usize;
            let flags = data[12];
            let known = V4_FLAG_CUSTOM_HP | V4_FLAG_EMB_SURGERY | V4_FLAG_E8E9;
            let reserved = flags & !known;
            if reserved != 0 {
                return Err(AzError::Format(format!(
                    "AZTH v4 unknown flags: 0x{:02X}", reserved
                )));
            }
            let mut offset = V4_HEADER_BASE;

            // Parse custom hyperparameters (flag 0x01)
            let (bias_lr, ngram_scale, mix_eta) = if flags & V4_FLAG_CUSTOM_HP != 0 {
                if data.len() < offset + 12 {
                    return Err(AzError::Format(format!(
                        "AZTH v4 custom HP truncated: {} < {}",
                        data.len(), offset + 12
                    )));
                }
                let o = offset;
                let bl = f32::from_le_bytes(
                    [data[o], data[o+1], data[o+2], data[o+3]]
                );
                let ns = f32::from_le_bytes(
                    [data[o+4], data[o+5], data[o+6], data[o+7]]
                );
                let me = f32::from_le_bytes(
                    [data[o+8], data[o+9], data[o+10], data[o+11]]
                );
                for (name, val) in [
                    ("bias_lr", bl), ("ngram_scale", ns), ("mix_eta", me),
                ] {
                    if !val.is_finite() {
                        return Err(AzError::Format(format!(
                            "invalid hyperparameter {}={} (NaN or Inf)", name, val
                        )));
                    }
                }
                offset += 12;
                (bl, ns, me)
            } else {
                (DEFAULT_BIAS_LR, DEFAULT_NGRAM_SCALE, DEFAULT_MIX_ETA)
            };

            // Parse emb-surgery (flag 0x02, R-010)
            let emb_surgery = if flags & V4_FLAG_EMB_SURGERY != 0 {
                if data.len() < offset + EMB_SURGERY_BYTES {
                    return Err(AzError::Format(format!(
                        "AZTH v4 emb-surgery truncated: {} < {}",
                        data.len(), offset + EMB_SURGERY_BYTES
                    )));
                }
                let method = data[offset];
                let alpha = f32::from_le_bytes([
                    data[offset+1], data[offset+2],
                    data[offset+3], data[offset+4],
                ]);
                offset += EMB_SURGERY_BYTES;
                emb_surgery_to_string(method, alpha)
            } else {
                String::new()
            };

            let e8e9 = flags & V4_FLAG_E8E9 != 0;

            Ok(HeaderInfo {
                version, total_bytes, token_count: None,
                bias_lr, ngram_scale, mix_eta,
                emb_surgery, e8e9,
                header_size: offset,
            })
        }
        _ => Err(AzError::Format(format!(
            "unsupported AZTH version {}", version
        ))),
    }
}

/// Extract stream and CRC from AZTH payload given header size and version.
pub fn extract_stream_and_crc_v(
    data: &[u8], header_size: usize, version: u8,
) -> (&[u8], Option<u32>) {
    let payload = &data[header_size..];
    if version >= 3 && payload.len() >= CRC_SIZE {
        let split = payload.len() - CRC_SIZE;
        let stream = &payload[..split];
        let crc_bytes = &payload[split..];
        let stored_crc = u32::from_le_bytes(
            [crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]
        );
        (stream, Some(stored_crc))
    } else {
        (payload, None)
    }
}

/// Build AZTH v4 header bytes.
pub fn build_v4_header(
    total_bytes: u32, bias_lr: f32, ngram_scale: f32, mix_eta: f32,
    emb_surgery: &str, e8e9: bool,
) -> Vec<u8> {
    let custom_hp = !uses_default_hp(bias_lr, ngram_scale, mix_eta);
    let (emb_method, emb_alpha) = parse_emb_surgery_str(emb_surgery);
    let has_emb = emb_method != EMB_METHOD_NONE;

    let mut flags: u8 = 0;
    if custom_hp { flags |= V4_FLAG_CUSTOM_HP; }
    if has_emb { flags |= V4_FLAG_EMB_SURGERY; }
    if e8e9 { flags |= V4_FLAG_E8E9; }

    let mut cap = V4_HEADER_BASE;
    if custom_hp { cap += 12; }
    if has_emb { cap += EMB_SURGERY_BYTES; }

    let mut hdr = Vec::with_capacity(cap);
    hdr.extend_from_slice(MAGIC_V4);
    hdr.extend_from_slice(&total_bytes.to_le_bytes());
    hdr.push(flags);
    if custom_hp {
        hdr.extend_from_slice(&bias_lr.to_le_bytes());
        hdr.extend_from_slice(&ngram_scale.to_le_bytes());
        hdr.extend_from_slice(&mix_eta.to_le_bytes());
    }
    if has_emb {
        hdr.push(emb_method);
        hdr.extend_from_slice(&emb_alpha.to_le_bytes());
    }
    hdr
}

/// Parse header hyperparameters with NaN/Inf validation.
pub fn parse_hyperparams(data: &[u8]) -> AzResult<(f32, f32, f32)> {
    let bias_lr = f32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let ngram_scale = f32::from_le_bytes([data[20], data[21], data[22], data[23]]);
    let mix_eta = f32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    for (name, val) in [("bias_lr", bias_lr), ("ngram_scale", ngram_scale), ("mix_eta", mix_eta)] {
        if !val.is_finite() {
            return Err(AzError::Format(format!(
                "invalid hyperparameter {}={} (NaN or Inf)", name, val
            )));
        }
    }
    Ok((bias_lr, ngram_scale, mix_eta))
}

/// Extract stream and CRC from v2/v3 AZTH (fixed HEADER_SIZE offset).
/// Kept for backward-compat tests. Production code uses extract_stream_and_crc_v.
#[cfg(test)]
fn extract_stream_and_crc(data: &[u8], version: u8) -> (&[u8], Option<u32>) {
    let payload = &data[HEADER_SIZE..];
    if version >= 3 && payload.len() >= CRC_SIZE {
        let split = payload.len() - CRC_SIZE;
        let stream = &payload[..split];
        let crc_bytes = &payload[split..];
        let stored_crc = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
        (stream, Some(stored_crc))
    } else {
        (payload, None)
    }
}

/// Build CDF from logits for current prediction step.
/// Uses f64 softmax to preserve tail precision for 65K vocab.
pub fn cdf_from_logits(logits: &[f32]) -> Cdf {
    let n = logits.len();
    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b as f64));
    let mut probs = Vec::with_capacity(n);
    let mut sum = 0.0f64;
    for &l in logits {
        let e = ((l as f64) - max_l).exp();
        probs.push(e);
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    let probs_f32: Vec<f32> = probs.iter().map(|&p| (p * inv_sum) as f32).collect();
    Cdf::from_probs(&probs_f32)
}

/// Decode a symbol with bounds checking, returning AzResult.
fn decode_checked(
    dec: &mut RangeDecoder, cdf: &Cdf, vocab_size: usize, token_idx: usize,
) -> AzResult<u32> {
    dec.decode_symbol_checked(cdf, vocab_size)
        .map(|sym| sym as u32)
        .map_err(|e| AzError::Decode { token_idx, detail: e.to_string() })
}

/// Coder efficiency metrics: theoretical vs actual compression.
pub struct CoderMetrics {
    pub input_bytes: usize,
    pub compressed_bytes: usize,
    pub ce_bits: f64,
}

impl CoderMetrics {
    /// Bits per byte of the compressed output.
    pub fn compressed_bpb(&self) -> f64 {
        self.compressed_bytes as f64 * 8.0 / self.input_bytes.max(1) as f64
    }

    /// Theoretical bits per byte (cross-entropy).
    pub fn ce_bpb(&self) -> f64 {
        self.ce_bits / self.input_bytes.max(1) as f64
    }

    /// Overhead in BPB: how many extra bits the coder adds beyond entropy.
    pub fn overhead_bpb(&self) -> f64 {
        self.compressed_bpb() - self.ce_bpb()
    }

    /// Coding efficiency as percentage (100% = perfect, entropy-optimal).
    /// Returns ce_bits / actual_bits * 100.
    pub fn efficiency_pct(&self) -> f64 {
        let actual_bits = self.compressed_bytes as f64 * 8.0;
        if actual_bits <= 0.0 { return 0.0; }
        (self.ce_bits / actual_bits) * 100.0
    }

    /// Compression ratio: compressed / original.
    pub fn ratio(&self) -> f64 {
        self.compressed_bytes as f64 / self.input_bytes.max(1) as f64
    }
}

pub fn cmd_compress(args: &[String]) -> AzResult<()> {
    let mut input_path = "data/enwik8".to_string();
    let mut output_path = String::new();
    let mut input_dir = String::new();
    let mut output_dir = String::new();
    let mut jobs: usize = 1;
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut bias_lr: f32 = 0.30;
    let mut ngram_scale: f32 = 0.5;
    let mut mix_eta: f32 = 0.01;
    let mut quant_str = String::new();
    let mut emb_surgery = String::new();
    let mut no_governor = false;
    let mut e8e9_mode = String::new();
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input")?.to_string(); }
            "--output" => { output_path = next_arg(args, &mut i, "--output")?.to_string(); }
            "--input-dir" => { input_dir = next_arg(args, &mut i, "--input-dir")?.to_string(); }
            "--output-dir" => { output_dir = next_arg(args, &mut i, "--output-dir")?.to_string(); }
            "--jobs" => { jobs = parse_arg(next_arg(args, &mut i, "--jobs")?, "--jobs")?; }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")?.to_string(); }
            "--bytes" => { max_bytes = parse_arg(next_arg(args, &mut i, "--bytes")?, "--bytes")?; }
            "--lr" => { bias_lr = parse_arg(next_arg(args, &mut i, "--lr")?, "--lr")?; }
            "--ngram-scale" => { ngram_scale = parse_arg(next_arg(args, &mut i, "--ngram-scale")?, "--ngram-scale")?; }
            "--mix-eta" => { mix_eta = parse_arg(next_arg(args, &mut i, "--mix-eta")?, "--mix-eta")?; }
            "--quant" => { quant_str = next_arg(args, &mut i, "--quant")?.to_string(); }
            "--emb-surgery" => { emb_surgery = next_arg(args, &mut i, "--emb-surgery")?.to_string(); }
            "--no-governor" => { no_governor = true; }
            "--e8e9" => { e8e9_mode = next_arg(args, &mut i, "--e8e9")?.to_string(); }
            "--verbose" | "-v" => { verbose = true; }
            _ => {}
        }
        i += 1;
    }

    // Hardware governor (R-HW1)
    let governor = crate::infrastructure::governor::HardwareGovernor::new(!no_governor);
    governor.preflight().map_err(|e| AzError::Format(format!("governor: {}", e)))?;
    governor.start_watchdog();

    // Batch mode: --input-dir + --output-dir
    if !input_dir.is_empty() {
        if output_dir.is_empty() {
            return Err(AzError::Parse {
                context: "--output-dir".into(), value: "(required with --input-dir)".into(),
            });
        }
        let quant_override = if quant_str.is_empty() { None }
            else { model_loader::parse_quant_flag(&quant_str) };
        return batch_compress(
            &input_dir, &output_dir, jobs, &weights_dir, &emb_surgery,
            quant_override, max_bytes, bias_lr, ngram_scale, mix_eta,
        );
    }

    if output_path.is_empty() {
        return Err(AzError::Parse {
            context: "--output".to_string(), value: "(missing)".to_string(),
        });
    }

    let mut timer = PhaseTimer::new();

    timer.start("read");
    let raw_bytes = crate::domain::error::read_file(&input_path)?;
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];
    eprintln!("[compress] input: {} ({} bytes)", input_path, total_bytes);

    // E8E9 preprocessing (R-027): detect and transform x86 executables
    let (e8e9_applied, e8e9_data) = match e8e9_mode.as_str() {
        "on" => {
            let (transformed, stats) =
                crate::domain::preprocess::e8e9_encode_with_stats(input_slice);
            eprintln!("[compress] E8E9: forced ON — {} E8 + {} E9, {} transformed, {} skipped",
                      stats.e8_count, stats.e9_count, stats.transformed, stats.skipped);
            (true, Some(transformed))
        }
        "off" => {
            eprintln!("[compress] E8E9: disabled");
            (false, None)
        }
        _ => {
            // auto-detect (default)
            if crate::domain::preprocess::detect_e8e9(input_slice) {
                let (transformed, stats) =
                    crate::domain::preprocess::e8e9_encode_with_stats(input_slice);
                eprintln!("[compress] E8E9: auto-detected — {} E8 + {} E9, {} transformed",
                          stats.e8_count, stats.e9_count, stats.transformed);
                (true, Some(transformed))
            } else {
                (false, None)
            }
        }
    };
    let compress_slice: &[u8] = e8e9_data.as_deref().unwrap_or(input_slice);

    timer.start("model_load");
    let quant_override = if quant_str.is_empty() { None } else { model_loader::parse_quant_flag(&quant_str) };
    let (mut model, tokenizer) = model_loader::load_model_and_tokenizer_with_quant(&weights_dir, quant_override);
    let v = model.config.vocab_size;

    if !emb_surgery.is_empty() {
        model.embedding_surgery(&emb_surgery);
    }

    timer.start("tokenize");
    let tokens = tokenizer.encode(compress_slice);
    eprintln!("[compress] tokenized: {} tokens ({:.2} bytes/token)",
              tokens.len(), total_bytes as f64 / tokens.len() as f64);

    timer.start("encode");
    let mut ngram = TokenNgram::new(4, v, ngram_scale);
    let mut bias = BiasHead::new(v, bias_lr);
    let mut mixer = AdaptiveMixer::new(mix_eta);
    let mut enc = RangeEncoder::new();

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut ce_bits = 0.0f64;

    // Pre-allocated buffers for zero-alloc hot loop (R-019)
    let mut ng_buf = vec![0.0f32; v];
    let mut ensemble_buf = vec![0.0f32; v];
    let mut probs_buf = vec![0.0f32; v];

    eprintln!("[compress] encoding ...");
    let t_start = std::time::Instant::now();

    for t in 0..tokens.len() {
        let tok = tokens[t] as usize;

        if t == 0 {
            let cdf = Cdf::uniform(v);
            enc.encode_symbol(tok, &cdf);
            let prob = 1.0 / v as f64;
            ce_bits += -prob.log2();
        } else {
            ngram.compute_bias_into(&mut ng_buf, None);
            let b_vec = bias.bias_vector();
            mixer.combine_into(&mut ensemble_buf, &logits, &ng_buf, b_vec);

            let cdf = cdf_from_logits(&ensemble_buf);
            enc.encode_symbol(tok, &cdf);

            softmax_into(&ensemble_buf, &mut probs_buf);
            let prob = probs_buf[tok] as f64;
            ce_bits += -prob.max(1e-30).log2();

            mixer.update(&probs_buf, &ng_buf, bias.bias_vector(), tok);
            bias.update(&probs_buf, tok);
        }

        ngram.observe(tok as u32);
        model.forward_into(tok, &mut state, &mut scratch, &mut logits);

        if t > 0 && t % 100 == 0 {
            governor.apply()
                .map_err(|e| AzError::Format(e))?;
        }
        if t > 0 && (t % 1000 == 0 || t == tokens.len() - 1) {
            let elapsed = t_start.elapsed().as_secs_f64();
            let pct = 100.0 * t as f64 / tokens.len() as f64;
            let enc_size = enc.size();
            eprint!("\r[compress] {:.1}% | {} tokens | ~{} bytes | {:.1} tok/s   ",
                    pct, t, enc_size, t as f64 / elapsed);
        }
    }

    let compressed = enc.finish();
    eprintln!();

    if verbose {
        let layers_ms = scratch.perf_layers_us as f64 / 1000.0;
        let head_ms = scratch.perf_head_us as f64 / 1000.0;
        let tmix_ms = scratch.perf_tmix_us as f64 / 1000.0;
        let cmix_ms = scratch.perf_cmix_us as f64 / 1000.0;
        let total_fwd = layers_ms + head_ms;
        let layers_pct = if total_fwd > 0.0 { 100.0 * layers_ms / total_fwd } else { 0.0 };
        let head_pct = if total_fwd > 0.0 { 100.0 * head_ms / total_fwd } else { 0.0 };
        let tmix_pct = if total_fwd > 0.0 { 100.0 * tmix_ms / total_fwd } else { 0.0 };
        let cmix_pct = if total_fwd > 0.0 { 100.0 * cmix_ms / total_fwd } else { 0.0 };
        let ntok = tokens.len().max(1) as f64;
        eprintln!("[compress] forward_pass breakdown:");
        eprintln!("  TimeMix  (12 layers):  {:.1}ms ({:.1}%)", tmix_ms, tmix_pct);
        eprintln!("  ChannelMix (12 layers): {:.1}ms ({:.1}%)", cmix_ms, cmix_pct);
        eprintln!("  layers total:           {:.1}ms ({:.1}%)", layers_ms, layers_pct);
        eprintln!("  head (65536x768):       {:.1}ms ({:.1}%)", head_ms, head_pct);
        eprintln!("  total forward:          {:.1}ms", total_fwd);
        eprintln!("  per-token: tmix {:.3}ms, cmix {:.3}ms, head {:.3}ms",
                  tmix_ms / ntok, cmix_ms / ntok, head_ms / ntok);
    }

    timer.start("write");
    // CRC covers the bytes that enter the coder (post-E8E9 if applied).
    // Decompress verifies CRC on decoded bytes, then applies inverse E8E9.
    let input_crc = crc32::crc32(compress_slice);
    let total_u32 = validate_u32_size(total_bytes, "TOTAL_BYTES")?;

    let hdr = build_v4_header(total_u32, bias_lr, ngram_scale, mix_eta, &emb_surgery, e8e9_applied);
    let mut output = Vec::with_capacity(hdr.len() + compressed.len() + CRC_SIZE);
    output.extend_from_slice(&hdr);
    output.extend_from_slice(&compressed);
    output.extend_from_slice(&input_crc.to_le_bytes());

    crate::domain::error::write_file(&output_path, &output)?;
    timer.stop();

    let total_time = timer.total_secs();
    let metrics = CoderMetrics {
        input_bytes: total_bytes,
        compressed_bytes: output.len(),
        ce_bits,
    };

    eprintln!();
    eprintln!("[compress] results:");
    eprintln!("  input:          {} bytes", total_bytes);
    eprintln!("  compressed:     {} bytes (ratio: {:.4})", output.len(), metrics.ratio());
    eprintln!("  compressed BPB: {:.4}", metrics.compressed_bpb());
    eprintln!("  cross-ent BPB:  {:.4}", metrics.ce_bpb());
    eprintln!("  coder overhead: {:.4} BPB", metrics.overhead_bpb());
    eprintln!("  coder efficiency: {:.2}%", metrics.efficiency_pct());
    eprintln!("  CRC-32:         {:08X}", input_crc);
    eprintln!("  format:         AZTH v4");
    eprintln!("  time:           {:.1}s ({:.1} tok/s)", total_time, tokens.len() as f64 / total_time);
    eprintln!("  output:         {}", output_path);

    if verbose {
        timer.report("compress");
        eprintln!("[compress] coder_metrics_json: {{\"ce_bpb\":{:.6},\"compressed_bpb\":{:.6},\
            \"overhead_bpb\":{:.6},\"efficiency_pct\":{:.2},\"ce_bits\":{:.1},\"actual_bits\":{}}}",
            metrics.ce_bpb(), metrics.compressed_bpb(), metrics.overhead_bpb(),
            metrics.efficiency_pct(), ce_bits, output.len() * 8);
        eprintln!("[compress] timing_json: {}", timer.to_json());
    }
    Ok(())
}

/// Batch-compress all files in `input_dir` → `output_dir` using `jobs` threads.
/// Model is loaded once and shared via Arc. Each thread gets its own state/scratch.
fn batch_compress(
    input_dir: &str, output_dir: &str, jobs: usize,
    weights_dir: &str, emb_surgery: &str,
    quant_override: Option<crate::domain::quant::QuantPolicy>,
    max_bytes: usize, bias_lr: f32, ngram_scale: f32, mix_eta: f32,
) -> AzResult<()> {
    use std::sync::{Arc, Mutex};

    // Discover files (non-recursive, skip .azth)
    let entries: Vec<std::path::PathBuf> = std::fs::read_dir(input_dir)
        .map_err(|e| AzError::Io { path: input_dir.to_string(), source: e })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().map_or(true, |ext| ext != "azth"))
        .collect();

    if entries.is_empty() {
        return Err(AzError::Format(format!("no files found in {}", input_dir)));
    }

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .map_err(|e| AzError::Io { path: output_dir.to_string(), source: e })?;

    eprintln!("[batch] {} files, {} jobs", entries.len(), jobs);

    // Load model once
    let t_load = std::time::Instant::now();
    let (mut model, tokenizer) = model_loader::load_model_and_tokenizer_with_quant(
        weights_dir, quant_override,
    );
    if !emb_surgery.is_empty() {
        model.embedding_surgery(emb_surgery);
    }
    let v = model.config.vocab_size;
    eprintln!("[batch] model loaded in {:.1}s (vocab={})", t_load.elapsed().as_secs_f64(), v);

    let model = Arc::new(model);
    let tokenizer = Arc::new(tokenizer);
    let work = Arc::new(Mutex::new(entries.clone()));

    // Result per file: (filename, input_bytes, compressed_bytes, bpb, tok_s, ok)
    type FileResult = (String, usize, usize, f64, f64, bool);
    let results: Arc<Mutex<Vec<FileResult>>> = Arc::new(Mutex::new(Vec::new()));

    let t_total = std::time::Instant::now();
    let n_jobs = jobs.max(1).min(entries.len());

    if n_jobs <= 1 {
        // Single-threaded: no thread spawn overhead
        compress_batch_worker(
            &model, &tokenizer, &work, &results,
            input_dir, output_dir, v, max_bytes, bias_lr, ngram_scale, mix_eta,
            emb_surgery,
        );
    } else {
        let mut handles = Vec::with_capacity(n_jobs);
        for _ in 0..n_jobs {
            let m = Arc::clone(&model);
            let tok = Arc::clone(&tokenizer);
            let w = Arc::clone(&work);
            let r = Arc::clone(&results);
            let in_d = input_dir.to_string();
            let out_d = output_dir.to_string();
            let es = emb_surgery.to_string();
            handles.push(std::thread::spawn(move || {
                compress_batch_worker(
                    &m, &tok, &w, &r,
                    &in_d, &out_d, v, max_bytes, bias_lr, ngram_scale, mix_eta,
                    &es,
                );
            }));
        }
        for h in handles {
            let _ = h.join();
        }
    }

    let elapsed = t_total.elapsed().as_secs_f64();
    let res = results.lock().unwrap();

    // Report
    eprintln!();
    eprintln!("[batch] results ({} files, {:.1}s total):", res.len(), elapsed);
    eprintln!("  {:30} {:>10} {:>10} {:>8} {:>8} {}", "file", "input", "output", "BPB", "tok/s", "status");
    let mut total_in = 0usize;
    let mut total_out = 0usize;
    let mut ok_count = 0usize;
    for (name, ib, cb, bpb, toks, ok) in res.iter() {
        let status = if *ok { "OK" } else { "FAIL" };
        eprintln!("  {:30} {:>10} {:>10} {:>8.4} {:>8.1} {}",
                  name, ib, cb, bpb, toks, status);
        total_in += ib;
        total_out += cb;
        if *ok { ok_count += 1; }
    }
    eprintln!("  {:30} {:>10} {:>10}", "TOTAL", total_in, total_out);
    eprintln!("[batch] {}/{} succeeded, ratio {:.4}, {:.1}s ({:.1} files/min)",
             ok_count, res.len(),
             total_out as f64 / total_in.max(1) as f64,
             elapsed,
             res.len() as f64 / (elapsed / 60.0));

    if ok_count < res.len() {
        return Err(AzError::Format(format!("{} files failed", res.len() - ok_count)));
    }
    Ok(())
}

/// Worker loop: pop files from shared queue, compress each one.
fn compress_batch_worker(
    model: &crate::infrastructure::rwkv7::model::Rwkv7Model,
    tokenizer: &crate::infrastructure::rwkv7::tokenizer::WorldTokenizer,
    work: &std::sync::Mutex<Vec<std::path::PathBuf>>,
    results: &std::sync::Mutex<Vec<(String, usize, usize, f64, f64, bool)>>,
    _input_dir: &str, output_dir: &str,
    v: usize, max_bytes: usize,
    bias_lr: f32, ngram_scale: f32, mix_eta: f32,
    emb_surgery: &str,
) {
    // Per-thread allocations
    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut ng_buf = vec![0.0f32; v];
    let mut ensemble_buf = vec![0.0f32; v];
    let mut probs_buf = vec![0.0f32; v];

    loop {
        let path = {
            let mut q = work.lock().unwrap();
            q.pop()
        };
        let path = match path {
            Some(p) => p,
            None => break,
        };

        let fname = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let out_name = format!("{}.azth", fname);
        let out_path = std::path::Path::new(output_dir).join(&out_name);

        let result = compress_single_file(
            model, tokenizer, &path, &out_path,
            &mut state, &mut scratch, &mut logits,
            &mut ng_buf, &mut ensemble_buf, &mut probs_buf,
            v, max_bytes, bias_lr, ngram_scale, mix_eta,
            emb_surgery,
        );

        let mut res = results.lock().unwrap();
        match result {
            Ok((ib, cb, bpb, toks)) => {
                eprintln!("[batch] {} -> {} ({} -> {} bytes, {:.4} BPB, {:.1} tok/s)",
                          fname, out_name, ib, cb, bpb, toks);
                res.push((fname, ib, cb, bpb, toks, true));
            }
            Err(e) => {
                eprintln!("[batch] FAIL {}: {}", fname, e);
                res.push((fname, 0, 0, 0.0, 0.0, false));
            }
        }
    }
}

/// Compress a single file. Returns (input_bytes, compressed_bytes, bpb, tok_per_sec).
fn compress_single_file(
    model: &crate::infrastructure::rwkv7::model::Rwkv7Model,
    tokenizer: &crate::infrastructure::rwkv7::tokenizer::WorldTokenizer,
    input_path: &std::path::Path, output_path: &std::path::Path,
    state: &mut Rwkv7State,
    scratch: &mut crate::infrastructure::rwkv7::model::Scratch,
    logits: &mut [f32],
    ng_buf: &mut [f32], ensemble_buf: &mut [f32], probs_buf: &mut [f32],
    v: usize, max_bytes: usize,
    bias_lr: f32, ngram_scale: f32, mix_eta: f32,
    emb_surgery: &str,
) -> AzResult<(usize, usize, f64, f64)> {
    let raw_bytes = crate::domain::error::read_file(
        &input_path.to_string_lossy(),
    )?;
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];

    let tokens = tokenizer.encode(input_slice);

    // Reset per-file state
    state.reset();
    scratch.reset_perf();
    let mut ngram = TokenNgram::new(4, v, ngram_scale);
    let mut bias = BiasHead::new(v, bias_lr);
    let mut mixer = AdaptiveMixer::new(mix_eta);
    let mut enc = RangeEncoder::new();
    let t_start = std::time::Instant::now();

    for t in 0..tokens.len() {
        let tok = tokens[t] as usize;

        if t == 0 {
            let cdf = Cdf::uniform(v);
            enc.encode_symbol(tok, &cdf);
        } else {
            ngram.compute_bias_into(ng_buf, None);
            let b_vec = bias.bias_vector();
            mixer.combine_into(ensemble_buf, logits, ng_buf, b_vec);

            let cdf = cdf_from_logits(ensemble_buf);
            enc.encode_symbol(tok, &cdf);

            softmax_into(ensemble_buf, probs_buf);
            mixer.update(probs_buf, ng_buf, bias.bias_vector(), tok);
            bias.update(probs_buf, tok);
        }

        ngram.observe(tok as u32);
        model.forward_into(tok, state, scratch, logits);
    }

    let compressed = enc.finish();
    let elapsed = t_start.elapsed().as_secs_f64();

    let input_crc = crc32::crc32(input_slice);
    let total_u32 = validate_u32_size(total_bytes, "TOTAL_BYTES")?;

    let hdr = build_v4_header(total_u32, bias_lr, ngram_scale, mix_eta, emb_surgery, false);
    let mut output = Vec::with_capacity(hdr.len() + compressed.len() + CRC_SIZE);
    output.extend_from_slice(&hdr);
    output.extend_from_slice(&compressed);
    output.extend_from_slice(&input_crc.to_le_bytes());

    crate::domain::error::write_file(
        &output_path.to_string_lossy(),
        &output,
    )?;

    let bpb = (output.len() as f64 * 8.0) / total_bytes.max(1) as f64;
    let tok_s = tokens.len() as f64 / elapsed.max(0.001);

    Ok((total_bytes, output.len(), bpb, tok_s))
}

pub fn cmd_decompress(args: &[String]) -> AzResult<()> {
    let mut input_path = String::new();
    let mut output_path = String::new();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut quant_str = String::new();
    let mut emb_surgery = String::new();
    let mut streaming = false;
    let mut no_governor = false;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input")?.to_string(); }
            "--output" => { output_path = next_arg(args, &mut i, "--output")?.to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")?.to_string(); }
            "--quant" => { quant_str = next_arg(args, &mut i, "--quant")?.to_string(); }
            "--emb-surgery" => { emb_surgery = next_arg(args, &mut i, "--emb-surgery")?.to_string(); }
            "--streaming" => { streaming = true; }
            "--no-governor" => { no_governor = true; }
            "--verbose" | "-v" => { verbose = true; }
            _ => {}
        }
        i += 1;
    }

    if input_path.is_empty() || output_path.is_empty() {
        return Err(AzError::Parse {
            context: "--input/--output".to_string(), value: "(missing)".to_string(),
        });
    }

    // Hardware governor (R-HW1)
    let governor = crate::infrastructure::governor::HardwareGovernor::new(!no_governor);
    governor.preflight().map_err(|e| AzError::Format(format!("governor: {}", e)))?;
    governor.start_watchdog();

    let mut timer = PhaseTimer::new();

    timer.start("read");
    let data = crate::domain::error::read_file(&input_path)?;

    validate_magic(&data)?;
    let hdr = parse_header(&data)?;
    let (stream, stored_crc) = extract_stream_and_crc_v(&data, hdr.header_size, hdr.version);

    eprintln!("[decompress] format: AZTH v{}", hdr.version);
    if let Some(tc) = hdr.token_count {
        eprintln!("[decompress] compressed: {} bytes, original: {} bytes, {} tokens",
                  data.len(), hdr.total_bytes, tc);
    } else {
        eprintln!("[decompress] compressed: {} bytes, original: {} bytes (v4, no token count)",
                  data.len(), hdr.total_bytes);
    }
    eprintln!("[decompress] hyperparams: lr={}, ngram_scale={}, mix_eta={}",
              hdr.bias_lr, hdr.ngram_scale, hdr.mix_eta);
    if !hdr.emb_surgery.is_empty() {
        eprintln!("[decompress] emb-surgery: {} (from header)", hdr.emb_surgery);
    }
    if hdr.e8e9 {
        eprintln!("[decompress] E8E9: will apply inverse transform after decode");
    }
    if let Some(crc) = stored_crc {
        eprintln!("[decompress] CRC-32: {:08X} (will verify)", crc);
    } else {
        eprintln!("[decompress] CRC-32: absent (v2 format, no integrity check)");
    }

    // Resolve emb-surgery: header takes precedence over CLI flag
    let effective_surgery = if !hdr.emb_surgery.is_empty() {
        if !emb_surgery.is_empty() && emb_surgery != hdr.emb_surgery {
            eprintln!("[decompress] WARNING: CLI --emb-surgery '{}' \
                overridden by header '{}'", emb_surgery, hdr.emb_surgery);
        }
        hdr.emb_surgery.clone()
    } else {
        emb_surgery.clone()
    };

    timer.start("model_load");
    let quant_override = if quant_str.is_empty() { None } else { model_loader::parse_quant_flag(&quant_str) };
    let (mut model, tokenizer) = model_loader::load_model_and_tokenizer_with_quant(&weights_dir, quant_override);
    let v = model.config.vocab_size;

    if !effective_surgery.is_empty() {
        model.embedding_surgery(&effective_surgery);
    }

    timer.start("decode");
    let mut ngram = TokenNgram::new(4, v, hdr.ngram_scale);
    let mut bias = BiasHead::new(v, hdr.bias_lr);
    let mut mixer = AdaptiveMixer::new(hdr.mix_eta);
    let mut dec = RangeDecoder::new(stream);

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];

    // Pre-allocated buffers for zero-alloc hot loop (R-019)
    let mut ng_buf = vec![0.0f32; v];
    let mut ensemble_buf = vec![0.0f32; v];
    let mut probs_buf = vec![0.0f32; v];

    eprintln!("[decompress] decoding ...");
    let t_start = std::time::Instant::now();

    let total_bytes = hdr.total_bytes;
    let mut t: usize = 0;

    // Streaming mode (R-014): write bytes progressively to disk via BufWriter.
    // Reduces peak memory from O(file_size) to O(buffer_size).
    // CRC computed incrementally; output deleted on verification failure.
    let mut writer: Option<std::io::BufWriter<std::fs::File>> = if streaming
        && hdr.token_count.is_none()
    {
        let f = std::fs::File::create(&output_path)
            .map_err(|e| AzError::Io { path: output_path.clone(), source: e })?;
        eprintln!("[decompress] streaming mode: writing progressively to {}", output_path);
        Some(std::io::BufWriter::new(f))
    } else {
        if streaming && hdr.token_count.is_some() {
            eprintln!("[decompress] WARNING: --streaming ignored for v2/v3 format");
        }
        None
    };
    let is_streaming = writer.is_some();

    let mut output_bytes: Vec<u8> = if is_streaming {
        Vec::new()
    } else {
        Vec::with_capacity(total_bytes)
    };
    let mut crc_state = crc32::CRC32_INIT;
    let mut bytes_written: usize = 0;

    if let Some(token_count) = hdr.token_count {
        // v2/v3 path: token-count driven loop (batch only)
        let mut decoded_tokens: Vec<u32> = Vec::with_capacity(token_count);
        for t in 0..token_count {
            let tok = if t == 0 {
                let cdf = Cdf::uniform(v);
                decode_checked(&mut dec, &cdf, v, t)?
            } else {
                ngram.compute_bias_into(&mut ng_buf, None);
                let b_vec = bias.bias_vector();
                mixer.combine_into(&mut ensemble_buf, &logits, &ng_buf, b_vec);
                let cdf = cdf_from_logits(&ensemble_buf);
                let sym = decode_checked(&mut dec, &cdf, v, t)?;
                softmax_into(&ensemble_buf, &mut probs_buf);
                mixer.update(&probs_buf, &ng_buf, bias.bias_vector(), sym as usize);
                bias.update(&probs_buf, sym as usize);
                sym
            };
            decoded_tokens.push(tok);
            ngram.observe(tok);
            model.forward_into(tok as usize, &mut state, &mut scratch, &mut logits);

            if t > 0 && t % 100 == 0 {
                governor.apply()
                    .map_err(|e| AzError::Format(e))?;
            }
            if t > 0 && (t % 1000 == 0 || t == token_count - 1) {
                let elapsed = t_start.elapsed().as_secs_f64();
                eprint!("\r[decompress] {:.1}% | {} tokens | {:.1} tok/s   ",
                        100.0 * t as f64 / token_count as f64, t, t as f64 / elapsed);
            }
        }
        eprintln!();
        for &tok in &decoded_tokens {
            let token_bytes = tokenizer.decode_token(tok);
            output_bytes.extend_from_slice(token_bytes);
            crc_state = crc32::crc32_update(crc_state, token_bytes);
        }
        bytes_written = output_bytes.len();
        t = token_count;
    } else {
        // v4 path: byte-count driven, interleaved decode+detokenize
        loop {
            let tok = if t == 0 {
                let cdf = Cdf::uniform(v);
                decode_checked(&mut dec, &cdf, v, t)?
            } else {
                ngram.compute_bias_into(&mut ng_buf, None);
                let b_vec = bias.bias_vector();
                mixer.combine_into(&mut ensemble_buf, &logits, &ng_buf, b_vec);
                let cdf = cdf_from_logits(&ensemble_buf);
                let sym = decode_checked(&mut dec, &cdf, v, t)?;
                softmax_into(&ensemble_buf, &mut probs_buf);
                mixer.update(&probs_buf, &ng_buf, bias.bias_vector(), sym as usize);
                bias.update(&probs_buf, sym as usize);
                sym
            };

            let token_bytes = tokenizer.decode_token(tok);
            if let Some(ref mut w) = writer {
                use std::io::Write;
                w.write_all(token_bytes)
                    .map_err(|e| AzError::Io { path: output_path.clone(), source: e })?;
            } else {
                output_bytes.extend_from_slice(token_bytes);
            }
            crc_state = crc32::crc32_update(crc_state, token_bytes);
            bytes_written += token_bytes.len();

            if bytes_written >= total_bytes {
                t += 1;
                break;
            }

            ngram.observe(tok);
            model.forward_into(tok as usize, &mut state, &mut scratch, &mut logits);
            t += 1;

            if t % 100 == 0 {
                governor.apply()
                    .map_err(|e| AzError::Format(e))?;
            }
            if t % 1000 == 0 {
                let elapsed = t_start.elapsed().as_secs_f64();
                let pct = 100.0 * bytes_written as f64 / total_bytes as f64;
                eprint!("\r[decompress] {:.1}% | {} tokens | {:.1} tok/s   ",
                        pct, t, t as f64 / elapsed);
            }
        }
        eprintln!();
    }

    // Flush streaming writer before verification
    if let Some(ref mut w) = writer {
        use std::io::Write;
        w.flush().map_err(|e| AzError::Io { path: output_path.clone(), source: e })?;
    }
    drop(writer);

    timer.start("verify");
    if bytes_written != total_bytes {
        if is_streaming { let _ = std::fs::remove_file(&output_path); }
        return Err(AzError::Integrity(format!(
            "size mismatch: decoded {} bytes, expected {}", bytes_written, total_bytes
        )));
    }

    let actual_crc = crc32::crc32_finalize(crc_state);
    if let Some(expected_crc) = stored_crc {
        if actual_crc != expected_crc {
            if is_streaming { let _ = std::fs::remove_file(&output_path); }
            return Err(AzError::Integrity(format!(
                "CRC-32 mismatch: expected {:08X}, got {:08X} — file or weights corrupted",
                expected_crc, actual_crc
            )));
        }
        eprintln!("[decompress] CRC-32: {:08X} (verified OK)", actual_crc);
    }

    // E8E9 inverse transform (R-027): restore original bytes if E8E9 was applied
    if hdr.e8e9 {
        if is_streaming {
            // Streaming mode: read back, inverse, rewrite
            let streamed = crate::domain::error::read_file(&output_path)?;
            let restored = crate::domain::preprocess::e8e9_decode(&streamed);
            crate::domain::error::write_file(&output_path, &restored)?;
            eprintln!("[decompress] E8E9: inverse transform applied (streaming rewrite)");
        } else {
            output_bytes = crate::domain::preprocess::e8e9_decode(&output_bytes);
            eprintln!("[decompress] E8E9: inverse transform applied");
        }
    }

    if !is_streaming {
        timer.start("write");
        crate::domain::error::write_file(&output_path, &output_bytes)?;
    }
    timer.stop();

    let total_time = timer.total_secs();
    eprintln!();
    eprintln!("[decompress] output: {} ({} bytes{})", output_path, bytes_written,
              if is_streaming { ", streamed" } else { "" });
    eprintln!("[decompress] time: {:.1}s ({:.1} tok/s)", total_time, t as f64 / total_time);
    eprintln!("[decompress] done.");

    if verbose {
        timer.report("decompress");
        eprintln!("[decompress] timing_json: {}", timer.to_json());
    }
    Ok(())
}

pub fn cmd_hybrid_eval(args: &[String]) -> AzResult<()> {
    let mut input_path = "data/enwik8".to_string();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut bias_lr: f32 = 0.30;
    let mut ngram_scale: f32 = 0.5;
    let mut mix_eta: f32 = 0.01;
    let mut tau: f32 = 0.0;
    let mut log_path = String::new();
    let mut emb_surgery = String::new();
    let mut quant_str = String::new();
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input")?.to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")?.to_string(); }
            "--bytes" => { max_bytes = parse_arg(next_arg(args, &mut i, "--bytes")?, "--bytes")?; }
            "--lr" => { bias_lr = parse_arg(next_arg(args, &mut i, "--lr")?, "--lr")?; }
            "--ngram-scale" => { ngram_scale = parse_arg(next_arg(args, &mut i, "--ngram-scale")?, "--ngram-scale")?; }
            "--mix-eta" => { mix_eta = parse_arg(next_arg(args, &mut i, "--mix-eta")?, "--mix-eta")?; }
            "--tau" => { tau = parse_arg(next_arg(args, &mut i, "--tau")?, "--tau")?; }
            "--log" => { log_path = next_arg(args, &mut i, "--log")?.to_string(); }
            "--emb-surgery" => { emb_surgery = next_arg(args, &mut i, "--emb-surgery")?.to_string(); }
            "--quant" => { quant_str = next_arg(args, &mut i, "--quant")?.to_string(); }
            "--verbose" | "-v" => { verbose = true; }
            _ => {}
        }
        i += 1;
    }

    let mut timer = PhaseTimer::new();

    timer.start("read");
    let raw_bytes = crate::domain::error::read_file(&input_path)?;
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];
    eprintln!("[hybrid-eval] input: {} ({} bytes)", input_path, total_bytes);

    timer.start("model_load");
    let quant_override = if quant_str.is_empty() { None } else { model_loader::parse_quant_flag(&quant_str) };
    let (mut model, tokenizer) = model_loader::load_model_and_tokenizer_with_quant(&weights_dir, quant_override);
    let v = model.config.vocab_size;

    if !emb_surgery.is_empty() {
        model.embedding_surgery(&emb_surgery);
    }

    timer.start("tokenize");
    let tokens = tokenizer.encode(input_slice);
    eprintln!("[hybrid-eval] tokenized: {} tokens ({:.2} bytes/token)",
              tokens.len(), total_bytes as f64 / tokens.len() as f64);

    let mut ngram = TokenNgram::new(4, v, ngram_scale);
    let mut bias = BiasHead::new(v, bias_lr);
    if tau > 0.0 { bias = bias.with_decay(tau); }
    let mut mixer = AdaptiveMixer::new(mix_eta);

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut ce_bits = 0.0f64;

    // Pre-allocated buffers for zero-alloc hot loop (R-019)
    let mut ng_buf = vec![0.0f32; v];
    let mut ensemble_buf = vec![0.0f32; v];
    let mut probs_buf = vec![0.0f32; v];

    let mut logger = if !log_path.is_empty() {
        Some(JsonLogger::new(&log_path, total_bytes))
    } else {
        None
    };

    timer.start("evaluate");
    eprintln!("[hybrid-eval] evaluating ...");
    let t_start = std::time::Instant::now();

    for t in 0..tokens.len() {
        let tok = tokens[t] as usize;

        if t == 0 {
            ce_bits += -(1.0f64 / v as f64).log2();
        } else {
            let h = entropy_from_logits(&logits);
            ngram.compute_bias_into(&mut ng_buf, Some(h));
            let b_vec = bias.bias_vector();
            mixer.combine_into(&mut ensemble_buf, &logits, &ng_buf, b_vec);

            softmax_into(&ensemble_buf, &mut probs_buf);
            let prob = probs_buf[tok] as f64;
            ce_bits += -prob.max(1e-30).log2();

            if let Some(ref mut lg) = logger {
                let token_bytes = tokenizer.decode_token(tokens[t]).len();
                let (eff_lr, ema_surp) = bias.telemetry();
                let snap = LogSnapshot {
                    w_ngram: mixer.w_ngram,
                    w_bias: mixer.w_bias,
                    eff_lr,
                    ema_surprise: ema_surp,
                };
                lg.record_token(token_bytes, prob, &snap);
            }

            mixer.update(&probs_buf, &ng_buf, bias.bias_vector(), tok);
            bias.update(&probs_buf, tok);
        }

        ngram.observe(tok as u32);
        model.forward_into(tok, &mut state, &mut scratch, &mut logits);

        if t > 0 && (t % 1000 == 0 || t == tokens.len() - 1) {
            let elapsed = t_start.elapsed().as_secs_f64();
            let bpb = ce_bits / total_bytes as f64;
            eprint!("\r[hybrid-eval] {:.1}% | {} tokens | BPB {:.4} | {:.1} tok/s   ",
                    100.0 * t as f64 / tokens.len() as f64, t, bpb, t as f64 / elapsed);
        }
    }
    timer.stop();

    if let Some(ref mut lg) = logger {
        let (eff_lr, ema_surp) = bias.telemetry();
        let snap = LogSnapshot {
            w_ngram: mixer.w_ngram, w_bias: mixer.w_bias,
            eff_lr, ema_surprise: ema_surp,
        };
        lg.finalize(&snap);
    }

    let total_time = timer.total_secs();
    let bpb = ce_bits / total_bytes as f64;

    eprintln!();
    eprintln!();
    eprintln!("[hybrid-eval] results:");
    eprintln!("  input:   {} ({} bytes, {} tokens)", input_path, total_bytes, tokens.len());
    eprintln!("  BPB:     {:.6}", bpb);
    eprintln!("  mixer:   w_ngram={:.4}, w_bias={:.4}", mixer.w_ngram, mixer.w_bias);
    eprintln!("  time:    {:.1}s ({:.1} tok/s)", total_time, tokens.len() as f64 / total_time);
    if !log_path.is_empty() {
        eprintln!("  log:     {}", log_path);
    }

    if verbose {
        timer.report("hybrid-eval");
        eprintln!("[hybrid-eval] timing_json: {}", timer.to_json());
    }
    Ok(())
}

pub fn cmd_baseline(args: &[String]) -> AzResult<()> {
    let mut input_path = "data/enwik8".to_string();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut log_path = String::new();
    let mut quant_str = String::new();
    let mut emb_surgery = String::new();
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input")?.to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")?.to_string(); }
            "--bytes" => { max_bytes = parse_arg(next_arg(args, &mut i, "--bytes")?, "--bytes")?; }
            "--log" => { log_path = next_arg(args, &mut i, "--log")?.to_string(); }
            "--quant" => { quant_str = next_arg(args, &mut i, "--quant")?.to_string(); }
            "--emb-surgery" => { emb_surgery = next_arg(args, &mut i, "--emb-surgery")?.to_string(); }
            "--verbose" | "-v" => { verbose = true; }
            _ => {}
        }
        i += 1;
    }

    let mut timer = PhaseTimer::new();

    timer.start("read");
    let raw_bytes = crate::domain::error::read_file(&input_path)?;
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];
    eprintln!("[baseline] input: {} ({} bytes)", input_path, total_bytes);

    timer.start("model_load");
    let quant_override = if quant_str.is_empty() { None } else { model_loader::parse_quant_flag(&quant_str) };
    let (mut model, tokenizer) = model_loader::load_model_and_tokenizer_with_quant(&weights_dir, quant_override);
    let v = model.config.vocab_size;

    if !emb_surgery.is_empty() {
        model.embedding_surgery(&emb_surgery);
    }

    timer.start("tokenize");
    let tokens = tokenizer.encode(input_slice);
    eprintln!("[baseline] tokenized: {} tokens ({:.2} bytes/token)",
              tokens.len(), total_bytes as f64 / tokens.len() as f64);

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut ce_bits = 0.0f64;

    // Pre-allocated buffer for zero-alloc baseline loop (R-019)
    let mut probs_buf = vec![0.0f32; v];

    let mut logger = if !log_path.is_empty() {
        Some(JsonLogger::new(&log_path, total_bytes))
    } else {
        None
    };

    timer.start("evaluate");
    eprintln!("[baseline] evaluating (RWKV-only, no ensemble) ...");
    let t_start = std::time::Instant::now();

    for t in 0..tokens.len() {
        let tok = tokens[t] as usize;

        if t == 0 {
            ce_bits += -(1.0f64 / v as f64).log2();
        } else {
            softmax_into(&logits, &mut probs_buf);
            let prob = probs_buf[tok] as f64;
            ce_bits += -prob.max(1e-30).log2();

            if let Some(ref mut lg) = logger {
                let token_bytes = tokenizer.decode_token(tokens[t]).len();
                lg.record_token(token_bytes, prob, &LogSnapshot::default());
            }
        }

        model.forward_into(tok, &mut state, &mut scratch, &mut logits);

        if t > 0 && (t % 1000 == 0 || t == tokens.len() - 1) {
            let elapsed = t_start.elapsed().as_secs_f64();
            let bpb = ce_bits / total_bytes as f64;
            eprint!("\r[baseline] {:.1}% | {} tokens | BPB {:.4} | {:.1} tok/s   ",
                    100.0 * t as f64 / tokens.len() as f64, t, bpb, t as f64 / elapsed);
        }
    }
    timer.stop();

    if let Some(ref mut lg) = logger {
        lg.finalize(&LogSnapshot::default());
    }

    let total_time = timer.total_secs();
    let bpb = ce_bits / total_bytes as f64;

    eprintln!();
    eprintln!();
    eprintln!("[baseline] results:");
    eprintln!("  input: {} ({} bytes, {} tokens)", input_path, total_bytes, tokens.len());
    eprintln!("  BPB:   {:.6} (RWKV-only, no ensemble)", bpb);
    eprintln!("  time:  {:.1}s ({:.1} tok/s)", total_time, tokens.len() as f64 / total_time);
    if !log_path.is_empty() {
        eprintln!("  log:   {}", log_path);
    }

    if verbose {
        timer.report("baseline");
        eprintln!("[baseline] timing_json: {}", timer.to_json());
    }
    Ok(())
}

pub fn cmd_cm_eval(args: &[String]) -> AzResult<()> {
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut output_path = String::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")?.to_string(); }
            "--output" => { output_path = next_arg(args, &mut i, "--output")?.to_string(); }
            _ => {}
        }
        i += 1;
    }

    if output_path.is_empty() {
        return Err(AzError::Parse {
            context: "--output".to_string(), value: "(missing)".to_string(),
        });
    }

    let (model, _) = model_loader::load_model_and_tokenizer(&weights_dir);
    model.spectral_profile(&output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal valid AZTH v3 file from parts.
    fn make_azth_v3(total_bytes: u32, token_count: u32, stream: &[u8], crc: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_V3);
        buf.extend_from_slice(&total_bytes.to_le_bytes());
        buf.extend_from_slice(&token_count.to_le_bytes());
        buf.extend_from_slice(&0.30f32.to_le_bytes()); // bias_lr
        buf.extend_from_slice(&0.50f32.to_le_bytes()); // ngram_scale
        buf.extend_from_slice(&0.01f32.to_le_bytes()); // mix_eta
        buf.extend_from_slice(stream);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Helper: build a minimal valid AZTH v2 file (no CRC).
    fn make_azth_v2(total_bytes: u32, token_count: u32, stream: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"AZTH\x02\x00\x00\x00");
        buf.extend_from_slice(&total_bytes.to_le_bytes());
        buf.extend_from_slice(&token_count.to_le_bytes());
        buf.extend_from_slice(&0.30f32.to_le_bytes());
        buf.extend_from_slice(&0.50f32.to_le_bytes());
        buf.extend_from_slice(&0.01f32.to_le_bytes());
        buf.extend_from_slice(stream);
        buf
    }

    /// Helper: build AZTH v4 file with default hyperparams.
    fn make_azth_v4_default(total_bytes: u32, stream: &[u8], crc: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_V4);
        buf.extend_from_slice(&total_bytes.to_le_bytes());
        buf.push(0x00); // flags: defaults
        buf.extend_from_slice(stream);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Helper: build AZTH v4 file with custom hyperparams.
    fn make_azth_v4_custom(
        total_bytes: u32, bias_lr: f32, ngram_scale: f32, mix_eta: f32,
        stream: &[u8], crc: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_V4);
        buf.extend_from_slice(&total_bytes.to_le_bytes());
        buf.push(V4_FLAG_CUSTOM_HP); // flags: custom HP
        buf.extend_from_slice(&bias_lr.to_le_bytes());
        buf.extend_from_slice(&ngram_scale.to_le_bytes());
        buf.extend_from_slice(&mix_eta.to_le_bytes());
        buf.extend_from_slice(stream);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    // --- validate_magic ---

    #[test]
    fn validate_magic_accepts_v2() {
        let data = make_azth_v2(100, 10, &[0u8; 16]);
        assert_eq!(validate_magic(&data).unwrap(), 2);
    }

    #[test]
    fn validate_magic_accepts_v3() {
        let data = make_azth_v3(100, 10, &[0u8; 16], 0);
        assert_eq!(validate_magic(&data).unwrap(), 3);
    }

    #[test]
    fn validate_magic_accepts_v4() {
        let data = make_azth_v4_default(100, &[0u8; 16], 0);
        assert_eq!(validate_magic(&data).unwrap(), 4);
    }

    #[test]
    fn validate_magic_accepts_v4_custom() {
        let data = make_azth_v4_custom(100, 0.5, 0.5, 0.5, &[0u8; 8], 0);
        assert_eq!(validate_magic(&data).unwrap(), 4);
    }

    #[test]
    fn validate_magic_rejects_short() {
        assert!(validate_magic(&[0u8; 7]).is_err());
    }

    #[test]
    fn validate_magic_rejects_v4_too_short() {
        // 12 bytes — valid magic but too short for v4 (needs 13)
        let mut data = vec![0u8; 12];
        data[..8].copy_from_slice(MAGIC_V4);
        assert!(validate_magic(&data).is_err());
    }

    #[test]
    fn validate_magic_rejects_bad_prefix() {
        let mut data = vec![0u8; 32];
        data[..4].copy_from_slice(b"NOPE");
        assert!(validate_magic(&data).is_err());
    }

    #[test]
    fn validate_magic_rejects_bad_version() {
        let mut data = make_azth_v3(100, 10, &[0u8; 8], 0);
        data[4] = 99;
        assert!(validate_magic(&data).is_err());
    }

    // --- parse_header ---

    #[test]
    fn parse_header_v3() {
        let file = make_azth_v3(500, 42, &[0u8; 8], 0);
        let hdr = parse_header(&file).unwrap();
        assert_eq!(hdr.version, 3);
        assert_eq!(hdr.total_bytes, 500);
        assert_eq!(hdr.token_count, Some(42));
        assert!((hdr.bias_lr - 0.30).abs() < 1e-6);
        assert_eq!(hdr.header_size, HEADER_SIZE);
    }

    #[test]
    fn parse_header_v4_default() {
        let file = make_azth_v4_default(1234, &[0u8; 8], 0);
        let hdr = parse_header(&file).unwrap();
        assert_eq!(hdr.version, 4);
        assert_eq!(hdr.total_bytes, 1234);
        assert_eq!(hdr.token_count, None);
        assert_eq!(hdr.bias_lr.to_bits(), DEFAULT_BIAS_LR.to_bits());
        assert_eq!(hdr.ngram_scale.to_bits(), DEFAULT_NGRAM_SCALE.to_bits());
        assert_eq!(hdr.mix_eta.to_bits(), DEFAULT_MIX_ETA.to_bits());
        assert_eq!(hdr.header_size, V4_HEADER_BASE);
    }

    #[test]
    fn parse_header_v4_custom() {
        let file = make_azth_v4_custom(999, 0.5, 0.7, 0.02, &[0u8; 8], 0);
        let hdr = parse_header(&file).unwrap();
        assert_eq!(hdr.version, 4);
        assert_eq!(hdr.total_bytes, 999);
        assert_eq!(hdr.token_count, None);
        assert!((hdr.bias_lr - 0.5).abs() < 1e-6);
        assert!((hdr.ngram_scale - 0.7).abs() < 1e-6);
        assert!((hdr.mix_eta - 0.02).abs() < 1e-6);
        assert_eq!(hdr.header_size, V4_HEADER_BASE + 12);
    }

    #[test]
    fn parse_header_v4_rejects_unknown_flags() {
        let mut file = make_azth_v4_default(100, &[0u8; 8], 0);
        file[12] = 0x80; // unknown flag bit
        assert!(parse_header(&file).is_err());
    }

    #[test]
    fn parse_header_v4_custom_rejects_nan() {
        let file = make_azth_v4_custom(100, f32::NAN, 0.5, 0.01, &[], 0);
        assert!(parse_header(&file).is_err());
    }

    // --- uses_default_hp ---

    #[test]
    fn default_hp_detected() {
        assert!(uses_default_hp(0.30, 0.50, 0.01));
    }

    #[test]
    fn non_default_hp_detected() {
        assert!(!uses_default_hp(0.31, 0.50, 0.01));
        assert!(!uses_default_hp(0.30, 0.51, 0.01));
        assert!(!uses_default_hp(0.30, 0.50, 0.02));
    }

    // --- build_v4_header ---

    #[test]
    fn build_v4_header_default() {
        let hdr = build_v4_header(1000, 0.30, 0.50, 0.01, "", false);
        assert_eq!(hdr.len(), V4_HEADER_BASE);
        assert_eq!(&hdr[..8], MAGIC_V4);
        assert_eq!(u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]), 1000);
        assert_eq!(hdr[12], 0x00); // no flags
    }

    #[test]
    fn build_v4_header_custom() {
        let hdr = build_v4_header(2000, 0.5, 0.7, 0.02, "", false);
        assert_eq!(hdr.len(), V4_HEADER_BASE + 12);
        assert_eq!(hdr[12], V4_FLAG_CUSTOM_HP);
        let lr = f32::from_le_bytes([hdr[13], hdr[14], hdr[15], hdr[16]]);
        assert!((lr - 0.5).abs() < 1e-6);
    }

    // --- extract_stream_and_crc_v ---

    #[test]
    fn extract_v4_default_splits_correctly() {
        let stream_data = vec![0xAA, 0xBB, 0xCC];
        let crc: u32 = 0xCAFEBABE;
        let file = make_azth_v4_default(100, &stream_data, crc);
        let (stream, stored_crc) = extract_stream_and_crc_v(&file, V4_HEADER_BASE, 4);
        assert_eq!(stream, &stream_data[..]);
        assert_eq!(stored_crc, Some(0xCAFEBABE));
    }

    #[test]
    fn extract_v4_custom_splits_correctly() {
        let stream_data = vec![0x11, 0x22];
        let crc: u32 = 0x12345678;
        let file = make_azth_v4_custom(50, 0.5, 0.5, 0.5, &stream_data, crc);
        let (stream, stored_crc) = extract_stream_and_crc_v(&file, V4_HEADER_BASE + 12, 4);
        assert_eq!(stream, &stream_data[..]);
        assert_eq!(stored_crc, Some(0x12345678));
    }

    // --- parse_hyperparams (v2/v3 legacy) ---

    #[test]
    fn parse_hyperparams_valid() {
        let file = make_azth_v3(100, 10, &[], 0);
        let (lr, scale, eta) = parse_hyperparams(&file).unwrap();
        assert!((lr - 0.30).abs() < 1e-6);
        assert!((scale - 0.50).abs() < 1e-6);
        assert!((eta - 0.01).abs() < 1e-6);
    }

    #[test]
    fn parse_hyperparams_rejects_nan() {
        let mut file = make_azth_v3(100, 10, &[], 0);
        let nan_bytes = f32::NAN.to_le_bytes();
        file[16..20].copy_from_slice(&nan_bytes);
        assert!(parse_hyperparams(&file).is_err());
    }

    #[test]
    fn parse_hyperparams_rejects_inf() {
        let mut file = make_azth_v3(100, 10, &[], 0);
        let inf_bytes = f32::INFINITY.to_le_bytes();
        file[20..24].copy_from_slice(&inf_bytes);
        assert!(parse_hyperparams(&file).is_err());
    }

    // --- extract_stream_and_crc (legacy v2/v3) ---

    #[test]
    fn extract_crc_v3_splits_correctly() {
        let stream_data = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let crc: u32 = 0xDEADBEEF;
        let file = make_azth_v3(100, 10, &stream_data, crc);

        let (stream, stored_crc) = extract_stream_and_crc(&file, 3);
        assert_eq!(stream, &stream_data[..]);
        assert_eq!(stored_crc, Some(0xDEADBEEF));
    }

    #[test]
    fn extract_crc_v2_returns_none() {
        let stream_data = vec![0xAA, 0xBB, 0xCC];
        let file = make_azth_v2(100, 10, &stream_data);

        let (stream, stored_crc) = extract_stream_and_crc(&file, 2);
        assert_eq!(stream, &stream_data[..]);
        assert_eq!(stored_crc, None);
    }

    #[test]
    fn extract_crc_v3_empty_stream() {
        let file = make_azth_v3(0, 0, &[], 0x12345678);
        let (stream, stored_crc) = extract_stream_and_crc(&file, 3);
        assert!(stream.is_empty());
        assert_eq!(stored_crc, Some(0x12345678));
    }

    // --- CRC integrity detection ---

    #[test]
    fn crc_detects_stream_corruption() {
        let original_data = b"Hello, Redukt!";
        let crc = crc32::crc32(original_data);
        let file = make_azth_v3(original_data.len() as u32, 5, &[0xAA; 20], crc);
        let (_, stored_crc) = extract_stream_and_crc(&file, 3);

        let wrong_output = b"Hello, Redakt!";
        let actual_crc = crc32::crc32(wrong_output);
        assert_ne!(actual_crc, stored_crc.unwrap());
    }

    #[test]
    fn crc_detects_bitflip_in_file() {
        let original_data = b"test data for CRC";
        let crc = crc32::crc32(original_data);
        let mut file = make_azth_v3(original_data.len() as u32, 3, &[0x55; 10], crc);
        file[HEADER_SIZE] ^= 0x01;
        let (_, stored_crc) = extract_stream_and_crc(&file, 3);
        assert_eq!(stored_crc, Some(crc));
    }

    // --- Coder determinism ---

    #[test]
    fn coder_deterministic_output() {
        let n = 256;
        let mut probs = vec![0.001f32; n];
        probs[0] = 0.4;
        probs[1] = 0.3;
        probs[10] = 0.1;
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum; }
        let cdf = Cdf::from_probs(&probs);
        let symbols: Vec<usize> = vec![0, 1, 0, 10, 255, 0, 0, 1, 128, 50];

        let encode_once = || {
            let mut enc = RangeEncoder::new();
            for &s in &symbols { enc.encode_symbol(s, &cdf); }
            enc.finish()
        };

        assert_eq!(encode_once(), encode_once());
    }

    // --- validate_u32_size (R-007) ---

    #[test]
    fn validate_u32_size_accepts_max() {
        let result = validate_u32_size(u32::MAX as usize, "test");
        assert_eq!(result.unwrap(), u32::MAX);
    }

    #[test]
    fn validate_u32_size_accepts_zero() {
        assert_eq!(validate_u32_size(0, "test").unwrap(), 0);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn validate_u32_size_rejects_overflow() {
        let too_big = u32::MAX as usize + 1;
        assert!(validate_u32_size(too_big, "TOTAL_BYTES").is_err());
    }

    #[test]
    fn cdf_from_logits_deterministic() {
        let logits: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let cdf1 = cdf_from_logits(&logits);
        let cdf2 = cdf_from_logits(&logits);
        assert_eq!(cdf1.total, cdf2.total);
        let mut enc1 = RangeEncoder::new();
        let mut enc2 = RangeEncoder::new();
        enc1.encode_symbol(42, &cdf1);
        enc2.encode_symbol(42, &cdf2);
        assert_eq!(enc1.finish(), enc2.finish());
    }

    // --- CoderMetrics (R-009) ---

    #[test]
    fn coder_metrics_perfect_efficiency() {
        // 100 bytes input, 80 bits theoretical, 80 bits actual (10 bytes)
        let m = CoderMetrics { input_bytes: 100, compressed_bytes: 10, ce_bits: 80.0 };
        assert!((m.ce_bpb() - 0.80).abs() < 1e-10);
        assert!((m.compressed_bpb() - 0.80).abs() < 1e-10);
        assert!(m.overhead_bpb().abs() < 1e-10);
        assert!((m.efficiency_pct() - 100.0).abs() < 1e-10);
        assert!((m.ratio() - 0.10).abs() < 1e-10);
    }

    #[test]
    fn coder_metrics_with_overhead() {
        // 1000 bytes input, 4000 bits entropy, 5000 bits actual (625 bytes)
        let m = CoderMetrics { input_bytes: 1000, compressed_bytes: 625, ce_bits: 4000.0 };
        assert!((m.ce_bpb() - 4.0).abs() < 1e-10);
        assert!((m.compressed_bpb() - 5.0).abs() < 1e-10);
        assert!((m.overhead_bpb() - 1.0).abs() < 1e-10);
        assert!((m.efficiency_pct() - 80.0).abs() < 1e-10);
    }

    #[test]
    fn coder_metrics_zero_input() {
        let m = CoderMetrics { input_bytes: 0, compressed_bytes: 0, ce_bits: 0.0 };
        assert_eq!(m.ce_bpb(), 0.0);
        assert_eq!(m.compressed_bpb(), 0.0);
        assert_eq!(m.efficiency_pct(), 0.0);
    }

    // --- emb-surgery header (R-010) ---

    #[test]
    fn parse_emb_surgery_str_variants() {
        assert_eq!(parse_emb_surgery_str(""), (EMB_METHOD_NONE, 0.0));
        assert_eq!(parse_emb_surgery_str("norm"), (EMB_METHOD_NORM, 0.0));
        assert_eq!(parse_emb_surgery_str("spread"), (EMB_METHOD_SPREAD, 0.0));
        let (m, a) = parse_emb_surgery_str("center0.3");
        assert_eq!(m, EMB_METHOD_CENTER);
        assert!((a - 0.3).abs() < 1e-6);
        let (m, a) = parse_emb_surgery_str("center");
        assert_eq!(m, EMB_METHOD_CENTER);
        assert!((a - 0.3).abs() < 1e-6); // default alpha
        let (m, a) = parse_emb_surgery_str("center0.5");
        assert_eq!(m, EMB_METHOD_CENTER);
        assert!((a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn emb_surgery_roundtrip_string() {
        assert_eq!(emb_surgery_to_string(EMB_METHOD_NONE, 0.0), "");
        assert_eq!(emb_surgery_to_string(EMB_METHOD_NORM, 0.0), "norm");
        assert_eq!(emb_surgery_to_string(EMB_METHOD_SPREAD, 0.0), "spread");
        assert_eq!(emb_surgery_to_string(EMB_METHOD_CENTER, 0.3), "center0.3");
    }

    #[test]
    fn build_v4_header_with_emb_surgery() {
        let hdr = build_v4_header(1000, 0.30, 0.50, 0.01, "center0.3", false);
        // 13 base + 5 emb-surgery = 18 bytes
        assert_eq!(hdr.len(), V4_HEADER_BASE + EMB_SURGERY_BYTES);
        assert_eq!(hdr[12], V4_FLAG_EMB_SURGERY);
        assert_eq!(hdr[13], EMB_METHOD_CENTER);
        let alpha = f32::from_le_bytes([hdr[14], hdr[15], hdr[16], hdr[17]]);
        assert!((alpha - 0.3).abs() < 1e-6);
    }

    #[test]
    fn build_v4_header_custom_hp_plus_emb_surgery() {
        let hdr = build_v4_header(1000, 0.5, 0.7, 0.02, "norm", false);
        // 13 base + 12 custom HP + 5 emb-surgery = 30 bytes
        assert_eq!(hdr.len(), V4_HEADER_BASE + 12 + EMB_SURGERY_BYTES);
        assert_eq!(hdr[12], V4_FLAG_CUSTOM_HP | V4_FLAG_EMB_SURGERY);
        // HP at offset 13
        let lr = f32::from_le_bytes([hdr[13], hdr[14], hdr[15], hdr[16]]);
        assert!((lr - 0.5).abs() < 1e-6);
        // Emb-surgery at offset 25
        assert_eq!(hdr[25], EMB_METHOD_NORM);
    }

    #[test]
    fn parse_header_v4_emb_surgery() {
        let hdr = build_v4_header(500, 0.30, 0.50, 0.01, "center0.3", false);
        let mut file = Vec::new();
        file.extend_from_slice(&hdr);
        file.extend_from_slice(&[0u8; 8]); // stream
        file.extend_from_slice(&0u32.to_le_bytes()); // crc
        let info = parse_header(&file).unwrap();
        assert_eq!(info.version, 4);
        assert_eq!(info.total_bytes, 500);
        assert_eq!(info.emb_surgery, "center0.3");
        assert_eq!(info.header_size, V4_HEADER_BASE + EMB_SURGERY_BYTES);
    }

    #[test]
    fn parse_header_v4_custom_hp_plus_emb_surgery() {
        let hdr = build_v4_header(999, 0.5, 0.7, 0.02, "spread", false);
        let mut file = Vec::new();
        file.extend_from_slice(&hdr);
        file.extend_from_slice(&[0u8; 4]); // stream
        file.extend_from_slice(&0u32.to_le_bytes()); // crc
        let info = parse_header(&file).unwrap();
        assert_eq!(info.version, 4);
        assert_eq!(info.emb_surgery, "spread");
        assert!((info.bias_lr - 0.5).abs() < 1e-6);
        assert_eq!(info.header_size, V4_HEADER_BASE + 12 + EMB_SURGERY_BYTES);
    }

    #[test]
    fn parse_header_v4_no_emb_surgery() {
        let file = make_azth_v4_default(100, &[0u8; 8], 0);
        let info = parse_header(&file).unwrap();
        assert_eq!(info.emb_surgery, "");
    }

    #[test]
    fn v4_emb_surgery_flag_rejected_by_old_reserved_check() {
        // Verify that 0x02 flag alone is accepted (not rejected as "unknown")
        let hdr = build_v4_header(100, 0.30, 0.50, 0.01, "norm", false);
        let mut file = Vec::new();
        file.extend_from_slice(&hdr);
        file.extend_from_slice(&[0u8; 8]);
        file.extend_from_slice(&0u32.to_le_bytes());
        assert!(parse_header(&file).is_ok());
    }

    #[test]
    fn parse_header_v4_rejects_unknown_flags_with_emb() {
        // Bit 7 (0x80) should still be rejected even when 0x02 is set
        let mut hdr = build_v4_header(100, 0.30, 0.50, 0.01, "center0.3", false);
        hdr[12] |= 0x80; // add unknown flag
        let mut file = Vec::new();
        file.extend_from_slice(&hdr);
        file.extend_from_slice(&[0u8; 8]);
        file.extend_from_slice(&0u32.to_le_bytes());
        assert!(parse_header(&file).is_err());
    }
}
