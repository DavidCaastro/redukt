//! Application commands — compress, decompress, hybrid-eval, baseline, cm-eval.

use crate::domain::error::{AzError, AzResult, parse_arg, next_arg, validate_u32_size};
use crate::domain::tensor::{softmax, entropy_from_logits, Tensor};
use crate::domain::ngram::TokenNgram;
use crate::domain::bias_head::BiasHead;
use crate::domain::mixer::AdaptiveMixer;
use crate::domain::coder::{Cdf, RangeEncoder, RangeDecoder};
use crate::domain::crc32;
use crate::application::model_loader;
use crate::application::telemetry::{JsonLogger, LogSnapshot, PhaseTimer};
use crate::infrastructure::rwkv7::model::Rwkv7State;

pub const MAGIC_V3: &[u8; 8] = b"AZTH\x03\x00\x00\x00";
pub const HEADER_SIZE: usize = 28; // 8 magic + 4 total_bytes + 4 token_count + 12 hyperparams
pub const CRC_SIZE: usize = 4; // CRC-32 appended after STREAM (v3 only)

/// Validate AZTH magic bytes. Returns format version (2 or 3).
pub fn validate_magic(data: &[u8]) -> AzResult<u8> {
    if data.len() < HEADER_SIZE {
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
    if version != 2 && version != 3 {
        return Err(AzError::Format(format!(
            "unsupported AZTH version {} (expected 2 or 3)", version
        )));
    }
    Ok(version)
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

/// Extract stream and CRC from AZTH payload based on version.
/// Returns (stream_slice, optional_stored_crc).
pub fn extract_stream_and_crc(data: &[u8], version: u8) -> (&[u8], Option<u32>) {
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
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut bias_lr: f32 = 0.30;
    let mut ngram_scale: f32 = 0.5;
    let mut mix_eta: f32 = 0.01;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input")?.to_string(); }
            "--output" => { output_path = next_arg(args, &mut i, "--output")?.to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")?.to_string(); }
            "--bytes" => { max_bytes = parse_arg(next_arg(args, &mut i, "--bytes")?, "--bytes")?; }
            "--lr" => { bias_lr = parse_arg(next_arg(args, &mut i, "--lr")?, "--lr")?; }
            "--ngram-scale" => { ngram_scale = parse_arg(next_arg(args, &mut i, "--ngram-scale")?, "--ngram-scale")?; }
            "--mix-eta" => { mix_eta = parse_arg(next_arg(args, &mut i, "--mix-eta")?, "--mix-eta")?; }
            "--verbose" | "-v" => { verbose = true; }
            _ => {}
        }
        i += 1;
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

    timer.start("model_load");
    let (model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
    let v = model.config.vocab_size;

    timer.start("tokenize");
    let tokens = tokenizer.encode(input_slice);
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
            let ng_bias = ngram.compute_bias(None);
            let b_vec = bias.bias_vector();
            let ensemble_logits = mixer.combine(&logits, &ng_bias, b_vec);

            let cdf = cdf_from_logits(&ensemble_logits);
            enc.encode_symbol(tok, &cdf);

            let tensor = Tensor::from_data(ensemble_logits, vec![v]);
            let probs = softmax(&tensor);
            let prob = probs.data[tok] as f64;
            ce_bits += -prob.max(1e-30).log2();

            mixer.update(&probs.data, &ngram.compute_bias(None), bias.bias_vector(), tok);
            bias.update(&probs.data, tok);
        }

        ngram.observe(tok as u32);
        model.forward_into(tok, &mut state, &mut scratch, &mut logits);

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

    timer.start("write");
    let input_crc = crc32::crc32(input_slice);
    let total_u32 = validate_u32_size(total_bytes, "TOTAL_BYTES")?;
    let token_u32 = validate_u32_size(tokens.len(), "TOKEN_COUNT")?;

    let mut output = Vec::with_capacity(HEADER_SIZE + compressed.len() + CRC_SIZE);
    output.extend_from_slice(MAGIC_V3);
    output.extend_from_slice(&total_u32.to_le_bytes());
    output.extend_from_slice(&token_u32.to_le_bytes());
    output.extend_from_slice(&bias_lr.to_le_bytes());
    output.extend_from_slice(&ngram_scale.to_le_bytes());
    output.extend_from_slice(&mix_eta.to_le_bytes());
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
    eprintln!("  format:         AZTH v3");
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

pub fn cmd_decompress(args: &[String]) -> AzResult<()> {
    let mut input_path = String::new();
    let mut output_path = String::new();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input")?.to_string(); }
            "--output" => { output_path = next_arg(args, &mut i, "--output")?.to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")?.to_string(); }
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

    let mut timer = PhaseTimer::new();

    timer.start("read");
    let data = crate::domain::error::read_file(&input_path)?;

    let version = validate_magic(&data)?;

    let total_bytes = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let token_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let (bias_lr, ngram_scale, mix_eta) = parse_hyperparams(&data)?;
    let (stream, stored_crc) = extract_stream_and_crc(&data, version);

    eprintln!("[decompress] format: AZTH v{}", version);
    eprintln!("[decompress] compressed: {} bytes, original: {} bytes, {} tokens",
              data.len(), total_bytes, token_count);
    eprintln!("[decompress] hyperparams: lr={}, ngram_scale={}, mix_eta={}",
              bias_lr, ngram_scale, mix_eta);
    if let Some(crc) = stored_crc {
        eprintln!("[decompress] CRC-32: {:08X} (will verify)", crc);
    } else {
        eprintln!("[decompress] CRC-32: absent (v2 format, no integrity check)");
    }

    timer.start("model_load");
    let (model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
    let v = model.config.vocab_size;

    timer.start("decode");
    let mut ngram = TokenNgram::new(4, v, ngram_scale);
    let mut bias = BiasHead::new(v, bias_lr);
    let mut mixer = AdaptiveMixer::new(mix_eta);
    let mut dec = RangeDecoder::new(stream);

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut decoded_tokens: Vec<u32> = Vec::with_capacity(token_count);

    eprintln!("[decompress] decoding ...");
    let t_start = std::time::Instant::now();

    for t in 0..token_count {
        let tok = if t == 0 {
            let cdf = Cdf::uniform(v);
            decode_checked(&mut dec, &cdf, v, t)?
        } else {
            let ng_bias = ngram.compute_bias(None);
            let b_vec = bias.bias_vector();
            let ensemble_logits = mixer.combine(&logits, &ng_bias, b_vec);

            let cdf = cdf_from_logits(&ensemble_logits);
            let sym = decode_checked(&mut dec, &cdf, v, t)?;

            let tensor = Tensor::from_data(ensemble_logits, vec![v]);
            let probs = softmax(&tensor);
            mixer.update(&probs.data, &ngram.compute_bias(None), bias.bias_vector(), sym as usize);
            bias.update(&probs.data, sym as usize);

            sym
        };

        decoded_tokens.push(tok);
        ngram.observe(tok);
        model.forward_into(tok as usize, &mut state, &mut scratch, &mut logits);

        if t > 0 && (t % 1000 == 0 || t == token_count - 1) {
            let elapsed = t_start.elapsed().as_secs_f64();
            let pct = 100.0 * t as f64 / token_count as f64;
            eprint!("\r[decompress] {:.1}% | {} tokens | {:.1} tok/s   ",
                    pct, t, t as f64 / elapsed);
        }
    }
    eprintln!();

    timer.start("detokenize");
    let mut output_bytes = Vec::with_capacity(total_bytes);
    for &tok in &decoded_tokens {
        let bytes = tokenizer.decode_token(tok);
        output_bytes.extend_from_slice(bytes);
    }

    timer.start("verify");
    if output_bytes.len() != total_bytes {
        return Err(AzError::Integrity(format!(
            "size mismatch: decoded {} bytes, expected {}", output_bytes.len(), total_bytes
        )));
    }

    if let Some(expected_crc) = stored_crc {
        let actual_crc = crc32::crc32(&output_bytes);
        if actual_crc != expected_crc {
            return Err(AzError::Integrity(format!(
                "CRC-32 mismatch: expected {:08X}, got {:08X} — file or weights corrupted",
                expected_crc, actual_crc
            )));
        }
        eprintln!("[decompress] CRC-32: {:08X} (verified OK)", actual_crc);
    }

    timer.start("write");
    crate::domain::error::write_file(&output_path, &output_bytes)?;
    timer.stop();

    let total_time = timer.total_secs();
    eprintln!();
    eprintln!("[decompress] output: {} ({} bytes)", output_path, output_bytes.len());
    eprintln!("[decompress] time: {:.1}s ({:.1} tok/s)", total_time, token_count as f64 / total_time);
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
    let (mut model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
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
            let ng_bias = ngram.compute_bias(Some(h));
            let b_vec = bias.bias_vector();
            let ensemble_logits = mixer.combine(&logits, &ng_bias, b_vec);

            let tensor = Tensor::from_data(ensemble_logits, vec![v]);
            let probs = softmax(&tensor);
            let prob = probs.data[tok] as f64;
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

            mixer.update(&probs.data, &ngram.compute_bias(Some(h)), bias.bias_vector(), tok);
            bias.update(&probs.data, tok);
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
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input")?.to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")?.to_string(); }
            "--bytes" => { max_bytes = parse_arg(next_arg(args, &mut i, "--bytes")?, "--bytes")?; }
            "--log" => { log_path = next_arg(args, &mut i, "--log")?.to_string(); }
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
    let (model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
    let v = model.config.vocab_size;

    timer.start("tokenize");
    let tokens = tokenizer.encode(input_slice);
    eprintln!("[baseline] tokenized: {} tokens ({:.2} bytes/token)",
              tokens.len(), total_bytes as f64 / tokens.len() as f64);

    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut ce_bits = 0.0f64;

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
            let tensor = Tensor::from_data(logits.clone(), vec![v]);
            let probs = softmax(&tensor);
            let prob = probs.data[tok] as f64;
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
    fn validate_magic_rejects_short() {
        assert!(validate_magic(&[0u8; 10]).is_err());
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

    // --- parse_hyperparams ---

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
        // Write NaN into bias_lr (bytes 16-19)
        let nan_bytes = f32::NAN.to_le_bytes();
        file[16..20].copy_from_slice(&nan_bytes);
        assert!(parse_hyperparams(&file).is_err());
    }

    #[test]
    fn parse_hyperparams_rejects_inf() {
        let mut file = make_azth_v3(100, 10, &[], 0);
        let inf_bytes = f32::INFINITY.to_le_bytes();
        file[20..24].copy_from_slice(&inf_bytes); // ngram_scale = Inf
        assert!(parse_hyperparams(&file).is_err());
    }

    // --- extract_stream_and_crc ---

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
}
