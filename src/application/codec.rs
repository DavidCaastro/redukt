//! Application commands — compress, decompress, hybrid-eval, baseline, cm-eval.

use crate::domain::error::{parse_arg, next_arg};
use crate::domain::tensor::{softmax, entropy_from_logits, Tensor};
use crate::domain::ngram::TokenNgram;
use crate::domain::bias_head::BiasHead;
use crate::domain::mixer::AdaptiveMixer;
use crate::domain::coder::{Cdf, RangeEncoder, RangeDecoder};
use crate::domain::crc32;
use crate::application::model_loader;
use crate::application::telemetry::{JsonLogger, LogSnapshot};
use crate::infrastructure::rwkv7::model::Rwkv7State;

pub const MAGIC_V3: &[u8; 8] = b"AZTH\x03\x00\x00\x00";
pub const HEADER_SIZE: usize = 28; // 8 magic + 4 total_bytes + 4 token_count + 12 hyperparams
pub const CRC_SIZE: usize = 4; // CRC-32 appended after STREAM (v3 only)

/// Validate AZTH magic bytes. Returns format version (2 or 3) or exits.
pub fn validate_magic(data: &[u8]) -> u8 {
    if data.len() < HEADER_SIZE {
        eprintln!("error: invalid AZTH file (too short: {} bytes)", data.len());
        std::process::exit(1);
    }
    if &data[..4] != b"AZTH" {
        eprintln!("error: invalid AZTH file (bad magic: {:?})", &data[..4.min(data.len())]);
        std::process::exit(1);
    }
    let version = data[4];
    if version != 2 && version != 3 {
        eprintln!("error: unsupported AZTH version {} (expected 2 or 3)", version);
        std::process::exit(1);
    }
    version
}

/// Parse header hyperparameters with NaN/Inf validation.
pub fn parse_hyperparams(data: &[u8]) -> (f32, f32, f32) {
    let bias_lr = f32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let ngram_scale = f32::from_le_bytes([data[20], data[21], data[22], data[23]]);
    let mix_eta = f32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    for (name, val) in [("bias_lr", bias_lr), ("ngram_scale", ngram_scale), ("mix_eta", mix_eta)] {
        if !val.is_finite() {
            eprintln!("error: invalid hyperparameter {}={} (NaN or Inf)", name, val);
            std::process::exit(1);
        }
    }
    (bias_lr, ngram_scale, mix_eta)
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

pub fn cmd_compress(args: &[String]) {
    let mut input_path = "data/enwik8".to_string();
    let mut output_path = String::new();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut bias_lr: f32 = 0.30;
    let mut ngram_scale: f32 = 0.5;
    let mut mix_eta: f32 = 0.01;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input").to_string(); }
            "--output" => { output_path = next_arg(args, &mut i, "--output").to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights").to_string(); }
            "--bytes" => { max_bytes = parse_arg(next_arg(args, &mut i, "--bytes"), "--bytes").unwrap_or_else(|e| panic!("{}", e)); }
            "--lr" => { bias_lr = parse_arg(next_arg(args, &mut i, "--lr"), "--lr").unwrap_or_else(|e| panic!("{}", e)); }
            "--ngram-scale" => { ngram_scale = parse_arg(next_arg(args, &mut i, "--ngram-scale"), "--ngram-scale").unwrap_or_else(|e| panic!("{}", e)); }
            "--mix-eta" => { mix_eta = parse_arg(next_arg(args, &mut i, "--mix-eta"), "--mix-eta").unwrap_or_else(|e| panic!("{}", e)); }
            _ => {}
        }
        i += 1;
    }

    if output_path.is_empty() {
        eprintln!("error: --output PATH required");
        std::process::exit(1);
    }

    // Load input
    let raw_bytes = crate::domain::error::read_file(&input_path)
        .unwrap_or_else(|e| panic!("{}", e));
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];
    eprintln!("[compress] input: {} ({} bytes)", input_path, total_bytes);

    // Load tokenizer and model
    let (model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
    let v = model.config.vocab_size;

    // Tokenize
    let tokens = tokenizer.encode(input_slice);
    eprintln!("[compress] tokenized: {} tokens ({:.2} bytes/token)",
              tokens.len(), total_bytes as f64 / tokens.len() as f64);

    // Initialize ensemble + coder
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
    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!();

    let input_crc = crc32::crc32(input_slice);

    let mut output = Vec::with_capacity(HEADER_SIZE + compressed.len() + CRC_SIZE);
    output.extend_from_slice(MAGIC_V3);
    output.extend_from_slice(&(total_bytes as u32).to_le_bytes());
    output.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    output.extend_from_slice(&bias_lr.to_le_bytes());
    output.extend_from_slice(&ngram_scale.to_le_bytes());
    output.extend_from_slice(&mix_eta.to_le_bytes());
    output.extend_from_slice(&compressed);
    output.extend_from_slice(&input_crc.to_le_bytes());

    crate::domain::error::write_file(&output_path, &output)
        .unwrap_or_else(|e| panic!("{}", e));

    let compressed_bpb = output.len() as f64 * 8.0 / total_bytes as f64;
    let ce_bpb = ce_bits / total_bytes as f64;
    let ratio = output.len() as f64 / total_bytes as f64;

    eprintln!();
    eprintln!("[compress] results:");
    eprintln!("  input:          {} bytes", total_bytes);
    eprintln!("  compressed:     {} bytes (ratio: {:.4})", output.len(), ratio);
    eprintln!("  compressed BPB: {:.4}", compressed_bpb);
    eprintln!("  cross-ent BPB:  {:.4}", ce_bpb);
    eprintln!("  coder overhead: {:.4} BPB", compressed_bpb - ce_bpb);
    eprintln!("  CRC-32:         {:08X}", input_crc);
    eprintln!("  format:         AZTH v3");
    eprintln!("  time:           {:.1}s ({:.1} tok/s)", elapsed, tokens.len() as f64 / elapsed);
    eprintln!("  output:         {}", output_path);
}

pub fn cmd_decompress(args: &[String]) {
    let mut input_path = String::new();
    let mut output_path = String::new();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input").to_string(); }
            "--output" => { output_path = next_arg(args, &mut i, "--output").to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights").to_string(); }
            _ => {}
        }
        i += 1;
    }

    if input_path.is_empty() || output_path.is_empty() {
        eprintln!("error: --input PATH and --output PATH required");
        std::process::exit(1);
    }

    let data = crate::domain::error::read_file(&input_path)
        .unwrap_or_else(|e| panic!("{}", e));

    let version = validate_magic(&data);

    let total_bytes = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let token_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let (bias_lr, ngram_scale, mix_eta) = parse_hyperparams(&data);
    let (stream, stored_crc) = extract_stream_and_crc(&data, version);

    eprintln!("[decompress] format: AZTH v{}", version);
    eprintln!("[decompress] compressed: {} bytes, original: {} bytes, {} tokens",
              data.len(), total_bytes, token_count);
    eprintln!("[decompress] hyperparams: lr={}, ngram_scale={}, mix_eta={}",
              bias_lr, ngram_scale, mix_eta);
    if stored_crc.is_some() {
        eprintln!("[decompress] CRC-32: {:08X} (will verify)", stored_crc.unwrap());
    } else {
        eprintln!("[decompress] CRC-32: absent (v2 format, no integrity check)");
    }

    let (model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
    let v = model.config.vocab_size;

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
            dec.decode_symbol(&cdf) as u32
        } else {
            let ng_bias = ngram.compute_bias(None);
            let b_vec = bias.bias_vector();
            let ensemble_logits = mixer.combine(&logits, &ng_bias, b_vec);

            let cdf = cdf_from_logits(&ensemble_logits);
            let sym = dec.decode_symbol(&cdf);

            let tensor = Tensor::from_data(ensemble_logits, vec![v]);
            let probs = softmax(&tensor);
            mixer.update(&probs.data, &ngram.compute_bias(None), bias.bias_vector(), sym);
            bias.update(&probs.data, sym);

            sym as u32
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
    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!();

    let mut output_bytes = Vec::with_capacity(total_bytes);
    for &tok in &decoded_tokens {
        let bytes = tokenizer.decode_token(tok);
        output_bytes.extend_from_slice(bytes);
    }

    if output_bytes.len() != total_bytes {
        eprintln!("error: size mismatch — decoded {} bytes, expected {}", output_bytes.len(), total_bytes);
        std::process::exit(1);
    }

    if let Some(expected_crc) = stored_crc {
        let actual_crc = crc32::crc32(&output_bytes);
        if actual_crc != expected_crc {
            eprintln!("error: CRC-32 mismatch — expected {:08X}, got {:08X}", expected_crc, actual_crc);
            eprintln!("  the compressed file or model weights may be corrupted");
            std::process::exit(1);
        }
        eprintln!("[decompress] CRC-32: {:08X} (verified OK)", actual_crc);
    }

    crate::domain::error::write_file(&output_path, &output_bytes)
        .unwrap_or_else(|e| panic!("{}", e));

    eprintln!();
    eprintln!("[decompress] output: {} ({} bytes)", output_path, output_bytes.len());
    eprintln!("[decompress] time: {:.1}s ({:.1} tok/s)", elapsed, token_count as f64 / elapsed);
    eprintln!("[decompress] done.");
}

pub fn cmd_hybrid_eval(args: &[String]) {
    let mut input_path = "data/enwik8".to_string();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut bias_lr: f32 = 0.30;
    let mut ngram_scale: f32 = 0.5;
    let mut mix_eta: f32 = 0.01;
    let mut tau: f32 = 0.0;
    let mut log_path = String::new();
    let mut emb_surgery = String::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input").to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights").to_string(); }
            "--bytes" => { max_bytes = parse_arg(next_arg(args, &mut i, "--bytes"), "--bytes").unwrap_or_else(|e| panic!("{}", e)); }
            "--lr" => { bias_lr = parse_arg(next_arg(args, &mut i, "--lr"), "--lr").unwrap_or_else(|e| panic!("{}", e)); }
            "--ngram-scale" => { ngram_scale = parse_arg(next_arg(args, &mut i, "--ngram-scale"), "--ngram-scale").unwrap_or_else(|e| panic!("{}", e)); }
            "--mix-eta" => { mix_eta = parse_arg(next_arg(args, &mut i, "--mix-eta"), "--mix-eta").unwrap_or_else(|e| panic!("{}", e)); }
            "--tau" => { tau = parse_arg(next_arg(args, &mut i, "--tau"), "--tau").unwrap_or_else(|e| panic!("{}", e)); }
            "--log" => { log_path = next_arg(args, &mut i, "--log").to_string(); }
            "--emb-surgery" => { emb_surgery = next_arg(args, &mut i, "--emb-surgery").to_string(); }
            _ => {}
        }
        i += 1;
    }

    let raw_bytes = crate::domain::error::read_file(&input_path)
        .unwrap_or_else(|e| panic!("{}", e));
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];
    eprintln!("[hybrid-eval] input: {} ({} bytes)", input_path, total_bytes);

    let (mut model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
    let v = model.config.vocab_size;

    if !emb_surgery.is_empty() {
        model.embedding_surgery(&emb_surgery);
    }

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

    if let Some(ref mut lg) = logger {
        let (eff_lr, ema_surp) = bias.telemetry();
        let snap = LogSnapshot {
            w_ngram: mixer.w_ngram, w_bias: mixer.w_bias,
            eff_lr, ema_surprise: ema_surp,
        };
        lg.finalize(&snap);
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let bpb = ce_bits / total_bytes as f64;

    eprintln!();
    eprintln!();
    eprintln!("[hybrid-eval] results:");
    eprintln!("  input:   {} ({} bytes, {} tokens)", input_path, total_bytes, tokens.len());
    eprintln!("  BPB:     {:.6}", bpb);
    eprintln!("  mixer:   w_ngram={:.4}, w_bias={:.4}", mixer.w_ngram, mixer.w_bias);
    eprintln!("  time:    {:.1}s ({:.1} tok/s)", elapsed, tokens.len() as f64 / elapsed);
    if !log_path.is_empty() {
        eprintln!("  log:     {}", log_path);
    }
}

pub fn cmd_baseline(args: &[String]) {
    let mut input_path = "data/enwik8".to_string();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 0;
    let mut log_path = String::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input_path = next_arg(args, &mut i, "--input").to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights").to_string(); }
            "--bytes" => { max_bytes = parse_arg(next_arg(args, &mut i, "--bytes"), "--bytes").unwrap_or_else(|e| panic!("{}", e)); }
            "--log" => { log_path = next_arg(args, &mut i, "--log").to_string(); }
            _ => {}
        }
        i += 1;
    }

    let raw_bytes = crate::domain::error::read_file(&input_path)
        .unwrap_or_else(|e| panic!("{}", e));
    let total_bytes = if max_bytes > 0 { max_bytes.min(raw_bytes.len()) } else { raw_bytes.len() };
    let input_slice = &raw_bytes[..total_bytes];
    eprintln!("[baseline] input: {} ({} bytes)", input_path, total_bytes);

    let (model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
    let v = model.config.vocab_size;

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

    if let Some(ref mut lg) = logger {
        lg.finalize(&LogSnapshot::default());
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let bpb = ce_bits / total_bytes as f64;

    eprintln!();
    eprintln!();
    eprintln!("[baseline] results:");
    eprintln!("  input: {} ({} bytes, {} tokens)", input_path, total_bytes, tokens.len());
    eprintln!("  BPB:   {:.6} (RWKV-only, no ensemble)", bpb);
    eprintln!("  time:  {:.1}s ({:.1} tok/s)", elapsed, tokens.len() as f64 / elapsed);
    if !log_path.is_empty() {
        eprintln!("  log:   {}", log_path);
    }
}

pub fn cmd_cm_eval(args: &[String]) {
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut output_path = String::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights").to_string(); }
            "--output" => { output_path = next_arg(args, &mut i, "--output").to_string(); }
            _ => {}
        }
        i += 1;
    }

    if output_path.is_empty() {
        eprintln!("error: --output PATH required");
        std::process::exit(1);
    }

    let (model, _) = model_loader::load_model_and_tokenizer(&weights_dir);
    model.spectral_profile(&output_path);
}
