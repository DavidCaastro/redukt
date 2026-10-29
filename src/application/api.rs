//! Minimal HTTP/1.1 API server — zero dependencies.
//!
//! Endpoints:
//!   POST /v1/compress     — raw bytes in, AZTH bytes out
//!   POST /v1/decompress   — AZTH bytes in, raw bytes out
//!   GET  /v1/status       — JSON health check
//!
//! Single-threaded: one request at a time (compression is CPU-bound).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::time::Instant;

use crate::domain::coder::{Cdf, RangeEncoder, RangeDecoder};
use crate::domain::ngram::TokenNgram;
use crate::domain::bias_head::BiasHead;
use crate::domain::mixer::AdaptiveMixer;
use crate::domain::tensor::softmax_into;
use crate::domain::error::next_arg;
use crate::domain::crc32;
use crate::application::model_loader;
use crate::application::codec::{
    CRC_SIZE, cdf_from_logits, build_v4_header,
    validate_magic, parse_header, extract_stream_and_crc_v,
};
use crate::infrastructure::rwkv7::model::{Rwkv7Model, Rwkv7State};
use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;

const MAX_BODY: usize = 256 * 1024 * 1024; // 256 MB request limit
const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn cmd_api(args: &[String]) {
    let mut host = "0.0.0.0".to_string();
    let mut port = "8080".to_string();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => { host = next_arg(args, &mut i, "--host")
                .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
                .to_string(); }
            "--port" => { port = next_arg(args, &mut i, "--port")
                .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
                .to_string(); }
            "--weights" => { weights_dir = next_arg(args, &mut i, "--weights")
                .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
                .to_string(); }
            _ => {}
        }
        i += 1;
    }

    eprintln!("[api] loading model from {} ...", weights_dir);
    let (model, tokenizer) = model_loader::load_model_and_tokenizer(&weights_dir);
    eprintln!("[api] model loaded (vocab={})", model.config.vocab_size);

    let bind = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&bind).unwrap_or_else(|e| {
        eprintln!("error: cannot bind to {}: {}", bind, e);
        std::process::exit(1);
    });
    eprintln!("[api] listening on http://{}", bind);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle_connection(stream, &model, &tokenizer),
            Err(e) => eprintln!("[api] accept error: {}", e),
        }
    }
}

fn handle_connection(
    mut stream: std::net::TcpStream,
    model: &Rwkv7Model,
    tokenizer: &WorldTokenizer,
) {
    let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    let req = match parse_request(&mut reader) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[api] {} parse error: {}", peer, e);
            let _ = send_response(&mut stream, 400, "text/plain", e.as_bytes());
            return;
        }
    };

    eprintln!("[api] {} {} {} ({} bytes)", peer, req.method, req.path, req.body.len());
    let t0 = Instant::now();

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/v1/status") => {
            let body = format!(
                "{{\"status\":\"ok\",\"version\":\"{}\",\"vocab_size\":{},\"max_body\":{}}}",
                VERSION, model.config.vocab_size, MAX_BODY
            );
            let _ = send_response(&mut stream, 200, "application/json", body.as_bytes());
        }
        ("POST", "/v1/compress") => {
            if req.body.is_empty() {
                let _ = send_response(&mut stream, 400, "text/plain", b"empty body");
                return;
            }
            match compress_bytes(&req.body, model, tokenizer, 0.30, 0.5, 0.01) {
                Ok(result) => {
                    let elapsed = t0.elapsed().as_secs_f64();
                    eprintln!("[api] compressed {} -> {} bytes ({:.1}s)",
                              req.body.len(), result.len(), elapsed);
                    let _ = send_response(&mut stream, 200, "application/octet-stream", &result);
                }
                Err(msg) => {
                    let _ = send_response(&mut stream, 422, "text/plain", msg.as_bytes());
                }
            }
        }
        ("POST", "/v1/decompress") => {
            if req.body.len() < 8 {
                let _ = send_response(&mut stream, 400, "text/plain", b"body too short for AZTH header");
                return;
            }
            match decompress_bytes(&req.body, model, tokenizer) {
                Ok(result) => {
                    let elapsed = t0.elapsed().as_secs_f64();
                    eprintln!("[api] decompressed {} -> {} bytes ({:.1}s)",
                              req.body.len(), result.len(), elapsed);
                    let _ = send_response(&mut stream, 200, "application/octet-stream", &result);
                }
                Err(msg) => {
                    let _ = send_response(&mut stream, 422, "text/plain", msg.as_bytes());
                }
            }
        }
        _ => {
            let _ = send_response(&mut stream, 404, "text/plain", b"not found");
        }
    }
}

// --- HTTP parsing (minimal, spec-sufficient) ---

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn parse_request(reader: &mut BufReader<std::net::TcpStream>) -> Result<HttpRequest, String> {
    let mut request_line = String::new();
    reader.read_line(&mut request_line).map_err(|e| format!("read error: {}", e))?;
    let parts: Vec<&str> = request_line.trim().splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err("malformed request line".into());
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| format!("header read: {}", e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() { break; }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().unwrap_or(0);
        } else if let Some(val) = trimmed.strip_prefix("content-length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
    }

    if content_length > MAX_BODY {
        return Err(format!("body too large: {} > {}", content_length, MAX_BODY));
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).map_err(|e| format!("body read: {}", e))?;
    }

    Ok(HttpRequest { method, path, body })
}

fn send_response(
    stream: &mut std::net::TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found",
        422 => "Unprocessable Entity", 500 => "Internal Server Error",
        _ => "Unknown",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status, reason, content_type, body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

// --- In-memory compress/decompress ---

fn compress_bytes(
    input: &[u8],
    model: &Rwkv7Model,
    tokenizer: &WorldTokenizer,
    bias_lr: f32,
    ngram_scale: f32,
    mix_eta: f32,
) -> Result<Vec<u8>, String> {
    let total_bytes = input.len();
    if total_bytes > u32::MAX as usize {
        return Err(format!(
            "input too large: {} bytes exceeds ~4 GB AZTH limit", total_bytes
        ));
    }
    let v = model.config.vocab_size;
    let tokens = tokenizer.encode(input);

    let mut ngram = TokenNgram::new(4, v, ngram_scale);
    let mut bias = BiasHead::new(v, bias_lr);
    let mut mixer = AdaptiveMixer::new(mix_eta);
    let mut enc = RangeEncoder::new();
    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];

    // Pre-allocated buffers for zero-alloc hot loop (R-019)
    let mut ng_buf = vec![0.0f32; v];
    let mut ensemble_buf = vec![0.0f32; v];
    let mut probs_buf = vec![0.0f32; v];

    for t in 0..tokens.len() {
        let tok = tokens[t] as usize;
        if t == 0 {
            enc.encode_symbol(tok, &Cdf::uniform(v));
        } else {
            ngram.compute_bias_into(&mut ng_buf, None);
            let b_vec = bias.bias_vector();
            mixer.combine_into(&mut ensemble_buf, &logits, &ng_buf, b_vec);
            let cdf = cdf_from_logits(&ensemble_buf);
            enc.encode_symbol(tok, &cdf);
            softmax_into(&ensemble_buf, &mut probs_buf);
            mixer.update(&probs_buf, &ng_buf, bias.bias_vector(), tok);
            bias.update(&probs_buf, tok);
        }
        ngram.observe(tok as u32);
        model.forward_into(tok, &mut state, &mut scratch, &mut logits);
    }

    let compressed = enc.finish();

    let input_crc = crc32::crc32(input);

    let hdr = build_v4_header(total_bytes as u32, bias_lr, ngram_scale, mix_eta, "");
    let mut output = Vec::with_capacity(hdr.len() + compressed.len() + CRC_SIZE);
    output.extend_from_slice(&hdr);
    output.extend_from_slice(&compressed);
    output.extend_from_slice(&input_crc.to_le_bytes());
    Ok(output)
}

/// Decode a symbol with bounds checking (API path — returns Err).
fn decode_checked(
    dec: &mut RangeDecoder, cdf: &Cdf, vocab_size: usize, token_idx: usize,
) -> Result<u32, String> {
    dec.decode_symbol_checked(cdf, vocab_size)
        .map(|sym| sym as u32)
        .map_err(|e| format!("decoder error at token {}: {}", token_idx, e))
}

pub fn decompress_bytes(
    data: &[u8],
    model: &Rwkv7Model,
    tokenizer: &WorldTokenizer,
) -> Result<Vec<u8>, String> {
    validate_magic(data).map_err(|e| e.to_string())?;
    let hdr = parse_header(data).map_err(|e| e.to_string())?;
    let (stream, stored_crc) = extract_stream_and_crc_v(data, hdr.header_size, hdr.version);

    let v = model.config.vocab_size;
    let total_bytes = hdr.total_bytes;
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

    let mut output = Vec::with_capacity(total_bytes);

    if let Some(token_count) = hdr.token_count {
        // v2/v3: token-count driven
        let mut decoded_tokens: Vec<u32> = Vec::with_capacity(token_count);
        for t in 0..token_count {
            let tok = if t == 0 {
                decode_checked(&mut dec, &Cdf::uniform(v), v, t)?
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
        }
        for &tok in &decoded_tokens {
            output.extend_from_slice(tokenizer.decode_token(tok));
        }
    } else {
        // v4: byte-count driven, interleaved
        let mut t: usize = 0;
        loop {
            let tok = if t == 0 {
                decode_checked(&mut dec, &Cdf::uniform(v), v, t)?
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
            output.extend_from_slice(tokenizer.decode_token(tok));
            if output.len() >= total_bytes {
                break;
            }
            ngram.observe(tok);
            model.forward_into(tok as usize, &mut state, &mut scratch, &mut logits);
            t += 1;
        }
    }

    if output.len() != total_bytes {
        return Err(format!("size mismatch: decoded {} bytes, expected {}", output.len(), total_bytes));
    }
    if let Some(expected_crc) = stored_crc {
        let actual_crc = crc32::crc32(&output);
        if actual_crc != expected_crc {
            return Err(format!(
                "CRC-32 mismatch: expected {:08X}, got {:08X}", expected_crc, actual_crc
            ));
        }
    }
    Ok(output)
}
