//! Minimal HTTP/1.1 API server — zero dependencies.
//!
//! Endpoints:
//!   POST /v1/compress     — raw bytes in, AZTH bytes out
//!   POST /v1/decompress   — AZTH bytes in, raw bytes out
//!   GET  /v1/status       — JSON health check (backbones, profile, formats, threads)
//!
//! Parity with the CLI: `/v1/compress` uses the same content detection, E8E9/shuffle/delta
//! preprocessing, backbone routing, weight profile and header as `compress --input`, and
//! runs the same lockstep engine (1 stream), so the body of the response is byte-identical
//! to the CLI archive for the same content and flags. `/v1/decompress` takes backbone and
//! weight profile from the archive header and answers 422 when that model is not loaded.
//!
//! Models are loaded once at start-up. Single-threaded accept loop: one request at a time
//! (compression is CPU-bound; the GEMV worker pool already uses every core).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use crate::application::codec::{
    extract_stream_and_crc_v, parse_header, preprocess_input, validate_magic, cdf_from_logits,
    HeaderInfo, DEFAULT_BIAS_LR, DEFAULT_MIX_ETA, DEFAULT_NGRAM_SCALE,
};
use crate::application::lockstep::{run_lockstep, Admission, EncodeMeta, StreamResult};
use crate::application::model_loader::{self, resolve_auto_profile, DynModel, DynTokenizer};
use crate::domain::backbone::{
    route_backbone, Backbone, BackboneChoice, BackendTokenizer, WeightProfile,
};
use crate::domain::cm_ensemble::CmEnsemble;
use crate::domain::coder::{Cdf, RangeDecoder};
use crate::domain::content_detect::ContentType;
use crate::domain::crc32;
use crate::domain::error::{next_arg, AzError, AzResult};
use crate::domain::tensor::softmax_into;

/// Request body limit (Cloud Run caps HTTP/1 requests and responses at 32 MiB).
const MAX_BODY: usize = 256 * 1024 * 1024;
const MAX_LINE: usize = 8192;
const MAX_HEADERS: usize = 100;
const IO_TIMEOUT: Duration = Duration::from_secs(60);
const VERSION: &str = env!("CARGO_PKG_VERSION");

// --- Configuration ---

/// Parsed flags of `redukt api`.
#[derive(Debug, PartialEq)]
struct ApiConfig {
    bind: String,
    port: u16,
    weights_dir: String,
    backbone: String,
    quant: String,
}

fn parse_config(args: &[String]) -> AzResult<ApiConfig> {
    let mut cfg = ApiConfig {
        bind: "0.0.0.0".into(), port: 8080, weights_dir: "weights/rwkv7-0.1b".into(),
        backbone: String::new(), quant: String::new(),
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        match flag.as_str() {
            "--bind" | "--host" => cfg.bind = next_arg(args, &mut i, &flag)?.to_string(),
            "--port" => {
                let v = next_arg(args, &mut i, "--port")?;
                cfg.port = v.parse().map_err(|_| AzError::Parse {
                    context: "--port".into(), value: v.to_string(),
                })?;
            }
            "--weights" => cfg.weights_dir = next_arg(args, &mut i, "--weights")?.to_string(),
            "--backbone" => cfg.backbone = next_arg(args, &mut i, "--backbone")?.to_string(),
            "--quant" => cfg.quant = next_arg(args, &mut i, "--quant")?.to_string(),
            other => return Err(AzError::Parse {
                context: "api".into(), value: format!("unknown flag '{}'", other),
            }),
        }
        i += 1;
    }
    Ok(cfg)
}

/// Backbones to load and, for a single explicit backbone, the forced routing choice.
/// "" / "auto" -> the default route (Mamba), routed per content; "a,b" loads several.
fn parse_backbones(spec: &str) -> AzResult<(Vec<BackboneChoice>, Option<BackboneChoice>)> {
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("auto") {
        return Ok((vec![route_backbone(&ContentType::Text)], None));
    }
    let mut list: Vec<BackboneChoice> = Vec::new();
    for part in spec.split(',') {
        let choice = BackboneChoice::from_str_opt(part.trim()).ok_or_else(|| AzError::Parse {
            context: "--backbone".into(),
            value: format!("'{}' (expected rwkv, mamba, mambabyte, plugin, auto or a comma list)", part),
        })?;
        if !list.contains(&choice) {
            list.push(choice);
        }
    }
    let forced = if list.len() == 1 { Some(list[0]) } else { None };
    Ok((list, forced))
}

// --- Server state ---

/// Public description of one loaded model (also what `/v1/status` reports).
struct LoadedInfo {
    choice: BackboneChoice,
    profile: WeightProfile,
    vocab: usize,
}

struct Server {
    info: Vec<LoadedInfo>,
    engines: Vec<(DynModel, DynTokenizer)>,
    forced: Option<BackboneChoice>,
    threads: usize,
}

/// Failure of a request: HTTP status plus a human-readable message.
#[derive(Debug)]
struct ApiError {
    status: u16,
    msg: String,
}

fn api_err(status: u16, msg: impl Into<String>) -> ApiError {
    ApiError { status, msg: msg.into() }
}

impl Server {
    fn loaded_names(&self) -> String {
        let names: Vec<String> = self.info.iter()
            .map(|i| format!("{}/{}", i.choice, i.profile)).collect();
        if names.is_empty() { "none".into() } else { names.join(", ") }
    }

    /// Index of the model for `choice`, requiring weight profile `profile` when given.
    fn engine_for(&self, choice: BackboneChoice, profile: Option<WeightProfile>) -> Result<usize, ApiError> {
        let idx = self.info.iter().position(|i| i.choice == choice).ok_or_else(|| {
            api_err(422, format!(
                "backbone {} is not loaded on this server (loaded: {}); restart `redukt api --backbone {}`",
                choice, self.loaded_names(), choice
            ))
        })?;
        match profile {
            Some(p) if p != self.info[idx].profile => Err(api_err(422, format!(
                "archive needs weight profile {} for backbone {} but this server loaded {} \
                 (restart `redukt api --quant {}`)",
                p, choice, self.info[idx].profile, p.to_string().to_lowercase()
            ))),
            _ => Ok(idx),
        }
    }

    fn engine(&self, idx: usize) -> Result<&(DynModel, DynTokenizer), ApiError> {
        self.engines.get(idx).ok_or_else(|| api_err(500, "model table inconsistent"))
    }

    fn status_json(&self) -> String {
        let models: Vec<String> = self.info.iter().map(|i| format!(
            "{{\"backbone\":\"{}\",\"profile\":\"{}\",\"vocab_size\":{},\"format_write\":\"v{}\"}}",
            i.choice, i.profile, i.vocab, if i.profile == WeightProfile::F32 { 4 } else { 5 }
        )).collect();
        let (backbone, profile, vocab) = self.info.first()
            .map_or(("none".to_string(), "none".to_string(), 0),
                    |i| (i.choice.to_string(), i.profile.to_string(), i.vocab));
        format!(
            "{{\"status\":\"ok\",\"version\":\"{}\",\"backbone\":\"{}\",\"profile\":\"{}\",\
             \"vocab_size\":{},\"backbones\":[{}],\"formats_read\":[\"v2\",\"v3\",\"v4\",\"v5\"],\
             \"threads\":{},\"max_body\":{}}}",
            VERSION, backbone, profile, vocab, models.join(","), self.threads, MAX_BODY
        )
    }
}

pub fn cmd_api(args: &[String]) -> AzResult<()> {
    let cfg = parse_config(args)?;
    let (choices, forced) = parse_backbones(&cfg.backbone)?;

    let mut info = Vec::new();
    let mut engines = Vec::new();
    for choice in choices {
        let profile = resolve_auto_profile(&cfg.quant, choice)
            .map_err(|e| AzError::Parse { context: "--quant".into(), value: e })?;
        eprintln!("[api] loading {} (profile {}) from {} ...", choice, profile, cfg.weights_dir);
        let t0 = Instant::now();
        let quant_override = model_loader::parse_quant_flag(&cfg.quant);
        let (model, tokenizer) = model_loader::load_dyn_model(choice, &cfg.weights_dir, quant_override, profile);
        if model.weight_profile() != profile {
            return Err(AzError::Format(format!(
                "loaded weight profile {} != requested {}", model.weight_profile(), profile
            )));
        }
        eprintln!("[api] {} loaded in {:.1}s (vocab={})", choice, t0.elapsed().as_secs_f64(), model.vocab_size());
        info.push(LoadedInfo { choice, profile, vocab: model.vocab_size() });
        engines.push((model, tokenizer));
    }
    let server = Server { info, engines, forced, threads: crate::domain::gemv::pool_threads() };

    let addr = format!("{}:{}", cfg.bind, cfg.port);
    let listener = TcpListener::bind(&addr).map_err(|e| AzError::Io { path: addr.clone(), source: e })?;
    eprintln!("[api] listening on http://{} ({} pool threads)", addr, server.threads);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle_connection(stream, &server),
            Err(e) => eprintln!("[api] accept error: {}", e),
        }
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, server: &Server) {
    let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let read_half = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[api] {} clone error: {}", peer, e);
            return;
        }
    };
    let mut reader = BufReader::new(read_half);

    let req = match parse_request(&mut reader, &mut stream) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[api] {} parse error: {}", peer, e.msg);
            let _ = send_response(&mut stream, e.status, "text/plain", e.msg.as_bytes());
            return;
        }
    };
    eprintln!("[api] {} {} {} ({} bytes)", peer, req.method, req.path, req.body.len());
    let t0 = Instant::now();
    let resp = dispatch(&req, server);
    eprintln!("[api] -> {} ({} bytes, {:.1}s)", resp.status, resp.body.len(), t0.elapsed().as_secs_f64());
    let _ = send_response(&mut stream, resp.status, resp.content_type, &resp.body);
}

// --- Routing ---

struct Response {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

fn text(status: u16, msg: &str) -> Response {
    Response { status, content_type: "text/plain", body: msg.as_bytes().to_vec() }
}

fn from_result(r: Result<Vec<u8>, ApiError>) -> Response {
    match r {
        Ok(body) => Response { status: 200, content_type: "application/octet-stream", body },
        Err(e) => text(e.status, &e.msg),
    }
}

fn dispatch(req: &HttpRequest, server: &Server) -> Response {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/v1/status") => Response {
            status: 200, content_type: "application/json", body: server.status_json().into_bytes(),
        },
        ("POST", "/v1/compress") => from_result(compress_request(&req.body, server)),
        ("POST", "/v1/decompress") => from_result(decompress_request(&req.body, server)),
        (_, "/v1/status") | (_, "/v1/compress") | (_, "/v1/decompress") => {
            text(405, "method not allowed")
        }
        _ => text(404, "not found"),
    }
}

/// Same pipeline as `compress --input` (single file): detect + preprocess, route backbone,
/// resolve the loaded profile, then one lockstep stream (identity contract with the CLI).
fn compress_request(body: &[u8], server: &Server) -> Result<Vec<u8>, ApiError> {
    if body.is_empty() {
        return Err(api_err(400, "empty body"));
    }
    if body.len() > u32::MAX as usize {
        return Err(api_err(413, format!("input too large: {} bytes exceeds ~4 GB AZTH limit", body.len())));
    }
    let prep = preprocess_input(body, "", "", "", "api");
    let choice = server.forced.unwrap_or_else(|| route_backbone(&prep.content_type));
    let idx = server.engine_for(choice, None)?;
    let (model, tokenizer) = server.engine(idx)?;
    let tokens = tokenizer.encode(&prep.data);
    let meta = EncodeMeta {
        total_bytes: body.len(),
        crc: crc32::crc32(&prep.data),
        e8e9: prep.e8e9,
        shuffle: prep.shuffle,
        delta: prep.delta,
        backbone: choice,
        profile: server.info[idx].profile,
        bias_lr: DEFAULT_BIAS_LR,
        ngram_scale: DEFAULT_NGRAM_SCALE,
        mix_eta: DEFAULT_MIX_ETA,
        emb_surgery: String::new(),
    };
    run_single(model, tokenizer, Admission::Encode { name: "api".into(), tokens, meta })
}

/// Run one admission through the lockstep engine with a single slot.
fn run_single(model: &DynModel, tokenizer: &DynTokenizer, adm: Admission) -> Result<Vec<u8>, ApiError> {
    let mut pending = Some(adm);
    let mut source = || pending.take().map(Ok);
    let mut result: Option<StreamResult> = None;
    let mut on_result = |r: StreamResult| result = Some(r);
    run_lockstep(model, tokenizer, 1, None, &mut source, &mut on_result)
        .map_err(|e| api_err(500, e.to_string()))?;
    let r = result.ok_or_else(|| api_err(500, "engine produced no result"))?;
    r.output.map_err(|e| api_err(422, e.to_string()))
}

/// Backbone and weight profile an archive needs, from its header only.
fn required_profile(hdr: &HeaderInfo) -> Result<WeightProfile, ApiError> {
    let mamba = matches!(hdr.backbone, BackboneChoice::Mamba | BackboneChoice::MambaByte);
    if !mamba && hdr.profile != WeightProfile::F32 {
        return Err(api_err(400, format!(
            "AZTH v{} declares weight profile {} for backbone {} (only Mamba supports profiles)",
            hdr.version, hdr.profile, hdr.backbone
        )));
    }
    Ok(hdr.profile)
}

/// Validate an archive against what this server can decode; returns the model index.
fn check_archive(data: &[u8], server: &Server) -> Result<(usize, HeaderInfo), ApiError> {
    validate_magic(data).map_err(|e| api_err(400, e.to_string()))?;
    let hdr = parse_header(data).map_err(|e| api_err(400, e.to_string()))?;
    let profile = required_profile(&hdr)?;
    if hdr.total_bytes > MAX_BODY {
        return Err(api_err(422, format!(
            "archive declares {} original bytes, above the server limit {}", hdr.total_bytes, MAX_BODY
        )));
    }
    if !hdr.emb_surgery.is_empty() {
        return Err(api_err(422, format!(
            "archive uses embedding surgery '{}', which the API cannot apply; use the CLI", hdr.emb_surgery
        )));
    }
    let idx = server.engine_for(hdr.backbone, Some(profile))?;
    Ok((idx, hdr))
}

fn decompress_request(data: &[u8], server: &Server) -> Result<Vec<u8>, ApiError> {
    if data.len() < 8 {
        return Err(api_err(400, "body too short for AZTH header"));
    }
    let (idx, _) = check_archive(data, server)?;
    let (model, tokenizer) = server.engine(idx)?;
    decompress_bytes(data, model, tokenizer).map_err(|e| api_err(422, e))
}

// --- HTTP parsing (minimal, spec-sufficient) ---

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn bad_request(msg: impl Into<String>) -> ApiError {
    api_err(400, msg)
}

/// Read one line of at most `MAX_LINE` bytes (without the terminator).
fn read_line_capped<R: BufRead>(reader: &mut R) -> Result<String, ApiError> {
    let mut line = String::new();
    let n = reader.by_ref().take(MAX_LINE as u64).read_line(&mut line)
        .map_err(|e| bad_request(format!("read error: {}", e)))?;
    if n == MAX_LINE && !line.ends_with('\n') {
        return Err(bad_request("header line too long"));
    }
    Ok(line.trim_end_matches(['\r', '\n']).to_string())
}

fn parse_request<R: BufRead, W: Write>(reader: &mut R, ack: &mut W) -> Result<HttpRequest, ApiError> {
    let request_line = read_line_capped(reader)?;
    let parts: Vec<&str> = request_line.trim().splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err(bad_request("malformed request line"));
    }
    let method = parts[0].to_string();
    let path = parts[1].split('?').next().unwrap_or("").to_string();

    let mut content_length: usize = 0;
    let mut expect_continue = false;
    for _ in 0..=MAX_HEADERS {
        let line = read_line_capped(reader)?;
        if line.trim().is_empty() {
            return finish_request(reader, ack, method, path, content_length, expect_continue);
        }
        let Some((name, value)) = line.split_once(':') else { continue };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => {
                content_length = value.parse()
                    .map_err(|_| bad_request(format!("invalid Content-Length '{}'", value)))?;
            }
            "transfer-encoding" => {
                return Err(bad_request("Transfer-Encoding is not supported; send Content-Length"));
            }
            "expect" => expect_continue = value.eq_ignore_ascii_case("100-continue"),
            _ => {}
        }
    }
    Err(bad_request("too many headers"))
}

fn finish_request<R: BufRead, W: Write>(
    reader: &mut R, ack: &mut W, method: String, path: String, content_length: usize,
    expect_continue: bool,
) -> Result<HttpRequest, ApiError> {
    if content_length > MAX_BODY {
        return Err(api_err(413, format!("body too large: {} > {}", content_length, MAX_BODY)));
    }
    if expect_continue && content_length > 0 {
        let _ = ack.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        let _ = ack.flush();
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body).map_err(|e| bad_request(format!("body read: {}", e)))?;
    Ok(HttpRequest { method, path, body })
}

fn render_response(status: u16, content_type: &str, body: &[u8]) -> Vec<u8> {
    let reason = match status {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found", 405 => "Method Not Allowed",
        413 => "Payload Too Large", 422 => "Unprocessable Entity", 500 => "Internal Server Error",
        _ => "Unknown",
    };
    let mut out = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status, reason, content_type, body.len()
    ).into_bytes();
    out.extend_from_slice(body);
    out
}

fn send_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    stream.write_all(&render_response(status, content_type, body))?;
    stream.flush()
}

// --- In-memory decompress (generic over backbone; also used by benchmark verify) ---

/// Decode a symbol with bounds checking (API path — returns Err).
fn decode_checked(
    dec: &mut RangeDecoder, cdf: &Cdf, vocab_size: usize, token_idx: usize,
) -> Result<u32, String> {
    dec.decode_symbol_checked(cdf, vocab_size)
        .map(|sym| sym as u32)
        .map_err(|e| format!("decoder error at token {}: {}", token_idx, e))
}

/// Undo the preprocessing recorded in the header, in the order of `decompress`.
fn invert_preprocessing(mut data: Vec<u8>, hdr: &HeaderInfo) -> Vec<u8> {
    if hdr.delta_stride > 0 {
        data = crate::domain::delta::delta_decode(&data, hdr.delta_stride as usize);
    }
    if hdr.shuffle_stride > 1 {
        data = crate::domain::byte_shuffle::byte_unshuffle(&data, hdr.shuffle_stride as usize);
    }
    if hdr.e8e9 {
        data = crate::domain::preprocess::e8e9_decode(&data);
    }
    data
}

pub fn decompress_bytes<M: Backbone, T: BackendTokenizer>(
    data: &[u8],
    model: &M,
    tokenizer: &T,
) -> Result<Vec<u8>, String> {
    validate_magic(data).map_err(|e| e.to_string())?;
    let hdr = parse_header(data).map_err(|e| e.to_string())?;
    let (stream, stored_crc) = extract_stream_and_crc_v(
        data, hdr.header_size, hdr.version,
    );

    let v = model.vocab_size();
    let total_bytes = hdr.total_bytes;
    let skip_threshold = hdr.skip_threshold;
    let entropy_thr_f = skip_threshold as f32 * 0.1;
    let mut consec_skips: usize = 0;
    const MAX_CONSEC_SKIP: usize = 3;

    let mut cm = CmEnsemble::new(
        v, hdr.ngram_scale, hdr.bias_lr, hdr.mix_eta, hdr.enhanced_cm,
    );
    let mut dec = RangeDecoder::new(stream);
    let mut state = model.create_state();
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut ensemble_buf = vec![0.0f32; v];
    let mut probs_buf = vec![0.0f32; v];

    let mut output = Vec::with_capacity(total_bytes.min(MAX_BODY));

    if let Some(token_count) = hdr.token_count {
        // v2/v3: token-count driven (no entropy gate)
        let mut decoded_tokens: Vec<u32> = Vec::with_capacity(token_count.min(MAX_BODY));
        for t in 0..token_count {
            let tok = if t == 0 {
                decode_checked(&mut dec, &Cdf::uniform(v), v, t)?
            } else {
                cm.compute_biases(None);
                cm.combine_into(&mut ensemble_buf, &logits);
                let cdf = cdf_from_logits(&ensemble_buf);
                let sym = decode_checked(&mut dec, &cdf, v, t)?;
                softmax_into(&ensemble_buf, &mut probs_buf);
                cm.update(&probs_buf, sym as usize);
                sym
            };
            decoded_tokens.push(tok);
            cm.observe(tok);
            model.forward_into(
                tok as usize, &mut state, &mut scratch, &mut logits,
            );
        }
        for &tok in &decoded_tokens {
            output.extend_from_slice(tokenizer.decode_token(tok));
        }
    } else {
        // v4/v5: byte-count driven, interleaved + R-077 entropy gate
        let mut t: usize = 0;
        loop {
            let mut gated = false;
            let tok = if t == 0 {
                decode_checked(&mut dec, &Cdf::uniform(v), v, t)?
            } else {
                cm.compute_biases(None);

                // R-077: entropy gate — identical to compress path
                let cm_h = cm.cm_only_entropy(&mut ensemble_buf);
                let do_gate = skip_threshold > 0
                    && consec_skips < MAX_CONSEC_SKIP
                    && cm_h < entropy_thr_f;

                if do_gate {
                    gated = true;
                } else {
                    cm.combine_into(&mut ensemble_buf, &logits);
                }
                let cdf = cdf_from_logits(&ensemble_buf);
                let sym = decode_checked(&mut dec, &cdf, v, t)?;
                softmax_into(&ensemble_buf, &mut probs_buf);
                cm.update(&probs_buf, sym as usize);
                sym
            };
            output.extend_from_slice(tokenizer.decode_token(tok));
            if output.len() >= total_bytes {
                break;
            }
            cm.observe(tok);
            if gated {
                consec_skips += 1;
            } else {
                model.forward_into(
                    tok as usize, &mut state, &mut scratch, &mut logits,
                );
                consec_skips = 0;
            }
            t += 1;
        }
    }

    if output.len() != total_bytes {
        return Err(format!(
            "size mismatch: decoded {} bytes, expected {}",
            output.len(), total_bytes
        ));
    }
    if let Some(expected_crc) = stored_crc {
        let actual_crc = crc32::crc32(&output);
        if actual_crc != expected_crc {
            return Err(format!(
                "CRC-32 mismatch: expected {:08X}, got {:08X}",
                expected_crc, actual_crc
            ));
        }
    }
    Ok(invert_preprocessing(output, &hdr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::codec::build_header;
    use std::io::Cursor;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn info(choice: BackboneChoice, profile: WeightProfile) -> LoadedInfo {
        LoadedInfo { choice, profile, vocab: 50280 }
    }

    /// Server with model metadata only (no weights): enough for routing/status/error paths.
    fn bare_server(loaded: Vec<LoadedInfo>, forced: Option<BackboneChoice>) -> Server {
        Server { info: loaded, engines: Vec::new(), forced, threads: 4 }
    }

    fn parse(raw: &str) -> Result<HttpRequest, ApiError> {
        let mut ack = Vec::new();
        parse_request(&mut Cursor::new(raw.as_bytes().to_vec()), &mut ack)
    }

    fn call(method: &str, path: &str, body: &[u8], srv: &Server) -> Response {
        let req = HttpRequest { method: method.into(), path: path.into(), body: body.to_vec() };
        dispatch(&req, srv)
    }

    fn archive(backbone: BackboneChoice, profile: WeightProfile, emb: &str) -> Vec<u8> {
        let mut a = build_header(
            100, 0.30, 0.50, 0.01, emb, false, 0, 0, 0, backbone, true, profile,
        );
        a.extend_from_slice(&[0u8; 16]);
        a
    }

    #[test]
    fn config_defaults_and_flags() {
        let d = parse_config(&[]).unwrap();
        assert_eq!((d.bind.as_str(), d.port), ("0.0.0.0", 8080));
        assert!(d.backbone.is_empty() && d.quant.is_empty());
        let c = parse_config(&s(&["--bind", "127.0.0.1", "--port", "9000", "--backbone", "mamba",
                                  "--quant", "q8", "--weights", "w"])).unwrap();
        assert_eq!(c, ApiConfig {
            bind: "127.0.0.1".into(), port: 9000, weights_dir: "w".into(),
            backbone: "mamba".into(), quant: "q8".into(),
        });
        assert_eq!(parse_config(&s(&["--host", "::1"])).unwrap().bind, "::1");
    }

    #[test]
    fn config_rejects_bad_input() {
        assert!(parse_config(&s(&["--port", "99999"])).is_err());
        assert!(parse_config(&s(&["--port"])).is_err());
        assert!(parse_config(&s(&["--bogus"])).is_err());
    }

    #[test]
    fn backbone_spec_parsing() {
        let (l, f) = parse_backbones("").unwrap();
        assert_eq!((l, f), (vec![BackboneChoice::Mamba], None));
        let (l, f) = parse_backbones("auto").unwrap();
        assert_eq!((l, f), (vec![BackboneChoice::Mamba], None));
        let (l, f) = parse_backbones("rwkv").unwrap();
        assert_eq!((l, f), (vec![BackboneChoice::Rwkv], Some(BackboneChoice::Rwkv)));
        let (l, f) = parse_backbones("mamba, rwkv,mamba").unwrap();
        assert_eq!((l, f), (vec![BackboneChoice::Mamba, BackboneChoice::Rwkv], None));
        assert!(parse_backbones("gpt").is_err());
    }

    #[test]
    fn request_parsing_basic() {
        let r = parse("POST /v1/compress?x=1 HTTP/1.1\r\ncontent-LENGTH: 3\r\nHost: a\r\n\r\nabc").unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str(), r.body.as_slice()), ("POST", "/v1/compress", &b"abc"[..]));
        let g = parse("GET /v1/status HTTP/1.1\r\n\r\n").unwrap();
        assert!(g.body.is_empty());
    }

    #[test]
    fn request_parsing_errors() {
        assert_eq!(parse("GARBAGE\r\n\r\n").unwrap_err().status, 400);
        assert_eq!(parse("POST / HTTP/1.1\r\nContent-Length: x\r\n\r\n").unwrap_err().status, 400);
        assert_eq!(parse("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap_err().status, 400);
        assert_eq!(parse("POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nab").unwrap_err().status, 400);
        let huge = format!("POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n", MAX_BODY + 1);
        assert_eq!(parse(&huge).unwrap_err().status, 413);
        let long = format!("GET /{} HTTP/1.1\r\n\r\n", "a".repeat(MAX_LINE));
        assert_eq!(parse(&long).unwrap_err().status, 400);
    }

    #[test]
    fn expect_continue_is_acknowledged() {
        let mut ack = Vec::new();
        let raw = "POST / HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\nhi";
        let r = parse_request(&mut Cursor::new(raw.as_bytes().to_vec()), &mut ack).unwrap();
        assert_eq!(r.body, b"hi");
        assert_eq!(ack, b"HTTP/1.1 100 Continue\r\n\r\n");
    }

    #[test]
    fn response_rendering() {
        let out = render_response(422, "text/plain", b"nope");
        let txt = String::from_utf8(out).unwrap();
        assert!(txt.starts_with("HTTP/1.1 422 Unprocessable Entity\r\n"));
        assert!(txt.contains("Content-Length: 4\r\n"));
        assert!(txt.ends_with("\r\n\r\nnope"));
    }

    #[test]
    fn status_reports_models_profile_formats_threads() {
        let srv = bare_server(vec![info(BackboneChoice::Mamba, WeightProfile::Q8)], None);
        let r = call("GET", "/v1/status", b"", &srv);
        assert_eq!((r.status, r.content_type), (200, "application/json"));
        let j = String::from_utf8(r.body).unwrap();
        for needle in [
            "\"status\":\"ok\"", "\"backbone\":\"mamba\"", "\"profile\":\"Q8\"", "\"threads\":4",
            "\"formats_read\":[\"v2\",\"v3\",\"v4\",\"v5\"]", "\"format_write\":\"v5\"",
            "\"version\":\"", "\"vocab_size\":50280",
        ] {
            assert!(j.contains(needle), "missing {} in {}", needle, j);
        }
    }

    #[test]
    fn routing_errors() {
        let srv = bare_server(vec![info(BackboneChoice::Mamba, WeightProfile::Q8)], None);
        assert_eq!(call("GET", "/nope", b"", &srv).status, 404);
        assert_eq!(call("GET", "/v1/compress", b"", &srv).status, 405);
        assert_eq!(call("POST", "/v1/compress", b"", &srv).status, 400);
        assert_eq!(call("POST", "/v1/decompress", b"AZTH", &srv).status, 400);
        assert_eq!(call("POST", "/v1/decompress", b"NOTAZTHFILE!", &srv).status, 400);
    }

    #[test]
    fn compress_with_backbone_not_loaded_is_422() {
        let srv = bare_server(vec![info(BackboneChoice::Rwkv, WeightProfile::F32)], None);
        let r = call("POST", "/v1/compress", b"hello world hello world", &srv);
        assert_eq!(r.status, 422);
        let msg = String::from_utf8(r.body).unwrap();
        assert!(msg.contains("backbone mamba is not loaded") && msg.contains("rwkv/F32"), "{}", msg);
    }

    #[test]
    fn decompress_requires_loaded_backbone_and_profile() {
        let srv = bare_server(vec![info(BackboneChoice::Mamba, WeightProfile::Q8)], None);
        let r = call("POST", "/v1/decompress", &archive(BackboneChoice::Rwkv, WeightProfile::F32, ""), &srv);
        assert_eq!(r.status, 422);
        assert!(String::from_utf8(r.body).unwrap().contains("backbone rwkv is not loaded"));

        let r = call("POST", "/v1/decompress", &archive(BackboneChoice::Mamba, WeightProfile::F32, ""), &srv);
        assert_eq!(r.status, 422);
        let msg = String::from_utf8(r.body).unwrap();
        let ok = msg.contains("weight profile F32") && msg.contains("loaded Q8");
        assert!(ok && msg.contains("--quant f32"), "{}", msg);
    }

    #[test]
    fn decompress_rejects_embedding_surgery_archives() {
        let srv = bare_server(vec![info(BackboneChoice::Rwkv, WeightProfile::F32)], None);
        let r = call("POST", "/v1/decompress", &archive(BackboneChoice::Rwkv, WeightProfile::F32, "norm"), &srv);
        assert_eq!(r.status, 422);
        assert!(String::from_utf8(r.body).unwrap().contains("embedding surgery"));
    }

    #[test]
    fn forced_backbone_overrides_routing() {
        let srv = bare_server(vec![info(BackboneChoice::Rwkv, WeightProfile::F32)], Some(BackboneChoice::Rwkv));
        assert_eq!(srv.engine_for(srv.forced.unwrap(), None).unwrap(), 0);
        // Model table without engines is reported as an internal error, never a panic.
        assert_eq!(call("POST", "/v1/compress", b"abc abc abc abc", &srv).status, 500);
    }
}
