//! Lockstep multi-stream engine (R-100): batch compress / decompress.
//!
//! N independent files advance one token each per step. Per step:
//! - Phase A (parallel over streams on the pool): CM mix + range coder for the
//!   current token of every active stream, on that stream's own CmEnsemble
//!   and coder.
//! - Phase B: `Backbone::forward_multi` for all active streams, so every row
//!   block of weights is read from DRAM once per step and applied to N
//!   activation vectors (GEMV -> GEMM) instead of once per stream.
//!
//! Continuous batching: when a stream finishes, its slot is refilled with the
//! next file of the group. Files are grouped by (backbone, weight profile,
//! embedding surgery) because all streams of a step share one set of weights.
//!
//! Identity contract: each stream executes exactly the single-file sequence of
//! `cmd_compress` / `cmd_decompress` (uniform first symbol, CM compute/combine/
//! code/update/observe, forward) on private state; the pool and the GEMM
//! kernels only decide which thread computes a unit, never how (see
//! `gemv::par_rows_multi`). Hence `compress --input-dir` produces, per file,
//! the same bytes as `compress --input`, for any `--jobs` and any batch
//! composition, and archives decode through either path.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use crate::application::codec::{
    build_header, cdf_from_logits, extract_stream_and_crc_v, parse_header, preprocess_input,
    resolve_compress_profile, resolve_decompress_profile, validate_magic, HeaderInfo, CRC_SIZE,
};
use crate::application::model_loader::{self, DynModel, DynTokenizer};
use crate::domain::backbone::{
    Backbone, BackboneChoice, BackboneScratch, BackboneState, BackendTokenizer, WeightProfile,
};
use crate::domain::cm_ensemble::CmEnsemble;
use crate::domain::coder::{Cdf, RangeDecoder, RangeEncoder};
use crate::domain::crc32;
use crate::domain::error::{validate_u32_size, AzError, AzResult};
use crate::domain::par_pool;
use crate::domain::tensor::softmax_into;
use crate::infrastructure::governor::HardwareGovernor;

/// Default number of lockstep streams for `--input-dir` (see `--jobs`).
pub const DEFAULT_SLOTS: usize = 8;
/// Hard cap on streams (memory: ~6 MB per stream for Mamba-130M).
pub const MAX_SLOTS: usize = 256;

/// Shared CLI options of the batch commands.
pub struct BatchOptions {
    pub slots: usize,
    pub weights_dir: String,
    pub quant: String,
    pub emb_surgery: String,
    pub max_bytes: usize,
    pub bias_lr: f32,
    pub ngram_scale: f32,
    pub mix_eta: f32,
    pub e8e9_mode: String,
    pub shuffle_mode: String,
    pub delta_mode: String,
    pub backbone: String,
}

/// Everything a stream needs besides the backbone state (Phase A inputs).
/// Kept apart from state/scratch so Phase A can run on the pool.
struct Work {
    name: String,
    logits: Vec<f32>,
    ensemble: Vec<f32>,
    probs: Vec<f32>,
    active: Option<Active>,
}

/// An admitted file in flight.
struct Active {
    cm: CmEnsemble,
    pos: usize,
    next_token: usize,
    done: bool,
    error: Option<AzError>,
    started: Instant,
    io: Io,
}

enum Io {
    Encode { tokens: Vec<u32>, enc: RangeEncoder, meta: EncodeMeta },
    Decode {
        dec: RangeDecoder<'static>,
        out: Vec<u8>,
        crc: u32,
        expected_crc: u32,
        written: usize,
        hdr: HeaderInfo,
    },
}

/// Header inputs of one file to compress (all decided before admission, with
/// the same functions as `cmd_compress`).
pub struct EncodeMeta {
    pub total_bytes: usize,
    pub crc: u32,
    pub e8e9: bool,
    pub shuffle: u8,
    pub delta: u8,
    pub backbone: BackboneChoice,
    pub profile: WeightProfile,
    pub bias_lr: f32,
    pub ngram_scale: f32,
    pub mix_eta: f32,
    pub emb_surgery: String,
}

/// A file ready to enter a slot.
pub enum Admission {
    Encode { name: String, tokens: Vec<u32>, meta: EncodeMeta },
    Decode { name: String, stream: Vec<u8>, expected_crc: u32, hdr: HeaderInfo },
}

/// Outcome of one stream.
pub struct StreamResult {
    pub name: String,
    pub input_bytes: usize,
    pub tokens: usize,
    pub secs: f64,
    /// Output bytes (archive, or restored file) or the error.
    pub output: AzResult<Vec<u8>>,
}

/// Aggregate statistics of one lockstep run.
pub struct RunStats {
    pub steps: usize,
    pub tokens: usize,
    pub secs: f64,
    pub files: usize,
    /// Seconds spent in Phase A (CM + coder) and Phase B (backbone forward).
    pub code_secs: f64,
    pub forward_secs: f64,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Approximate resident bytes of one stream (state + scratch + logit buffers +
/// CM buffers). Excludes the n-gram hash tables, which grow with the data.
pub fn stream_bytes_estimate(backbone_bytes: usize, vocab: usize) -> usize {
    const MATCH_TABLE_BYTES: usize = (1 << 16) * 4;
    // logits/ensemble/probs + 5 CM bias buffers + bias head + segment counts (+ word separators)
    backbone_bytes + 3 * vocab * 4 + 7 * vocab * 4 + vocab + MATCH_TABLE_BYTES
}

// ---- Engine ----

/// Run streams in lockstep until `source` is exhausted and every stream has
/// finished. `source` yields the next admission (or an admission error, which
/// is reported through `on_result`); `on_result` receives each finished file.
pub fn run_lockstep<M: Backbone, T: BackendTokenizer>(
    model: &M, tokenizer: &T, slots: usize, governor: Option<&HardwareGovernor>,
    source: &mut dyn FnMut() -> Option<AzResult<Admission>>,
    on_result: &mut dyn FnMut(StreamResult),
) -> AzResult<RunStats> {
    let v = model.vocab_size();
    let slots = slots.clamp(1, MAX_SLOTS);
    let mut states: Vec<M::State> = (0..slots).map(|_| model.create_state()).collect();
    let mut scratches: Vec<M::Scratch> = (0..slots).map(|_| model.create_scratch()).collect();
    let mut works: Vec<Work> = (0..slots).map(|_| Work {
        name: String::new(), logits: vec![0.0; v], ensemble: vec![0.0; v], probs: vec![0.0; v], active: None,
    }).collect();
    let mut exhausted = false;
    let mut stats = RunStats { steps: 0, tokens: 0, secs: 0.0, files: 0, code_secs: 0.0, forward_secs: 0.0 };
    let t0 = Instant::now();
    let mut last_report = Instant::now();

    loop {
        // Admission: fill empty slots (continuous batching).
        for i in 0..slots {
            if works[i].active.is_some() || exhausted {
                continue;
            }
            match source() {
                None => exhausted = true,
                Some(Err(e)) => on_result(StreamResult {
                    name: String::new(), input_bytes: 0, tokens: 0, secs: 0.0, output: Err(e),
                }),
                Some(Ok(adm)) => {
                    states[i].reset();
                    scratches[i].reset_perf();
                    admit(&mut works[i], adm, v);
                }
            }
        }
        let n_active = works.iter().filter(|w| w.active.is_some()).count();
        if n_active == 0 {
            break;
        }

        // Phase A: CM + coder, one unit per active stream, on the pool.
        let t_a = Instant::now();
        {
            let parts: Vec<Mutex<&mut Work>> = works.iter_mut()
                .filter(|w| w.active.is_some()).map(Mutex::new).collect();
            par_pool::global().run(parts.len(), &|i| {
                let mut g = lock(&parts[i]);
                code_step(&mut g, v, tokenizer);
            });
        }
        stats.tokens += n_active;
        stats.code_secs += t_a.elapsed().as_secs_f64();

        // Finished streams leave their slot.
        for w in works.iter_mut() {
            let done = w.active.as_ref().map_or(false, |a| a.done);
            if done {
                stats.files += 1;
                on_result(finish(w, tokenizer));
            }
        }

        // Phase B: one lockstep forward for every stream still running.
        let t_b = Instant::now();
        {
            let mut tokens = Vec::with_capacity(slots);
            let mut st: Vec<&mut M::State> = Vec::with_capacity(slots);
            let mut sc: Vec<&mut M::Scratch> = Vec::with_capacity(slots);
            let mut lo: Vec<&mut [f32]> = Vec::with_capacity(slots);
            let iter = works.iter_mut().zip(states.iter_mut()).zip(scratches.iter_mut());
            for ((w, s), c) in iter {
                if let Some(a) = w.active.as_ref() {
                    tokens.push(a.next_token);
                    st.push(s);
                    sc.push(c);
                    lo.push(w.logits.as_mut_slice());
                }
            }
            if !tokens.is_empty() {
                model.forward_multi(&tokens, &mut st, &mut sc, &mut lo);
            }
        }
        stats.forward_secs += t_b.elapsed().as_secs_f64();
        stats.steps += 1;

        if stats.steps % 100 == 0 {
            if let Some(g) = governor {
                g.apply().map_err(AzError::Format)?;
            }
        }
        if last_report.elapsed().as_secs_f64() >= 1.0 {
            last_report = Instant::now();
            let el = t0.elapsed().as_secs_f64();
            eprint!("\r[lockstep] {} files done | {} active | {} tokens | {:.1} tok/s aggregate   ",
                stats.files, n_active, stats.tokens, stats.tokens as f64 / el);
        }
    }
    eprintln!();
    stats.secs = t0.elapsed().as_secs_f64();
    Ok(stats)
}

/// Install an admission into an empty slot.
fn admit(w: &mut Work, adm: Admission, v: usize) {
    let started = Instant::now();
    let (name, cm, io) = match adm {
        Admission::Encode { name, tokens, meta } => {
            let cm = CmEnsemble::new(v, meta.ngram_scale, meta.bias_lr, meta.mix_eta, true);
            (name, cm, Io::Encode { tokens, enc: RangeEncoder::new(), meta })
        }
        Admission::Decode { name, stream, expected_crc, hdr } => {
            let cm = CmEnsemble::new(v, hdr.ngram_scale, hdr.bias_lr, hdr.mix_eta, hdr.enhanced_cm);
            let dec = RangeDecoder::new_owned(stream);
            let out = Vec::with_capacity(hdr.total_bytes);
            let crc = crc32::CRC32_INIT;
            (name, cm, Io::Decode { dec, out, crc, expected_crc, written: 0, hdr })
        }
    };
    w.name = name;
    w.active = Some(Active { cm, pos: 0, next_token: 0, done: false, error: None, started, io });
}

/// Phase A for one stream: code the token at `pos` with the logits of the
/// previous forward, then advance the CM — exactly the single-file sequence.
fn code_step<T: BackendTokenizer>(w: &mut Work, v: usize, tokenizer: &T) {
    let Work { logits, ensemble, probs, active, .. } = w;
    let a = match active.as_mut() {
        Some(a) => a,
        None => return,
    };
    match &mut a.io {
        Io::Encode { tokens, enc, .. } => {
            let Some(&tok) = tokens.get(a.pos) else {
                a.done = true;
                return;
            };
            let tok = tok as usize;
            if a.pos == 0 {
                enc.encode_symbol(tok, &Cdf::uniform(v));
            } else {
                a.cm.compute_biases(None);
                a.cm.combine_into(ensemble, logits);
                enc.encode_symbol(tok, &cdf_from_logits(ensemble));
                softmax_into(ensemble, probs);
                a.cm.update(probs, tok);
            }
            a.cm.observe(tok as u32);
            a.next_token = tok;
            a.pos += 1;
            a.done = a.pos >= tokens.len();
        }
        Io::Decode { dec, out, crc, written, hdr, .. } => {
            let decoded = if a.pos == 0 {
                dec.decode_symbol_checked(&Cdf::uniform(v), v)
            } else {
                a.cm.compute_biases(None);
                a.cm.combine_into(ensemble, logits);
                let cdf = cdf_from_logits(ensemble);
                let r = dec.decode_symbol_checked(&cdf, v);
                if let Ok(sym) = r {
                    softmax_into(ensemble, probs);
                    a.cm.update(probs, sym);
                }
                r
            };
            let tok = match decoded {
                Ok(t) => t,
                Err(e) => {
                    a.error = Some(AzError::Decode { token_idx: a.pos, detail: e.to_string() });
                    a.done = true;
                    return;
                }
            };
            let bytes = tokenizer.decode_token(tok as u32);
            out.extend_from_slice(bytes);
            *crc = crc32::crc32_update(*crc, bytes);
            *written += bytes.len();
            a.pos += 1;
            if *written >= hdr.total_bytes {
                a.done = true;
                return;
            }
            // Every encoded token covers >= 1 input byte, so a valid stream has
            // at most `total_bytes` tokens. Stops a corrupt stream that keeps
            // decoding zero-length ids from spinning its slot forever.
            if a.pos >= hdr.total_bytes {
                a.error = Some(AzError::Decode {
                    token_idx: a.pos,
                    detail: format!("{} tokens decoded but only {} of {} bytes", a.pos, written, hdr.total_bytes),
                });
                a.done = true;
                return;
            }
            a.cm.observe(tok as u32);
            a.next_token = tok;
        }
    }
}

/// Take a finished stream out of its slot and build its output.
fn finish<T: BackendTokenizer>(w: &mut Work, _tokenizer: &T) -> StreamResult {
    let name = std::mem::take(&mut w.name);
    let a = match w.active.take() {
        Some(a) => a,
        None => return StreamResult {
            name, input_bytes: 0, tokens: 0, secs: 0.0,
            output: Err(AzError::Format("lockstep: finish on empty slot".into())),
        },
    };
    let secs = a.started.elapsed().as_secs_f64();
    let tokens = a.pos;
    if let Some(e) = a.error {
        return StreamResult { name, input_bytes: 0, tokens, secs, output: Err(e) };
    }
    match a.io {
        Io::Encode { enc, meta, tokens: toks } => {
            let output = encode_output(enc, &meta);
            StreamResult { name, input_bytes: meta.total_bytes, tokens: toks.len(), secs, output }
        }
        Io::Decode { out, crc, expected_crc, written, hdr, .. } => {
            let input_bytes = hdr.total_bytes;
            let output = decode_output(out, crc, expected_crc, written, &hdr);
            StreamResult { name, input_bytes, tokens, secs, output }
        }
    }
}

/// Header + stream + CRC, exactly as `cmd_compress` writes them.
fn encode_output(enc: RangeEncoder, m: &EncodeMeta) -> AzResult<Vec<u8>> {
    let compressed = enc.finish();
    let total_u32 = validate_u32_size(m.total_bytes, "TOTAL_BYTES")?;
    let hdr = build_header(
        total_u32, m.bias_lr, m.ngram_scale, m.mix_eta, &m.emb_surgery,
        m.e8e9, m.shuffle, m.delta, 0, m.backbone, true, m.profile,
    );
    let mut output = Vec::with_capacity(hdr.len() + compressed.len() + CRC_SIZE);
    output.extend_from_slice(&hdr);
    output.extend_from_slice(&compressed);
    output.extend_from_slice(&m.crc.to_le_bytes());
    Ok(output)
}

/// Size + CRC verification and inverse transforms (delta, shuffle, E8E9), in
/// the order of `cmd_decompress`.
fn decode_output(
    out: Vec<u8>, crc_state: u32, expected: u32, written: usize, hdr: &HeaderInfo,
) -> AzResult<Vec<u8>> {
    if written != hdr.total_bytes {
        return Err(AzError::Integrity(format!(
            "size mismatch: decoded {} bytes, expected {}", written, hdr.total_bytes
        )));
    }
    let actual = crc32::crc32_finalize(crc_state);
    if actual != expected {
        return Err(AzError::Integrity(format!(
            "CRC-32 mismatch: expected {:08X}, got {:08X} — file or weights corrupted", expected, actual
        )));
    }
    let mut data = out;
    if hdr.delta_stride > 0 {
        data = crate::domain::delta::delta_decode(&data, hdr.delta_stride as usize);
    }
    if hdr.shuffle_stride > 1 {
        data = crate::domain::byte_shuffle::byte_unshuffle(&data, hdr.shuffle_stride as usize);
    }
    if hdr.e8e9 {
        data = crate::domain::preprocess::e8e9_decode(&data);
    }
    Ok(data)
}

// ---- Directory drivers ----

/// Files of `dir` (non-recursive, sorted by name) with `keep(path)`.
fn list_files(dir: &str, keep: &dyn Fn(&Path) -> bool) -> AzResult<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| AzError::Io { path: dir.to_string(), source: e })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && keep(p))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(AzError::Format(format!("no files found in {}", dir)));
    }
    Ok(files)
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

/// Prefix an admission error with its file name (the run reports it nameless).
fn named(p: &Path, e: AzError) -> AzError {
    AzError::Format(format!("{}: {}", file_name(p), e))
}

/// Write `dir/name` through `name.part` + rename, so a failed or interrupted
/// write never leaves a truncated file under the final name.
fn write_output(dir: &str, name: &str, bytes: &[u8]) -> AzResult<usize> {
    if name.is_empty() {
        return Err(AzError::Format("empty output file name".into()));
    }
    let final_path = Path::new(dir).join(name);
    let tmp_path = Path::new(dir).join(format!("{}.part", name));
    if let Err(e) = crate::domain::error::write_file(&tmp_path.to_string_lossy(), bytes) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    std::fs::rename(&tmp_path, &final_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        AzError::Io { path: final_path.to_string_lossy().to_string(), source: e }
    })?;
    Ok(bytes.len())
}

fn read_capped(path: &Path, max_bytes: usize) -> AzResult<Vec<u8>> {
    let mut raw = crate::domain::error::read_file(&path.to_string_lossy())?;
    if max_bytes > 0 && raw.len() > max_bytes {
        raw.truncate(max_bytes);
    }
    Ok(raw)
}

/// Model group key: all streams of a step share one set of weights.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GroupKey {
    pub backbone: BackboneChoice,
    pub profile: WeightProfile,
    pub emb_surgery: String,
}

impl GroupKey {
    fn ord_tuple(&self) -> (u8, u8, &str) {
        (self.backbone.header_byte(), self.profile.header_code(), &self.emb_surgery)
    }
}

/// Group paths by key, preserving each group's order of first appearance.
pub fn group_by_key(items: Vec<(GroupKey, PathBuf)>) -> Vec<(GroupKey, Vec<PathBuf>)> {
    let mut groups: BTreeMap<(u8, u8, String), (GroupKey, Vec<PathBuf>)> = BTreeMap::new();
    for (key, path) in items {
        let (b, p, s) = key.ord_tuple();
        groups.entry((b, p, s.to_string())).or_insert_with(|| (key.clone(), Vec::new())).1.push(path);
    }
    groups.into_values().collect()
}

/// Decide backbone + profile for one input exactly like `cmd_compress`.
fn compress_key(input: &[u8], opts: &BatchOptions) -> AzResult<GroupKey> {
    let (content_type, _) = crate::domain::content_detect::detect_content_type(input);
    let backbone = BackboneChoice::from_str_opt(&opts.backbone)
        .unwrap_or_else(|| crate::domain::backbone::route_backbone(&content_type));
    let profile = resolve_compress_profile(&opts.quant, backbone)?;
    Ok(GroupKey { backbone, profile, emb_surgery: opts.emb_surgery.clone() })
}

/// Preprocess + tokenize one file for admission (same functions as `cmd_compress`).
fn compress_admission(
    path: &Path, opts: &BatchOptions, key: &GroupKey, tokenizer: &DynTokenizer,
) -> AzResult<Admission> {
    let raw = read_capped(path, opts.max_bytes)?;
    if raw.is_empty() {
        return Err(AzError::Format("empty input".into()));
    }
    // Fail before tokenizing/compressing, not after hours of work in encode_output.
    validate_u32_size(raw.len(), "TOTAL_BYTES")?;
    let name = file_name(path);
    let prep = preprocess_input(&raw, &opts.e8e9_mode, &opts.shuffle_mode, &opts.delta_mode, &name);
    let tokens = tokenizer.encode(&prep.data);
    let meta = EncodeMeta {
        total_bytes: raw.len(),
        crc: crc32::crc32(&prep.data),
        e8e9: prep.e8e9,
        shuffle: prep.shuffle,
        delta: prep.delta,
        backbone: key.backbone,
        profile: key.profile,
        bias_lr: opts.bias_lr,
        ngram_scale: opts.ngram_scale,
        mix_eta: opts.mix_eta,
        emb_surgery: opts.emb_surgery.clone(),
    };
    Ok(Admission::Encode { name, tokens, meta })
}

/// Load the model of a group and report per-stream memory.
fn load_group_model(key: &GroupKey, opts: &BatchOptions, slots: usize) -> AzResult<(DynModel, DynTokenizer)> {
    let quant_override = if opts.quant.is_empty() { None } else { model_loader::parse_quant_flag(&opts.quant) };
    let t_load = Instant::now();
    let (mut model, tokenizer) = model_loader::load_dyn_model(
        key.backbone, &opts.weights_dir, quant_override, key.profile,
    );
    if model.weight_profile() != key.profile {
        return Err(AzError::Format(format!(
            "loaded weight profile {} != required profile {}", model.weight_profile(), key.profile
        )));
    }
    if !key.emb_surgery.is_empty() {
        model.embedding_surgery(&key.emb_surgery);
    }
    let v = model.vocab_size();
    let per_stream = stream_bytes_estimate(model.stream_bytes(), v);
    eprintln!(
        "[lockstep] model loaded in {:.1}s: backbone {}, profile {}, vocab {}; {} streams x ~{:.1} MB \
         (state+scratch {:.1} MB, buffers+CM {:.1} MB) = ~{:.0} MB, {} pool threads",
        t_load.elapsed().as_secs_f64(), key.backbone, key.profile, v, slots,
        per_stream as f64 / 1048576.0, model.stream_bytes() as f64 / 1048576.0,
        (per_stream - model.stream_bytes()) as f64 / 1048576.0,
        per_stream as f64 * slots as f64 / 1048576.0, crate::domain::gemv::pool_threads(),
    );
    Ok((model, tokenizer))
}

/// Per-file report row.
struct Row {
    name: String,
    input: usize,
    output: usize,
    tokens: usize,
    secs: f64,
    ok: bool,
}

impl Row {
    fn from_result(r: &StreamResult, output: usize, ok: bool) -> Row {
        Row { name: r.name.clone(), input: r.input_bytes, output, tokens: r.tokens, secs: r.secs, ok }
    }

    fn failed(name: String) -> Row {
        Row { name, input: 0, output: 0, tokens: 0, secs: 0.0, ok: false }
    }
}

fn report(tag: &str, rows: &[Row], groups: usize, wall: f64, engine_tokens: usize, engine_secs: f64) -> AzResult<()> {
    let compress = tag == "compress";
    eprintln!();
    eprintln!("[{}] results ({} files, {} model group(s), {:.1}s total):", tag, rows.len(), groups, wall);
    eprintln!("  {:34} {:>10} {:>10} {:>8} {:>8} {:>9} {}",
        "file", "input", "output", if compress { "BPB" } else { "" }, "tokens", "tok/s", "status");
    let (mut tin, mut tout, mut ok) = (0usize, 0usize, 0usize);
    for r in rows {
        let bpb = if compress && r.input > 0 { format!("{:8.4}", r.output as f64 * 8.0 / r.input as f64) }
            else { format!("{:>8}", "") };
        eprintln!("  {:34} {:>10} {:>10} {} {:>8} {:>9.1} {}",
            r.name, r.input, r.output, bpb, r.tokens, r.tokens as f64 / r.secs.max(1e-9),
            if r.ok { "OK" } else { "FAIL" });
        tin += r.input;
        tout += r.output;
        ok += r.ok as usize;
    }
    eprintln!("  {:34} {:>10} {:>10}", "TOTAL", tin, tout);
    eprintln!("[{}] {}/{} succeeded; {} tokens in {:.1}s of lockstep = {:.1} tok/s aggregate \
               ({:.1} s/file wall)",
        tag, ok, rows.len(), engine_tokens, engine_secs, engine_tokens as f64 / engine_secs.max(1e-9),
        wall / rows.len().max(1) as f64);
    if ok < rows.len() {
        return Err(AzError::Format(format!("{} files failed", rows.len() - ok)));
    }
    Ok(())
}

/// `compress --input-dir DIR --output-dir DIR --jobs N`: every file becomes
/// `<name>.azth`, byte-identical to `compress --input <file>`.
pub fn compress_dir(
    input_dir: &str, output_dir: &str, opts: &BatchOptions, governor: Option<&HardwareGovernor>,
) -> AzResult<()> {
    let files = list_files(input_dir, &|p| p.extension().map_or(true, |e| e != "azth"))?;
    std::fs::create_dir_all(output_dir)
        .map_err(|e| AzError::Io { path: output_dir.to_string(), source: e })?;
    let slots = opts.slots.clamp(1, MAX_SLOTS);
    eprintln!("[compress] lockstep batch: {} files, {} streams", files.len(), slots);
    let t_wall = Instant::now();

    // Pass 1: routing per file (content type -> backbone, --quant -> profile).
    let mut keyed = Vec::with_capacity(files.len());
    let mut rows: Vec<Row> = Vec::new();
    for p in &files {
        match read_capped(p, opts.max_bytes).and_then(|raw| compress_key(&raw, opts)) {
            Ok(key) => keyed.push((key, p.clone())),
            Err(e) => {
                eprintln!("[compress] FAIL {}: {}", file_name(p), e);
                rows.push(Row::failed(file_name(p)));
            }
        }
    }
    let groups = group_by_key(keyed);
    let (mut eng_tokens, mut eng_secs) = (0usize, 0.0f64);
    let (mut code_secs, mut fwd_secs) = (0.0f64, 0.0f64);

    // Pass 2: one lockstep run per model group.
    for (key, paths) in &groups {
        // No more streams than files: unused slots would only allocate state.
        let group_slots = slots.min(paths.len()).max(1);
        let (model, tokenizer) = load_group_model(key, opts, group_slots)?;
        let mut it = paths.iter();
        let mut source = || it.next().map(|p| compress_admission(p, opts, key, &tokenizer).map_err(|e| named(p, e)));
        let mut on_result = |mut r: StreamResult| {
            let out_name = format!("{}.azth", r.name);
            let output = std::mem::replace(&mut r.output, Ok(Vec::new()));
            let written = output.and_then(|bytes| write_output(output_dir, &out_name, &bytes));
            match written {
                Ok(n) => {
                    eprintln!("\r[compress] {} -> {} ({} -> {} bytes, {:.4} BPB, {} tokens, {:.1} tok/s)   ",
                        r.name, out_name, r.input_bytes, n, n as f64 * 8.0 / r.input_bytes.max(1) as f64,
                        r.tokens, r.tokens as f64 / r.secs.max(1e-9));
                    rows.push(Row::from_result(&r, n, true));
                }
                Err(e) => {
                    eprintln!("\r[compress] FAIL {}: {}", r.name, e);
                    rows.push(Row::from_result(&r, 0, false));
                }
            }
        };
        let st = run_lockstep(&model, &tokenizer, group_slots, governor, &mut source, &mut on_result)?;
        eng_tokens += st.tokens;
        eng_secs += st.secs;
        code_secs += st.code_secs;
        fwd_secs += st.forward_secs;
    }
    let wall = t_wall.elapsed().as_secs_f64();
    eprintln!("[compress] lockstep phases: CM+coder {:.1}s, forward {:.1}s, other {:.1}s",
        code_secs, fwd_secs, eng_secs - code_secs - fwd_secs);
    report("compress", &rows, groups.len(), wall, eng_tokens, eng_secs)
}

/// Parse one archive for grouping: header fields fix backbone and profile.
fn decompress_key(data: &[u8], quant: &str, emb_surgery_cli: &str) -> AzResult<(GroupKey, HeaderInfo)> {
    validate_magic(data)?;
    let hdr = parse_header(data)?;
    if hdr.token_count.is_some() {
        return Err(AzError::Format(format!(
            "AZTH v{} (token-count driven) is not supported in batch mode; use `decompress --input`",
            hdr.version
        )));
    }
    if hdr.skip_threshold > 0 {
        return Err(AzError::Format(
            "adaptive-skip archives are not supported in batch mode; use `decompress --input`".into(),
        ));
    }
    let profile = resolve_decompress_profile(quant, &hdr)?;
    let emb_surgery = if hdr.emb_surgery.is_empty() { emb_surgery_cli.to_string() } else { hdr.emb_surgery.clone() };
    Ok((GroupKey { backbone: hdr.backbone, profile, emb_surgery }, hdr))
}

fn decompress_admission(path: &Path, quant: &str, emb_surgery_cli: &str) -> AzResult<Admission> {
    let data = crate::domain::error::read_file(&path.to_string_lossy())?;
    let (_, hdr) = decompress_key(&data, quant, emb_surgery_cli)?;
    let (stream, stored_crc) = extract_stream_and_crc_v(&data, hdr.header_size, hdr.version);
    let expected_crc = stored_crc
        .ok_or_else(|| AzError::Format("archive without CRC-32 (v2) is not supported in batch mode".into()))?;
    Ok(Admission::Decode { name: file_name(path), stream: stream.to_vec(), expected_crc, hdr })
}

/// `decompress --input-dir DIR --output-dir DIR --jobs N`: every `<name>.azth`
/// is restored to `<name>` (CRC-32 verified), grouped by header backbone/profile.
pub fn decompress_dir(
    input_dir: &str, output_dir: &str, opts: &BatchOptions, governor: Option<&HardwareGovernor>,
) -> AzResult<()> {
    let files = list_files(input_dir, &|p| p.extension().map_or(false, |e| e == "azth"))?;
    std::fs::create_dir_all(output_dir)
        .map_err(|e| AzError::Io { path: output_dir.to_string(), source: e })?;
    let slots = opts.slots.clamp(1, MAX_SLOTS);
    eprintln!("[decompress] lockstep batch: {} archives, {} streams", files.len(), slots);
    let t_wall = Instant::now();

    let mut keyed = Vec::with_capacity(files.len());
    let mut rows: Vec<Row> = Vec::new();
    for p in &files {
        let parsed = crate::domain::error::read_file(&p.to_string_lossy())
            .and_then(|d| decompress_key(&d, &opts.quant, &opts.emb_surgery));
        match parsed {
            Ok((key, _)) => keyed.push((key, p.clone())),
            Err(e) => {
                eprintln!("[decompress] FAIL {}: {}", file_name(p), e);
                rows.push(Row::failed(file_name(p)));
            }
        }
    }
    let groups = group_by_key(keyed);
    let (mut eng_tokens, mut eng_secs) = (0usize, 0.0f64);
    let (mut code_secs, mut fwd_secs) = (0.0f64, 0.0f64);

    for (key, paths) in &groups {
        // No more streams than files: unused slots would only allocate state.
        let group_slots = slots.min(paths.len()).max(1);
        let (model, tokenizer) = load_group_model(key, opts, group_slots)?;
        let mut it = paths.iter();
        let mut source = || {
            it.next().map(|p| decompress_admission(p, &opts.quant, &opts.emb_surgery).map_err(|e| named(p, e)))
        };
        let mut on_result = |mut r: StreamResult| {
            let out_name = r.name.strip_suffix(".azth").unwrap_or(&r.name).to_string();
            let output = std::mem::replace(&mut r.output, Ok(Vec::new()));
            let written = output.and_then(|bytes| write_output(output_dir, &out_name, &bytes));
            match written {
                Ok(n) => {
                    eprintln!("\r[decompress] {} -> {} ({} bytes, CRC-32 verified OK, {} tokens, {:.1} tok/s)   ",
                        r.name, out_name, n, r.tokens, r.tokens as f64 / r.secs.max(1e-9));
                    rows.push(Row::from_result(&r, n, true));
                }
                Err(e) => {
                    eprintln!("\r[decompress] FAIL {}: {}", r.name, e);
                    rows.push(Row::from_result(&r, 0, false));
                }
            }
        };
        let st = run_lockstep(&model, &tokenizer, group_slots, governor, &mut source, &mut on_result)?;
        eng_tokens += st.tokens;
        eng_secs += st.secs;
        code_secs += st.code_secs;
        fwd_secs += st.forward_secs;
    }
    let wall = t_wall.elapsed().as_secs_f64();
    eprintln!("[decompress] lockstep phases: CM+coder {:.1}s, forward {:.1}s, other {:.1}s",
        code_secs, fwd_secs, eng_secs - code_secs - fwd_secs);
    report("decompress", &rows, groups.len(), wall, eng_tokens, eng_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::backbone::BackboneScratch;
    use crate::infrastructure::mamba::byte_tokenizer::ByteTokenizer;

    /// Deterministic toy backbone: a 4-float recurrent state, logits a smooth
    /// function of (state, token). Vocab 256 so ByteTokenizer applies.
    struct MockBackbone;
    struct MockState([f32; 4]);
    struct MockScratch;
    impl BackboneState for MockState {
        fn reset(&mut self) { self.0 = [0.0; 4]; }
    }
    impl BackboneScratch for MockScratch {}
    impl Backbone for MockBackbone {
        type State = MockState;
        type Scratch = MockScratch;
        fn vocab_size(&self) -> usize { 256 }
        fn create_state(&self) -> MockState { MockState([0.0; 4]) }
        fn create_scratch(&self) -> MockScratch { MockScratch }
        fn forward_into(&self, token: usize, st: &mut MockState, _: &mut MockScratch, out: &mut [f32]) {
            let s = &mut st.0;
            s[0] = s[0] * 0.9 + token as f32 * 0.1;
            s[1] = s[1] * 0.5 + (token % 7) as f32;
            s[2] += 1.0;
            s[3] = (s[0] + s[1]) * 0.01;
            for (i, o) in out.iter_mut().enumerate() {
                let d = i as f32 - s[0] * 1.3 - s[1];
                *o = -(d * d) * 0.002 + s[3] * ((i % 3) as f32);
            }
        }
    }

    fn meta(total: usize, crc: u32) -> EncodeMeta {
        EncodeMeta {
            total_bytes: total, crc, e8e9: false, shuffle: 0, delta: 0,
            backbone: BackboneChoice::Mamba, profile: WeightProfile::F32,
            bias_lr: 0.30, ngram_scale: 0.50, mix_eta: 0.01, emb_surgery: String::new(),
        }
    }

    fn inputs() -> Vec<Vec<u8>> {
        let mut v = Vec::new();
        for (len, seed) in [(300usize, 3u32), (37, 5), (900, 7), (1, 9), (120, 11)] {
            let mut s = seed;
            v.push((0..len).map(|_| {
                s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                (s % 40) as u8 + b'a'
            }).collect());
        }
        v
    }

    fn compress_all(slots: usize) -> Vec<Vec<u8>> {
        let tok = ByteTokenizer::new();
        let data = inputs();
        let mut idx = 0;
        let mut source = || {
            let d = data.get(idx)?;
            idx += 1;
            Some(Ok(Admission::Encode {
                name: format!("f{}", idx), tokens: tok.encode(d), meta: meta(d.len(), crc32::crc32(d)),
            }))
        };
        let mut outs: Vec<(String, Vec<u8>)> = Vec::new();
        let mut on_result = |r: StreamResult| outs.push((r.name, r.output.expect("encode ok")));
        let st = run_lockstep(&MockBackbone, &tok, slots, None, &mut source, &mut on_result).unwrap();
        assert_eq!(st.files, data.len());
        outs.sort();
        outs.into_iter().map(|(_, b)| b).collect()
    }

    #[test]
    fn lockstep_output_independent_of_slot_count() {
        let one = compress_all(1);
        for slots in [2usize, 3, 8] {
            assert_eq!(compress_all(slots), one, "slots={}", slots);
        }
        // Lengths differ (300, 37, 900, 1, 120 bytes): slots are refilled mid-run.
        assert!(one.iter().all(|a| a.len() > 13));
    }

    #[test]
    fn lockstep_decode_restores_inputs_for_any_slot_count() {
        let archives = compress_all(3);
        let tok = ByteTokenizer::new();
        for slots in [1usize, 2, 5] {
            let mut idx = 0;
            let mut source = || {
                let a = archives.get(idx)?;
                idx += 1;
                validate_magic(a).unwrap();
                let hdr = parse_header(a).unwrap();
                let (stream, crc) = extract_stream_and_crc_v(a, hdr.header_size, hdr.version);
                Some(Ok(Admission::Decode {
                    name: format!("f{}.azth", idx), stream: stream.to_vec(), expected_crc: crc.unwrap(), hdr,
                }))
            };
            let mut outs: Vec<(String, Vec<u8>)> = Vec::new();
            let mut on_result = |r: StreamResult| outs.push((r.name, r.output.expect("decode ok")));
            run_lockstep(&MockBackbone, &tok, slots, None, &mut source, &mut on_result).unwrap();
            outs.sort();
            let restored: Vec<Vec<u8>> = outs.into_iter().map(|(_, b)| b).collect();
            assert_eq!(restored, inputs(), "slots={}", slots);
        }
    }

    #[test]
    fn corrupted_archive_fails_with_crc_not_panic() {
        let mut archives = compress_all(2);
        let last = archives[0].len() - 1;
        archives[0][last] ^= 0xFF; // flip CRC byte
        let tok = ByteTokenizer::new();
        let a = archives[0].clone();
        let mut given = false;
        let mut source = || {
            if given { return None; }
            given = true;
            let hdr = parse_header(&a).unwrap();
            let (stream, crc) = extract_stream_and_crc_v(&a, hdr.header_size, hdr.version);
            Some(Ok(Admission::Decode { name: "x".into(), stream: stream.to_vec(), expected_crc: crc.unwrap(), hdr }))
        };
        let mut failed = false;
        let mut on_result = |r: StreamResult| failed = matches!(r.output, Err(AzError::Integrity(_)));
        run_lockstep(&MockBackbone, &tok, 2, None, &mut source, &mut on_result).unwrap();
        assert!(failed, "CRC corruption must surface as an Integrity error");
    }

    #[test]
    fn admission_errors_are_reported_and_run_continues() {
        let tok = ByteTokenizer::new();
        let data = inputs();
        let mut idx = 0;
        let mut source = || {
            idx += 1;
            match idx {
                1 => Some(Err(AzError::Format("boom".into()))),
                2 => Some(Ok(Admission::Encode {
                    name: "ok".into(), tokens: tok.encode(&data[0]), meta: meta(data[0].len(), 0),
                })),
                _ => None,
            }
        };
        let mut names = Vec::new();
        let mut on_result = |r: StreamResult| names.push((r.name, r.output.is_ok()));
        let st = run_lockstep(&MockBackbone, &tok, 4, None, &mut source, &mut on_result).unwrap();
        assert_eq!(st.files, 1);
        assert_eq!(names, vec![(String::new(), false), ("ok".to_string(), true)]);
    }

    /// Tokenizer whose ids all decode to zero bytes (models a corrupt stream
    /// that keeps landing on empty/special ids).
    struct EmptyTokenizer;
    impl BackendTokenizer for EmptyTokenizer {
        fn encode(&self, data: &[u8]) -> Vec<u32> { data.iter().map(|&b| b as u32).collect() }
        fn decode_token(&self, _: u32) -> &[u8] { &[] }
    }

    #[test]
    fn zero_length_tokens_terminate_with_error_instead_of_spinning() {
        let a = compress_all(1).swap_remove(0);
        let hdr = parse_header(&a).unwrap();
        let total = hdr.total_bytes;
        let (stream, crc) = extract_stream_and_crc_v(&a, hdr.header_size, hdr.version);
        let mut adm = Some(Admission::Decode {
            name: "z.azth".into(), stream: stream.to_vec(), expected_crc: crc.unwrap(), hdr,
        });
        let mut source = || adm.take().map(Ok);
        let mut res: Vec<(usize, bool)> = Vec::new();
        let mut on_result = |r: StreamResult| res.push((r.tokens, matches!(r.output, Err(AzError::Decode { .. }))));
        let st = run_lockstep(&MockBackbone, &EmptyTokenizer, 2, None, &mut source, &mut on_result).unwrap();
        assert_eq!(res, vec![(total, true)], "must stop after total_bytes tokens with a Decode error");
        assert_eq!(st.steps, total);
    }

    #[test]
    fn write_output_is_atomic_and_rejects_empty_name() {
        let dir = std::env::temp_dir().join(format!("redukt_lockstep_wo_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_string_lossy().to_string();
        assert_eq!(write_output(&d, "a b ñ.azth", b"xyz").unwrap(), 3);
        assert_eq!(std::fs::read(dir.join("a b ñ.azth")).unwrap(), b"xyz");
        assert!(!dir.join("a b ñ.azth.part").exists());
        assert!(write_output(&d, "", b"x").is_err());
        let missing = dir.join("no").join("such").to_string_lossy().to_string();
        assert!(write_output(&missing, "f", b"x").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn group_by_key_merges_same_model_and_keeps_order() {
        let k = |b: BackboneChoice, p: WeightProfile| GroupKey { backbone: b, profile: p, emb_surgery: String::new() };
        let items = vec![
            (k(BackboneChoice::Mamba, WeightProfile::Q8), PathBuf::from("a")),
            (k(BackboneChoice::Rwkv, WeightProfile::F32), PathBuf::from("b")),
            (k(BackboneChoice::Mamba, WeightProfile::Q8), PathBuf::from("c")),
            (k(BackboneChoice::Mamba, WeightProfile::F32), PathBuf::from("d")),
        ];
        let groups = group_by_key(items);
        assert_eq!(groups.len(), 3);
        let q8 = groups.iter().find(|(k, _)| k.profile == WeightProfile::Q8).unwrap();
        assert_eq!(q8.1, vec![PathBuf::from("a"), PathBuf::from("c")]);
        let mut surg = k(BackboneChoice::Rwkv, WeightProfile::F32);
        surg.emb_surgery = "norm".into();
        let plain = (k(BackboneChoice::Rwkv, WeightProfile::F32), PathBuf::from("y"));
        let g2 = group_by_key(vec![(surg, PathBuf::from("x")), plain]);
        assert_eq!(g2.len(), 2, "embedding surgery changes the weights -> separate group");
    }

    #[test]
    fn stream_estimate_is_plausible_for_mamba_130m() {
        let mb = stream_bytes_estimate(24 * 1536 * 19 * 4, 50280) as f64 / 1048576.0;
        assert!(mb > 4.0 && mb < 8.0, "{} MB", mb);
    }

}
