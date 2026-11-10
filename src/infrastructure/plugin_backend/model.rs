//! GPU plugin backend — Backbone trait implementation (R-074).
//!
//! Wraps PluginLibrary + BackendVTable into Backbone/BackendTokenizer impls.
//! PluginTokenizer: identity byte-level (vocab 256, MambaByte-style models) or the
//! GPT-NeoX BPE tokenizer of mamba-130m (vocab 50280), selected from the `.rbin`
//! header (v2 tokenizer field, or vocab size for v1 files).
//! FFI v2 batch API (R-096): `batch_init` / `batch_reset` / `forward_batch`, exposed
//! only when the loaded plugin implements it.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::domain::backbone::{
    Backbone, BackboneState, BackboneScratch, BackendTokenizer,
};
use crate::domain::backend_ffi::{
    validate_batch_args, BackendInfo, BackendResult, BackendResetFn, BackendVTableV2,
};
use crate::infrastructure::mamba::tokenizer::BpeTokenizer;
use super::loader::PluginLibrary;

// ---------------------------------------------------------------------------
// State & Scratch (opaque — plugin manages GPU state internally)
// ---------------------------------------------------------------------------

/// Recurrent state managed by the plugin (opaque from host side).
///
/// R-109: with the FFI v2 batch API every state owns a device-side **slot** of the pool
/// allocated by `prepare_streams`, so N lockstep streams are independent. Without a pool
/// (single-file path, or a v1-only plugin) the state is the plugin's process-global v1
/// state and `slot` is `None`.
pub struct PluginState {
    reset_fn: BackendResetFn,
    slot: Option<u32>,
    v2: Option<BackendVTableV2>,
}

// SAFETY: Plugin functions are process-global, called single-threaded.
unsafe impl Send for PluginState {}

impl BackboneState for PluginState {
    fn reset(&mut self) {
        match (self.slot, self.v2.as_ref()) {
            (Some(slot), Some(v2)) => {
                let slots = [slot];
                // SAFETY: the vtable fn is valid while the library is loaded (the backend that
                // created this state keeps it alive); `slots` is a live 1-element array.
                let code = unsafe { (v2.batch_reset)(slots.as_ptr(), 1) };
                if BackendResult::from_code(code) != BackendResult::Ok {
                    panic!("[plugin] batch_reset(slot {}) failed: {}", slot, code);
                }
            }
            // SAFETY: process-global v1 reset, no caller memory.
            _ => unsafe { (self.reset_fn)() },
        }
    }
}

/// Host-side bookkeeping of the FFI v2 slot pool (R-109): `init(n)` after `batch_init(n)`,
/// then one `take()` per `create_state`.
#[derive(Default)]
struct SlotPool {
    cap: u32,
    next: u32,
    seen: Vec<bool>,
}

impl SlotPool {
    fn init(&mut self, cap: u32) {
        self.cap = cap;
        self.next = 0;
        self.seen = vec![false; cap as usize];
    }

    /// Next free slot, or None when the pool is exhausted (or absent).
    fn take(&mut self) -> Option<u32> {
        if self.next < self.cap {
            self.next += 1;
            Some(self.next - 1)
        } else {
            None
        }
    }
}

/// Scratch workspace — unused (plugin manages its own buffers).
pub struct PluginScratch;

impl BackboneScratch for PluginScratch {}

// ---------------------------------------------------------------------------
// .rbin header metadata & tokenizer selection
// ---------------------------------------------------------------------------

/// "RGPU" little-endian — must match scripts/convert-weights-gpu.py and mamba_cuda.cu.
const RBIN_MAGIC: u32 = 0x5550_4752;
const RBIN_HEADER_LEN: usize = 44;
const RBIN_TOK_BYTE: u32 = 0;
const RBIN_TOK_BPE: u32 = 1;
/// Default location of the mamba-130m `tokenizer.json` (same as the CPU Mamba path).
const BPE_FALLBACK_DIR: &str = "weights/mamba-130m";

/// Metadata read from the first 44 bytes of a `.rbin` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RbinMeta {
    pub version: u32,
    pub vocab_size: u32,
    /// Explicit tokenizer kind (v2 header word 10); `None` for v1 files.
    pub tokenizer: Option<u32>,
}

/// Tokenizer family used by the plugin backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    Byte,
    Bpe,
}

/// Fully resolved tokenizer choice (BPE carries the directory of `tokenizer.json`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSpec {
    Byte,
    Bpe(PathBuf),
}

/// Tokenizer chosen by the last successful `PluginBackend::try_new`.  The plugin is a
/// process-global singleton (one loaded model per process), so this mirrors that.
/// It lets `PluginTokenizer::new()` (called by the model loader right after
/// `try_new`) build the matching tokenizer without changing its signature.
static LAST_SPEC: Mutex<Option<TokenizerSpec>> = Mutex::new(None);

fn u32_le(b: &[u8], word: usize) -> u32 {
    let o = word * 4;
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Parse a `.rbin` header (v1 or v2).  Returns `None` on bad magic/version/length.
pub fn parse_rbin_header(bytes: &[u8]) -> Option<RbinMeta> {
    if bytes.len() < RBIN_HEADER_LEN || u32_le(bytes, 0) != RBIN_MAGIC {
        return None;
    }
    let version = u32_le(bytes, 1);
    let tokenizer = match version {
        1 => None,
        2 => Some(u32_le(bytes, 10)),
        _ => return None,
    };
    Some(RbinMeta { version, vocab_size: u32_le(bytes, 2), tokenizer })
}

/// Same path rule as the plugin: `<weights>/model.rbin` unless it already ends in `.rbin`.
pub fn rbin_path(weights: &str) -> PathBuf {
    if weights.ends_with(".rbin") {
        PathBuf::from(weights)
    } else {
        Path::new(weights).join("model.rbin")
    }
}

fn read_rbin_meta(path: &Path) -> Option<RbinMeta> {
    use std::io::Read;
    let mut buf = [0u8; RBIN_HEADER_LEN];
    let mut f = std::fs::File::open(path).ok()?;
    f.read_exact(&mut buf).ok()?;
    parse_rbin_header(&buf)
}

/// Decide the tokenizer family from the header (if readable) and the plugin's vocab.
pub fn select_tokenizer_kind(
    meta: Option<RbinMeta>, info_vocab: u32,
) -> Result<TokenizerKind, String> {
    if let Some(m) = meta {
        if m.vocab_size != info_vocab {
            return Err(format!(
                "rbin vocab {} != plugin vocab {}", m.vocab_size, info_vocab,
            ));
        }
    }
    match meta.and_then(|m| m.tokenizer) {
        Some(RBIN_TOK_BYTE) if info_vocab == 256 => Ok(TokenizerKind::Byte),
        Some(RBIN_TOK_BPE) if info_vocab > 256 => Ok(TokenizerKind::Bpe),
        Some(t) => Err(format!("rbin tokenizer {} incompatible with vocab {}", t, info_vocab)),
        None if info_vocab == 256 => Ok(TokenizerKind::Byte),
        None if info_vocab > 256 => Ok(TokenizerKind::Bpe),
        None => Err(format!("unsupported plugin vocab {}", info_vocab)),
    }
}

/// First candidate directory that contains `tokenizer.json`.
fn find_bpe_dir(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|d| d.join("tokenizer.json").is_file()).cloned()
}

/// Resolve the tokenizer for `weights` (dir with model.rbin, or a .rbin path).
/// BPE looks for `tokenizer.json` next to the `.rbin`, then in `weights/mamba-130m`.
pub fn resolve_tokenizer(weights: &str, info_vocab: u32) -> Result<TokenizerSpec, String> {
    let rbin = rbin_path(weights);
    let kind = select_tokenizer_kind(read_rbin_meta(&rbin), info_vocab)?;
    if kind == TokenizerKind::Byte {
        return Ok(TokenizerSpec::Byte);
    }
    let mut candidates = Vec::new();
    if let Some(parent) = rbin.parent() {
        candidates.push(parent.to_path_buf());
    }
    candidates.push(PathBuf::from(BPE_FALLBACK_DIR));
    find_bpe_dir(&candidates)
        .map(TokenizerSpec::Bpe)
        .ok_or_else(|| format!(
            "BPE plugin model (vocab {}) needs tokenizer.json next to {} or in {}",
            info_vocab, rbin.display(), BPE_FALLBACK_DIR,
        ))
}

fn set_last_spec(spec: TokenizerSpec) {
    if let Ok(mut g) = LAST_SPEC.lock() {
        *g = Some(spec);
    }
}

fn last_spec() -> TokenizerSpec {
    LAST_SPEC.lock().ok().and_then(|g| g.clone()).unwrap_or(TokenizerSpec::Byte)
}

// ---------------------------------------------------------------------------
// PluginBackend
// ---------------------------------------------------------------------------

/// GPU backend wrapping a loaded plugin via FFI.
pub struct PluginBackend {
    _lib: PluginLibrary,
    info: BackendInfo,
    forward_fn: crate::domain::backend_ffi::BackendForwardFn,
    reset_fn: BackendResetFn,
    tokenizer: TokenizerSpec,
    v2: Option<BackendVTableV2>,
    pool: Mutex<SlotPool>,
    /// Row-major `n * vocab` logits buffer reused by `forward_multi` (R-109).
    multi_buf: Mutex<Vec<f32>>,
}

// SAFETY: Plugin functions are process-global, called single-threaded in codec.
unsafe impl Send for PluginBackend {}
unsafe impl Sync for PluginBackend {}

impl PluginBackend {
    /// Try to load and initialize the GPU plugin.
    /// Returns None if no plugin found, initialization fails, or the matching
    /// tokenizer cannot be resolved (reason printed to stderr).
    pub fn try_new(weights_path: &str, device_id: i32) -> Option<Self> {
        let lib = PluginLibrary::try_load()?;
        let info = lib.init(weights_path, device_id).ok()?;
        let tokenizer = match resolve_tokenizer(weights_path, info.vocab_size) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[plugin] tokenizer selection failed: {}", e);
                return None;
            }
        };
        let v2 = lib.vtable_v2.clone();
        eprintln!(
            "[plugin] loaded: {} (vocab={}, d_model={}, tokenizer={:?}, abi={})",
            info.name_str(), info.vocab_size, info.hidden_dim, tokenizer,
            v2.as_ref().map_or(1, |v| v.abi_version),
        );
        set_last_spec(tokenizer.clone());
        Some(Self {
            forward_fn: lib.vtable.forward,
            reset_fn: lib.vtable.reset_state,
            _lib: lib,
            info,
            tokenizer,
            v2,
            pool: Mutex::new(SlotPool::default()),
            multi_buf: Mutex::new(Vec::new()),
        })
    }

    /// Backend information.
    pub fn info(&self) -> &BackendInfo { &self.info }
}

/// FFI v2 batch path (R-096), wired into the lockstep codec by R-109 (`prepare_streams`,
/// slot states, `forward_multi`).
impl PluginBackend {
    /// Tokenizer selected for this plugin model.
    #[allow(dead_code)] // diagnostic accessor used by tests/notebooks
    pub fn tokenizer_spec(&self) -> &TokenizerSpec { &self.tokenizer }

    /// ABI revision if the plugin implements the batch API, else `None` (v1 only).
    pub fn batch_abi(&self) -> Option<u32> { self.v2.as_ref().map(|v| v.abi_version) }

    /// Allocate `max_streams` zeroed state slots on the device (0 releases the pool) and
    /// restart slot assignment: the next `max_streams` `create_state` calls get slots 0..N.
    pub fn batch_init(&self, max_streams: u32) -> Result<(), BackendResult> {
        let v2 = self.v2.as_ref().ok_or(BackendResult::VersionMismatch)?;
        // SAFETY: `v2` was resolved from the loaded library, which `self` keeps alive, so the
        // fn pointer is valid; batch_init takes only a scalar and no caller memory.
        let code = unsafe { (v2.batch_init)(max_streams) };
        match BackendResult::from_code(code) {
            BackendResult::Ok => {
                lock(&self.pool).init(max_streams);
                Ok(())
            }
            err => Err(err),
        }
    }

    /// One step for `slots.len()` streams; `out` = `n * vocab` logits, row b = slots[b].
    pub fn forward_batch(
        &self, slots: &[u32], tokens: &[u32], out: &mut [f32],
    ) -> Result<(), BackendResult> {
        let v2 = self.v2.as_ref().ok_or(BackendResult::VersionMismatch)?;
        let vocab = self.info.vocab_size;
        {
            let mut pool = lock(&self.pool);
            let cap = pool.cap;
            validate_batch_args(slots, tokens, out.len(), vocab, cap, &mut pool.seen)?;
        }
        if slots.is_empty() {
            return Ok(());
        }
        // SAFETY: the vtable pointer is valid while the library is loaded. validate_batch_args
        // checked slots.len() == tokens.len(), out.len() == n * vocab, slots < batch_cap and
        // no duplicate slots, so all three pointers cover the lengths the plugin reads/writes.
        let code = unsafe {
            (v2.forward_batch)(
                slots.len() as u32, slots.as_ptr(), tokens.as_ptr(), out.as_mut_ptr(), vocab,
            )
        };
        match BackendResult::from_code(code) {
            BackendResult::Ok => Ok(()),
            err => Err(err),
        }
    }
}

/// Lock a poisoned-tolerant mutex (the plugin calls never unwind across it).
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Backbone for PluginBackend {
    type State = PluginState;
    type Scratch = PluginScratch;

    fn vocab_size(&self) -> usize { self.info.vocab_size as usize }

    /// After `prepare_streams(n)` the next `n` states own device slots 0..n (independent
    /// streams); otherwise the state is the plugin's process-global v1 state.
    fn create_state(&self) -> PluginState {
        let slot = if self.v2.is_some() { lock(&self.pool).take() } else { None };
        PluginState { reset_fn: self.reset_fn, slot, v2: self.v2.clone() }
    }

    fn create_scratch(&self) -> PluginScratch { PluginScratch }

    fn forward_into(
        &self, token: usize, state: &mut PluginState,
        _scratch: &mut PluginScratch, out_logits: &mut [f32],
    ) {
        if let Some(slot) = state.slot {
            // Slot state: v2 batch of one. The FFI v2 contract makes v1 and v2 bit-identical.
            if let Err(e) = self.forward_batch(&[slot], &[token as u32], out_logits) {
                panic!("[plugin] forward_batch(slot {}) failed: {:?}", slot, e);
            }
            return;
        }
        // SAFETY: process-global v1 forward; `out_logits` is a live slice of `len` f32.
        let result = unsafe {
            (self.forward_fn)(
                token as u32,
                out_logits.as_mut_ptr(),
                out_logits.len() as u32,
            )
        };
        if result != BackendResult::Ok {
            panic!("[plugin] forward failed: {:?}", result);
        }
    }

    /// R-109: one `forward_batch` call for every slot state (sequential v1 fallback only for
    /// states without a slot, i.e. when `prepare_streams` was not called).
    fn forward_multi(
        &self, tokens: &[usize], states: &mut [&mut PluginState],
        scratches: &mut [&mut PluginScratch], out_logits: &mut [&mut [f32]],
    ) {
        let n = tokens.len();
        let slots: Option<Vec<u32>> = states.iter().map(|s| s.slot).collect();
        let slots = match slots {
            Some(s) if n > 0 => s,
            _ => {
                for i in 0..n {
                    self.forward_into(tokens[i], states[i], scratches[i], out_logits[i]);
                }
                return;
            }
        };
        let vocab = self.info.vocab_size as usize;
        let toks: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let mut buf = lock(&self.multi_buf);
        buf.resize(n * vocab, 0.0);
        if let Err(e) = self.forward_batch(&slots, &toks, &mut buf[..n * vocab]) {
            panic!("[plugin] forward_batch({} streams) failed: {:?}", n, e);
        }
        for (i, out) in out_logits.iter_mut().enumerate() {
            out.copy_from_slice(&buf[i * vocab..(i + 1) * vocab]);
        }
    }

    /// R-109: allocate the device slot pool for `n` lockstep streams. A v1-only plugin can
    /// serve exactly one stream (its single global state).
    fn prepare_streams(&self, n: usize) -> Result<(), String> {
        match (self.v2.as_ref(), n) {
            (_, 0 | 1) if self.v2.is_none() => Ok(()),
            (None, _) => Err(format!(
                "the GPU plugin implements FFI v1 only (one global state): --jobs {} needs the v2 batch API", n
            )),
            (Some(_), _) => {
                let n = u32::try_from(n).map_err(|_| "too many streams".to_string())?;
                self.batch_init(n).map_err(|e| format!("plugin batch_init({}) failed: {:?}", n, e))?;
                eprintln!("[plugin] batch pool: {} slots (FFI v2, abi {})", n, self.batch_abi().unwrap_or(2));
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PluginTokenizer — identity byte-level (vocab 256) or GPT-NeoX BPE (vocab 50280)
// ---------------------------------------------------------------------------

/// Tokenizer for plugin backends.  Byte mode maps each byte to token 0..255; BPE
/// mode delegates to the mamba-130m `BpeTokenizer` (identical to the CPU Mamba path).
pub struct PluginTokenizer {
    byte_table: [[u8; 1]; 256],
    bpe: Option<BpeTokenizer>,
}

impl PluginTokenizer {
    /// Tokenizer matching the last successfully initialized plugin (byte-level if no
    /// plugin has been initialized in this process).
    pub fn new() -> Self {
        Self::from_spec(&last_spec())
    }

    /// Build a tokenizer for an explicit spec.  The BPE directory was checked to
    /// contain `tokenizer.json` by `resolve_tokenizer`.
    pub fn from_spec(spec: &TokenizerSpec) -> Self {
        let mut table = [[0u8; 1]; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            slot[0] = i as u8;
        }
        let bpe = match spec {
            TokenizerSpec::Byte => None,
            TokenizerSpec::Bpe(dir) => Some(BpeTokenizer::load(dir)),
        };
        Self { byte_table: table, bpe }
    }
}

impl BackendTokenizer for PluginTokenizer {
    fn encode(&self, data: &[u8]) -> Vec<u32> {
        match &self.bpe {
            Some(t) => t.encode(data),
            None => data.iter().map(|&b| b as u32).collect(),
        }
    }

    fn decode_token(&self, id: u32) -> &[u8] {
        match &self.bpe {
            Some(t) => t.decode_token(id),
            None => &self.byte_table[(id as usize) & 0xFF],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn header(version: u32, vocab: u32, word10: u32) -> Vec<u8> {
        let words = [RBIN_MAGIC, version, vocab, 768, 1536, 16, 4, 48, 24, 0, word10];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("redukt-plugin-{}-{}", tag, std::process::id()));
        std::fs::create_dir_all(&d).expect("create temp dir");
        d
    }

    #[test]
    fn slot_pool_hands_out_each_slot_once_then_none() {
        let mut p = SlotPool::default();
        assert_eq!(p.take(), None); // no pool
        p.init(3);
        assert_eq!((p.take(), p.take(), p.take(), p.take()), (Some(0), Some(1), Some(2), None));
        assert_eq!(p.seen.len(), 3);
        p.init(1); // re-init restarts assignment
        assert_eq!((p.take(), p.take()), (Some(0), None));
    }

    #[test]
    fn plugin_tokenizer_roundtrip() {
        let tok = PluginTokenizer::from_spec(&TokenizerSpec::Byte);
        let data = b"Hello\x00\xFF world";
        let encoded = tok.encode(data);
        let mut decoded = Vec::new();
        for id in &encoded {
            decoded.extend_from_slice(tok.decode_token(*id));
        }
        assert_eq!(decoded, data);
    }

    #[test]
    fn plugin_tokenizer_all_bytes() {
        let tok = PluginTokenizer::from_spec(&TokenizerSpec::Byte);
        for b in 0u8..=255 {
            let enc = tok.encode(&[b]);
            assert_eq!(enc, vec![b as u32]);
            assert_eq!(tok.decode_token(b as u32), &[b]);
        }
    }

    #[test]
    fn new_defaults_to_byte_without_plugin() {
        // No plugin initialized in the test process -> byte-level identity.
        let tok = PluginTokenizer::new();
        assert_eq!(tok.encode(b"ab"), vec![97, 98]);
    }

    #[test]
    fn try_new_returns_none_without_plugin() {
        let result = PluginBackend::try_new("weights/mamba-130m", 0);
        assert!(result.is_none());
    }

    #[test]
    fn parse_header_v1_and_v2() {
        let v1 = parse_rbin_header(&header(1, 50280, 0)).expect("v1");
        assert_eq!(v1, RbinMeta { version: 1, vocab_size: 50280, tokenizer: None });
        let v2 = parse_rbin_header(&header(2, 256, RBIN_TOK_BYTE)).expect("v2");
        assert_eq!(v2.tokenizer, Some(RBIN_TOK_BYTE));
        assert!(parse_rbin_header(&header(3, 256, 0)).is_none());
        assert!(parse_rbin_header(&header(1, 256, 0)[..40]).is_none());
        let mut bad = header(1, 256, 0);
        bad[0] ^= 0xFF;
        assert!(parse_rbin_header(&bad).is_none());
    }

    #[test]
    fn select_by_vocab_for_v1_or_missing_header() {
        let v1 = parse_rbin_header(&header(1, 50280, 0));
        assert_eq!(select_tokenizer_kind(v1, 50280), Ok(TokenizerKind::Bpe));
        let v1b = parse_rbin_header(&header(1, 256, 0));
        assert_eq!(select_tokenizer_kind(v1b, 256), Ok(TokenizerKind::Byte));
        assert_eq!(select_tokenizer_kind(None, 256), Ok(TokenizerKind::Byte));
        assert_eq!(select_tokenizer_kind(None, 50280), Ok(TokenizerKind::Bpe));
        assert!(select_tokenizer_kind(None, 100).is_err());
    }

    #[test]
    fn select_by_v2_field_and_reject_inconsistent() {
        let bpe = parse_rbin_header(&header(2, 50280, RBIN_TOK_BPE));
        assert_eq!(select_tokenizer_kind(bpe, 50280), Ok(TokenizerKind::Bpe));
        let byte = parse_rbin_header(&header(2, 256, RBIN_TOK_BYTE));
        assert_eq!(select_tokenizer_kind(byte, 256), Ok(TokenizerKind::Byte));
        // byte tokenizer declared with a BPE-sized vocab
        let wrong = parse_rbin_header(&header(2, 50280, RBIN_TOK_BYTE));
        assert!(select_tokenizer_kind(wrong, 50280).is_err());
        // unknown tokenizer kind
        let unknown = parse_rbin_header(&header(2, 50280, 7));
        assert!(select_tokenizer_kind(unknown, 50280).is_err());
        // header vocab disagrees with what the plugin reports
        assert!(select_tokenizer_kind(bpe, 256).is_err());
    }

    #[test]
    fn rbin_path_rule_matches_plugin() {
        assert_eq!(rbin_path("w/m.rbin"), PathBuf::from("w/m.rbin"));
        assert_eq!(rbin_path("w/mamba"), Path::new("w/mamba").join("model.rbin"));
    }

    #[test]
    fn resolve_byte_from_v2_file() {
        let dir = temp_dir("byte");
        std::fs::write(dir.join("model.rbin"), header(2, 256, RBIN_TOK_BYTE)).expect("write");
        let spec = resolve_tokenizer(dir.to_str().expect("utf8"), 256);
        assert_eq!(spec, Ok(TokenizerSpec::Byte));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_bpe_prefers_tokenizer_next_to_rbin() {
        let dir = temp_dir("bpe");
        std::fs::write(dir.join("model.rbin"), header(2, 50280, RBIN_TOK_BPE)).expect("write");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write");
        let spec = resolve_tokenizer(dir.to_str().expect("utf8"), 50280);
        assert_eq!(spec, Ok(TokenizerSpec::Bpe(dir.clone())));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_bpe_dir_skips_missing_candidates() {
        let dir = temp_dir("cand");
        std::fs::write(dir.join("tokenizer.json"), b"{}").expect("write");
        let missing = dir.join("does-not-exist");
        assert_eq!(find_bpe_dir(&[missing.clone(), dir.clone()]), Some(dir.clone()));
        assert_eq!(find_bpe_dir(&[missing]), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bpe_mode_matches_cpu_mamba_tokenizer() {
        // Only runs where the mamba-130m tokenizer is present (dev machine / Kaggle).
        let dir = PathBuf::from(BPE_FALLBACK_DIR);
        if !dir.join("tokenizer.json").is_file() {
            return;
        }
        let plugin_tok = PluginTokenizer::from_spec(&TokenizerSpec::Bpe(dir.clone()));
        let cpu_tok = BpeTokenizer::load(&dir);
        let data = b"<page><title>Redukt</title> lossless \xC0\xFF bytes</page>";
        let ids = plugin_tok.encode(data);
        assert_eq!(ids, cpu_tok.encode(data));
        assert!(ids.len() < data.len());
        assert!(ids.iter().all(|&id| id < 50280));
        let decoded: Vec<u8> = ids.iter().flat_map(|&id| plugin_tok.decode_token(id).to_vec()).collect();
        assert_eq!(decoded, data);
    }
}
