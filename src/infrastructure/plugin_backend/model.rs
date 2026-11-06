//! GPU plugin backend — Backbone trait implementation (R-074).
//!
//! Wraps PluginLibrary + BackendVTable into Backbone/BackendTokenizer impls.
//! PluginTokenizer: identity byte-level (vocab 256) for MambaByte-style models.

use crate::domain::backbone::{
    Backbone, BackboneState, BackboneScratch, BackendTokenizer,
};
use crate::domain::backend_ffi::{BackendInfo, BackendResult, BackendResetFn};
use super::loader::PluginLibrary;

// ---------------------------------------------------------------------------
// State & Scratch (opaque — plugin manages GPU state internally)
// ---------------------------------------------------------------------------

/// Recurrent state managed by the plugin (opaque from host side).
pub struct PluginState {
    reset_fn: BackendResetFn,
}

// SAFETY: Plugin functions are process-global, called single-threaded.
unsafe impl Send for PluginState {}

impl BackboneState for PluginState {
    fn reset(&mut self) {
        unsafe { (self.reset_fn)(); }
    }
}

/// Scratch workspace — unused (plugin manages its own buffers).
pub struct PluginScratch;

impl BackboneScratch for PluginScratch {}

// ---------------------------------------------------------------------------
// PluginBackend
// ---------------------------------------------------------------------------

/// GPU backend wrapping a loaded plugin via FFI.
pub struct PluginBackend {
    _lib: PluginLibrary,
    info: BackendInfo,
    forward_fn: crate::domain::backend_ffi::BackendForwardFn,
    reset_fn: BackendResetFn,
}

// SAFETY: Plugin functions are process-global, called single-threaded in codec.
unsafe impl Send for PluginBackend {}
unsafe impl Sync for PluginBackend {}

impl PluginBackend {
    /// Try to load and initialize the GPU plugin.
    /// Returns None if no plugin found or initialization fails.
    pub fn try_new(weights_path: &str, device_id: i32) -> Option<Self> {
        let lib = PluginLibrary::try_load()?;
        let info = lib.init(weights_path, device_id).ok()?;
        let forward_fn = lib.vtable.forward;
        let reset_fn = lib.vtable.reset_state;
        eprintln!(
            "[plugin] loaded: {} (vocab={}, d_model={})",
            info.name_str(), info.vocab_size, info.hidden_dim,
        );
        Some(Self { _lib: lib, info, forward_fn, reset_fn })
    }

    /// Backend information.
    pub fn info(&self) -> &BackendInfo { &self.info }
}

impl Backbone for PluginBackend {
    type State = PluginState;
    type Scratch = PluginScratch;

    fn vocab_size(&self) -> usize { self.info.vocab_size as usize }

    fn create_state(&self) -> PluginState {
        PluginState { reset_fn: self.reset_fn }
    }

    fn create_scratch(&self) -> PluginScratch { PluginScratch }

    fn forward_into(
        &self, token: usize, _state: &mut PluginState,
        _scratch: &mut PluginScratch, out_logits: &mut [f32],
    ) {
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
}

// ---------------------------------------------------------------------------
// PluginTokenizer — identity byte-level (vocab 256)
// ---------------------------------------------------------------------------

/// Identity byte-level tokenizer for plugin backends.
/// Each byte maps to token ID 0..255.
pub struct PluginTokenizer {
    byte_table: [[u8; 1]; 256],
}

impl PluginTokenizer {
    pub fn new() -> Self {
        let mut table = [[0u8; 1]; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            slot[0] = i as u8;
        }
        Self { byte_table: table }
    }
}

impl BackendTokenizer for PluginTokenizer {
    fn encode(&self, data: &[u8]) -> Vec<u32> {
        data.iter().map(|&b| b as u32).collect()
    }

    fn decode_token(&self, id: u32) -> &[u8] {
        &self.byte_table[(id as usize) & 0xFF]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_tokenizer_roundtrip() {
        let tok = PluginTokenizer::new();
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
        let tok = PluginTokenizer::new();
        for b in 0u8..=255 {
            let enc = tok.encode(&[b]);
            assert_eq!(enc, vec![b as u32]);
            assert_eq!(tok.decode_token(b as u32), &[b]);
        }
    }

    #[test]
    fn try_new_returns_none_without_plugin() {
        let result = PluginBackend::try_new("weights/mamba-130m", 0);
        assert!(result.is_none());
    }
}
