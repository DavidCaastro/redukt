//! Backbone abstraction — zero-cost trait interface for neural language models.
//!
//! Decouples the compression pipeline from any specific model implementation.
//! Implemented by RWKV-7 (byte-level) and Mamba (BPE). Router selects per content type.

use std::fmt;

/// Which backbone to use for compression (R-039).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneChoice {
    Rwkv,
    Mamba,
}

impl BackboneChoice {
    /// Encode as header byte: 0=RWKV, 1=Mamba.
    pub fn header_byte(self) -> u8 {
        match self { Self::Rwkv => 0, Self::Mamba => 1 }
    }

    /// Decode from header byte.
    pub fn from_header_byte(b: u8) -> Option<Self> {
        match b { 0 => Some(Self::Rwkv), 1 => Some(Self::Mamba), _ => None }
    }

    /// Parse CLI string ("rwkv", "mamba", "auto").
    /// Returns None for "auto" (caller decides).
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rwkv" => Some(Self::Rwkv),
            "mamba" => Some(Self::Mamba),
            _ => None,
        }
    }
}

impl fmt::Display for BackboneChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rwkv => write!(f, "rwkv"),
            Self::Mamba => write!(f, "mamba"),
        }
    }
}

/// Route content type to backbone (R-039, updated R-040).
/// Mamba-130M wins 10/12 file types. RWKV only wins on GenericBinary (+4.6%).
pub fn route_backbone(ct: &crate::domain::content_detect::ContentType) -> BackboneChoice {
    use crate::domain::content_detect::ContentType;
    match ct {
        ContentType::GenericBinary => BackboneChoice::Rwkv,
        _ => BackboneChoice::Mamba,
    }
}

/// Recurrent state for a backbone model.
pub trait BackboneState: Send {
    /// Reset all state to zeros (for batch reuse without reallocation).
    fn reset(&mut self);
}

/// Pre-allocated scratch workspace for zero-alloc forward passes.
pub trait BackboneScratch {
    /// Reset performance counters (default no-op).
    fn reset_perf(&mut self) {}
}

/// Neural language model backbone — single-step RNN-mode inference.
pub trait Backbone: Sync {
    type State: BackboneState;
    type Scratch: BackboneScratch;

    /// Vocabulary size (number of output logits).
    fn vocab_size(&self) -> usize;

    /// Create a fresh zero-initialized state.
    fn create_state(&self) -> Self::State;

    /// Create a pre-allocated scratch workspace.
    fn create_scratch(&self) -> Self::Scratch;

    /// Single-step forward pass: token in, logits out.
    fn forward_into(
        &self, token: usize, state: &mut Self::State,
        scratch: &mut Self::Scratch, out_logits: &mut [f32],
    );
}

/// Tokenizer backend — encode bytes to tokens and decode back.
pub trait BackendTokenizer: Sync {
    fn encode(&self, data: &[u8]) -> Vec<u32>;
    fn decode_token(&self, id: u32) -> &[u8];
}
