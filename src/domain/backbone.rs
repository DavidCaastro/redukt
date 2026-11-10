//! Backbone abstraction — zero-cost trait interface for neural language models.
//!
//! Decouples the compression pipeline from any specific model implementation.
//! Implemented by RWKV-7 (byte-level), Mamba (BPE), and MambaByte (byte-level).
//! Router selects per content type.

use std::fmt;

/// Which backbone to use for compression (R-039, R-082).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneChoice {
    Rwkv,
    Mamba,
    MambaByte,
    Plugin,
}

impl BackboneChoice {
    /// Encode as header byte: 0=RWKV, 1=Mamba, 2=MambaByte, 3=Plugin.
    pub fn header_byte(self) -> u8 {
        match self {
            Self::Rwkv => 0,
            Self::Mamba => 1,
            Self::MambaByte => 2,
            Self::Plugin => 3,
        }
    }

    /// Decode from header byte.
    pub fn from_header_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Rwkv),
            1 => Some(Self::Mamba),
            2 => Some(Self::MambaByte),
            3 => Some(Self::Plugin),
            _ => None,
        }
    }

    /// Parse CLI string ("rwkv", "mamba", "mambabyte", "plugin", "auto").
    /// Returns None for "auto" (caller decides).
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rwkv" => Some(Self::Rwkv),
            "mamba" => Some(Self::Mamba),
            "mambabyte" => Some(Self::MambaByte),
            "plugin" => Some(Self::Plugin),
            _ => None,
        }
    }
}

impl fmt::Display for BackboneChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rwkv => write!(f, "rwkv"),
            Self::Mamba => write!(f, "mamba"),
            Self::MambaByte => write!(f, "mambabyte"),
            Self::Plugin => write!(f, "plugin"),
        }
    }
}

/// Numeric weight profile of the Mamba-family backbone (R-099).
/// Stored in the AZTH v5 PROFILE byte (bits 0-3). Decompression requires the
/// identical profile: it changes the logits, hence the coded stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightProfile {
    /// All weights f32 (AZTH v4 compatible, default).
    F32,
    /// Weight-only int8, per-row f32 scale, on in_proj/out_proj/embedding/head.
    /// Activations and all other tensors stay f32.
    Q8,
}

impl WeightProfile {
    /// PROFILE quant code: 0=F32, 1=Q8 (2=Q6 reserved, 3-15 reserved).
    pub fn header_code(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::Q8 => 1,
        }
    }

    /// Decode PROFILE quant code. None for reserved/unknown codes.
    pub fn from_header_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(Self::F32),
            1 => Some(Self::Q8),
            _ => None,
        }
    }

    /// Parse a `--quant` value for the Mamba family.
    /// `Ok(None)` = "auto"/unspecified (caller decides), `Err` = unsupported level.
    pub fn from_cli(s: &str) -> Result<Option<Self>, String> {
        match s.to_lowercase().as_str() {
            "" | "auto" => Ok(None),
            "f32" => Ok(Some(Self::F32)),
            "q8" => Ok(Some(Self::Q8)),
            other => Err(format!(
                "--quant {} not supported for the Mamba backbone (use f32, q8 or auto)", other
            )),
        }
    }
}

impl fmt::Display for WeightProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::Q8 => write!(f, "Q8"),
        }
    }
}

/// Route content type to backbone (R-039, updated R-040, simplified post-BUG-001).
/// Mamba-130M wins 10/12 file types. RWKV marginal on GenericBinary (+4.6%),
/// but samba misclassified as GenericBinary lost -79% with RWKV routing.
/// Simplified: Mamba always. Users can override with --backbone rwkv.
pub fn route_backbone(_ct: &crate::domain::content_detect::ContentType) -> BackboneChoice {
    BackboneChoice::Mamba
}

/// Recurrent state for a backbone model.
pub trait BackboneState: Send {
    /// Reset all state to zeros (for batch reuse without reallocation).
    fn reset(&mut self);
    /// R-108: zero the recurrent state every `every` forwards (0 = never). Recorded in the
    /// AZTH header; backbones without length drift ignore it (default no-op).
    fn set_reset_every(&mut self, _every: u64) {}
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

    /// Lockstep forward of N independent streams, one token each (R-100).
    /// `out_logits[i]` must be bit-identical to what `forward_into` produces
    /// for stream `i` alone. Default: N sequential single-stream calls;
    /// backbones override it to share each weight fetch across streams.
    fn forward_multi(
        &self, tokens: &[usize], states: &mut [&mut Self::State],
        scratches: &mut [&mut Self::Scratch], out_logits: &mut [&mut [f32]],
    ) {
        for i in 0..tokens.len() {
            self.forward_into(tokens[i], states[i], scratches[i], out_logits[i]);
        }
    }

    /// Approximate bytes of one stream's state + scratch (0 = unknown).
    /// Used for the lockstep memory report (R-100).
    fn stream_bytes(&self) -> usize {
        0
    }
}

/// Tokenizer backend — encode bytes to tokens and decode back.
pub trait BackendTokenizer: Sync {
    fn encode(&self, data: &[u8]) -> Vec<u32>;
    fn decode_token(&self, id: u32) -> &[u8];
}
