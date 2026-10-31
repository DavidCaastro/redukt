//! Centralized model + tokenizer loading.
//!
//! Supports RWKV-7 (byte-level) and Mamba (BPE) backbones.
//! DynModel/DynTokenizer provide unified dynamic dispatch (R-039).

use std::path::{Path, PathBuf};

use crate::domain::backbone::{
    Backbone, BackboneState, BackboneScratch, BackendTokenizer, BackboneChoice,
};
use crate::domain::quant::{QuantLevel, QuantPolicy};
use crate::infrastructure::rwkv7::model::Rwkv7Model;
use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;
use crate::infrastructure::mamba::model::{MambaModel, MambaState, MambaScratch};
use crate::infrastructure::mamba::tokenizer::BpeTokenizer;

// --- Path resolution ---

/// Resolve the tokenizer vocab file path given a weights directory or GGUF file.
pub fn resolve_vocab_path(weights_path: &Path) -> PathBuf {
    if weights_path.is_dir() {
        weights_path.join("rwkv_vocab_v20230424.txt")
    } else {
        let parent = weights_path.parent().unwrap_or(Path::new("."));
        let vocab_beside = parent.join("rwkv_vocab_v20230424.txt");
        if vocab_beside.exists() {
            vocab_beside
        } else {
            Path::new("weights/rwkv7-0.1b/rwkv_vocab_v20230424.txt").to_path_buf()
        }
    }
}

/// Default Mamba weights directory.
const MAMBA_WEIGHTS_DIR: &str = "weights/mamba-130m-hf";

// --- RWKV loading ---

/// Load model and tokenizer from a weights path (directory or GGUF file).
pub fn load_model_and_tokenizer(weights_dir: &str) -> (Rwkv7Model, WorldTokenizer) {
    load_model_and_tokenizer_with_quant(weights_dir, None)
}

/// Load RWKV model and tokenizer with optional forced quantization policy.
pub fn load_model_and_tokenizer_with_quant(
    weights_dir: &str,
    quant_override: Option<QuantPolicy>,
) -> (Rwkv7Model, WorldTokenizer) {
    let weights_path = Path::new(weights_dir);
    let vocab_path = resolve_vocab_path(weights_path);
    let tokenizer = WorldTokenizer::load(&vocab_path);
    let model = Rwkv7Model::load_auto_with_quant(weights_path, quant_override);
    (model, tokenizer)
}

// --- Mamba loading ---

/// Load Mamba model and BPE tokenizer.
pub fn load_mamba_model_and_tokenizer(weights_dir: &str) -> (MambaModel, BpeTokenizer) {
    let path = Path::new(weights_dir);
    let model = MambaModel::load(path);
    let tokenizer = BpeTokenizer::load(path);
    (model, tokenizer)
}

// --- Unified dynamic dispatch (R-039) ---

/// Dynamic model wrapper — dispatches Backbone trait to RWKV or Mamba.
pub enum DynModel {
    Rwkv(Rwkv7Model),
    Mamba(MambaModel),
}

/// Dynamic state wrapper.
pub enum DynState {
    Rwkv(<Rwkv7Model as Backbone>::State),
    Mamba(MambaState),
}

/// Dynamic scratch wrapper.
pub enum DynScratch {
    Rwkv(<Rwkv7Model as Backbone>::Scratch),
    Mamba(MambaScratch),
}

/// Dynamic tokenizer wrapper.
pub enum DynTokenizer {
    World(WorldTokenizer),
    Bpe(BpeTokenizer),
}

impl BackboneState for DynState {
    fn reset(&mut self) {
        match self {
            Self::Rwkv(s) => s.reset(),
            Self::Mamba(s) => s.reset(),
        }
    }
}

impl BackboneScratch for DynScratch {
    fn reset_perf(&mut self) {
        match self {
            Self::Rwkv(s) => s.reset_perf(),
            Self::Mamba(_) => {}
        }
    }
}

impl Backbone for DynModel {
    type State = DynState;
    type Scratch = DynScratch;

    fn vocab_size(&self) -> usize {
        match self {
            Self::Rwkv(m) => m.vocab_size(),
            Self::Mamba(m) => m.vocab_size(),
        }
    }

    fn create_state(&self) -> DynState {
        match self {
            Self::Rwkv(m) => DynState::Rwkv(m.create_state()),
            Self::Mamba(m) => DynState::Mamba(m.create_state()),
        }
    }

    fn create_scratch(&self) -> DynScratch {
        match self {
            Self::Rwkv(m) => DynScratch::Rwkv(m.create_scratch()),
            Self::Mamba(m) => DynScratch::Mamba(m.create_scratch()),
        }
    }

    fn forward_into(
        &self, token: usize, state: &mut DynState,
        scratch: &mut DynScratch, out_logits: &mut [f32],
    ) {
        match (self, state, scratch) {
            (Self::Rwkv(m), DynState::Rwkv(s), DynScratch::Rwkv(sc)) => {
                m.forward_into(token, s, sc, out_logits);
            }
            (Self::Mamba(m), DynState::Mamba(s), DynScratch::Mamba(sc)) => {
                m.forward_into(token, s, sc, out_logits);
            }
            _ => panic!("DynModel/DynState/DynScratch type mismatch"),
        }
    }
}

impl BackendTokenizer for DynTokenizer {
    fn encode(&self, data: &[u8]) -> Vec<u32> {
        match self {
            Self::World(t) => t.encode(data),
            Self::Bpe(t) => t.encode(data),
        }
    }

    fn decode_token(&self, id: u32) -> &[u8] {
        match self {
            Self::World(t) => t.decode_token(id),
            Self::Bpe(t) => t.decode_token(id),
        }
    }
}

impl DynModel {
    /// Apply embedding surgery (RWKV-only, no-op for Mamba).
    pub fn embedding_surgery(&mut self, method: &str) {
        if let Self::Rwkv(m) = self {
            m.embedding_surgery(method);
        }
    }

    /// RWKV perf counters from scratch, if available.
    pub fn rwkv_perf(scratch: &DynScratch) -> Option<(u64, u64, u64, u64)> {
        if let DynScratch::Rwkv(s) = scratch {
            Some((s.perf_layers_us, s.perf_head_us, s.perf_tmix_us, s.perf_cmix_us))
        } else {
            None
        }
    }
}

/// Load a DynModel + DynTokenizer for the given backbone choice.
pub fn load_dyn_model(
    backbone: BackboneChoice,
    rwkv_weights: &str,
    quant_override: Option<QuantPolicy>,
) -> (DynModel, DynTokenizer) {
    match backbone {
        BackboneChoice::Rwkv => {
            let (m, t) = load_model_and_tokenizer_with_quant(rwkv_weights, quant_override);
            (DynModel::Rwkv(m), DynTokenizer::World(t))
        }
        BackboneChoice::Mamba => {
            let mamba_dir = if Path::new(MAMBA_WEIGHTS_DIR).exists() {
                MAMBA_WEIGHTS_DIR
            } else {
                rwkv_weights // fallback: user may pass mamba path via --weights
            };
            let (m, t) = load_mamba_model_and_tokenizer(mamba_dir);
            (DynModel::Mamba(m), DynTokenizer::Bpe(t))
        }
    }
}

/// Parse a quantization level string ("f32", "q8", "q4", "auto") into a policy.
pub fn parse_quant_flag(s: &str) -> Option<QuantPolicy> {
    match s.to_lowercase().as_str() {
        "f32" => Some(QuantPolicy::Uniform(QuantLevel::F32)),
        "q8" => Some(QuantPolicy::Uniform(QuantLevel::Q8)),
        "q4" => Some(QuantPolicy::Uniform(QuantLevel::Q4 { group_size: 64 })),
        _ => None,
    }
}
