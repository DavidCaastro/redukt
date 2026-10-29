//! Centralized model + tokenizer loading.
//!
//! Eliminates 5× duplication of vocab-path resolution and model auto-detection
//! across cmd_generate, cmd_chat, cmd_compress, cmd_baseline, cmd_rwkv_test.

use std::path::{Path, PathBuf};

use crate::domain::quant::{QuantLevel, QuantPolicy};
use crate::infrastructure::rwkv7::model::Rwkv7Model;
use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;

/// Resolve the tokenizer vocab file path given a weights directory or GGUF file.
///
/// Strategy:
/// 1. If weights_path is a directory: look for vocab file inside it
/// 2. If weights_path is a file (.gguf): look next to it, then fallback to default
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

/// Load model and tokenizer from a weights path (directory or GGUF file).
/// Auto-detects format and quantization.
pub fn load_model_and_tokenizer(weights_dir: &str) -> (Rwkv7Model, WorldTokenizer) {
    load_model_and_tokenizer_with_quant(weights_dir, None)
}

/// Load model and tokenizer with an optional forced quantization policy.
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

/// Load only the tokenizer (skip model weights). Used in CM-only mode (R-032).
pub fn load_tokenizer_only(weights_dir: &str) -> WorldTokenizer {
    let weights_path = Path::new(weights_dir);
    let vocab_path = resolve_vocab_path(weights_path);
    WorldTokenizer::load(&vocab_path)
}

/// Parse a quantization level string ("f32", "q8", "q4", "auto") into a policy.
/// Returns None for "auto" (let the system decide).
pub fn parse_quant_flag(s: &str) -> Option<QuantPolicy> {
    match s.to_lowercase().as_str() {
        "f32" => Some(QuantPolicy::Uniform(QuantLevel::F32)),
        "q8" => Some(QuantPolicy::Uniform(QuantLevel::Q8)),
        "q4" => Some(QuantPolicy::Uniform(QuantLevel::Q4 { group_size: 64 })),
        _ => None, // "auto" or unrecognized → auto-detect
    }
}
