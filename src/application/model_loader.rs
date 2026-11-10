//! Centralized model + tokenizer loading.
//!
//! Supports RWKV-7 (byte-level) and Mamba (BPE) backbones.
//! DynModel/DynTokenizer provide unified dynamic dispatch (R-039).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};

use crate::domain::backbone::{
    Backbone, BackboneState, BackboneScratch, BackendTokenizer, BackboneChoice, WeightProfile,
};
use crate::domain::fmath::NumericContract;
use crate::domain::gemv::Kernel;
use crate::domain::quant::{QuantLevel, QuantPolicy};
use crate::infrastructure::rwkv7::model::Rwkv7Model;
use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;
use crate::infrastructure::mamba::model::{MambaModel, MambaState, MambaScratch};
use crate::infrastructure::mamba::tokenizer::BpeTokenizer;
use crate::infrastructure::mamba::byte_tokenizer::ByteTokenizer;
use crate::infrastructure::plugin_backend::model::{
    PluginBackend, PluginState, PluginScratch, PluginTokenizer,
};

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
const MAMBA_WEIGHTS_DIR: &str = "weights/mamba-130m";

/// Default MambaByte weights directory (R-082).
const MAMBABYTE_WEIGHTS_DIR: &str = "weights/mambabyte-96m";

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

/// Load Mamba model (given weight profile, R-099) and BPE tokenizer.
pub fn load_mamba_model_and_tokenizer(
    weights_dir: &str, profile: WeightProfile,
) -> (MambaModel, BpeTokenizer) {
    let path = Path::new(weights_dir);
    let model = MambaModel::load_with_profile(path, profile);
    let tokenizer = BpeTokenizer::load(path);
    (model, tokenizer)
}

// --- Unified dynamic dispatch (R-039) ---

/// Dynamic model wrapper — dispatches Backbone trait to RWKV, Mamba, or MambaByte.
pub enum DynModel {
    Rwkv(Rwkv7Model),
    Mamba(MambaModel),
    MambaByte(MambaModel),
    Plugin(PluginBackend),
}

/// Dynamic state wrapper.
pub enum DynState {
    Rwkv(<Rwkv7Model as Backbone>::State),
    Mamba(MambaState),
    MambaByte(MambaState),
    Plugin(PluginState),
}

/// Dynamic scratch wrapper.
pub enum DynScratch {
    Rwkv(<Rwkv7Model as Backbone>::Scratch),
    Mamba(MambaScratch),
    MambaByte(MambaScratch),
    Plugin(PluginScratch),
}

/// Dynamic tokenizer wrapper.
pub enum DynTokenizer {
    World(WorldTokenizer),
    Bpe(BpeTokenizer),
    Byte(ByteTokenizer),
    Plugin(PluginTokenizer),
}

impl BackboneState for DynState {
    fn reset(&mut self) {
        match self {
            Self::Rwkv(s) => s.reset(),
            Self::Mamba(s) => s.reset(),
            Self::MambaByte(s) => s.reset(),
            Self::Plugin(s) => s.reset(),
        }
    }
}

impl BackboneScratch for DynScratch {
    fn reset_perf(&mut self) {
        match self {
            Self::Rwkv(s) => s.reset_perf(),
            Self::Mamba(s) => s.reset_perf(),
            Self::MambaByte(s) => s.reset_perf(),
            Self::Plugin(s) => s.reset_perf(),
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
            Self::MambaByte(m) => m.vocab_size(),
            Self::Plugin(m) => m.vocab_size(),
        }
    }

    fn create_state(&self) -> DynState {
        match self {
            Self::Rwkv(m) => DynState::Rwkv(m.create_state()),
            Self::Mamba(m) => DynState::Mamba(m.create_state()),
            Self::MambaByte(m) => DynState::MambaByte(m.create_state()),
            Self::Plugin(m) => DynState::Plugin(m.create_state()),
        }
    }

    fn create_scratch(&self) -> DynScratch {
        match self {
            Self::Rwkv(m) => DynScratch::Rwkv(m.create_scratch()),
            Self::Mamba(m) => DynScratch::Mamba(m.create_scratch()),
            Self::MambaByte(m) => DynScratch::MambaByte(m.create_scratch()),
            Self::Plugin(m) => DynScratch::Plugin(m.create_scratch()),
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
            (Self::MambaByte(m), DynState::MambaByte(s), DynScratch::MambaByte(sc)) => {
                m.forward_into(token, s, sc, out_logits);
            }
            (Self::Plugin(m), DynState::Plugin(s), DynScratch::Plugin(sc)) => {
                m.forward_into(token, s, sc, out_logits);
            }
            _ => panic!("DynModel/DynState/DynScratch type mismatch"),
        }
    }

    /// Lockstep forward (R-100): Mamba/MambaByte share weight fetches across
    /// streams; RWKV and plugin backbones fall back to sequential forwards.
    fn forward_multi(
        &self, tokens: &[usize], states: &mut [&mut DynState],
        scratches: &mut [&mut DynScratch], out_logits: &mut [&mut [f32]],
    ) {
        let m = match self {
            Self::Mamba(m) | Self::MambaByte(m) => m,
            _ => {
                for i in 0..tokens.len() {
                    self.forward_into(tokens[i], states[i], scratches[i], out_logits[i]);
                }
                return;
            }
        };
        let mut st: Vec<&mut MambaState> = Vec::with_capacity(tokens.len());
        let mut sc: Vec<&mut MambaScratch> = Vec::with_capacity(tokens.len());
        for (s, c) in states.iter_mut().zip(scratches.iter_mut()) {
            match (&mut **s, &mut **c) {
                (DynState::Mamba(s), DynScratch::Mamba(c))
                | (DynState::MambaByte(s), DynScratch::MambaByte(c)) => {
                    st.push(s);
                    sc.push(c);
                }
                _ => panic!("DynModel/DynState/DynScratch type mismatch"),
            }
        }
        m.forward_multi(tokens, &mut st, &mut sc, out_logits);
    }

    fn stream_bytes(&self) -> usize {
        match self {
            Self::Mamba(m) | Self::MambaByte(m) => m.stream_bytes(),
            Self::Rwkv(m) => m.stream_bytes(),
            Self::Plugin(m) => m.stream_bytes(),
        }
    }
}

impl BackendTokenizer for DynTokenizer {
    fn encode(&self, data: &[u8]) -> Vec<u32> {
        match self {
            Self::World(t) => t.encode(data),
            Self::Bpe(t) => t.encode(data),
            Self::Byte(t) => t.encode(data),
            Self::Plugin(t) => t.encode(data),
        }
    }

    fn decode_token(&self, id: u32) -> &[u8] {
        match self {
            Self::World(t) => t.decode_token(id),
            Self::Bpe(t) => t.decode_token(id),
            Self::Byte(t) => t.decode_token(id),
            Self::Plugin(t) => t.decode_token(id),
        }
    }
}

impl DynModel {
    /// Whether split-head pipeline is supported (R-060, Mamba only).
    pub fn supports_pipeline(&self) -> bool {
        matches!(self, Self::Mamba(_) | Self::MambaByte(_))
    }

    /// Hidden dimension for pipeline buffer allocation (R-060).
    pub fn hidden_dim(&self) -> usize {
        match self {
            Self::Rwkv(m) => m.vocab_size(),
            Self::Mamba(m) => m.config.d_model,
            Self::MambaByte(m) => m.config.d_model,
            Self::Plugin(m) => m.info().hidden_dim as usize,
        }
    }

    /// Run layers + RMSNorm only; copies `scratch.x` into `hidden_out` (R-060).
    /// Mamba only — panics if called on RWKV.
    pub fn forward_layers_normed(
        &self, token: usize, state: &mut DynState, scratch: &mut DynScratch,
        hidden_out: &mut [f32],
    ) {
        match (self, state, scratch) {
            (Self::Mamba(m), DynState::Mamba(s), DynScratch::Mamba(sc)) => {
                m.forward_layers_normed(token, s, sc);
                hidden_out.copy_from_slice(&sc.x);
            }
            (Self::MambaByte(m), DynState::MambaByte(s), DynScratch::MambaByte(sc)) => {
                m.forward_layers_normed(token, s, sc);
                hidden_out.copy_from_slice(&sc.x);
            }
            _ => panic!("pipeline only supported for Mamba/MambaByte"),
        }
    }

    /// Compute LM head from a hidden buffer (R-060).
    /// Mamba/MambaByte only — panics if called on RWKV.
    pub fn compute_head_into(&self, hidden: &[f32], out_logits: &mut [f32]) {
        match self {
            Self::Mamba(m) => m.compute_head_into(hidden, out_logits),
            Self::MambaByte(m) => m.compute_head_into(hidden, out_logits),
            _ => panic!("pipeline only supported for Mamba/MambaByte"),
        }
    }

    /// Apply embedding surgery (RWKV-only, no-op for Mamba/MambaByte/Plugin).
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

    /// Weight profile of the loaded backbone (R-099). RWKV/Plugin report F32:
    /// their quantization is not part of the AZTH profile.
    pub fn weight_profile(&self) -> WeightProfile {
        match self {
            Self::Mamba(m) | Self::MambaByte(m) => m.profile(),
            Self::Rwkv(_) | Self::Plugin(_) => WeightProfile::F32,
        }
    }

    /// Configure the GEMV/GEMM kernels for numeric contract `c` (R-107): the
    /// FMA kernels for contract 2, the plain ones otherwise. Contract 2 is only
    /// defined for the Mamba family; RWKV/Plugin reject it.
    pub fn set_numeric_contract(&mut self, c: NumericContract) -> Result<(), String> {
        match self {
            Self::Mamba(m) | Self::MambaByte(m) => {
                m.set_kernel(Kernel::for_contract(c));
                Ok(())
            }
            Self::Rwkv(_) | Self::Plugin(_) if c.uses_fma() => Err(format!(
                "numeric contract {} is only defined for the Mamba family", c.name()
            )),
            Self::Rwkv(_) | Self::Plugin(_) => Ok(()),
        }
    }

    /// Numeric contract an encoder running this model must write (R-107).
    pub fn write_contract(&self) -> NumericContract {
        match self {
            Self::Mamba(m) | Self::MambaByte(m) => m.kernel().write_contract(),
            Self::Rwkv(_) | Self::Plugin(_) => NumericContract::CURRENT,
        }
    }

    /// Mamba activation sparsity stats (R-059).
    pub fn mamba_sparsity(scratch: &DynScratch) -> Option<(u64, u64)> {
        match scratch {
            DynScratch::Mamba(s) | DynScratch::MambaByte(s) => {
                Some((s.sparsity_total_channels, s.sparsity_near_zero))
            }
            _ => None,
        }
    }
}

/// Numeric contract of the Mamba-family models of this process (R-107), as a
/// PROFILE contract code. Set once by the CLI before any model is loaded
/// (`compress --numeric`, or the archives of a `decompress --input-dir` batch);
/// `load_dyn_model` applies it and `codec::build_header` writes it. Default:
/// contract 1 (fmath-v1, plain kernels), so the default bitstream never changes.
static PROCESS_CONTRACT: AtomicU8 = AtomicU8::new(1);

/// Set the process numeric contract (see `PROCESS_CONTRACT`).
pub fn set_process_contract(c: NumericContract) {
    PROCESS_CONTRACT.store(c.header_code(), Ordering::Relaxed);
}

/// Process numeric contract for Mamba-family models (see `PROCESS_CONTRACT`).
pub fn process_contract() -> NumericContract {
    NumericContract::from_header_code(PROCESS_CONTRACT.load(Ordering::Relaxed)).unwrap_or(NumericContract::CURRENT)
}

/// Load a DynModel + DynTokenizer for the given backbone choice.
/// `quant_override` applies to RWKV only; `mamba_profile` to Mamba/MambaByte (R-099).
/// Mamba-family models get the kernels of `process_contract()` (R-107).
pub fn load_dyn_model(
    backbone: BackboneChoice,
    rwkv_weights: &str,
    quant_override: Option<QuantPolicy>,
    mamba_profile: WeightProfile,
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
            let (mut m, t) = load_mamba_model_and_tokenizer(mamba_dir, mamba_profile);
            m.set_kernel(Kernel::for_contract(process_contract()));
            (DynModel::Mamba(m), DynTokenizer::Bpe(t))
        }
        BackboneChoice::MambaByte => {
            let mb_dir = if Path::new(MAMBABYTE_WEIGHTS_DIR).exists() {
                MAMBABYTE_WEIGHTS_DIR
            } else {
                rwkv_weights // fallback: user may pass path via --weights
            };
            let path = Path::new(mb_dir);
            let mut model = MambaModel::load_with_profile(path, mamba_profile);
            model.set_kernel(Kernel::for_contract(process_contract()));
            let tokenizer = ByteTokenizer::new();
            (DynModel::MambaByte(model), DynTokenizer::Byte(tokenizer))
        }
        BackboneChoice::Plugin => {
            let pb = PluginBackend::try_new(rwkv_weights, 0)
                .expect("[plugin] GPU backend not found or init failed");
            (DynModel::Plugin(pb), DynTokenizer::Plugin(PluginTokenizer::new()))
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

/// Weight profile that `--quant auto` (or no `--quant`) selects for the Mamba family.
/// Q8 weight-only (block 64) measures ~3.3x faster than F32 at +0.6%..+1.2% BPB,
/// inside the +5% budget. `--quant f32` gives ~1% better BPB and is the recommended
/// profile for batches of >=16 streams, where aggregate throughput is equal.
pub const AUTO_MAMBA_PROFILE: WeightProfile = WeightProfile::Q8;

/// Resolve the weight profile for COMPRESSION from `--quant` and the backbone.
/// Mamba/MambaByte: explicit f32|q8, "auto"/empty -> `AUTO_MAMBA_PROFILE`.
/// RWKV/Plugin: always F32 (their quantization is not part of the AZTH profile).
/// Decompression never uses this: the profile comes from the archive header.
pub fn resolve_auto_profile(quant: &str, backbone: BackboneChoice) -> Result<WeightProfile, String> {
    if !matches!(backbone, BackboneChoice::Mamba | BackboneChoice::MambaByte) {
        return Ok(WeightProfile::F32);
    }
    Ok(WeightProfile::from_cli(quant)?.unwrap_or(AUTO_MAMBA_PROFILE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_maps_to_q8_for_mamba_family() {
        for b in [BackboneChoice::Mamba, BackboneChoice::MambaByte] {
            assert_eq!(resolve_auto_profile("auto", b), Ok(WeightProfile::Q8));
            assert_eq!(resolve_auto_profile("", b), Ok(WeightProfile::Q8));
            assert_eq!(resolve_auto_profile("AUTO", b), Ok(WeightProfile::Q8));
        }
    }

    #[test]
    fn explicit_quant_is_respected_for_mamba() {
        let b = BackboneChoice::Mamba;
        assert_eq!(resolve_auto_profile("f32", b), Ok(WeightProfile::F32));
        assert_eq!(resolve_auto_profile("q8", b), Ok(WeightProfile::Q8));
        assert!(resolve_auto_profile("q4", b).is_err());
    }

    #[test]
    fn rwkv_and_plugin_are_always_f32() {
        for b in [BackboneChoice::Rwkv, BackboneChoice::Plugin] {
            for q in ["auto", "", "f32", "q8", "q4"] {
                assert_eq!(resolve_auto_profile(q, b), Ok(WeightProfile::F32));
            }
        }
    }
}
