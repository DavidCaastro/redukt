//! Backbone abstraction — zero-cost trait interface for neural language models.
//!
//! Decouples the compression pipeline from any specific model implementation.
//! Currently implemented by RWKV-7; designed for future Mamba/Griffin backends.

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
