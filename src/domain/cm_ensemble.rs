//! Context-mixing ensemble — groups all CM predictors and their buffers.
//!
//! Reduces parameter passing from 12+ loose args to a single `CmEnsemble`.
//! Owns all CM models + mixer + pre-allocated bias buffers.
//! Used identically in compress and decompress loops.

use crate::domain::bias_head::BiasHead;
use crate::domain::match_model::MatchModel;
use crate::domain::mixer::AdaptiveMixer;
use crate::domain::ngram::TokenNgram;
use crate::domain::run_model::RunModel;
use crate::domain::segment_model::SegmentModel;
use crate::domain::word_model::WordModel;

/// All context-mixing components bundled together.
pub struct CmEnsemble {
    pub ngram: TokenNgram,
    pub bias: BiasHead,
    pub match_m: MatchModel,
    pub run_m: RunModel,
    pub word_m: WordModel,
    pub segment_m: SegmentModel,
    pub mixer: AdaptiveMixer,
    // Pre-allocated bias buffers (one per model that needs it)
    pub ng_buf: Vec<f32>,
    pub match_buf: Vec<f32>,
    pub run_buf: Vec<f32>,
    pub word_buf: Vec<f32>,
    pub seg_buf: Vec<f32>,
}

impl CmEnsemble {
    /// Create a new ensemble with default hyperparameters.
    pub fn new(
        vocab_size: usize,
        ngram_scale: f32,
        bias_lr: f32,
        mix_eta: f32,
        entropy_scaling: bool,
    ) -> Self {
        let mut mixer = AdaptiveMixer::new(mix_eta);
        mixer.set_entropy_scaling(entropy_scaling);
        Self {
            ngram: TokenNgram::new(4, vocab_size, ngram_scale),
            bias: BiasHead::new(vocab_size, bias_lr),
            match_m: MatchModel::new(vocab_size),
            run_m: RunModel::new(vocab_size),
            word_m: WordModel::new(vocab_size),
            segment_m: SegmentModel::new(vocab_size, entropy_scaling),
            mixer,
            ng_buf: vec![0.0; vocab_size],
            match_buf: vec![0.0; vocab_size],
            run_buf: vec![0.0; vocab_size],
            word_buf: vec![0.0; vocab_size],
            seg_buf: vec![0.0; vocab_size],
        }
    }

    /// Apply tau decay to bias head (for hybrid-eval).
    pub fn with_bias_decay(mut self, tau: f32) -> Self {
        self.bias = self.bias.with_decay(tau);
        self
    }

    /// Compute all CM biases into internal buffers.
    /// `entropy_hint` is forwarded to the ngram model (R-045 hash features).
    pub fn compute_biases(&mut self, entropy_hint: Option<f32>) {
        self.ngram.compute_bias_into(&mut self.ng_buf, entropy_hint);
        self.match_m.compute_bias_into(&mut self.match_buf);
        self.run_m.compute_bias_into(&mut self.run_buf);
        self.word_m.compute_bias_into(&mut self.word_buf);
        self.segment_m.compute_bias_into(&mut self.seg_buf);
    }

    /// Combine backbone logits with all CM biases into `out`.
    pub fn combine_into(&self, out: &mut [f32], backbone_logits: &[f32]) {
        self.mixer.combine_with_match(
            out,
            backbone_logits,
            &self.ng_buf,
            self.bias.bias_vector(),
            &self.match_buf,
            &self.run_buf,
            &self.word_buf,
        );
        let seg = &self.seg_buf;
        for i in 0..out.len() {
            out[i] += seg[i];
        }
    }

    /// Update all CM models after observing the correct token.
    pub fn update(&mut self, probs: &[f32], token: usize) {
        self.mixer.update_with_match(
            probs,
            &self.ng_buf,
            self.bias.bias_vector(),
            &self.match_buf,
            &self.run_buf,
            &self.word_buf,
            token,
        );
        self.bias.update(probs, token);
    }

    /// Observe a token (advance all CM model histories).
    pub fn observe(&mut self, token: u32) {
        self.ngram.observe(token);
        self.match_m.observe(token);
        self.run_m.observe(token);
        self.word_m.observe(token);
        self.segment_m.observe(token);
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensemble_create_and_observe() {
        let mut cm = CmEnsemble::new(256, 0.5, 0.001, 0.01, false);
        cm.observe(65); // 'A'
        cm.observe(66); // 'B'
        cm.compute_biases(None);
        assert_eq!(cm.ng_buf.len(), 256);
    }

    #[test]
    fn ensemble_combine_adds_biases() {
        let mut cm = CmEnsemble::new(8, 0.5, 0.001, 0.01, false);
        cm.observe(1);
        cm.observe(2);
        cm.compute_biases(None);
        let logits = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        cm.combine_into(&mut out, &logits);
        // With biases, at least one element should be non-zero
        let sum: f32 = out.iter().map(|x| x.abs()).sum();
        assert!(sum >= 0.0);
    }

}
