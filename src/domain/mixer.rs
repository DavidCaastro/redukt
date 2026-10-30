//! Adaptive component mixer via online gradient descent.
//!
//! Learns optimal weights for ensemble components (N-gram, bias head, match model)
//! by tracking each component's gradient contribution to cross-entropy.
//!
//! At each step:
//!   1. Ensemble: final[i] = rwkv[i] + w_ng * ng[i] + w_b * b[i] + w_m * m[i]
//!   2. After observing correct token, compute gradient of loss w.r.t. weights:
//!      dL/dw = sum(prob[i] * comp[i]) - comp[correct]
//!   3. SGD update: w -= eta * gradient
//!
//! Negative gradient = component is helping → weight increases.
//! Positive gradient = component is hurting → weight decreases.

pub struct AdaptiveMixer {
    pub w_ngram: f32,
    pub w_bias: f32,
    pub w_match: f32,
    eta: f32,
}

impl AdaptiveMixer {
    pub fn new(eta: f32) -> Self {
        Self {
            w_ngram: 1.0,
            w_bias: 1.0,
            w_match: 1.0,
            eta,
        }
    }

    /// Zero-alloc combine: RWKV logits + ngram + bias + match.
    pub fn combine_with_match(
        &self, out: &mut [f32], rwkv_logits: &[f32],
        ngram_bias: &[f32], bias_vec: &[f32], match_bias: &[f32],
    ) {
        debug_assert_eq!(out.len(), rwkv_logits.len());
        for i in 0..rwkv_logits.len() {
            out[i] = rwkv_logits[i]
                + self.w_ngram * ngram_bias[i]
                + self.w_bias * bias_vec[i]
                + self.w_match * match_bias[i];
        }
    }

    /// Update all weights including match model.
    pub fn update_with_match(
        &mut self, probs: &[f32], ngram_bias: &[f32], bias_vec: &[f32],
        match_bias: &[f32], correct_token: usize,
    ) {
        let mut grad_ng = -ngram_bias[correct_token];
        let mut grad_b = -bias_vec[correct_token];
        let mut grad_m = -match_bias[correct_token];
        for i in 0..probs.len() {
            grad_ng += probs[i] * ngram_bias[i];
            grad_b += probs[i] * bias_vec[i];
            grad_m += probs[i] * match_bias[i];
        }
        self.w_ngram -= self.eta * grad_ng;
        self.w_bias -= self.eta * grad_b;
        self.w_match -= self.eta * grad_m;
        self.w_ngram = self.w_ngram.clamp(0.01, 5.0);
        self.w_bias = self.w_bias.clamp(0.01, 5.0);
        self.w_match = self.w_match.clamp(0.01, 5.0);
    }
}
