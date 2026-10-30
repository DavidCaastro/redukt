//! Adaptive component mixer via online gradient descent.
//!
//! Learns optimal weights for ensemble components (N-gram, bias head,
//! match model, run-length model, word model) by tracking each
//! component's gradient contribution to cross-entropy.
//!
//! At each step:
//!   1. Ensemble: final[i] = rwkv[i] + Σ w_k * comp_k[i]
//!   2. After observing correct token, compute gradient of loss w.r.t. weights:
//!      dL/dw_k = sum(prob[i] * comp_k[i]) - comp_k[correct]
//!   3. SGD update: w_k -= eta * gradient
//!
//! Negative gradient = component is helping → weight increases.
//! Positive gradient = component is hurting → weight decreases.

/// Number of mixer components.
const N_COMPONENTS: usize = 5;

pub struct AdaptiveMixer {
    pub w_ngram: f32,
    pub w_bias: f32,
    pub w_match: f32,
    pub w_run: f32,
    pub w_word: f32,
    eta: f32,
}

impl AdaptiveMixer {
    pub fn new(eta: f32) -> Self {
        Self {
            w_ngram: 1.0,
            w_bias: 1.0,
            w_match: 1.0,
            w_run: 1.0,
            w_word: 1.0,
            eta,
        }
    }

    /// Zero-alloc combine: RWKV logits + all CM components.
    pub fn combine_with_match(
        &self, out: &mut [f32], rwkv_logits: &[f32],
        ngram_bias: &[f32], bias_vec: &[f32], match_bias: &[f32],
        run_bias: &[f32], word_bias: &[f32],
    ) {
        debug_assert_eq!(out.len(), rwkv_logits.len());
        for i in 0..rwkv_logits.len() {
            out[i] = rwkv_logits[i]
                + self.w_ngram * ngram_bias[i]
                + self.w_bias * bias_vec[i]
                + self.w_match * match_bias[i]
                + self.w_run * run_bias[i]
                + self.w_word * word_bias[i];
        }
    }

    /// Update all weights via SGD.
    pub fn update_with_match(
        &mut self, probs: &[f32], ngram_bias: &[f32], bias_vec: &[f32],
        match_bias: &[f32], run_bias: &[f32], word_bias: &[f32],
        correct_token: usize,
    ) {
        let mut grads = [0.0f32; N_COMPONENTS];
        grads[0] = -ngram_bias[correct_token];
        grads[1] = -bias_vec[correct_token];
        grads[2] = -match_bias[correct_token];
        grads[3] = -run_bias[correct_token];
        grads[4] = -word_bias[correct_token];
        for i in 0..probs.len() {
            grads[0] += probs[i] * ngram_bias[i];
            grads[1] += probs[i] * bias_vec[i];
            grads[2] += probs[i] * match_bias[i];
            grads[3] += probs[i] * run_bias[i];
            grads[4] += probs[i] * word_bias[i];
        }
        self.w_ngram = (self.w_ngram - self.eta * grads[0]).clamp(0.01, 5.0);
        self.w_bias = (self.w_bias - self.eta * grads[1]).clamp(0.01, 5.0);
        self.w_match = (self.w_match - self.eta * grads[2]).clamp(0.01, 5.0);
        self.w_run = (self.w_run - self.eta * grads[3]).clamp(0.01, 5.0);
        self.w_word = (self.w_word - self.eta * grads[4]).clamp(0.01, 5.0);
    }
}
