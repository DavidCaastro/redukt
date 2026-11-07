//! Adaptive component mixer via online gradient descent.
//!
//! Learns optimal weights for ensemble components (N-gram, bias head,
//! match model, run-length model, word model) by tracking each
//! component's gradient contribution to cross-entropy.
//!
//! At each step:
//!   1. Ensemble: final[i] = rwkv[i] + s * Σ w_k * comp_k[i]
//!   2. After observing correct token, compute gradient of loss w.r.t. weights:
//!      dL/dw_k = sum(prob[i] * comp_k[i]) - comp_k[correct]
//!   3. SGD update: w_k -= eta * gradient
//!
//! R-043: entropy-adaptive CM scaling — `s` (cm_scale) modulates all CM
//! contributions based on the model's per-token cross-entropy.  When the
//! model is more uncertain than average (high CE), CM gets boosted; when
//! confident, CM is reduced.

/// Number of mixer components.
const N_COMPONENTS: usize = 5;

pub struct AdaptiveMixer {
    pub w_ngram: f32,
    pub w_bias: f32,
    pub w_match: f32,
    pub w_run: f32,
    pub w_word: f32,
    eta: f32,
    // R-043: entropy-adaptive CM scaling
    cm_scale: f32,
    ce_ema: f32,
    entropy_scaling: bool,
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
            cm_scale: 1.0,
            ce_ema: 3.0,
            entropy_scaling: false,
        }
    }

    /// Enable or disable entropy-adaptive CM scaling (R-043).
    pub fn set_entropy_scaling(&mut self, enabled: bool) {
        self.entropy_scaling = enabled;
    }

    /// Current CM scale factor (R-077: needed by entropy gate).
    pub fn cm_scale(&self) -> f32 { self.cm_scale }

    /// Zero-alloc combine: RWKV logits + all CM components.
    /// When entropy_scaling is enabled, CM contributions are scaled by cm_scale.
    pub fn combine_with_match(
        &self, out: &mut [f32], rwkv_logits: &[f32],
        ngram_bias: &[f32], bias_vec: &[f32], match_bias: &[f32],
        run_bias: &[f32], word_bias: &[f32],
    ) {
        debug_assert_eq!(out.len(), rwkv_logits.len());
        let s = self.cm_scale;
        for i in 0..rwkv_logits.len() {
            out[i] = rwkv_logits[i]
                + s * self.w_ngram * ngram_bias[i]
                + s * self.w_bias * bias_vec[i]
                + s * self.w_match * match_bias[i]
                + s * self.w_run * run_bias[i]
                + s * self.w_word * word_bias[i];
        }
    }

    /// Update all weights via SGD, then update entropy tracking (R-043).
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

        // R-043: update entropy-adaptive scaling for next step
        if self.entropy_scaling {
            let prob = probs[correct_token].max(1e-30);
            let ce = -crate::domain::tensor::portable_lnf(prob);
            self.ce_ema = 0.99 * self.ce_ema + 0.01 * ce;
            self.cm_scale = (ce / self.ce_ema.max(0.01)).clamp(1.0, 1.2);
        }
    }
}
