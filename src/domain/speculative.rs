//! Speculative execution for compression — CM-draft with backbone verify.
//!
//! Strategy B from execution-architecture-analysis.md:
//! Use the 6 CM models (0.3% of compute) to draft K tokens ahead.
//! Then run backbone forward sequentially to get exact logits for coding.
//! On repetitive data (logs, XML), CM predicts correctly 60-80% of the time,
//! allowing pipelining optimizations and future batch verification.
//!
//! KEY CONSTRAINT: For lossless arithmetic coding, we ALWAYS need backbone
//! logits for every token. Speculation helps by:
//! 1. Pre-computing CM biases for K tokens ahead (amortized)
//! 2. Enabling future batch-verify on GPU (parallel scan)
//! 3. Measuring CM draft accuracy to inform skip decisions
//!
//! This module is purely additive — does NOT modify any existing model.

use crate::domain::cm_ensemble::CmEnsemble;

/// Result of a speculative draft attempt.
#[derive(Debug)]
pub struct DraftResult {
    /// Drafted tokens (up to K).
    pub tokens: Vec<u32>,
    /// Confidence per drafted token (max bias score).
    pub confidences: Vec<f32>,
}

/// Tracks speculative accuracy over time for adaptive K selection.
pub struct SpeculativeTracker {
    /// Window of recent draft outcomes (true = correct, false = wrong).
    history: Vec<bool>,
    /// Rolling acceptance rate.
    acceptance_rate: f32,
    /// Maximum draft length.
    max_k: usize,
    /// Current adaptive K.
    current_k: usize,
    /// Minimum confidence to include in draft.
    min_confidence: f32,
}

impl SpeculativeTracker {
    pub fn new(max_k: usize, min_confidence: f32) -> Self {
        Self {
            history: Vec::with_capacity(1024),
            acceptance_rate: 0.0,
            max_k,
            current_k: max_k,
            min_confidence,
        }
    }

    /// Get current adaptive draft length.
    pub fn draft_k(&self) -> usize {
        self.current_k
    }

    /// Get rolling acceptance rate.
    pub fn acceptance_rate(&self) -> f32 {
        self.acceptance_rate
    }

    /// Record outcome of a draft token.
    pub fn record(&mut self, correct: bool) {
        self.history.push(correct);
        // EMA update
        let alpha = if self.history.len() < 100 { 0.1 } else { 0.01 };
        let val = if correct { 1.0 } else { 0.0 };
        self.acceptance_rate = (1.0 - alpha) * self.acceptance_rate + alpha * val;

        // Adaptive K: increase if doing well, decrease if not
        if self.history.len() % 64 == 0 {
            if self.acceptance_rate > 0.7 && self.current_k < self.max_k {
                self.current_k += 1;
            } else if self.acceptance_rate < 0.3 && self.current_k > 1 {
                self.current_k -= 1;
            }
        }
    }

    /// Minimum confidence threshold for drafting.
    pub fn min_confidence(&self) -> f32 {
        self.min_confidence
    }
}

/// Generate a draft of up to K tokens using CM models only.
///
/// The CM ensemble is advanced tentatively for each drafted token.
/// If the draft is rejected, the caller must undo these observations.
///
/// Returns early if confidence drops below threshold.
pub fn draft_tokens(
    cm: &mut CmEnsemble,
    k: usize,
    min_confidence: f32,
) -> DraftResult {
    let mut tokens = Vec::with_capacity(k);
    let mut confidences = Vec::with_capacity(k);

    for _ in 0..k {
        let (tok, conf) = cm.draft_next();
        if conf < min_confidence && !tokens.is_empty() {
            break; // Stop drafting if confidence drops (but always draft at least 1)
        }
        tokens.push(tok);
        confidences.push(conf);
        // Tentatively advance CM state
        cm.observe(tok);
    }

    DraftResult { tokens, confidences }
}

/// Verify a draft against actual tokens from the file.
///
/// Returns the number of consecutive correct predictions from the start.
/// Does NOT run the backbone — just compares draft tokens to actual tokens.
pub fn verify_draft(
    draft: &DraftResult,
    actual_tokens: &[u32],
    start_pos: usize,
) -> usize {
    let mut accepted = 0;
    for (i, &drafted) in draft.tokens.iter().enumerate() {
        let actual_pos = start_pos + i;
        if actual_pos >= actual_tokens.len() {
            break;
        }
        if drafted == actual_tokens[actual_pos] {
            accepted += 1;
        } else {
            break;
        }
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_adaptive_k() {
        let mut tracker = SpeculativeTracker::new(8, 0.0);
        assert_eq!(tracker.draft_k(), 8);
        // Feed lots of failures
        for _ in 0..256 {
            tracker.record(false);
        }
        assert!(tracker.draft_k() < 8);
        assert!(tracker.acceptance_rate() < 0.1);
    }

    #[test]
    fn tracker_high_acceptance() {
        let mut tracker = SpeculativeTracker::new(4, 0.0);
        // Feed lots of successes
        for _ in 0..256 {
            tracker.record(true);
        }
        assert!(tracker.acceptance_rate() > 0.8);
    }

    #[test]
    fn verify_draft_matches() {
        let draft = DraftResult {
            tokens: vec![10, 20, 30, 40],
            confidences: vec![1.0; 4],
        };
        let actual = vec![5, 10, 20, 30, 40, 50];
        // Starting at position 1 in actual: [10, 20, 30, 40] matches draft
        assert_eq!(verify_draft(&draft, &actual, 1), 4);
    }

    #[test]
    fn verify_draft_partial_match() {
        let draft = DraftResult {
            tokens: vec![10, 20, 99, 40],
            confidences: vec![1.0; 4],
        };
        let actual = vec![5, 10, 20, 30, 40, 50];
        // Matches 10, 20 but fails at 99 vs 30
        assert_eq!(verify_draft(&draft, &actual, 1), 2);
    }

    #[test]
    fn verify_draft_no_match() {
        let draft = DraftResult {
            tokens: vec![99],
            confidences: vec![0.5],
        };
        let actual = vec![1, 2, 3];
        assert_eq!(verify_draft(&draft, &actual, 0), 0);
    }
}
