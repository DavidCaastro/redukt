//! Word-level context predictor for context mixing.
//!
//! Tracks "words" (sequences between separator tokens) and predicts
//! the next token based on what followed the same word before.
//! Effective for natural language and structured text where word-level
//! repetitions are common (variable names, XML tags, dictionary words).

use std::collections::HashMap;

const MAX_WORD_LEN: usize = 32;
const MAX_ENTRIES: usize = 100_000;

/// Word-level context predictor.
pub struct WordModel {
    vocab_size: usize,
    /// Current word being built (as token sequence).
    current_word: Vec<u32>,
    /// Word hash → (next_token → count, total_count).
    table: HashMap<u64, WordEntry>,
    /// Set of separator token IDs.
    separators: Vec<bool>,
    /// Whether we just completed a word (for prediction).
    last_word_hash: u64,
    has_prediction: bool,
    frozen: bool,
}

struct WordEntry {
    counts: HashMap<u32, u32>,
    total: u32,
}

impl WordModel {
    pub fn new(vocab_size: usize) -> Self {
        // Common separator tokens: space (32), newline (10), tab (9), CR (13)
        // For byte-level tokenizers, these are single bytes.
        // For BPE tokenizers, we mark by checking decode.
        // As a simple heuristic, we'll accept a separator callback.
        let mut separators = vec![false; vocab_size];
        // Mark common ASCII separators for byte-value tokens
        if vocab_size > 256 {
            // Likely a subword tokenizer — we can't know which tokens are separators
            // without the tokenizer. Use a simple heuristic: mark nothing,
            // and treat "word boundary" as any token that starts with space.
            // For now, we'll just use the first 256 entries.
        }
        for &sep in &[9u32, 10, 13, 32] {
            if (sep as usize) < vocab_size {
                separators[sep as usize] = true;
            }
        }

        Self {
            vocab_size,
            current_word: Vec::with_capacity(MAX_WORD_LEN),
            table: HashMap::new(),
            separators,
            last_word_hash: 0,
            has_prediction: false,
            frozen: false,
        }
    }

    /// Mark a token ID as a word separator (e.g., space, newline tokens).
    #[allow(dead_code)]
    pub fn add_separator(&mut self, token_id: u32) {
        if (token_id as usize) < self.separators.len() {
            self.separators[token_id as usize] = true;
        }
    }

    /// Write word-context bias into `buf` (must be vocab_size).
    pub fn compute_bias_into(&self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), self.vocab_size);
        for v in buf.iter_mut() { *v = 0.0; }

        if !self.has_prediction { return; }

        if let Some(entry) = self.table.get(&self.last_word_hash) {
            if entry.total < 2 { return; }
            let inv_total = 1.0 / entry.total as f32;
            for (&tok, &count) in &entry.counts {
                if (tok as usize) < self.vocab_size && count > 0 {
                    // Log-odds bias: ln(count / (total - count + 1))
                    // Simpler: proportional bias scaled by confidence
                    let p = count as f32 * inv_total;
                    let bias = crate::domain::tensor::portable_lnf(
                        1.0 + p * entry.total as f32
                    );
                    buf[tok as usize] = bias;
                }
            }
        }
    }

    /// Observe a new token: update word state and table.
    pub fn observe(&mut self, token: u32) {
        let is_sep = (token as usize) < self.separators.len()
            && self.separators[token as usize];

        if is_sep || self.current_word.len() >= MAX_WORD_LEN {
            // Word boundary: finalize current word
            if self.current_word.len() >= 2 {
                let h = word_hash(&self.current_word);
                // Record that `token` follows this word
                if !self.frozen {
                    let entry = self.table.entry(h).or_insert_with(|| {
                        WordEntry { counts: HashMap::new(), total: 0 }
                    });
                    *entry.counts.entry(token).or_insert(0) += 1;
                    entry.total += 1;
                    if self.table.len() > MAX_ENTRIES {
                        self.frozen = true;
                    }
                }
                self.last_word_hash = h;
                self.has_prediction = true;
            } else {
                self.has_prediction = false;
            }
            self.current_word.clear();
            if !is_sep {
                self.current_word.push(token);
            }
        } else {
            // Update prediction: current word is being built
            if self.current_word.is_empty() && self.has_prediction {
                // First token of new word — prediction from last word
                // (already set has_prediction and last_word_hash above)
            }
            self.current_word.push(token);

            // Try to predict from partial word too (if we've seen it before)
            if self.current_word.len() >= 2 {
                let h = word_hash(&self.current_word);
                if self.table.contains_key(&h) {
                    self.last_word_hash = h;
                    self.has_prediction = true;
                }
            }
        }
    }
}

fn word_hash(tokens: &[u32]) -> u64 {
    let mut h = 0x9e3779b97f4a7c15u64;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_model_no_prediction_initially() {
        let wm = WordModel::new(256);
        let mut buf = vec![0.0f32; 256];
        wm.compute_bias_into(&mut buf);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn word_model_learns_pattern() {
        let mut wm = WordModel::new(256);
        // "hello " repeated → after "hello" should predict space
        // h=104, e=101, l=108, o=111, space=32
        for &tok in &[104u32, 101, 108, 108, 111, 32, 104, 101, 108, 108, 111, 32] {
            wm.observe(tok);
        }
        // After second "hello", should have learned space follows "hello"
        let mut buf = vec![0.0f32; 256];
        wm.compute_bias_into(&mut buf);
        // Space (32) should have positive bias
        assert!(buf[32] > 0.0, "expected bias for space after repeated 'hello'");
    }

    #[test]
    fn word_model_short_words_ignored() {
        let mut wm = WordModel::new(256);
        // Single-char "words" between spaces
        for &tok in &[65u32, 32, 66, 32, 65, 32] {
            wm.observe(tok);
        }
        let mut buf = vec![0.0f32; 256];
        wm.compute_bias_into(&mut buf);
        // Single-char words too short for meaningful prediction
        assert!(buf.iter().all(|&v| v == 0.0));
    }
}
