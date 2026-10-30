//! Longest-match predictor for context mixing.
//!
//! Maintains a hash-chain index over the token history. When an active
//! match is found, predicts the next token with confidence proportional
//! to the match length (log-space bias, like TokenNgram).
//!
//! Inspired by the match model in paq8/cmix — the single most impactful
//! CM predictor after n-grams for structured/repetitive data.

/// Token-level match predictor using hash-chain indexing.
pub struct MatchModel {
    vocab_size: usize,
    history: Vec<u32>,
    /// Direct-mapped hash table: bigram_hash → most recent position.
    heads: Vec<u32>,
    /// Hash chain: position → previous position with same bigram hash.
    chain: Vec<u32>,
    table_mask: usize,
    /// Active match: position in history to predict from (next token).
    match_pos: usize,
    /// Length of current active match (0 = none).
    match_len: usize,
}

const EMPTY: u32 = u32::MAX;
const TABLE_BITS: usize = 16;
const MAX_CHAIN_WALK: usize = 64;

impl MatchModel {
    pub fn new(vocab_size: usize) -> Self {
        let table_size = 1 << TABLE_BITS;
        Self {
            vocab_size,
            history: Vec::with_capacity(8192),
            heads: vec![EMPTY; table_size],
            chain: Vec::with_capacity(8192),
            table_mask: table_size - 1,
            match_pos: 0,
            match_len: 0,
        }
    }

    /// Write match prediction bias into `buf` (must be vocab_size).
    /// Buffer is zeroed first. If an active match exists, the predicted
    /// token gets a positive log-space bias proportional to match length.
    pub fn compute_bias_into(&self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), self.vocab_size);
        for v in buf.iter_mut() { *v = 0.0; }

        if self.match_len > 0 && self.match_pos < self.history.len() {
            let predicted = self.history[self.match_pos] as usize;
            if predicted < self.vocab_size {
                // Log-space bias: ln(1 + match_len). Higher match = stronger prediction.
                let confidence = crate::domain::tensor::portable_lnf(
                    1.0 + self.match_len as f32
                );
                buf[predicted] = confidence;
            }
        }
    }

    /// Observe a new token: update history, hash index, and match state.
    pub fn observe(&mut self, token: u32) {
        let pos = self.history.len();
        self.history.push(token);
        self.chain.push(EMPTY);

        // Update hash chain for bigram ending at pos (need at least 2 tokens)
        if pos >= 1 {
            let h = bigram_hash(self.history[pos - 1], token) & self.table_mask;
            self.chain[pos] = self.heads[h];
            self.heads[h] = pos as u32;
        }

        // Try to extend active match
        if self.match_len > 0 {
            if self.match_pos > 0
                && self.history[self.match_pos - 1] == token
            {
                // Match still alive: the token before match_pos matches
                // Actually, we need to check: did the token we just saw
                // match our prediction? If history[match_pos-1] was the
                // predicted position, now advance.
                // Correction: match_pos points to the token we predicted.
                // If the actual token matches, extend.
                // Wait, let me reconsider the match semantics.
            }
        }

        // Re-evaluate match state after observing this token:
        // Try to extend current match first
        let extended = self.match_len > 0
            && self.match_pos > 0
            && self.match_pos <= pos
            && self.history[self.match_pos - 1] == token;

        if extended {
            // Match continues — advance prediction position
            self.match_pos += 0; // match_pos already points to next prediction
            // Actually we predicted history[match_pos], and token matched.
            // Now predict history[match_pos + 1].
            // Hmm, let me rethink the state machine.
        }

        // Clean re-implementation of match state:
        // After observing token at position `pos`, we want to find the longest
        // suffix of history[0..=pos] that appears earlier in the history.
        // Then match_pos = the position right after that earlier occurrence.

        // Try extending existing match
        if self.match_len > 0 && self.match_pos < pos {
            if self.history[self.match_pos] == token {
                self.match_len += 1;
                self.match_pos += 1;
                return;
            }
        }

        // Search for new match using hash chain
        self.match_len = 0;
        self.match_pos = 0;

        if pos < 1 { return; }

        let h = bigram_hash(self.history[pos - 1], token) & self.table_mask;
        let mut candidate = self.heads[h];
        let mut best_len: usize = 0;
        let mut best_next: usize = 0;
        let mut walks = 0;

        while candidate != EMPTY && walks < MAX_CHAIN_WALK {
            let c = candidate as usize;
            if c < pos && c >= 1
                && self.history[c] == token
                && self.history[c - 1] == self.history[pos - 1]
            {
                // Bigram matches. Extend backward to find full match length.
                let mut len = 2usize;
                while len < pos && c >= len
                    && self.history[c - len] == self.history[pos - len]
                {
                    len += 1;
                }
                if len > best_len && c + 1 < self.history.len() {
                    best_len = len;
                    best_next = c + 1;
                }
            }
            candidate = self.chain[c];
            walks += 1;
        }

        if best_len >= 2 {
            self.match_len = best_len;
            self.match_pos = best_next;
        }
    }
}

#[inline]
fn bigram_hash(a: u32, b: u32) -> usize {
    let mut h = 0xcbf29ce484222325u64;
    h ^= a as u64;
    h = h.wrapping_mul(0x100000001b3);
    h ^= b as u64;
    h = h.wrapping_mul(0x100000001b3);
    h as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_model_basic() {
        let mut mm = MatchModel::new(10);
        // Feed: 1 2 3 4 1 2 3
        for &tok in &[1u32, 2, 3, 4, 1, 2, 3] {
            mm.observe(tok);
        }
        // After seeing 1,2,3 again, should predict 4 (what followed 1,2,3 before)
        let mut buf = vec![0.0f32; 10];
        mm.compute_bias_into(&mut buf);
        assert!(mm.match_len >= 2, "expected match_len >= 2, got {}", mm.match_len);
        assert!(buf[4] > 0.0, "expected positive bias for token 4");
    }

    #[test]
    fn match_model_no_match() {
        let mut mm = MatchModel::new(10);
        for &tok in &[1u32, 2, 3] {
            mm.observe(tok);
        }
        let mut buf = vec![0.0f32; 10];
        mm.compute_bias_into(&mut buf);
        // No repeated bigram → no match → all zeros
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn match_model_extends() {
        let mut mm = MatchModel::new(10);
        // Feed: 5 6 7 8 5 6 7
        for &tok in &[5u32, 6, 7, 8, 5, 6, 7] {
            mm.observe(tok);
        }
        // Match [5,6,7] → predict 8
        let mut buf = vec![0.0f32; 10];
        mm.compute_bias_into(&mut buf);
        assert!(buf[8] > 0.0, "expected bias for token 8, got {:?}", buf);
        assert!(mm.match_len >= 3);
    }

    #[test]
    fn match_model_confidence_grows() {
        let mut mm = MatchModel::new(10);
        // Long repeated pattern
        for &tok in &[1u32, 2, 3, 4, 5, 1, 2, 3, 4] {
            mm.observe(tok);
        }
        let mut buf = vec![0.0f32; 10];
        mm.compute_bias_into(&mut buf);
        // Should predict 5 with high confidence (match len 4)
        assert!(buf[5] > 1.0, "expected strong bias for long match");
    }
}
