//! Neural Prediction Cache (R-048).
//!
//! Not yet wired into codec loops (R-049). Allow dead_code until integration.
#![allow(dead_code)]
//!
//! Hash table mapping context fingerprints (last N tokens) to top-K
//! backbone logit distributions. Enables skipping the expensive backbone
//! forward pass (~29ms) for previously-seen contexts.
//!
//! Both compressor and decompressor build identical caches from the same
//! token stream, so skip decisions are deterministic without header flags.
//!
//! Design: `docs/internal/research/prediction-cache-research.md`

/// Number of top tokens stored per cache entry.
const TOP_K: usize = 32;

/// Hash table size (power of 2). 64K entries × ~210B = ~13MB max.
const TABLE_BITS: usize = 16;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: usize = TABLE_SIZE - 1;

/// Default context order (tokens hashed for lookup key).
const DEFAULT_ORDER: usize = 8;

/// Minimum observations before trusting a cache entry.
const DEFAULT_MIN_OBS: u16 = 2;

/// Maximum consecutive backbone skips (bounds state divergence).
const DEFAULT_MAX_SKIPS: u16 = 8;

/// Sentinel for empty top-K slots.
const EMPTY_TOKEN: u16 = u16::MAX;

/// Single cache entry: top-K logit distribution + metadata.
#[derive(Clone)]
struct CacheEntry {
    /// Top-K token IDs sorted by logit (descending).
    tokens: [u16; TOP_K],
    /// Corresponding logits.
    logits: [f32; TOP_K],
    /// Default logit for tokens not in top-K.
    base_logit: f32,
    /// Full 64-bit context hash (for collision detection).
    hash: u64,
    /// Number of times this context was observed via backbone.
    observations: u16,
}

impl CacheEntry {
    fn empty() -> Self {
        Self {
            tokens: [EMPTY_TOKEN; TOP_K],
            logits: [0.0; TOP_K],
            base_logit: 0.0,
            hash: 0,
            observations: 0,
        }
    }
}

/// Neural prediction cache for backbone logit distributions.
pub struct PredCache {
    table: Vec<Option<CacheEntry>>,
    order: usize,
    min_obs: u16,
    max_skips: u16,
    enabled: bool,
    // Stats
    hits: u64,
    misses: u64,
}

impl PredCache {
    /// Create a new prediction cache.
    pub fn new(enabled: bool) -> Self {
        Self {
            table: vec![None; TABLE_SIZE],
            order: DEFAULT_ORDER,
            min_obs: DEFAULT_MIN_OBS,
            max_skips: DEFAULT_MAX_SKIPS,
            enabled,
            hits: 0,
            misses: 0,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        enabled: bool, order: usize, min_obs: u16, max_skips: u16,
    ) -> Self {
        Self {
            table: vec![None; TABLE_SIZE],
            order: order.max(1),
            min_obs: min_obs.max(1),
            max_skips,
            enabled,
            hits: 0,
            misses: 0,
        }
    }

    /// Whether the cache is enabled.
    pub fn enabled(&self) -> bool { self.enabled }

    /// Maximum consecutive skips allowed.
    pub fn max_skips(&self) -> u16 { self.max_skips }

    /// Cache hit count.
    pub fn hits(&self) -> u64 { self.hits }

    /// Cache miss count.
    pub fn misses(&self) -> u64 { self.misses }

    /// Hit rate as fraction [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Compute context hash from the last `order` tokens.
    pub fn context_hash(&self, tokens: &[u32], pos: usize) -> u64 {
        context_hash_fnv(tokens, pos, self.order)
    }

    /// Look up cached logits for a context. Returns true if a trusted
    /// entry exists (observations >= min_obs). If true, `out` is filled
    /// with the reconstructed logit distribution.
    pub fn lookup(&mut self, hash: u64, out: &mut [f32]) -> bool {
        if !self.enabled { return false; }
        let idx = (hash as usize) & TABLE_MASK;
        if let Some(ref entry) = self.table[idx] {
            if entry.hash == hash && entry.observations >= self.min_obs {
                reconstruct_logits(entry, out);
                self.hits += 1;
                return true;
            }
        }
        self.misses += 1;
        false
    }

    /// Insert or update a cache entry after a backbone forward pass.
    /// `logits` must be the full vocab-size logit array from the backbone.
    pub fn insert(&mut self, hash: u64, logits: &[f32]) {
        if !self.enabled { return; }
        let idx = (hash as usize) & TABLE_MASK;

        if let Some(ref mut entry) = self.table[idx] {
            if entry.hash == hash {
                // Same context: increment observations, update logits
                // (running update: keep most recent logits)
                entry.observations = entry.observations.saturating_add(1);
                extract_top_k(logits, entry);
                return;
            }
            // Collision: replace if new entry or if existing has few obs
            if entry.observations <= 1 {
                let mut new_entry = CacheEntry::empty();
                new_entry.hash = hash;
                new_entry.observations = 1;
                extract_top_k(logits, &mut new_entry);
                self.table[idx] = Some(new_entry);
            }
            // else: keep existing high-observation entry (LFU eviction)
        } else {
            let mut new_entry = CacheEntry::empty();
            new_entry.hash = hash;
            new_entry.observations = 1;
            extract_top_k(logits, &mut new_entry);
            self.table[idx] = Some(new_entry);
        }
    }

    /// Number of occupied entries in the table.
    pub fn occupied(&self) -> usize {
        self.table.iter().filter(|e| e.is_some()).count()
    }
}

/// FNV-1a hash of the last `order` tokens before position `pos`.
fn context_hash_fnv(tokens: &[u32], pos: usize, order: usize) -> u64 {
    let start = pos.saturating_sub(order);
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in &tokens[start..pos] {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Extract top-K tokens and logits from a full logit array.
fn extract_top_k(logits: &[f32], entry: &mut CacheEntry) {
    // Find top-K by partial sort (selection)
    // For K=32 and vocab=50K, linear scan with min-heap is efficient.
    let mut top: [(f32, u16); TOP_K] = [(f32::NEG_INFINITY, EMPTY_TOKEN); TOP_K];
    let mut min_val = f32::NEG_INFINITY;
    let mut min_idx = 0usize;

    for (i, &logit) in logits.iter().enumerate() {
        if logit > min_val {
            top[min_idx] = (logit, i as u16);
            // Find new minimum
            min_val = f32::INFINITY;
            for (j, &(v, _)) in top.iter().enumerate() {
                if v < min_val {
                    min_val = v;
                    min_idx = j;
                }
            }
        }
    }

    // Sort top-K by logit descending for consistent ordering
    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));

    // Compute base_logit: average of non-top-K logits
    let top_sum: f64 = top.iter().map(|&(v, _)| v as f64).sum();
    let total_sum: f64 = logits.iter().map(|&v| v as f64).sum();
    let non_top_count = logits.len().saturating_sub(TOP_K);
    let base = if non_top_count > 0 {
        ((total_sum - top_sum) / non_top_count as f64) as f32
    } else {
        0.0
    };

    for i in 0..TOP_K {
        entry.tokens[i] = top[i].1;
        entry.logits[i] = top[i].0;
    }
    entry.base_logit = base;
}

/// Reconstruct a full logit array from a cache entry's top-K representation.
fn reconstruct_logits(entry: &CacheEntry, out: &mut [f32]) {
    // Fill with base logit
    for v in out.iter_mut() { *v = entry.base_logit; }
    // Overwrite top-K positions
    for i in 0..TOP_K {
        let tok = entry.tokens[i] as usize;
        if tok < out.len() {
            out[tok] = entry.logits[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_hash_deterministic() {
        let tokens = vec![10u32, 20, 30, 40, 50];
        let h1 = context_hash_fnv(&tokens, 5, 3);
        let h2 = context_hash_fnv(&tokens, 5, 3);
        assert_eq!(h1, h2);
    }

    #[test]
    fn context_hash_different_contexts() {
        let tokens = vec![10u32, 20, 30, 40, 50];
        let h1 = context_hash_fnv(&tokens, 3, 3); // [10,20,30]
        let h2 = context_hash_fnv(&tokens, 5, 3); // [30,40,50]
        assert_ne!(h1, h2);
    }

    #[test]
    fn context_hash_short_history() {
        // Fewer tokens than order — should still work (uses available)
        let tokens = vec![10u32, 20];
        let h = context_hash_fnv(&tokens, 2, 8);
        assert_ne!(h, 0);
    }

    #[test]
    fn context_hash_empty_pos() {
        let tokens = vec![10u32];
        let h = context_hash_fnv(&tokens, 0, 4);
        // Empty range → FNV offset basis
        assert_eq!(h, 0xcbf29ce484222325);
    }

    #[test]
    fn insert_and_lookup_basic() {
        let mut cache = PredCache::with_params(true, 4, 1, 8);
        let logits: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let hash = 0x12345678u64;

        cache.insert(hash, &logits);
        let mut out = vec![0.0f32; 100];
        assert!(cache.lookup(hash, &mut out));
        // Top token should be 99 (logit 9.9)
        assert!(out[99] > out[50], "top token should have highest logit");
    }

    #[test]
    fn lookup_miss_returns_false() {
        let mut cache = PredCache::new(true);
        let mut out = vec![0.0f32; 100];
        assert!(!cache.lookup(0xDEADBEEF, &mut out));
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn lookup_below_min_obs_returns_false() {
        let mut cache = PredCache::with_params(true, 4, 3, 8);
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let hash = 0xAAAAu64;

        // Insert once (observations = 1, min_obs = 3)
        cache.insert(hash, &logits);
        let mut out = vec![0.0f32; 100];
        assert!(!cache.lookup(hash, &mut out));

        // Insert again (observations = 2, still < 3)
        cache.insert(hash, &logits);
        assert!(!cache.lookup(hash, &mut out));

        // Insert third time (observations = 3, now trusted)
        cache.insert(hash, &logits);
        assert!(cache.lookup(hash, &mut out));
    }

    #[test]
    fn disabled_cache_never_hits() {
        let mut cache = PredCache::new(false);
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
        cache.insert(0x1234, &logits);
        let mut out = vec![0.0f32; 100];
        assert!(!cache.lookup(0x1234, &mut out));
    }

    #[test]
    fn extract_top_k_accuracy() {
        let mut entry = CacheEntry::empty();
        let mut logits = vec![0.0f32; 256];
        // Set a few high values
        logits[42] = 10.0;
        logits[100] = 8.0;
        logits[200] = 6.0;
        logits[0] = 4.0;

        extract_top_k(&logits, &mut entry);

        // Check that top tokens are captured
        let top_tokens: Vec<u16> = entry.tokens.iter()
            .take(4)
            .copied()
            .collect();
        assert!(top_tokens.contains(&42), "should contain token 42");
        assert!(top_tokens.contains(&100), "should contain token 100");
        assert!(top_tokens.contains(&200), "should contain token 200");

        // Verify ordering: first should be highest
        assert_eq!(entry.tokens[0], 42);
        assert_eq!(entry.logits[0], 10.0);
    }

    #[test]
    fn reconstruct_preserves_top_k() {
        let mut entry = CacheEntry::empty();
        let mut logits = vec![1.0f32; 256];
        logits[42] = 10.0;
        logits[100] = 8.0;

        extract_top_k(&logits, &mut entry);

        let mut out = vec![0.0f32; 256];
        reconstruct_logits(&entry, &mut out);

        // Top tokens should have their original logits
        assert!((out[42] - 10.0).abs() < 1e-6);
        assert!((out[100] - 8.0).abs() < 1e-6);

        // Non-top tokens should have base_logit (close to 1.0 avg)
        // base_logit = avg of (256 - 32) non-top values
        assert!(out[50].is_finite());
    }

    #[test]
    fn collision_eviction() {
        let mut cache = PredCache::with_params(true, 4, 1, 8);
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Two hashes that map to same table slot
        let hash_a = 0u64;
        let hash_b = TABLE_SIZE as u64; // same lower bits
        assert_eq!((hash_a as usize) & TABLE_MASK,
                    (hash_b as usize) & TABLE_MASK);

        cache.insert(hash_a, &logits);
        // hash_a has 1 observation → hash_b can evict it
        cache.insert(hash_b, &logits);

        let mut out = vec![0.0f32; 100];
        // hash_b should be present
        assert!(cache.lookup(hash_b, &mut out));
    }

    #[test]
    fn high_obs_entry_survives_collision() {
        let mut cache = PredCache::with_params(true, 4, 1, 8);
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let hash_a = 0u64;
        let hash_b = TABLE_SIZE as u64;

        // Build up observations for hash_a
        cache.insert(hash_a, &logits);
        cache.insert(hash_a, &logits); // obs = 2

        // hash_b tries to evict but hash_a has obs > 1
        cache.insert(hash_b, &logits);

        let mut out = vec![0.0f32; 100];
        // hash_a should still be present
        assert!(cache.lookup(hash_a, &mut out));
        // hash_b was dropped (collision, couldn't evict)
        assert!(!cache.lookup(hash_b, &mut out));
    }

    #[test]
    fn hit_rate_tracking() {
        let mut cache = PredCache::with_params(true, 4, 1, 8);
        let logits: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 50];

        cache.insert(0x1111, &logits);
        assert!(cache.lookup(0x1111, &mut out));  // hit
        assert!(!cache.lookup(0x2222, &mut out)); // miss
        assert!(!cache.lookup(0x3333, &mut out)); // miss

        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 2);
        let rate = cache.hit_rate();
        assert!((rate - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn occupied_count() {
        let mut cache = PredCache::new(true);
        assert_eq!(cache.occupied(), 0);

        let logits: Vec<f32> = (0..50).map(|i| i as f32).collect();
        cache.insert(0xAAAA, &logits);
        cache.insert(0xBBBB, &logits);
        assert_eq!(cache.occupied(), 2);
    }

    #[test]
    fn large_vocab_reconstruct() {
        // Simulate Mamba's 50280 vocab
        let mut entry = CacheEntry::empty();
        let mut logits = vec![-5.0f32; 50280];
        logits[0] = 15.0;
        logits[256] = 12.0;
        logits[50000] = 10.0;

        extract_top_k(&logits, &mut entry);

        let mut out = vec![0.0f32; 50280];
        reconstruct_logits(&entry, &mut out);

        assert!((out[0] - 15.0).abs() < 1e-6);
        assert!((out[256] - 12.0).abs() < 1e-6);
        assert!((out[50000] - 10.0).abs() < 1e-6);
        // Non-top tokens should get base_logit (≈ -5.0)
        assert!((out[1000] - entry.base_logit).abs() < 1e-6);
        assert!(out[1000] < -4.0, "base logit should be close to -5.0");
    }

    #[test]
    fn end_to_end_with_token_stream() {
        let mut cache = PredCache::with_params(true, 4, 2, 8);
        let vocab = 100;

        // Repeated context: [10, 20, 30, 40] appears at positions 0..4 and 4..8
        let tokens = vec![10u32, 20, 30, 40, 10, 20, 30, 40, 50];
        let fake_logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0f32; vocab];

        // First occurrence at pos=4: context = [10,20,30,40]
        let h1 = cache.context_hash(&tokens, 4);
        assert!(!cache.lookup(h1, &mut out), "cold cache should miss");
        cache.insert(h1, &fake_logits);

        // Second occurrence at pos=8: context = [10,20,30,40] (same!)
        let h2 = cache.context_hash(&tokens, 8);
        assert_eq!(h1, h2, "same context should produce same hash");

        cache.insert(h2, &fake_logits); // obs = 2 now
        let hit = cache.lookup(h2, &mut out);
        assert!(hit, "should hit after 2 observations");
    }
}
