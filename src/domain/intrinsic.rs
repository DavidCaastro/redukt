//! Intrinsic telemetry — runtime capture of mathematical properties.
//!
//! Captures properties that our mathematical objects inherently possess
//! but that the prediction pipeline currently ignores. Zero-copy piggyback
//! on existing computations where possible.
//!
//! R70: designed for <4 μs/byte overhead (~0.04% of 10ms/byte budget).
//!
//! Tier 0: piggyback on existing computation (0 extra cost)
//! Tier 1: running accumulators (1-3 μs/byte)
//! Tier 2: periodic computation (<1ms every N bytes)
//! Tier 3: snapshot (<10ms every 100K bytes)
//!
//! All data persisted as JSONL for offline analysis (Tier 4).

/// Maximum number of CM models tracked.
const MAX_MODELS: usize = 20;
/// Maximum LSTM hidden dimension tracked.
const MAX_HIDDEN: usize = 256;
/// Number of LSTM gates (f, g, o — coupled i=1-f not stored separately).
const N_GATES: usize = 3;

/// Intrinsic telemetry accumulator.
///
/// Lightweight struct that accumulates running statistics from the
/// prediction pipeline. Each field captures a mathematical property
/// that the pipeline computes but currently discards.
#[allow(dead_code)] // Phase 2 fields: adam_snr_high, table counters
pub struct IntrinsicTelemetry {
    // ─── Counters ───
    pub n_bits: u64,
    pub n_bytes: u64,

    // ─── Tier 0: Ensemble (pred_buf) statistics ───
    // Accumulated per-bit, piggyback on existing pred_buf construction.
    // Captures: consensus (variance), per-model reliability (mean prediction).
    pred_sum: [f64; MAX_MODELS],
    pred_sq_sum: [f64; MAX_MODELS],
    n_models_active: usize,

    // ─── Tier 0: CM hash table hit/miss tracking ───
    // Piggyback on existing table_predict/table_update calls.
    // Captures: occupancy, collision quality, information density.
    pub table_hits: [u64; MAX_MODELS],
    pub table_misses: [u64; MAX_MODELS],
    pub table_inserts: [u64; MAX_MODELS],
    pub table_evicts: [u64; MAX_MODELS],
    n_cm_models: usize,

    // ─── Tier 0: LSTM gate saturation ───
    // Piggyback on existing gate computation in LSTM forward pass.
    // Captures: what fraction of gates are "decided" (near 0 or 1).
    pub gate_sat_low: [u64; N_GATES],   // gate < 0.05
    pub gate_sat_high: [u64; N_GATES],  // gate > 0.95
    pub gate_total: u64,                // total gate evaluations

    // ─── Tier 0: Match model statistics ───
    // Piggyback on existing predict() call.
    pub match_len_hist: [u64; 129],  // histogram of match lengths 0-128
    pub match_miss_count: u64,       // no match found

    // ─── Tier 1: LSTM hidden state geometry ───
    // Computed per-byte from h vector already in L1 cache.
    // Captures: effective dimensionality, utilization.
    h_norm_sum: f64,
    h_norm_sq_sum: f64,
    h_norm_count: u64,

    // ─── Tier 1: LSTM cell state range ───
    // Captures: dynamic range, dead dimensions, saturation.
    c_min: [f32; MAX_HIDDEN],
    c_max: [f32; MAX_HIDDEN],
    c_sum: [f64; MAX_HIDDEN],
    c_count: u64,
    hidden_dim: usize,

    // ─── Tier 1: Low-confidence CM prediction count ───
    // Captures: what fraction of CM lookups have very few observations.
    pub low_confidence_count: u64,
    pub lookup_count: u64,

    // ─── Tier 2: Adam optimizer SNR (sampled) ───
    // Computed during adam_step, piggyback on existing m/v access.
    pub adam_snr_sum: f64,
    pub adam_snr_count: u64,
    pub adam_snr_low: u64,   // SNR < 0.01 (noise-dominated)
    pub adam_snr_high: u64,  // SNR > 1.0 (signal-dominated)
    pub adam_lr_eff_sum: f64,
    pub adam_lr_eff_count: u64,

    // ─── Snapshot ring buffer ───
    snapshots: Vec<IntrinsicSnapshot>,
    snapshot_interval: u64,
    last_snapshot_byte: u64,
}

/// A single point-in-time snapshot of intrinsic properties.
#[derive(Clone)]
#[allow(dead_code)] // Phase 2 field: occupancy_approx
pub struct IntrinsicSnapshot {
    pub byte_offset: u64,
    // Ensemble
    pub ensemble_consensus_var: f64,
    pub pred_means: Vec<f64>,
    pub pred_vars: Vec<f64>,
    // LSTM
    pub h_norm_mean: f64,
    pub h_norm_var: f64,
    pub c_dead_dims: usize,
    pub c_saturated_dims: usize,
    pub c_range_mean: f64,
    pub gate_sat_pct: [f64; N_GATES],
    // Adam
    pub adam_snr_mean: f64,
    pub adam_snr_low_pct: f64,
    pub adam_lr_eff_mean: f64,
    // CM
    pub hit_rates: Vec<f64>,
    pub occupancy_approx: Vec<f64>,
    pub low_confidence_pct: f64,
    // Match
    pub match_miss_pct: f64,
    pub match_len_median: usize,
}

#[allow(dead_code)] // Phase 2 methods: record_table_*, record_prediction, record_adam_state, record_slot_confidence
impl IntrinsicTelemetry {
    pub fn new(n_cm_models: usize, hidden_dim: usize, snapshot_interval: u64) -> Self {
        Self {
            n_bits: 0,
            n_bytes: 0,
            pred_sum: [0.0; MAX_MODELS],
            pred_sq_sum: [0.0; MAX_MODELS],
            n_models_active: 0,
            table_hits: [0; MAX_MODELS],
            table_misses: [0; MAX_MODELS],
            table_inserts: [0; MAX_MODELS],
            table_evicts: [0; MAX_MODELS],
            n_cm_models: n_cm_models.min(MAX_MODELS),
            gate_sat_low: [0; N_GATES],
            gate_sat_high: [0; N_GATES],
            gate_total: 0,
            match_len_hist: [0; 129],
            match_miss_count: 0,
            h_norm_sum: 0.0,
            h_norm_sq_sum: 0.0,
            h_norm_count: 0,
            c_min: [f32::INFINITY; MAX_HIDDEN],
            c_max: [f32::NEG_INFINITY; MAX_HIDDEN],
            c_sum: [0.0; MAX_HIDDEN],
            c_count: 0,
            hidden_dim: hidden_dim.min(MAX_HIDDEN),
            low_confidence_count: 0,
            lookup_count: 0,
            adam_snr_sum: 0.0,
            adam_snr_count: 0,
            adam_snr_low: 0,
            adam_snr_high: 0,
            adam_lr_eff_sum: 0.0,
            adam_lr_eff_count: 0,
            snapshots: Vec::with_capacity(1024),
            snapshot_interval,
            last_snapshot_byte: 0,
        }
    }

    // ─── Tier 0: piggyback calls (inline, nanoseconds) ───

    /// Record a prediction from model k. Call once per model per bit.
    #[inline]
    pub fn record_prediction(&mut self, model_idx: usize, p: f32) {
        if model_idx < MAX_MODELS {
            let pf = p as f64;
            self.pred_sum[model_idx] += pf;
            self.pred_sq_sum[model_idx] += pf * pf;
        }
        self.n_bits += 1;
    }

    /// Batch-record all predictions for one bit. More efficient than per-model.
    #[inline]
    pub fn record_predictions_batch(&mut self, preds: &[f32]) {
        let n = preds.len().min(MAX_MODELS);
        self.n_models_active = n;
        for k in 0..n {
            let pf = preds[k] as f64;
            self.pred_sum[k] += pf;
            self.pred_sq_sum[k] += pf * pf;
        }
        self.n_bits += 1;
    }

    /// Record CM table lookup result. Call from table_predict.
    #[inline]
    pub fn record_table_hit(&mut self, model_idx: usize) {
        if model_idx < MAX_MODELS {
            self.table_hits[model_idx] += 1;
        }
    }

    #[inline]
    pub fn record_table_miss(&mut self, model_idx: usize) {
        if model_idx < MAX_MODELS {
            self.table_misses[model_idx] += 1;
        }
    }

    /// Record slot effective sample size from a lookup.
    #[inline]
    pub fn record_slot_confidence(&mut self, c0: u16, c1: u16, scale: u16) {
        self.lookup_count += 1;
        let n_eff = (c0 as f32 + c1 as f32) / scale as f32;
        if n_eff < 2.0 {
            self.low_confidence_count += 1;
        }
    }

    /// Record CM table eviction. Call from table_update.
    #[inline]
    pub fn record_table_evict(&mut self, model_idx: usize) {
        if model_idx < MAX_MODELS {
            self.table_evicts[model_idx] += 1;
        }
    }

    #[inline]
    pub fn record_table_insert(&mut self, model_idx: usize) {
        if model_idx < MAX_MODELS {
            self.table_inserts[model_idx] += 1;
        }
    }

    /// Record LSTM gate values for saturation tracking.
    /// f_gates, g_gates, o_gates are slices of length hidden_dim.
    #[inline]
    pub fn record_gate_saturation(&mut self, f_gates: &[f32], o_gates: &[f32]) {
        for &f in f_gates {
            if f < 0.05 { self.gate_sat_low[0] += 1; }
            if f > 0.95 { self.gate_sat_high[0] += 1; }
        }
        // g_gate (tanh) saturates at ±1, tracked differently — skip for now
        for &o in o_gates {
            if o < 0.05 { self.gate_sat_low[2] += 1; }
            if o > 0.95 { self.gate_sat_high[2] += 1; }
        }
        self.gate_total += f_gates.len() as u64;
    }

    /// Record match model prediction result.
    #[inline]
    pub fn record_match(&mut self, match_len: usize) {
        if match_len == 0 {
            self.match_miss_count += 1;
        }
        let idx = match_len.min(128);
        self.match_len_hist[idx] += 1;
    }

    // ─── Tier 1: per-byte accumulators (microseconds) ───

    /// Record LSTM hidden state norm. Call once per byte (after last bit).
    /// h is already in L1 cache from the forward pass.
    #[inline]
    pub fn record_h_state(&mut self, h: &[f32]) {
        let mut norm_sq = 0.0f64;
        for &v in h {
            norm_sq += (v as f64) * (v as f64);
        }
        self.h_norm_sum += norm_sq.sqrt();
        self.h_norm_sq_sum += norm_sq;
        self.h_norm_count += 1;
    }

    /// Record LSTM cell state. Call once per byte.
    #[inline]
    pub fn record_c_state(&mut self, c: &[f32]) {
        let hd = c.len().min(self.hidden_dim);
        for i in 0..hd {
            if c[i] < self.c_min[i] { self.c_min[i] = c[i]; }
            if c[i] > self.c_max[i] { self.c_max[i] = c[i]; }
            self.c_sum[i] += c[i] as f64;
        }
        self.c_count += 1;
    }

    /// Record byte boundary (call once per byte processed).
    pub fn record_byte(&mut self) {
        self.n_bytes += 1;
        if self.snapshot_interval > 0 && self.n_bytes - self.last_snapshot_byte >= self.snapshot_interval {
            self.take_snapshot();
            self.last_snapshot_byte = self.n_bytes;
        }
    }

    // ─── Tier 2: Adam SNR sampling ───

    /// Record Adam optimizer state for a sampled weight.
    /// Call during adam_step for a subset of weights.
    #[inline]
    pub fn record_adam_state(&mut self, m: f32, v: f32, lr: f32) {
        if !m.is_finite() || !v.is_finite() || v <= 0.0 { return; }
        let snr = (m * m) as f64 / v as f64;
        self.adam_snr_sum += snr;
        self.adam_snr_count += 1;
        if snr < 0.01 { self.adam_snr_low += 1; }
        if snr > 1.0 { self.adam_snr_high += 1; }
        let lr_eff = (lr * m.abs() / (v.sqrt() + 1e-6)) as f64;
        self.adam_lr_eff_sum += lr_eff;
        self.adam_lr_eff_count += 1;
    }

    // ─── Tier 3: Snapshots ───

    fn take_snapshot(&mut self) {
        let nb = self.n_bits.max(1) as f64;
        let n_m = self.n_models_active.max(1);

        // Ensemble stats
        let mut pred_means = Vec::with_capacity(n_m);
        let mut pred_vars = Vec::with_capacity(n_m);
        let mut total_var = 0.0;
        for k in 0..n_m {
            let mean = self.pred_sum[k] / nb;
            let var = (self.pred_sq_sum[k] / nb) - mean * mean;
            pred_means.push(mean);
            pred_vars.push(var.max(0.0));
            total_var += var.max(0.0);
        }
        let consensus_var = total_var / n_m as f64;

        // LSTM h stats
        let h_count = self.h_norm_count.max(1) as f64;
        let h_norm_mean = self.h_norm_sum / h_count;
        let h_norm_var = (self.h_norm_sq_sum / h_count) - (self.h_norm_sum / h_count).powi(2);

        // Cell state stats
        let hd = self.hidden_dim;
        let cc = self.c_count.max(1) as f64;
        let mut c_dead = 0usize;
        let mut c_sat = 0usize;
        let mut range_sum = 0.0f64;
        for i in 0..hd {
            let range = (self.c_max[i] - self.c_min[i]) as f64;
            range_sum += range;
            if range < 0.01 { c_dead += 1; }
            let mean = self.c_sum[i] / cc;
            if mean.abs() > 5.0 { c_sat += 1; }
        }

        // Gate saturation
        let gt = self.gate_total.max(1) as f64;
        let gate_sat_pct = [
            self.gate_sat_high[0] as f64 / gt,
            self.gate_sat_low[1] as f64 / gt,  // g_gate: not tracked, 0
            self.gate_sat_high[2] as f64 / gt,
        ];

        // Adam stats
        let adam_c = self.adam_snr_count.max(1) as f64;
        let adam_snr_mean = self.adam_snr_sum / adam_c;
        let adam_snr_low_pct = self.adam_snr_low as f64 / adam_c;
        let adam_lr_c = self.adam_lr_eff_count.max(1) as f64;
        let adam_lr_eff_mean = self.adam_lr_eff_sum / adam_lr_c;

        // CM hit rates
        let mut hit_rates = Vec::with_capacity(self.n_cm_models);
        let mut occupancy_approx = Vec::with_capacity(self.n_cm_models);
        for m in 0..self.n_cm_models {
            let total = self.table_hits[m] + self.table_misses[m];
            hit_rates.push(if total > 0 { self.table_hits[m] as f64 / total as f64 } else { 0.0 });
            let ins = self.table_inserts[m];
            let ev = self.table_evicts[m];
            occupancy_approx.push((ins as f64 - ev as f64).max(0.0));
        }

        // Low confidence
        let lc = self.lookup_count.max(1) as f64;
        let low_conf_pct = self.low_confidence_count as f64 / lc;

        // Match stats
        let total_matches: u64 = self.match_len_hist.iter().sum();
        let tm = total_matches.max(1);
        let miss_pct = self.match_miss_count as f64 / tm as f64;
        let median = self.compute_match_median();

        self.snapshots.push(IntrinsicSnapshot {
            byte_offset: self.n_bytes,
            ensemble_consensus_var: consensus_var,
            pred_means,
            pred_vars,
            h_norm_mean,
            h_norm_var: h_norm_var.max(0.0),
            c_dead_dims: c_dead,
            c_saturated_dims: c_sat,
            c_range_mean: if hd > 0 { range_sum / hd as f64 } else { 0.0 },
            gate_sat_pct,
            adam_snr_mean,
            adam_snr_low_pct,
            adam_lr_eff_mean,
            hit_rates,
            occupancy_approx,
            low_confidence_pct: low_conf_pct,
            match_miss_pct: miss_pct,
            match_len_median: median,
        });
    }

    fn compute_match_median(&self) -> usize {
        let total: u64 = self.match_len_hist.iter().sum();
        if total == 0 { return 0; }
        let half = total / 2;
        let mut acc = 0u64;
        for (i, &count) in self.match_len_hist.iter().enumerate() {
            acc += count;
            if acc >= half { return i; }
        }
        128
    }

    /// Number of snapshots accumulated.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Serialize all snapshots to JSONL format.
    pub fn to_jsonl(&self) -> String {
        let mut lines = Vec::with_capacity(self.snapshots.len());
        for snap in &self.snapshots {
            lines.push(snap.to_json());
        }
        lines.join("\n")
    }

    /// Summary statistics as JSON (for RESULT line or final output).
    pub fn summary_json(&self) -> String {
        let nb = self.n_bits.max(1) as f64;
        let n_m = self.n_models_active.max(1);

        let mut total_var = 0.0;
        for k in 0..n_m {
            let mean = self.pred_sum[k] / nb;
            let var = (self.pred_sq_sum[k] / nb) - mean * mean;
            total_var += var.max(0.0);
        }
        let consensus = total_var / n_m as f64;

        let gt = self.gate_total.max(1) as f64;
        let f_sat = self.gate_sat_high[0] as f64 / gt;
        let o_sat = self.gate_sat_high[2] as f64 / gt;

        let h_count = self.h_norm_count.max(1) as f64;
        let h_mean = self.h_norm_sum / h_count;

        let lc = self.lookup_count.max(1) as f64;
        let low_conf = self.low_confidence_count as f64 / lc;

        let tm: u64 = self.match_len_hist.iter().sum();
        let miss_pct = if tm > 0 { self.match_miss_count as f64 / tm as f64 } else { 1.0 };

        format!(
            r#"{{"n_bytes":{},"n_bits":{},"ensemble_consensus_var":{:.6},"gate_f_sat_pct":{:.4},"gate_o_sat_pct":{:.4},"h_norm_mean":{:.4},"cm_low_confidence_pct":{:.4},"match_miss_pct":{:.4},"match_len_median":{},"snapshots":{}}}"#,
            self.n_bytes, self.n_bits,
            consensus, f_sat, o_sat,
            h_mean, low_conf, miss_pct,
            self.compute_match_median(),
            self.snapshots.len(),
        )
    }
}

impl IntrinsicSnapshot {
    fn to_json(&self) -> String {
        let pm: Vec<String> = self.pred_means.iter().map(|v| format!("{:.4}", v)).collect();
        let pv: Vec<String> = self.pred_vars.iter().map(|v| format!("{:.6}", v)).collect();
        let hr: Vec<String> = self.hit_rates.iter().map(|v| format!("{:.4}", v)).collect();

        format!(
            r#"{{"byte":{},"ensemble_var":{:.6},"pred_means":[{}],"pred_vars":[{}],"h_norm_mean":{:.4},"h_norm_var":{:.6},"c_dead":{},"c_sat":{},"c_range":{:.4},"gate_sat":[{:.4},{:.4},{:.4}],"adam_snr":{:.4},"adam_snr_low_pct":{:.4},"adam_lr_eff":{:.6},"hit_rates":[{}],"low_conf_pct":{:.4},"match_miss_pct":{:.4},"match_median":{}}}"#,
            self.byte_offset,
            self.ensemble_consensus_var,
            pm.join(","), pv.join(","),
            self.h_norm_mean, self.h_norm_var,
            self.c_dead_dims, self.c_saturated_dims, self.c_range_mean,
            self.gate_sat_pct[0], self.gate_sat_pct[1], self.gate_sat_pct[2],
            self.adam_snr_mean, self.adam_snr_low_pct, self.adam_lr_eff_mean,
            hr.join(","),
            self.low_confidence_pct,
            self.match_miss_pct, self.match_len_median,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_accumulation() {
        let mut t = IntrinsicTelemetry::new(3, 128, 0);
        let preds = [0.3f32, 0.7, 0.5];
        t.record_predictions_batch(&preds);
        assert_eq!(t.n_bits, 1);
        assert!((t.pred_sum[0] - 0.3).abs() < 1e-6);
        assert!((t.pred_sum[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_gate_saturation() {
        let mut t = IntrinsicTelemetry::new(3, 4, 0);
        let f = [0.01, 0.5, 0.99, 0.03];
        let o = [0.96, 0.5, 0.04, 0.97];
        t.record_gate_saturation(&f, &o);
        assert_eq!(t.gate_sat_low[0], 2);  // f < 0.05: 0.01, 0.03
        assert_eq!(t.gate_sat_high[0], 1); // f > 0.95: 0.99
        assert_eq!(t.gate_sat_high[2], 2); // o > 0.95: 0.96, 0.97
        assert_eq!(t.gate_sat_low[2], 1);  // o < 0.05: 0.04
    }

    #[test]
    fn test_h_state() {
        let mut t = IntrinsicTelemetry::new(3, 4, 0);
        let h = [1.0f32, 0.0, 0.0, 0.0]; // norm = 1.0
        t.record_h_state(&h);
        assert!((t.h_norm_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_match_histogram() {
        let mut t = IntrinsicTelemetry::new(3, 4, 0);
        t.record_match(0);  // miss
        t.record_match(4);
        t.record_match(16);
        t.record_match(16);
        assert_eq!(t.match_miss_count, 1);
        assert_eq!(t.match_len_hist[0], 1);
        assert_eq!(t.match_len_hist[4], 1);
        assert_eq!(t.match_len_hist[16], 2);
        assert_eq!(t.compute_match_median(), 4); // 0, 4, 16, 16 → median at index 2
    }

    #[test]
    fn test_snapshot() {
        let mut t = IntrinsicTelemetry::new(3, 4, 10);
        let preds = [0.3f32, 0.7, 0.5];
        for _ in 0..80 {
            t.record_predictions_batch(&preds);
        }
        let h = [1.0f32, 0.0, 0.0, 0.0];
        let c = [0.5f32, -0.5, 3.0, -3.0];
        for _ in 0..10 {
            t.record_h_state(&h);
            t.record_c_state(&c);
            t.record_byte();
        }
        assert!(t.snapshot_count() >= 1);
        let jsonl = t.to_jsonl();
        assert!(jsonl.contains("\"byte\":"));
    }

    #[test]
    fn test_summary_json() {
        let mut t = IntrinsicTelemetry::new(2, 4, 0);
        let preds = [0.5f32, 0.5];
        t.record_predictions_batch(&preds);
        t.record_byte();
        let json = t.summary_json();
        assert!(json.contains("\"n_bytes\":1"));
    }
}
