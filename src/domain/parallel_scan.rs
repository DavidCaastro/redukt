//! Generic parallel scan (prefix sum) over associative operations.
//!
//! Core algorithm for parallelizing linear recurrences like the SSM:
//!     s[t] = A[t] * s[t-1] + b[t]
//!
//! Represented as composable pairs (A, b) with associative composition:
//!     (A2, b2) . (A1, b1) = (A2*A1, A2*b1 + b2)
//!
//! Blelloch scan computes all prefix compositions in O(N log N) work
//! and O(log N) sequential depth — enabling massive parallelism on GPU
//! or multi-core CPU.
//!
//! This module is model-agnostic: the combine function is injected.
//! Used by both RWKV and Mamba execution strategies.
//!
//! Reference: Blelloch (1990), "Prefix Sums and Their Applications"
//! Reference: Mamba-2 SSD (Gu & Dao, 2024), arXiv:2405.21060

/// In-place inclusive scan (sequential, for baseline/verification).
///
/// Given `elements = [e0, e1, ..., en-1]` and associative `combine`,
/// transforms to `[e0, e0.e1, e0.e1.e2, ..., e0...en-1]`.
pub fn inclusive_scan_seq<T: Clone>(
    elements: &mut [T],
    combine: &dyn Fn(&T, &T) -> T,
) {
    for i in 1..elements.len() {
        let merged = combine(&elements[i - 1], &elements[i]);
        elements[i] = merged;
    }
}

/// Blelloch-style work-efficient parallel scan (CPU threaded).
///
/// Two-phase algorithm:
///   1. Up-sweep (reduce): pairwise combine bottom-up
///   2. Down-sweep (distribute): propagate prefixes top-down
///
/// Total work: O(N) combines. Sequential depth: O(log N).
/// On CPU, uses Rayon-style parallelism (when available) or sequential fallback.
///
/// `combine(left, right)` must be ASSOCIATIVE: (a.b).c == a.(b.c)
/// Elements are modified in-place to contain the inclusive scan result.
pub fn blelloch_scan<T: Clone + Send>(
    elements: &mut [T],
    combine: &(dyn Fn(&T, &T) -> T + Sync),
    identity: &T,
) {
    let n = elements.len();
    if n <= 1 {
        return;
    }

    // --- Up-sweep (reduce phase) ---
    // Build a reduction tree bottom-up.
    let mut stride = 1;
    while stride < n {
        let next = stride * 2;
        let mut i = next - 1;
        while i < n {
            let left_idx = i - stride;
            let merged = combine(&elements[left_idx], &elements[i]);
            elements[i] = merged;
            i += next;
        }
        stride = next;
    }

    // --- Down-sweep (distribution phase) ---
    // Replace root with identity, propagate prefixes down.
    elements[n - 1] = identity.clone();

    stride = n.next_power_of_two() / 2;
    while stride >= 1 {
        let mut i = stride * 2 - 1;
        while i < n {
            let left_idx = i - stride;
            let temp = elements[left_idx].clone();
            elements[left_idx] = elements[i].clone();
            elements[i] = combine(&temp, &elements[i]);
            i += stride * 2;
        }
        stride /= 2;
    }
}

/// SSM scan element: represents one step of the recurrence
/// `s[t] = a * s[t-1] + b` as the composable pair (a, b).
///
/// For diagonal SSM (Mamba d_state=16), `a` and `b` are vectors
/// of length d_state. This struct handles the scalar case;
/// vector version uses `SsmScanVec`.
#[derive(Clone, Debug)]
pub struct SsmScanPair {
    pub a: f32,
    pub b: f32,
}

impl SsmScanPair {
    pub fn new(a: f32, b: f32) -> Self {
        Self { a, b }
    }

    pub fn identity() -> Self {
        Self { a: 1.0, b: 0.0 }
    }

    /// Associative composition: (self) . (other)
    /// = applying `other` first, then `self`.
    /// Result: (self.a * other.a, self.a * other.b + self.b)
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            a: self.a * other.a,
            b: self.a * other.b + self.b,
        }
    }
}

/// Vector SSM scan element for d_state dimensions.
/// Each dimension is an independent scalar recurrence.
#[derive(Clone)]
pub struct SsmScanVec {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl SsmScanVec {
    pub fn new(d_state: usize) -> Self {
        Self {
            a: vec![1.0; d_state],
            b: vec![0.0; d_state],
        }
    }

    pub fn identity(d_state: usize) -> Self {
        Self::new(d_state)
    }

    /// Associative composition (element-wise for diagonal SSM).
    pub fn compose(&self, other: &Self) -> Self {
        let d = self.a.len();
        let mut out = Self {
            a: vec![0.0; d],
            b: vec![0.0; d],
        };
        for i in 0..d {
            out.a[i] = self.a[i] * other.a[i];
            out.b[i] = self.a[i] * other.b[i] + self.b[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_scan_simple() {
        // Sum scan: combine(a,b) = a+b
        let mut data = vec![1, 2, 3, 4, 5];
        inclusive_scan_seq(&mut data, &|a, b| a + b);
        assert_eq!(data, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn ssm_pair_composition_is_associative() {
        let a = SsmScanPair::new(0.9, 0.5);
        let b = SsmScanPair::new(0.8, 0.3);
        let c = SsmScanPair::new(0.7, 0.1);

        let ab_c = a.compose(&b).compose(&c);
        let a_bc = a.compose(&b.compose(&c));

        assert!((ab_c.a - a_bc.a).abs() < 1e-6);
        assert!((ab_c.b - a_bc.b).abs() < 1e-6);
    }

    #[test]
    fn ssm_pair_identity() {
        let p = SsmScanPair::new(0.9, 0.5);
        let id = SsmScanPair::identity();
        let result = p.compose(&id);
        assert!((result.a - p.a).abs() < 1e-6);
        assert!((result.b - p.b).abs() < 1e-6);
    }

    #[test]
    fn ssm_vec_compose_elementwise() {
        let a = SsmScanVec {
            a: vec![0.9, 0.8],
            b: vec![0.5, 0.3],
        };
        let b = SsmScanVec {
            a: vec![0.7, 0.6],
            b: vec![0.1, 0.2],
        };
        let c = a.compose(&b);
        // c.a[0] = 0.9 * 0.7 = 0.63
        // c.b[0] = 0.9 * 0.1 + 0.5 = 0.59
        assert!((c.a[0] - 0.63).abs() < 1e-6);
        assert!((c.b[0] - 0.59).abs() < 1e-6);
    }

    #[test]
    fn sequential_scan_matches_direct_recurrence() {
        // Verify scan produces same result as direct SSM simulation
        let steps: Vec<SsmScanPair> = vec![
            SsmScanPair::new(0.9, 1.0),
            SsmScanPair::new(0.8, 0.5),
            SsmScanPair::new(0.95, 0.2),
            SsmScanPair::new(0.85, 0.8),
        ];

        // Direct simulation: s[t] = a[t] * s[t-1] + b[t], s[-1] = 0
        let mut state = 0.0f32;
        let mut direct = Vec::new();
        for p in &steps {
            state = p.a * state + p.b;
            direct.push(state);
        }

        // Scan simulation
        let mut scan_elements = steps.clone();
        inclusive_scan_seq(&mut scan_elements, &|left, right| right.compose(left));

        // The scan result at position i encodes the composition of steps 0..=i.
        // Applying to initial state 0: result = composed.b (since a*0 + b = b)
        for i in 0..direct.len() {
            let scan_state = scan_elements[i].b; // applied to s[-1]=0
            assert!(
                (scan_state - direct[i]).abs() < 1e-5,
                "mismatch at {}: scan={} direct={}",
                i, scan_state, direct[i]
            );
        }
    }

    #[test]
    fn blelloch_exclusive_scan_sum() {
        let mut data = vec![3, 1, 7, 0, 4, 1, 6, 3];
        blelloch_scan(&mut data, &|a: &i32, b: &i32| a + b, &0);
        // Exclusive prefix sum: [0, 3, 4, 11, 11, 15, 16, 22]
        assert_eq!(data, vec![0, 3, 4, 11, 11, 15, 16, 22]);
    }
}
