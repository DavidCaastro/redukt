//! Run-length predictor for context mixing.
//!
//! Detects consecutive repetitions of the same token and predicts
//! continuation with confidence proportional to run length.
//! Highly effective for structured data with repeated patterns
//! (e.g., XML tags, log formats, binary null regions).

/// Run-length predictor: bias toward repeating the current token.
pub struct RunModel {
    vocab_size: usize,
    last_token: u32,
    run_length: usize,
}

impl RunModel {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size, last_token: u32::MAX, run_length: 0 }
    }

    /// Write run-length bias into `buf` (must be vocab_size).
    /// If current run >= 2, biases the repeated token with ln(run_length).
    pub fn compute_bias_into(&self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), self.vocab_size);
        for v in buf.iter_mut() { *v = 0.0; }

        if self.run_length >= 2 {
            let tok = self.last_token as usize;
            if tok < self.vocab_size {
                buf[tok] = crate::domain::tensor::portable_lnf(self.run_length as f32);
            }
        }
    }

    /// Observe a new token: update run state.
    pub fn observe(&mut self, token: u32) {
        if token == self.last_token {
            self.run_length += 1;
        } else {
            self.last_token = token;
            self.run_length = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_no_repeat() {
        let mut rm = RunModel::new(10);
        rm.observe(1);
        rm.observe(2);
        let mut buf = vec![0.0f32; 10];
        rm.compute_bias_into(&mut buf);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn run_repeat_detected() {
        let mut rm = RunModel::new(10);
        rm.observe(3);
        rm.observe(3);
        let mut buf = vec![0.0f32; 10];
        rm.compute_bias_into(&mut buf);
        assert!(buf[3] > 0.0, "expected positive bias for repeated token");
    }

    #[test]
    fn run_grows() {
        let mut rm = RunModel::new(10);
        rm.observe(5);
        rm.observe(5);
        let mut b1 = vec![0.0f32; 10];
        rm.compute_bias_into(&mut b1);
        let bias_2 = b1[5];

        rm.observe(5);
        rm.observe(5);
        let mut b2 = vec![0.0f32; 10];
        rm.compute_bias_into(&mut b2);
        let bias_4 = b2[5];

        assert!(bias_4 > bias_2, "longer run should give stronger bias");
    }

    #[test]
    fn run_broken() {
        let mut rm = RunModel::new(10);
        rm.observe(3);
        rm.observe(3);
        rm.observe(3);
        rm.observe(7); // break
        let mut buf = vec![0.0f32; 10];
        rm.compute_bias_into(&mut buf);
        assert!(buf.iter().all(|&v| v == 0.0), "run broken → no bias");
    }
}
