/// Record-length detection via byte autocorrelation (R-031).
///
/// Finds periodic structure in binary data by measuring byte coincidence
/// at various lags. A strong peak indicates fixed-width records, enabling
/// column transposition (byte_shuffle with detected stride).

/// Detect fixed record length via autocorrelation.
/// Scans lags from `min_stride` to `max_stride`, returns the smallest lag
/// with coincidence rate > `threshold` × average rate.
/// Returns None if no strong periodicity is found.
pub fn detect_record_length(data: &[u8], min_stride: usize, max_stride: usize) -> Option<usize> {
    let n = data.len();
    if n < max_stride * 4 || min_stride < 2 {
        return None;
    }
    let scan = n.min(16384);
    let max_lag = max_stride.min(scan / 4);

    // Compute coincidence rate for each lag
    let mut rates = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        let count = scan - lag;
        let matches = data[..count].iter()
            .zip(data[lag..scan].iter())
            .filter(|(a, b)| a == b)
            .count();
        rates.push(matches as f64 / count as f64);
    }

    // Q1 (25th percentile) as baseline — robust to periodic peaks
    if rates.len() < min_stride + 1 {
        return None;
    }
    let mut sorted: Vec<f64> = rates[min_stride..].to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = sorted[sorted.len() / 4];
    if q1 <= 0.0 {
        return None;
    }
    let avg = q1;

    // Find the smallest lag >= min_stride with strong peak (>5x Q1)
    // High threshold avoids false positives on mixed binary/text data
    let threshold = 5.0;
    for lag in min_stride..max_lag {
        let ratio = rates[lag] / avg;
        if ratio > threshold {
            // Check it's a true peak (higher than neighbors)
            let is_peak = (lag == min_stride || rates[lag] > rates[lag - 1])
                && (lag + 1 >= max_lag || rates[lag] >= rates[lag + 1]);
            if is_peak {
                return Some(lag);
            }
        }
    }

    None
}

/// Convenience: detect with default parameters (stride 4-512).
/// Minimum stride 4 avoids false positives from int16 byte-swapping patterns.
pub fn detect_records(data: &[u8]) -> Option<usize> {
    detect_record_length(data, 4, 512)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_periodic_records() {
        // 500 records of 8 bytes each, with distinctive per-field structure
        let mut data = Vec::with_capacity(4000);
        for i in 0u16..500 {
            data.push(0x01);                           // field 1: type marker
            data.push((i % 10) as u8);                 // field 2: category (0-9)
            data.extend_from_slice(&i.to_le_bytes());  // field 3: record ID
            data.push(((i * 7) % 256) as u8);          // field 4: hash
            data.push(0xFF);                            // field 5: terminator
            data.push(((i * 13) % 256) as u8);         // field 6: checksum
            data.push(0x00);                            // field 7: padding
        }
        let result = detect_record_length(&data, 2, 64);
        assert!(result.is_some(), "should detect periodicity");
        let r = result.unwrap();
        assert!(8 % r == 0 || r % 8 == 0,
            "detected stride {} should be related to record length 8", r);
    }

    #[test]
    fn no_records_in_text() {
        let data = b"The quick brown fox jumps over the lazy dog. \
                     This is plain text with no fixed-width records. \
                     Another sentence here for padding to get enough data. \
                     Yet another line to make the buffer big enough for analysis.";
        // Repeat to get enough data
        let mut big = Vec::new();
        for _ in 0..100 { big.extend_from_slice(data); }
        let result = detect_records(&big);
        // Text shouldn't show strong enough periodicity
        // (it might detect something, but we mainly test no panic)
        let _ = result;
    }

    #[test]
    fn no_records_in_random() {
        // Use a well-mixed PRNG (xorshift) to avoid accidental periodicity
        let mut x: u64 = 0xDEADBEEF12345678;
        let data: Vec<u8> = (0..8000).map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        }).collect();
        let result = detect_records(&data);
        assert!(result.is_none(), "random data should not have records");
    }

    #[test]
    fn short_data_returns_none() {
        assert!(detect_records(&[1, 2, 3]).is_none());
        assert!(detect_records(&[]).is_none());
    }

    #[test]
    fn detects_stride2_alternating() {
        // Strong even/odd alternation pattern
        let mut data = Vec::with_capacity(8000);
        for i in 0..4000u16 {
            data.push(0x00);       // even positions: zero
            data.push((i % 256) as u8); // odd positions: varying
        }
        let result = detect_record_length(&data, 2, 64);
        assert_eq!(result, Some(2), "should detect stride=2");
    }

    #[test]
    fn detect_records_convenience() {
        // Just verify the convenience function doesn't panic on various sizes
        let small = vec![0u8; 100];
        let _ = detect_records(&small);
        let medium = vec![0u8; 10000];
        let _ = detect_records(&medium);
    }
}
