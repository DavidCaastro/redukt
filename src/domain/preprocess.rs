/// E8/E9 preprocessing transform for x86 executables.
///
/// Converts CALL (0xE8) and JMP (0xE9) relative addresses to absolute,
/// improving compression by clustering addresses to common targets.

/// Statistics from E8/E9 transform application.
pub struct E8E9Stats {
    pub e8_count: usize,
    pub e9_count: usize,
    pub transformed: usize,
    pub skipped: usize,
}

/// Apply E8/E9 forward transform with statistics tracking.
pub fn e8e9_encode_with_stats(data: &[u8]) -> (Vec<u8>, E8E9Stats) {
    let mut out = data.to_vec();
    let n = out.len();
    let mut stats = E8E9Stats {
        e8_count: 0,
        e9_count: 0,
        transformed: 0,
        skipped: 0,
    };
    let mut i = 0;
    while i < n.saturating_sub(4) {
        if out[i] == 0xE8 || out[i] == 0xE9 {
            if out[i] == 0xE8 { stats.e8_count += 1; } else { stats.e9_count += 1; }
            let rel = i32::from_le_bytes([out[i + 1], out[i + 2], out[i + 3], out[i + 4]]);
            let abs = rel.wrapping_add(i as i32);
            if abs >= 0 && (abs as usize) < n {
                let bytes = abs.to_le_bytes();
                out[i + 1] = bytes[0];
                out[i + 2] = bytes[1];
                out[i + 3] = bytes[2];
                out[i + 4] = bytes[3];
                stats.transformed += 1;
            } else {
                stats.skipped += 1;
            }
            i += 5;
        } else {
            i += 1;
        }
    }
    (out, stats)
}

/// Apply E8/E9 forward transform (relative → absolute addresses).
#[cfg(test)]
pub fn e8e9_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0;
    while i < n.saturating_sub(4) {
        if out[i] == 0xE8 || out[i] == 0xE9 {
            let rel = i32::from_le_bytes([out[i + 1], out[i + 2], out[i + 3], out[i + 4]]);
            let abs = rel.wrapping_add(i as i32);
            if abs >= 0 && (abs as usize) < n {
                let bytes = abs.to_le_bytes();
                out[i + 1] = bytes[0];
                out[i + 2] = bytes[1];
                out[i + 3] = bytes[2];
                out[i + 4] = bytes[3];
            }
            i += 5;
        } else {
            i += 1;
        }
    }
    out
}

/// Apply E8/E9 reverse transform (absolute → relative addresses).
#[cfg(test)]
pub fn e8e9_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0;
    while i < n.saturating_sub(4) {
        if out[i] == 0xE8 || out[i] == 0xE9 {
            let abs = i32::from_le_bytes([out[i + 1], out[i + 2], out[i + 3], out[i + 4]]);
            if abs >= 0 && (abs as usize) < n {
                let rel = abs.wrapping_sub(i as i32);
                let bytes = rel.to_le_bytes();
                out[i + 1] = bytes[0];
                out[i + 2] = bytes[1];
                out[i + 3] = bytes[2];
                out[i + 4] = bytes[3];
            }
            i += 5;
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8e9_roundtrip_identity() {
        let data = b"Hello, world! This is a test of non-executable data.";
        let encoded = e8e9_encode(data);
        let decoded = e8e9_decode(&encoded);
        assert_eq!(&decoded, data.as_slice());
    }

    #[test]
    fn e8e9_roundtrip_with_e8() {
        let mut data = vec![0u8; 100];
        data[10] = 0xE8;
        data[11] = 20;
        data[12] = 0;
        data[13] = 0;
        data[14] = 0;
        let encoded = e8e9_encode(&data);
        assert_eq!(i32::from_le_bytes([encoded[11], encoded[12], encoded[13], encoded[14]]), 30);
        let decoded = e8e9_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn e8e9_out_of_range_not_transformed() {
        let mut data = vec![0u8; 50];
        data[10] = 0xE8;
        data[11] = 100;
        data[12] = 0;
        data[13] = 0;
        data[14] = 0;
        let encoded = e8e9_encode(&data);
        assert_eq!(encoded[11], 100);
    }

    #[test]
    fn e8e9_negative_address() {
        let mut data = vec![0u8; 50];
        data[10] = 0xE8;
        data[11] = 0xFF;
        data[12] = 0xFF;
        data[13] = 0xFF;
        data[14] = 0xFF;
        let encoded = e8e9_encode(&data);
        assert_eq!(i32::from_le_bytes([encoded[11], encoded[12], encoded[13], encoded[14]]), 9);
        let decoded = e8e9_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn e8e9_stats_tracking() {
        let mut data = vec![0u8; 100];
        data[0] = 0xE8;
        data[1] = 10;
        data[5] = 0xE9;
        data[6] = 5;
        data[10] = 0xE8;
        data[11] = 200;
        let (_, stats) = e8e9_encode_with_stats(&data);
        assert_eq!(stats.e8_count, 2);
        assert_eq!(stats.e9_count, 1);
        assert_eq!(stats.transformed, 2);
        assert_eq!(stats.skipped, 1);
    }
}
