/// Delta coding for structured binary data (R-030).
///
/// Computes byte-level differences: `out[i] = in[i] - in[i-stride]` (wrapping).
/// Applied AFTER byte-plane shuffle for maximum effect — consecutive same-plane
/// bytes become small deltas, highly compressible.

/// Delta encode: `out[i] = in[i].wrapping_sub(in[i - stride])` for i >= stride.
/// First `stride` bytes are copied unchanged.
pub fn delta_encode(data: &[u8], stride: usize) -> Vec<u8> {
    if stride == 0 || stride >= data.len() {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..stride]);
    for i in stride..data.len() {
        out.push(data[i].wrapping_sub(data[i - stride]));
    }
    out
}

/// Delta decode (inverse): `out[i] = in[i].wrapping_add(out[i - stride])` for i >= stride.
pub fn delta_decode(data: &[u8], stride: usize) -> Vec<u8> {
    if stride == 0 || stride >= data.len() {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..stride]);
    for i in stride..data.len() {
        out.push(data[i].wrapping_add(out[i - stride]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_stride1() {
        let data: Vec<u8> = (0..100).collect();
        let encoded = delta_encode(&data, 1);
        assert_ne!(&encoded, &data);
        let decoded = delta_decode(&encoded, 1);
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_stride4() {
        let data: Vec<u8> = (0..200).map(|i| (i * 7 % 256) as u8).collect();
        let decoded = delta_decode(&delta_encode(&data, 4), 4);
        assert_eq!(decoded, data);
    }

    #[test]
    fn constant_data_becomes_zeros() {
        let data = vec![42u8; 50];
        let encoded = delta_encode(&data, 1);
        // First byte unchanged, rest are 0 (42-42=0)
        assert_eq!(encoded[0], 42);
        assert!(encoded[1..].iter().all(|&b| b == 0));
    }

    #[test]
    fn linear_ramp_becomes_constant() {
        // [0,1,2,3,...,99] → delta stride=1 → [0,1,1,1,...,1]
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let encoded = delta_encode(&data, 1);
        assert_eq!(encoded[0], 0);
        assert!(encoded[1..].iter().all(|&b| b == 1));
    }

    #[test]
    fn wrapping_arithmetic() {
        // 0 - 255 = 1 (wrapping)
        let data = vec![255u8, 0];
        let encoded = delta_encode(&data, 1);
        assert_eq!(encoded, vec![255, 1]); // 0 - 255 = 1 (wrapping_sub)
        let decoded = delta_decode(&encoded, 1);
        assert_eq!(decoded, data);
    }

    #[test]
    fn empty_and_short() {
        assert_eq!(delta_encode(&[], 1), Vec::<u8>::new());
        assert_eq!(delta_decode(&[], 1), Vec::<u8>::new());
        assert_eq!(delta_encode(&[5], 1), vec![5]);
        assert_eq!(delta_decode(&[5], 1), vec![5]);
    }

    #[test]
    fn stride_larger_than_data() {
        let data = vec![1, 2, 3];
        assert_eq!(delta_encode(&data, 10), data);
        assert_eq!(delta_decode(&data, 10), data);
    }

    #[test]
    fn large_roundtrip() {
        let data: Vec<u8> = (0..10000).map(|i| ((i * 13 + 7) % 256) as u8).collect();
        for stride in [1, 2, 4, 8] {
            let decoded = delta_decode(&delta_encode(&data, stride), stride);
            assert_eq!(decoded, data, "failed for stride {}", stride);
        }
    }

    #[test]
    fn shuffle_then_delta_roundtrip() {
        // Simulate float32 data: shuffle stride=4 then delta stride=1
        use crate::domain::byte_shuffle::{byte_shuffle, byte_unshuffle};
        let data: Vec<u8> = (0..400).map(|i| ((i * 3 + i / 4) % 256) as u8).collect();
        let shuffled = byte_shuffle(&data, 4);
        let deltad = delta_encode(&shuffled, 1);
        // Reverse
        let undelta = delta_decode(&deltad, 1);
        let unshuffled = byte_unshuffle(&undelta, 4);
        assert_eq!(unshuffled, data);
    }
}
