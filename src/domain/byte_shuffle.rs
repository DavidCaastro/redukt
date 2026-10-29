/// Byte-plane shuffle for structured binary data (R-029).
///
/// Groups bytes by position within each N-byte stride:
/// [a0,a1,a2,a3, b0,b1,b2,b3, c0,c1,c2,c3]
/// → [a0,b0,c0, a1,b1,c1, a2,b2,c2, a3,b3,c3]
///
/// For float32 arrays, this clusters exponent bytes (low entropy)
/// together, dramatically improving compression.

/// Shuffle bytes by stride: group byte-plane 0, then plane 1, etc.
/// Trailing bytes (len % stride != 0) are appended unchanged.
pub fn byte_shuffle(data: &[u8], stride: usize) -> Vec<u8> {
    if stride <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let n = data.len();
    let full = n / stride;

    let mut out = Vec::with_capacity(n);

    // Shuffle complete groups: plane by plane
    for plane in 0..stride {
        for group in 0..full {
            out.push(data[group * stride + plane]);
        }
    }
    // Append any trailing bytes unchanged
    out.extend_from_slice(&data[full * stride..]);
    out
}

/// Inverse of byte_shuffle: restore original byte order from shuffled planes.
pub fn byte_unshuffle(data: &[u8], stride: usize) -> Vec<u8> {
    if stride <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let n = data.len();
    let full = n / stride;

    let mut out = vec![0u8; n];

    // Unshuffle: read plane by plane, write interleaved
    let mut src = 0;
    for plane in 0..stride {
        for group in 0..full {
            out[group * stride + plane] = data[src];
            src += 1;
        }
    }
    // Copy trailing bytes
    for i in (full * stride)..n {
        out[i] = data[src];
        src += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_stride4() {
        let data: Vec<u8> = (0..100).collect();
        let shuffled = byte_shuffle(&data, 4);
        assert_ne!(&shuffled, &data);
        let restored = byte_unshuffle(&shuffled, 4);
        assert_eq!(restored, data);
    }

    #[test]
    fn roundtrip_with_remainder() {
        let data: Vec<u8> = (0..103).collect(); // 103 = 25*4 + 3
        let shuffled = byte_shuffle(&data, 4);
        let restored = byte_unshuffle(&shuffled, 4);
        assert_eq!(restored, data);
    }

    #[test]
    fn roundtrip_stride2() {
        let data: Vec<u8> = (0..50).collect();
        let shuffled = byte_shuffle(&data, 2);
        let restored = byte_unshuffle(&shuffled, 2);
        assert_eq!(restored, data);
    }

    #[test]
    fn roundtrip_stride8() {
        let data: Vec<u8> = (0..64).collect();
        let shuffled = byte_shuffle(&data, 8);
        let restored = byte_unshuffle(&shuffled, 8);
        assert_eq!(restored, data);
    }

    #[test]
    fn stride1_is_identity() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(byte_shuffle(&data, 1), data);
        assert_eq!(byte_unshuffle(&data, 1), data);
    }

    #[test]
    fn empty_data() {
        assert_eq!(byte_shuffle(&[], 4), Vec::<u8>::new());
        assert_eq!(byte_unshuffle(&[], 4), Vec::<u8>::new());
    }

    #[test]
    fn shuffle_known_output() {
        // 3 groups of stride 4: [0,1,2,3, 4,5,6,7, 8,9,10,11]
        // Plane 0: [0,4,8], Plane 1: [1,5,9], Plane 2: [2,6,10], Plane 3: [3,7,11]
        let data: Vec<u8> = (0..12).collect();
        let expected = vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11];
        assert_eq!(byte_shuffle(&data, 4), expected);
    }

    #[test]
    fn float32_exponent_clustering() {
        // Simulate 3 float32 values near 1.0 (exponent byte = 0x3F at offset 3)
        let data = vec![
            0x00, 0x00, 0x80, 0x3F, // 1.0f32
            0x00, 0x00, 0xA0, 0x3F, // 1.25f32
            0x00, 0x00, 0xC0, 0x3F, // 1.5f32
        ];
        let shuffled = byte_shuffle(&data, 4);
        // Plane 3 (exponent bytes) should be clustered: [0x3F, 0x3F, 0x3F]
        assert_eq!(&shuffled[9..12], &[0x3F, 0x3F, 0x3F]);
        // Roundtrip
        assert_eq!(byte_unshuffle(&shuffled, 4), data);
    }

    #[test]
    fn large_roundtrip() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        for stride in [2, 4, 8, 16] {
            let restored = byte_unshuffle(&byte_shuffle(&data, stride), stride);
            assert_eq!(restored, data, "failed for stride {}", stride);
        }
    }
}
