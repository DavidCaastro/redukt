//! Byte-level tokenizer for MambaByte (vocab_size=256, identity mapping).
//!
//! Each byte maps directly to its u32 value. No merges, no tokenizer.json needed.

use crate::domain::backbone::BackendTokenizer;

pub struct ByteTokenizer {
    /// Precomputed single-byte slices for decode_token.
    bytes: [[u8; 1]; 256],
}

impl ByteTokenizer {
    pub fn new() -> Self {
        let mut bytes = [[0u8; 1]; 256];
        for i in 0..256 {
            bytes[i] = [i as u8];
        }
        Self { bytes }
    }

    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize { 256 }
}

impl BackendTokenizer for ByteTokenizer {
    fn encode(&self, data: &[u8]) -> Vec<u32> {
        data.iter().map(|&b| b as u32).collect()
    }

    fn decode_token(&self, id: u32) -> &[u8] {
        if (id as usize) < 256 {
            &self.bytes[id as usize]
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_roundtrip() {
        let tok = ByteTokenizer::new();
        let data = b"Hello, world! \x00\xff\x80";
        let encoded = tok.encode(data);
        let decoded: Vec<u8> = encoded.iter()
            .flat_map(|&id| tok.decode_token(id))
            .copied()
            .collect();
        assert_eq!(decoded, data);
    }

    #[test]
    fn byte_tokenizer_vocab_size() {
        let tok = ByteTokenizer::new();
        assert_eq!(tok.vocab_size(), 256);
    }

    #[test]
    fn byte_tokenizer_edge_cases() {
        let tok = ByteTokenizer::new();
        // Test all 256 byte values roundtrip
        let all_bytes: Vec<u8> = (0u16..256).map(|b| b as u8).collect();
        let encoded = tok.encode(&all_bytes);
        assert_eq!(encoded.len(), 256);
        for (i, &id) in encoded.iter().enumerate() {
            assert_eq!(id, i as u32);
            let dec = tok.decode_token(id);
            assert_eq!(dec, &[i as u8]);
        }
        // Out-of-range token returns empty
        assert_eq!(tok.decode_token(256), &[] as &[u8]);
        assert_eq!(tok.decode_token(u32::MAX), &[] as &[u8]);
    }
}
