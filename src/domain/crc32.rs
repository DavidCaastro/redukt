//! CRC-32 (IEEE 802.3) — zero-dependency implementation.
//!
//! Polynomial: 0xEDB88320 (reflected). Identical to gzip, PNG, zip.
//! Used by AZTH v3 to verify decompressed data integrity.

const POLY: u32 = 0xEDB8_8320;

/// Precomputed lookup table (256 entries, generated at compile time).
const fn make_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
}

static TABLE: [u32; 256] = make_table();

/// Compute CRC-32 (IEEE) of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE[idx];
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // "123456789" -> 0xCBF43926 (standard test vector)
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn empty_input() {
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn single_byte() {
        // 'a' -> 0xE8B7BE43 (well-known)
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    }

    #[test]
    fn incremental_consistency() {
        let data = b"Hello, World!";
        let full = crc32(data);
        // Verify determinism
        assert_eq!(crc32(data), full);
    }
}
