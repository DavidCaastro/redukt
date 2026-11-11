/// Content-type detection for preprocessing dispatch (R-028).
///
/// Analyzes the first 4-64KB of input data to classify content type,
/// enabling type-specific preprocessing (R-029 byte shuffle, R-030 delta, etc.).

use std::fmt;

use crate::domain::fmath::log_f64;

/// Detected content type for preprocessing dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Natural language text (dickens, webster, nci, source code)
    Text,
    /// Structured text with tags or fixed format (XML, logs, CSV)
    StructuredText,
    /// x86 executable (MZ/ELF magic detected)
    Executable,
    /// Float array — concentrated exponent bytes (MRI, x-ray)
    FloatArray,
    /// Near-random data, entropy > 7.0 bits/byte (sao, encrypted)
    HighEntropy,
    /// Generic binary not matching other categories
    GenericBinary,
}

impl fmt::Display for ContentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContentType::Text => write!(f, "text"),
            ContentType::StructuredText => write!(f, "structured-text"),
            ContentType::Executable => write!(f, "executable"),
            ContentType::FloatArray => write!(f, "float-array"),
            ContentType::HighEntropy => write!(f, "high-entropy"),
            ContentType::GenericBinary => write!(f, "generic-binary"),
        }
    }
}

/// Extracted features used for classification.
pub struct ContentFeatures {
    pub entropy: f64,
    pub alphabet_size: usize,
    pub printable_ratio: f64,
    pub xml_density: f64,
    pub exp_entropy: f64,
    pub exp_alphabet: usize,
    pub magic: MagicType,
}

/// Known file magic signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagicType {
    None,
    MZ,
    ELF,
    Xml,
}

impl fmt::Display for ContentFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "entropy={:.2} alpha={} print={:.3} xml={:.4} exp_ent={:.2} exp_a={} magic={:?}",
            self.entropy, self.alphabet_size, self.printable_ratio,
            self.xml_density, self.exp_entropy, self.exp_alphabet, self.magic,
        )
    }
}

/// Detect content type from raw bytes. Scans up to 8KB.
pub fn detect_content_type(data: &[u8]) -> (ContentType, ContentFeatures) {
    let features = extract_features(data);
    let ctype = classify(&features);
    (ctype, features)
}

/// Extract statistical features from data (first 8KB).
fn extract_features(data: &[u8]) -> ContentFeatures {
    let scan_len = data.len().min(8192);
    if scan_len == 0 {
        return ContentFeatures {
            entropy: 0.0, alphabet_size: 0, printable_ratio: 1.0,
            xml_density: 0.0, exp_entropy: 8.0, exp_alphabet: 256,
            magic: MagicType::None,
        };
    }
    let scan = &data[..scan_len];

    let mut freq = [0u32; 256];
    for &b in scan {
        freq[b as usize] += 1;
    }

    let n = scan_len as f64;
    let entropy = shannon_entropy(&freq, n);
    let alphabet_size = freq.iter().filter(|&&f| f > 0).count();

    let printable = scan.iter().filter(|&&b| is_printable(b)).count();
    let printable_ratio = printable as f64 / n;

    let xml_chars = scan.iter().filter(|&&b| b == b'<' || b == b'>').count();
    let xml_density = xml_chars as f64 / n;

    let (exp_entropy, exp_alphabet) = float_exponent_features(scan);
    let magic = detect_magic(scan);

    ContentFeatures {
        entropy,
        alphabet_size,
        printable_ratio,
        xml_density,
        exp_entropy,
        exp_alphabet,
        magic,
    }
}

/// Shannon entropy in bits/byte.
fn shannon_entropy(freq: &[u32; 256], n: f64) -> f64 {
    let mut h = 0.0f64;
    for &f in freq {
        if f > 0 {
            let p = f as f64 / n;
            // R-104 F3: fmath, not libm — the routing features must be identical on every platform.
            h -= p * (log_f64(p) * std::f64::consts::LOG2_E);
        }
    }
    h
}

/// Check if byte is printable ASCII (0x20-0x7E) or whitespace (tab, LF, CR).
fn is_printable(b: u8) -> bool {
    (0x20..=0x7E).contains(&b) || b == 0x0A || b == 0x0D || b == 0x09
}

/// Analyze float32 exponent byte pattern.
/// For IEEE 754 float32 arrays, byte at offset 3 of each 4-byte group
/// (sign+exponent) has very low entropy when values are in similar ranges.
fn float_exponent_features(data: &[u8]) -> (f64, usize) {
    if data.len() < 16 {
        return (8.0, 256);
    }
    let mut freq = [0u32; 256];
    let mut count = 0u32;
    for i in (3..data.len().min(4096)).step_by(4) {
        freq[data[i] as usize] += 1;
        count += 1;
    }
    if count == 0 {
        return (8.0, 256);
    }
    let entropy = shannon_entropy(&freq, count as f64);
    let alphabet = freq.iter().filter(|&&f| f > 0).count();
    (entropy, alphabet)
}

/// Detect known magic signatures.
fn detect_magic(data: &[u8]) -> MagicType {
    if data.len() < 4 {
        return MagicType::None;
    }
    // MZ (PE executable)
    if data[0] == 0x4D && data[1] == 0x5A {
        return MagicType::MZ;
    }
    // ELF
    if data.len() >= 4 && data[..4] == [0x7F, 0x45, 0x4C, 0x46] {
        return MagicType::ELF;
    }
    // XML declaration: <?xml
    if data.len() >= 5 && &data[..5] == b"<?xml" {
        return MagicType::Xml;
    }
    MagicType::None
}

/// Classify content type from extracted features.
fn classify(f: &ContentFeatures) -> ContentType {
    // 1. Magic number overrides
    if f.magic == MagicType::MZ || f.magic == MagicType::ELF {
        return ContentType::Executable;
    }
    if f.magic == MagicType::Xml {
        return ContentType::StructuredText;
    }

    // 2. Text family: printable ratio >= 0.90
    if f.printable_ratio >= 0.90 {
        if f.xml_density >= 0.04 {
            return ContentType::StructuredText;
        }
        // Low alphabet + high printable + low entropy = structured logs/data
        if f.alphabet_size < 48 && f.printable_ratio >= 0.99 && f.entropy < 5.2 {
            return ContentType::StructuredText;
        }
        return ContentType::Text;
    }

    // 3. High entropy (near-random or encrypted)
    if f.entropy >= 7.0 {
        return ContentType::HighEntropy;
    }

    // 4. Binary family (printable < 0.90)
    // Float-array: concentrated exponent bytes
    if f.exp_entropy < 4.0 && f.exp_alphabet < 45 {
        return ContentType::FloatArray;
    }

    ContentType::GenericBinary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_plain_text() {
        // Realistic text: varied vocabulary (>48 unique chars)
        let data = b"Hello, world! This is a test of plain English text. \
                     The quick brown fox jumps over the lazy dog. 0123456789 \
                     ABCDEFGHIJKLMNOPQRSTUVWXYZ ~@#$%^&*()_+-={}[]|;':\"<>?\n";
        let (ct, f) = detect_content_type(data);
        assert_eq!(ct, ContentType::Text);
        assert!(f.printable_ratio > 0.95);
        assert!(f.alphabet_size >= 48);
    }

    #[test]
    fn detect_xml() {
        let data = b"<?xml version=\"1.0\"?><root><item>value</item><item>other</item></root>";
        let (ct, _) = detect_content_type(data);
        assert_eq!(ct, ContentType::StructuredText);
    }

    #[test]
    fn detect_xml_by_density() {
        // No <?xml magic but high tag density
        let data = b"<html><head><title>Test</title></head><body><p>Hello</p></body></html>";
        let (ct, f) = detect_content_type(data);
        assert_eq!(ct, ContentType::StructuredText);
        assert!(f.xml_density >= 0.04);
    }

    #[test]
    fn detect_mz_executable() {
        let mut data = vec![0u8; 256];
        data[0] = 0x4D; // M
        data[1] = 0x5A; // Z
        let (ct, _) = detect_content_type(&data);
        assert_eq!(ct, ContentType::Executable);
    }

    #[test]
    fn detect_elf_executable() {
        let mut data = vec![0u8; 256];
        data[0] = 0x7F;
        data[1] = 0x45; // E
        data[2] = 0x4C; // L
        data[3] = 0x46; // F
        let (ct, _) = detect_content_type(&data);
        assert_eq!(ct, ContentType::Executable);
    }

    #[test]
    fn detect_float_array() {
        // Simulate float32 array: values near 1.0 (exponent byte = 0x3F)
        let mut data = vec![0u8; 4096];
        for i in (0..data.len()).step_by(4) {
            // Float ~1.0: mantissa varies, exponent stays 0x3F
            data[i] = (i & 0xFF) as u8;
            data[i + 1] = ((i >> 2) & 0xFF) as u8;
            data[i + 2] = 0x80;
            data[i + 3] = 0x3F;
        }
        let (ct, f) = detect_content_type(&data);
        assert_eq!(ct, ContentType::FloatArray);
        assert!(f.exp_entropy < 1.0);
    }

    #[test]
    fn detect_high_entropy() {
        // Near-random data: all 256 byte values roughly equal
        let mut data = vec![0u8; 8192];
        for (i, b) in data.iter_mut().enumerate() {
            // Simple PRNG-like distribution
            *b = ((i.wrapping_mul(1103515245).wrapping_add(12345)) >> 8) as u8;
        }
        let (ct, f) = detect_content_type(&data);
        assert_eq!(ct, ContentType::HighEntropy);
        assert!(f.entropy >= 7.0);
    }

    #[test]
    fn detect_structured_logs() {
        // Low alphabet, high printable, no XML
        let line = b"2026-01-01 12:00:00 INFO server started on port 8080\n";
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(line);
        }
        let (ct, f) = detect_content_type(&data);
        assert_eq!(ct, ContentType::StructuredText);
        assert!(f.alphabet_size < 55);
    }

    #[test]
    fn detect_generic_binary() {
        // Mixed binary content, moderate entropy (concentrated in 0-80 range)
        let mut data = vec![0u8; 1024];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i * 3 + 7) % 80) as u8;
        }
        let (ct, f) = detect_content_type(&data);
        assert_eq!(ct, ContentType::GenericBinary);
        assert!(f.entropy < 7.0);
        assert!(f.printable_ratio < 0.90);
    }

    #[test]
    fn display_content_type() {
        assert_eq!(format!("{}", ContentType::Text), "text");
        assert_eq!(format!("{}", ContentType::FloatArray), "float-array");
        assert_eq!(format!("{}", ContentType::HighEntropy), "high-entropy");
    }

    #[test]
    fn short_data_does_not_panic() {
        // Empty and tiny inputs should not panic; exact type is best-effort
        let (_, f) = detect_content_type(b"");
        assert_eq!(f.alphabet_size, 0);
        let (_, f2) = detect_content_type(b"ab");
        assert_eq!(f2.alphabet_size, 2);
    }

    #[test]
    fn features_display() {
        let (_, f) = detect_content_type(b"Hello world test data for display formatting");
        let s = format!("{}", f);
        assert!(s.contains("entropy="));
        assert!(s.contains("alpha="));
    }
}
