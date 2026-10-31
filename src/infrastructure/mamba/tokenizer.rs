//! BPE tokenizer for GPT-NeoX / Mamba models (50,280 vocab).
//!
//! Loads from HuggingFace tokenizer.json format (vocab + merges).
//! Zero external dependencies — parses JSON manually.

use std::collections::HashMap;
use std::path::Path;

use crate::domain::backbone::BackendTokenizer;

pub struct BpeTokenizer {
    /// Token string → token ID
    encoder: HashMap<Vec<u8>, u32>,
    /// Token ID → token bytes
    decoder: Vec<Vec<u8>>,
    /// Merge pairs: (token_a, token_b) → merged result, ordered by priority
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Byte-to-unicode mapping (GPT-2 style)
    byte_encoder: [u8; 256],
    byte_decoder: HashMap<char, u8>,
    /// Fallback: byte → token ID for bytes not in encoder (0xC0, 0xC1, 0xF5-0xFF)
    byte_fallback: [u32; 256],
}

impl BpeTokenizer {
    /// Load from a directory containing tokenizer.json (HuggingFace format).
    pub fn load(path: &Path) -> Self {
        let tok_path = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        eprintln!("[mamba-bpe] loading from {} ...", tok_path.display());

        let content = std::fs::read_to_string(&tok_path)
            .unwrap_or_else(|e| panic!("[mamba-bpe] failed to read: {}", e));

        // Build byte<->unicode mapping (GPT-2 bytes_to_unicode)
        let (byte_enc, byte_dec) = build_byte_unicode_map();

        // Parse vocab and merges from tokenizer.json
        let (vocab_map, merges_raw) = parse_tokenizer_json(&content);

        let vocab_size = vocab_map.iter().map(|(_, id)| *id).max().unwrap_or(0) as usize + 1;
        let mut decoder: Vec<Vec<u8>> = vec![Vec::new(); vocab_size];
        let mut encoder: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab_map.len());

        for (token_str, id) in &vocab_map {
            // Convert from unicode representation back to bytes
            let bytes: Vec<u8> = token_str.chars()
                .map(|c| *byte_dec.get(&c).unwrap_or(&0))
                .collect();
            if (*id as usize) < decoder.len() {
                decoder[*id as usize] = bytes.clone();
            }
            encoder.insert(bytes, *id);
        }

        let merges: Vec<(Vec<u8>, Vec<u8>)> = merges_raw.iter().map(|(a, b)| {
            let a_bytes: Vec<u8> = a.chars()
                .map(|c| *byte_dec.get(&c).unwrap_or(&0))
                .collect();
            let b_bytes: Vec<u8> = b.chars()
                .map(|c| *byte_dec.get(&c).unwrap_or(&0))
                .collect();
            (a_bytes, b_bytes)
        }).collect();

        // Ensure all 256 single-byte tokens exist in encoder.
        // GPT-NeoX mamba-130m is missing 13 bytes (0xC0, 0xC1, 0xF5-0xFF).
        // Add synthetic tokens so binary data roundtrips losslessly.
        let mut next_id = decoder.len() as u32;
        let mut added = 0u32;
        for b in 0u16..256 {
            let key = vec![b as u8];
            if !encoder.contains_key(&key) {
                // Extend decoder if needed
                while (next_id as usize) >= decoder.len() {
                    decoder.push(Vec::new());
                }
                decoder[next_id as usize] = key.clone();
                encoder.insert(key, next_id);
                next_id += 1;
                added += 1;
            }
        }
        if added > 0 {
            eprintln!("[mamba-bpe] added {} synthetic byte tokens (IDs {}..{})",
                      added, next_id - added, next_id - 1);
        }

        // Build byte_fallback lookup: byte → token ID for fast single-byte encoding
        let mut byte_fallback = [0u32; 256];
        for b in 0u16..256 {
            byte_fallback[b as usize] = *encoder.get(&vec![b as u8])
                .expect("all 256 bytes must have tokens after synthetic addition");
        }

        eprintln!(
            "[mamba-bpe] loaded: {} vocab entries, {} merges",
            encoder.len(), merges.len()
        );

        Self { encoder, decoder, merges, byte_encoder: byte_enc, byte_decoder: byte_dec, byte_fallback }
    }

    /// Encode bytes to token IDs using BPE.
    pub fn encode(&self, data: &[u8]) -> Vec<u32> {
        if data.is_empty() { return Vec::new(); }

        // Split into words on whitespace boundaries (simplified GPT-2 pre-tokenization)
        let words = pre_tokenize(data);
        let mut all_tokens = Vec::new();

        for word in &words {
            // Convert each byte to its unicode character representation
            let mut symbols: Vec<Vec<u8>> = word.iter()
                .map(|&b| vec![b])
                .collect();

            // Apply merges iteratively
            self.apply_merges(&mut symbols);

            // Look up each symbol in the encoder
            for sym in &symbols {
                if let Some(&id) = self.encoder.get(sym) {
                    all_tokens.push(id);
                } else {
                    // Fallback: encode individual bytes via guaranteed byte_fallback
                    for &b in sym.iter() {
                        all_tokens.push(self.byte_fallback[b as usize]);
                    }
                }
            }
        }

        all_tokens
    }

    fn apply_merges(&self, symbols: &mut Vec<Vec<u8>>) {
        if symbols.len() < 2 { return; }

        // For each merge rule (in priority order), scan and apply
        for (a, b) in &self.merges {
            let mut i = 0;
            while i + 1 < symbols.len() {
                if symbols[i] == *a && symbols[i + 1] == *b {
                    let mut merged = symbols[i].clone();
                    merged.extend_from_slice(&symbols[i + 1]);
                    symbols[i] = merged;
                    symbols.remove(i + 1);
                    // Don't increment — check if new symbol can merge with next
                } else {
                    i += 1;
                }
            }
            if symbols.len() < 2 { break; }
        }
    }

    pub fn decode_token(&self, id: u32) -> &[u8] {
        if (id as usize) < self.decoder.len() {
            &self.decoder[id as usize]
        } else {
            &[]
        }
    }

    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.decoder.len()
    }

    /// Verify encode→decode roundtrip. Returns None if OK, or
    /// Some((expected_len, actual_len, first_diff_pos)) on failure.
    #[cfg(test)]
    pub fn verify_roundtrip(&self, data: &[u8]) -> Option<(usize, usize, usize)> {
        let tokens = self.encode(data);
        let mut decoded = Vec::with_capacity(data.len());
        for &tok in &tokens {
            decoded.extend_from_slice(self.decode_token(tok));
        }
        if decoded.len() != data.len() {
            return Some((data.len(), decoded.len(), 0));
        }
        for (i, (&a, &b)) in data.iter().zip(decoded.iter()).enumerate() {
            if a != b {
                return Some((data.len(), decoded.len(), i));
            }
        }
        None
    }
}

impl BackendTokenizer for BpeTokenizer {
    fn encode(&self, data: &[u8]) -> Vec<u32> { self.encode(data) }
    fn decode_token(&self, id: u32) -> &[u8] { self.decode_token(id) }
}

/// GPT-2 bytes_to_unicode mapping.
/// Maps byte values 0-255 to unicode characters, avoiding control characters.
fn build_byte_unicode_map() -> ([u8; 256], HashMap<char, u8>) {
    // The GPT-2 bytes_to_unicode mapping:
    // Printable ASCII bytes map to themselves as unicode codepoints.
    // Non-printable bytes map to codepoints starting at 256.
    let mut byte_to_char: [char; 256] = ['\0'; 256];
    let mut next_cp: u32 = 256;

    for i in 0u32..256 {
        let b = i as u8;
        let ch = if (b'!'..=b'~').contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b)
        {
            char::from_u32(i).unwrap()
        } else {
            let c = char::from_u32(next_cp).unwrap();
            next_cp += 1;
            c
        };
        byte_to_char[i as usize] = ch;
    }

    // Build reverse map: char -> byte
    let mut char_to_byte = HashMap::with_capacity(256);
    let mut byte_enc = [0u8; 256];
    for i in 0..256 {
        char_to_byte.insert(byte_to_char[i], i as u8);
        byte_enc[i] = i as u8; // identity — actual encoding is via char mapping
    }

    (byte_enc, char_to_byte)
}

/// Simple pre-tokenization: split on whitespace boundaries.
/// Leading space is attached to the following word (GPT-2 style).
fn pre_tokenize(data: &[u8]) -> Vec<Vec<u8>> {
    let mut words: Vec<Vec<u8>> = Vec::new();
    let mut current = Vec::new();

    for &b in data {
        if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            // Space becomes part of next word (GPT-2 style: Ġ prefix)
            current.push(b);
        } else {
            current.push(b);
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Minimal JSON parser for tokenizer.json — extracts vocab and merges.
fn parse_tokenizer_json(json: &str) -> (Vec<(String, u32)>, Vec<(String, String)>) {
    let mut vocab = Vec::new();
    let mut merges = Vec::new();

    // Find "vocab": { ... } section
    if let Some(vocab_start) = json.find("\"vocab\"") {
        let after = &json[vocab_start..];
        if let Some(brace_start) = after.find('{') {
            let inner = &after[brace_start + 1..];
            let brace_end = find_matching_brace(inner);
            let vocab_str = &inner[..brace_end];
            parse_vocab_entries(vocab_str, &mut vocab);
        }
    }

    // Find "merges": [ ... ] section
    if let Some(merges_start) = json.find("\"merges\"") {
        let after = &json[merges_start..];
        if let Some(bracket_start) = after.find('[') {
            let inner = &after[bracket_start + 1..];
            let bracket_end = find_matching_bracket(inner);
            let merges_str = &inner[..bracket_end];
            parse_merge_entries(merges_str, &mut merges);
        }
    }

    (vocab, merges)
}

fn find_matching_brace(s: &str) -> usize {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, c) in s.char_indices() {
        if escape_next { escape_next = false; continue; }
        if c == '\\' && in_string { escape_next = true; continue; }
        if c == '"' { in_string = !in_string; continue; }
        if in_string { continue; }
        if c == '{' { depth += 1; }
        if c == '}' {
            if depth == 0 { return i; }
            depth -= 1;
        }
    }
    s.len()
}

fn find_matching_bracket(s: &str) -> usize {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, c) in s.char_indices() {
        if escape_next { escape_next = false; continue; }
        if c == '\\' && in_string { escape_next = true; continue; }
        if c == '"' { in_string = !in_string; continue; }
        if in_string { continue; }
        if c == '[' { depth += 1; }
        if c == ']' {
            if depth == 0 { return i; }
            depth -= 1;
        }
    }
    s.len()
}

fn parse_vocab_entries(s: &str, vocab: &mut Vec<(String, u32)>) {
    // Parse entries like: "token_string": id, ...
    let mut pos = 0;
    let bytes = s.as_bytes();
    while pos < bytes.len() {
        // Find next quoted string
        let Some(q_start) = find_next_char(bytes, pos, b'"') else { break };
        let Some(q_end) = find_string_end(bytes, q_start + 1) else { break };
        let key = unescape_json_string(&s[q_start + 1..q_end]);

        // Find colon then number
        let Some(colon) = find_next_char(bytes, q_end + 1, b':') else { break };
        let (num, num_end) = parse_number(bytes, colon + 1);

        vocab.push((key, num));
        pos = num_end;
    }
}

fn parse_merge_entries(s: &str, merges: &mut Vec<(String, String)>) {
    // Parse entries like: "token_a token_b", ...
    let mut pos = 0;
    let bytes = s.as_bytes();
    while pos < bytes.len() {
        let Some(q_start) = find_next_char(bytes, pos, b'"') else { break };
        let Some(q_end) = find_string_end(bytes, q_start + 1) else { break };
        let entry = unescape_json_string(&s[q_start + 1..q_end]);

        // Split on first space
        if let Some(space_pos) = entry.find(' ') {
            let a = entry[..space_pos].to_string();
            let b = entry[space_pos + 1..].to_string();
            merges.push((a, b));
        }

        pos = q_end + 1;
    }
}

fn find_next_char(bytes: &[u8], start: usize, target: u8) -> Option<usize> {
    for i in start..bytes.len() {
        if bytes[i] == target { return Some(i); }
    }
    None
}

fn find_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i < bytes.len() {
        if bytes[i] == b'\\' { i += 2; continue; }
        if bytes[i] == b'"' { return Some(i); }
        i += 1;
    }
    None
}

fn parse_number(bytes: &[u8], start: usize) -> (u32, usize) {
    let mut i = start;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r') {
        i += 1;
    }
    let num_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let num_str = std::str::from_utf8(&bytes[num_start..i]).unwrap_or("0");
    (num_str.parse().unwrap_or(0), i)
}

fn unescape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('/') => result.push('/'),
                Some('u') => {
                    let mut hex = String::new();
                    for _ in 0..4 {
                        if let Some(h) = chars.next() { hex.push(h); }
                    }
                    let cp = u32::from_str_radix(&hex, 16).unwrap_or(0);
                    if let Some(ch) = char::from_u32(cp) {
                        result.push(ch);
                    }
                }
                Some(other) => { result.push('\\'); result.push(other); }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_unicode_map_roundtrip() {
        let (_enc, dec) = build_byte_unicode_map();
        // All 256 bytes must be mappable
        assert_eq!(dec.len(), 256);
    }

    #[test]
    fn test_pre_tokenize_basic() {
        let words = pre_tokenize(b"hello world");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], b"hello");
        assert_eq!(words[1], b" world");
    }

    #[test]
    fn test_unescape_json() {
        assert_eq!(unescape_json_string(r#"hello\"world"#), "hello\"world");
        assert_eq!(unescape_json_string(r#"\n\t"#), "\n\t");
    }
}
