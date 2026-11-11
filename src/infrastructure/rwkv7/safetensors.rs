//! SafeTensors parser — zero-dependency, read-only.
//!
//! SafeTensors format (https://github.com/huggingface/safetensors):
//!   [8 bytes LE u64: header_len]
//!   [header_len bytes: JSON header]
//!   [tensor data...]
//!
//! Header JSON maps tensor names to { dtype, shape, data_offsets: [start, end] }.
//! Offsets are relative to the start of the data section (after header).

use std::collections::HashMap;
use std::path::Path;

use crate::domain::error::{AzError, AzResult};
use crate::domain::tensor::Tensor;

#[derive(Debug)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

pub struct SafeTensorsFile {
    metas: HashMap<String, TensorMeta>,
    data_offset: usize, // byte offset in file where tensor data starts
    raw: Vec<u8>,
}

impl SafeTensorsFile {
    pub fn open(path: &Path) -> AzResult<Self> {
        let path_str = path.display().to_string();
        let raw = crate::domain::error::read_file_capped(&path_str)?;

        if raw.len() < 8 {
            return Err(AzError::ModelLoad(format!("{}: file too small for safetensors ({} bytes)", path_str, raw.len())));
        }

        // SAFETY: we just checked raw.len() >= 8
        let header_len = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;

        if raw.len() < header_end {
            return Err(AzError::ModelLoad(format!("{}: truncated, need {} bytes for header but file is {}", path_str, header_end, raw.len())));
        }

        let header_str = std::str::from_utf8(&raw[8..header_end])
            .map_err(|_| AzError::ModelLoad(format!("{}: header is not valid UTF-8", path_str)))?;

        let metas = parse_header(header_str)
            .map_err(|msg| AzError::ModelLoad(format!("{}: {}", path_str, msg)))?;

        Ok(Self { metas, data_offset: header_end, raw })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.metas.keys().map(|s| s.as_str()).collect()
    }

    /// Load a tensor by name, converting to f32.
    pub fn load_tensor(&self, name: &str) -> AzResult<Tensor> {
        let meta = self.metas.get(name)
            .ok_or_else(|| AzError::ModelLoad(format!("tensor '{}' not found in safetensors", name)))?;

        let start = self.data_offset + meta.data_start;
        let end = self.data_offset + meta.data_end;
        let bytes = &self.raw[start..end];

        let data = match meta.dtype.as_str() {
            "F32" => {
                bytes.chunks_exact(4)
                    // SAFETY: chunks_exact(4) guarantees exactly 4 bytes
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            }
            "F16" => {
                bytes.chunks_exact(2)
                    // SAFETY: chunks_exact(2) guarantees exactly 2 bytes
                    .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                    .collect()
            }
            "BF16" => {
                bytes.chunks_exact(2)
                    // SAFETY: chunks_exact(2) guarantees exactly 2 bytes
                    .map(|b| bf16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                    .collect()
            }
            other => return Err(AzError::ModelLoad(format!("unsupported dtype: {}", other))),
        };

        Ok(Tensor::from_data(data, meta.shape.clone()))
    }

    /// Load a tensor, panicking with a descriptive message on failure.
    /// Use this in model loading where all tensors are required.
    pub fn get_tensor(&self, name: &str) -> Tensor {
        self.load_tensor(name)
            .unwrap_or_else(|e| panic!("[safetensors] {}", e))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.metas.contains_key(name)
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16 → normal f32
        let mut f = frac as f32 / 1024.0;
        f *= 1.0 / 16384.0; // 2^-14, exact (no powi: R-104 F3 lint)
        if sign == 1 { -f } else { f }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::NAN
        }
    } else {
        let f32_exp = exp + 112; // bias adjustment: 127 - 15
        let f32_frac = frac << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac)
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---- Minimal JSON parser for safetensors header ----
// We only need to parse the specific structure safetensors uses.

fn parse_header(json: &str) -> Result<HashMap<String, TensorMeta>, String> {
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err("invalid safetensors header JSON".to_string());
    }

    let inner = &json[1..json.len() - 1];
    let entries = split_top_level_entries(inner);
    let mut result = HashMap::new();

    for entry in entries {
        let (key, value) = split_key_value(&entry)?;
        if key == "__metadata__" {
            continue;
        }
        if let Some(meta) = parse_tensor_meta(&value) {
            result.insert(key, meta);
        }
    }

    Ok(result)
}

fn split_top_level_entries(s: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let mut in_string = false;
    let mut escape = false;
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        if escape {
            escape = false;
            continue;
        }
        match chars[i] {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' | '[' if !in_string => depth += 1,
            '}' | ']' if !in_string => depth -= 1,
            ',' if !in_string && depth == 0 => {
                entries.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() {
        entries.push(last.to_string());
    }
    entries
}

fn split_key_value(s: &str) -> Result<(String, String), String> {
    // Find first ':' outside strings
    let mut in_string = false;
    let mut escape = false;
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        if escape { escape = false; continue; }
        match chars[i] {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            ':' if !in_string => {
                let key = extract_string(s[..i].trim());
                let value = s[i + 1..].trim().to_string();
                return Ok((key, value));
            }
            _ => {}
        }
    }
    Err(format!("no ':' found in entry: {}", s))
}

fn extract_string(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn parse_tensor_meta(json: &str) -> Option<TensorMeta> {
    let json = json.trim();
    if !json.starts_with('{') { return None; }

    let inner = &json[1..json.len() - 1];
    let entries = split_top_level_entries(inner);

    let mut dtype = String::new();
    let mut shape = Vec::new();
    let mut offsets = [0usize; 2];

    for entry in entries {
        let (key, value) = split_key_value(&entry).ok()?;
        match key.as_str() {
            "dtype" => dtype = extract_string(&value),
            "shape" => shape = parse_usize_array(&value),
            "data_offsets" => {
                let arr = parse_usize_array(&value);
                if arr.len() == 2 {
                    offsets = [arr[0], arr[1]];
                }
            }
            _ => {}
        }
    }

    if dtype.is_empty() { return None; }

    Some(TensorMeta {
        dtype,
        shape,
        data_start: offsets[0],
        data_end: offsets[1],
    })
}

fn parse_usize_array(s: &str) -> Vec<usize> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') { return Vec::new(); }
    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() { return Vec::new(); }
    inner.split(',')
        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
    }

    #[test]
    fn test_parse_header() {
        let header = r#"{"tensor1": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]}, "__metadata__": {"format": "pt"}}"#;
        let metas = parse_header(header).unwrap();
        assert!(metas.contains_key("tensor1"));
        assert!(!metas.contains_key("__metadata__"));
        let t = &metas["tensor1"];
        assert_eq!(t.dtype, "F32");
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(t.data_start, 0);
        assert_eq!(t.data_end, 48);
    }
}
