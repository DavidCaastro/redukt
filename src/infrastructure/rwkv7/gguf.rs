//! GGUF parser — zero-dependency, read-only.
//!
//! GGUF format (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md):
//!   [4 bytes: magic "GGUF"]
//!   [4 bytes LE u32: version]
//!   [8 bytes LE u64: tensor_count]
//!   [8 bytes LE u64: metadata_kv_count]
//!   [metadata key-value pairs...]
//!   [tensor info entries...]
//!   [alignment padding]
//!   [tensor data...]
//!
//! Supports versions 2 and 3.
//! Tensor dequantization: F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, Q4_K, Q5_K, Q6_K → f32.

use std::collections::HashMap;
use std::path::Path;

use crate::domain::error::{AzError, AzResult};
use crate::domain::tensor::Tensor;
use crate::domain::quant::{GgufQuantTensor, ggml_block_info};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in LE

// GGML tensor types we support
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q4_1: u32 = 3;
const GGML_TYPE_Q5_0: u32 = 6;
const GGML_TYPE_Q5_1: u32 = 7;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_BF16: u32 = 30;

// GGUF metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// Alignment for tensor data (standard: 32 bytes).
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// Security limits to prevent OOM from crafted GGUF files
const MAX_TENSOR_COUNT: usize = 100_000;
const MAX_KV_COUNT: usize = 100_000;
const MAX_ARRAY_COUNT: usize = 10_000_000;
const MAX_ARRAY_DEPTH: usize = 4;
const MAX_NDIMS: usize = 8;
const MAX_NUMEL: usize = 1_000_000_000; // ~4 GB of f32

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

#[allow(dead_code)]
impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::U64(v) => u32::try_from(*v).ok(),
            GgufValue::I32(v) => if *v >= 0 { Some(*v as u32) } else { None },
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

struct TensorInfo {
    dims: Vec<u64>,
    dtype: u32,
    offset: u64, // relative to data section start
}

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, TensorInfo>,
    data_offset: usize, // absolute byte offset where tensor data starts
    raw: Vec<u8>,
}

/// Cursor for reading binary data sequentially.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> AzResult<u8> {
        if self.remaining() < 1 {
            return Err(AzError::ModelLoad("GGUF: unexpected EOF reading u8".into()));
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> AzResult<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> AzResult<u16> {
        if self.remaining() < 2 {
            return Err(AzError::ModelLoad("GGUF: unexpected EOF reading u16".into()));
        }
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> AzResult<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> AzResult<u32> {
        if self.remaining() < 4 {
            return Err(AzError::ModelLoad("GGUF: unexpected EOF reading u32".into()));
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> AzResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> AzResult<u64> {
        if self.remaining() < 8 {
            return Err(AzError::ModelLoad("GGUF: unexpected EOF reading u64".into()));
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> AzResult<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> AzResult<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> AzResult<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> AzResult<String> {
        let len = self.read_u64()? as usize;
        if self.remaining() < len {
            return Err(AzError::ModelLoad(format!(
                "GGUF: unexpected EOF reading string of len {}", len
            )));
        }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|_| AzError::ModelLoad("GGUF: invalid UTF-8 in string".into()))?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn read_bool(&mut self) -> AzResult<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_value(&mut self, vtype: u32) -> AzResult<GgufValue> {
        self.read_value_depth(vtype, 0)
    }

    fn read_value_depth(&mut self, vtype: u32, depth: usize) -> AzResult<GgufValue> {
        match vtype {
            GGUF_TYPE_UINT8 => Ok(GgufValue::U8(self.read_u8()?)),
            GGUF_TYPE_INT8 => Ok(GgufValue::I8(self.read_i8()?)),
            GGUF_TYPE_UINT16 => Ok(GgufValue::U16(self.read_u16()?)),
            GGUF_TYPE_INT16 => Ok(GgufValue::I16(self.read_i16()?)),
            GGUF_TYPE_UINT32 => Ok(GgufValue::U32(self.read_u32()?)),
            GGUF_TYPE_INT32 => Ok(GgufValue::I32(self.read_i32()?)),
            GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(self.read_f32()?)),
            GGUF_TYPE_BOOL => Ok(GgufValue::Bool(self.read_bool()?)),
            GGUF_TYPE_STRING => Ok(GgufValue::Str(self.read_string()?)),
            GGUF_TYPE_UINT64 => Ok(GgufValue::U64(self.read_u64()?)),
            GGUF_TYPE_INT64 => Ok(GgufValue::I64(self.read_i64()?)),
            GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(self.read_f64()?)),
            GGUF_TYPE_ARRAY => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(AzError::ModelLoad(format!(
                        "GGUF: array nesting too deep (max {})", MAX_ARRAY_DEPTH
                    )));
                }
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                if count > MAX_ARRAY_COUNT {
                    return Err(AzError::ModelLoad(format!(
                        "GGUF: array count {} exceeds limit {}", count, MAX_ARRAY_COUNT
                    )));
                }
                let mut arr = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    arr.push(self.read_value_depth(elem_type, depth + 1)?);
                }
                Ok(GgufValue::Array(arr))
            }
            other => Err(AzError::ModelLoad(format!("GGUF: unknown value type {}", other))),
        }
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> AzResult<Self> {
        let path_str = path.display().to_string();
        let raw = crate::domain::error::read_file_capped(&path_str)?;

        if raw.len() < 24 {
            return Err(AzError::ModelLoad(format!(
                "{}: file too small for GGUF ({} bytes)", path_str, raw.len()
            )));
        }

        let mut cur = Cursor::new(&raw);

        // Magic
        let magic = cur.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(AzError::ModelLoad(format!(
                "{}: not a GGUF file (magic {:08x}, expected {:08x})", path_str, magic, GGUF_MAGIC
            )));
        }

        // Version
        let version = cur.read_u32()?;
        if version < 2 || version > 3 {
            return Err(AzError::ModelLoad(format!(
                "{}: unsupported GGUF version {} (supported: 2, 3)", path_str, version
            )));
        }

        let tensor_count = cur.read_u64()? as usize;
        let kv_count = cur.read_u64()? as usize;

        if tensor_count > MAX_TENSOR_COUNT {
            return Err(AzError::ModelLoad(format!(
                "{}: tensor_count {} exceeds safety limit {}", path_str, tensor_count, MAX_TENSOR_COUNT
            )));
        }
        if kv_count > MAX_KV_COUNT {
            return Err(AzError::ModelLoad(format!(
                "{}: kv_count {} exceeds safety limit {}", path_str, kv_count, MAX_KV_COUNT
            )));
        }

        eprintln!("[gguf] version={}, tensors={}, metadata_kv={}", version, tensor_count, kv_count);

        // Read metadata
        let mut metadata = HashMap::new();
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        for _ in 0..kv_count {
            let key = cur.read_string()?;
            let vtype = cur.read_u32()?;
            let value = cur.read_value(vtype)?;

            if key == "general.alignment" {
                if let Some(a) = value.as_u32() {
                    alignment = a as usize;
                }
            }
            metadata.insert(key, value);
        }

        // Read tensor info
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()?;
            if n_dims as usize > MAX_NDIMS {
                return Err(AzError::ModelLoad(format!(
                    "tensor '{}': n_dims {} exceeds limit {}", name, n_dims, MAX_NDIMS
                )));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            let dtype = cur.read_u32()?;
            let offset = cur.read_u64()?;

            tensors.insert(name, TensorInfo { dims, dtype, offset });
        }

        // Validate alignment (must be power of 2 and > 0)
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(AzError::ModelLoad(format!(
                "{}: invalid alignment {} (must be power of 2)", path_str, alignment
            )));
        }

        // Data section starts after header, aligned
        let data_offset = align_up(cur.pos, alignment);

        if data_offset > raw.len() {
            return Err(AzError::ModelLoad(format!(
                "{}: data offset {} exceeds file size {}", path_str, data_offset, raw.len()
            )));
        }

        eprintln!("[gguf] data starts at offset {} ({:.1} MB header)",
                  data_offset, data_offset as f64 / 1_048_576.0);

        // Report architecture info
        if let Some(arch) = metadata.get("general.architecture") {
            eprintln!("[gguf] architecture: {:?}", arch);
        }
        if let Some(name) = metadata.get("general.name") {
            eprintln!("[gguf] model name: {:?}", name);
        }

        Ok(Self { metadata, tensors, data_offset, raw })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Load a tensor by name, dequantizing to f32.
    pub fn load_tensor(&self, name: &str) -> AzResult<Tensor> {
        let info = self.tensors.get(name)
            .ok_or_else(|| AzError::ModelLoad(format!("tensor '{}' not found in GGUF", name)))?;

        let numel: usize = info.dims.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d as usize)
        }).ok_or_else(|| AzError::ModelLoad(format!(
            "tensor '{}': dimension product overflow ({:?})", name, info.dims
        )))?;
        if numel > MAX_NUMEL {
            return Err(AzError::ModelLoad(format!(
                "tensor '{}': numel {} exceeds limit {}", name, numel, MAX_NUMEL
            )));
        }
        let abs_offset = self.data_offset.checked_add(info.offset as usize)
            .ok_or_else(|| AzError::ModelLoad(format!(
                "tensor '{}': offset overflow (data_offset={}, tensor_offset={})",
                name, self.data_offset, info.offset
            )))?;

        let data = match info.dtype {
            GGML_TYPE_F32 => {
                let byte_len = numel * 4;
                self.check_bounds(name, abs_offset, byte_len)?;
                let bytes = &self.raw[abs_offset..abs_offset + byte_len];
                bytes.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            }
            GGML_TYPE_F16 => {
                let byte_len = numel * 2;
                self.check_bounds(name, abs_offset, byte_len)?;
                let bytes = &self.raw[abs_offset..abs_offset + byte_len];
                bytes.chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                    .collect()
            }
            GGML_TYPE_BF16 => {
                let byte_len = numel * 2;
                self.check_bounds(name, abs_offset, byte_len)?;
                let bytes = &self.raw[abs_offset..abs_offset + byte_len];
                bytes.chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                    .collect()
            }
            GGML_TYPE_Q4_0 => {
                dequantize_q4_0(&self.raw, abs_offset, numel)?
            }
            GGML_TYPE_Q4_1 => {
                dequantize_q4_1(&self.raw, abs_offset, numel)?
            }
            GGML_TYPE_Q5_0 => {
                dequantize_q5_0(&self.raw, abs_offset, numel)?
            }
            GGML_TYPE_Q5_1 => {
                dequantize_q5_1(&self.raw, abs_offset, numel)?
            }
            GGML_TYPE_Q8_0 => {
                dequantize_q8_0(&self.raw, abs_offset, numel)?
            }
            GGML_TYPE_Q4_K => {
                dequantize_q4_k(&self.raw, abs_offset, numel)?
            }
            GGML_TYPE_Q5_K => {
                dequantize_q5_k(&self.raw, abs_offset, numel)?
            }
            GGML_TYPE_Q6_K => {
                dequantize_q6_k(&self.raw, abs_offset, numel)?
            }
            other => {
                return Err(AzError::ModelLoad(format!(
                    "tensor '{}': unsupported GGML type {} (supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K)",
                    name, other
                )));
            }
        };

        // GGUF ne[] is reversed PyTorch shape (ne[0]=innermost dim).
        // Data is stored in PyTorch/C row-major order, so reverse dims to get correct shape.
        let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        Ok(Tensor::from_data(data, shape))
    }

    /// Load a tensor, panicking with a descriptive message on failure.
    pub fn get_tensor(&self, name: &str) -> Tensor {
        self.load_tensor(name)
            .unwrap_or_else(|e| panic!("[gguf] {}", e))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get a metadata value by key.
    pub fn get_meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get model config from GGUF metadata.
    /// Returns (n_layer, n_embd, n_head, head_size, vocab_size) if available.
    pub fn detect_config(&self) -> Option<(usize, usize, usize, usize, usize)> {
        // Try RWKV-specific keys first (llama.cpp RWKV converter)
        let arch = self.get_meta("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("rwkv");

        let prefix = arch;
        let n_layer = self.get_meta(&format!("{}.block_count", prefix))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)?;
        let n_embd = self.get_meta(&format!("{}.embedding_length", prefix))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)?;

        // head_count may not always be present — derive from n_embd / head_size
        let head_size = self.get_meta(&format!("{}.attention.key_length", prefix))
            .or_else(|| self.get_meta(&format!("{}.wkv.head_size", prefix)))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(64);
        let n_head = self.get_meta(&format!("{}.attention.head_count", prefix))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .filter(|&v| v > 0)
            .unwrap_or(n_embd / head_size);
        let vocab_size = self.get_meta(&format!("{}.vocab_size", prefix))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(65536);

        Some((n_layer, n_embd, n_head, head_size, vocab_size))
    }

    /// Load a tensor's raw quantized bytes without dequantizing to f32.
    /// Returns a GgufQuantTensor for native quantized matmul.
    /// Only supports 2D tensors with Q8_0, Q4_K, Q5_K, Q6_K types.
    pub fn load_tensor_raw(&self, name: &str) -> AzResult<GgufQuantTensor> {
        let info = self.tensors.get(name)
            .ok_or_else(|| AzError::ModelLoad(format!("tensor '{}' not found in GGUF", name)))?;

        if info.dims.len() != 2 {
            return Err(AzError::ModelLoad(format!(
                "tensor '{}': native quant requires 2D, got {}D", name, info.dims.len()
            )));
        }

        // GGUF ne[]: ne[0]=cols(innermost), ne[1]=rows
        let cols = info.dims[0] as usize;
        let rows = info.dims[1] as usize;

        let (block_size, block_bytes) = ggml_block_info(info.dtype)
            .ok_or_else(|| AzError::ModelLoad(format!(
                "tensor '{}': unsupported GGML type {} for native quant", name, info.dtype
            )))?;

        if cols % block_size != 0 {
            return Err(AzError::ModelLoad(format!(
                "tensor '{}': cols {} not divisible by block_size {}", name, cols, block_size
            )));
        }

        let blocks_per_row = cols / block_size;
        let row_bytes = blocks_per_row * block_bytes;
        let total_bytes = rows * row_bytes;

        let abs_offset = self.data_offset.checked_add(info.offset as usize)
            .ok_or_else(|| AzError::ModelLoad(format!(
                "tensor '{}': offset overflow", name
            )))?;
        self.check_bounds(name, abs_offset, total_bytes)?;

        let raw = self.raw[abs_offset..abs_offset + total_bytes].to_vec();

        Ok(GgufQuantTensor { raw, ggml_type: info.dtype, rows, cols, row_bytes })
    }

    /// Get the GGML type ID for a tensor (for deciding native vs dequant loading).
    pub fn tensor_ggml_type(&self, name: &str) -> Option<u32> {
        self.tensors.get(name).map(|info| info.dtype)
    }

    fn check_bounds(&self, name: &str, offset: usize, len: usize) -> AzResult<()> {
        let end = offset.checked_add(len).ok_or_else(|| AzError::ModelLoad(format!(
            "tensor '{}': bounds overflow (offset={}, len={})", name, offset, len
        )))?;
        if end > self.raw.len() {
            Err(AzError::ModelLoad(format!(
                "tensor '{}': data [{}, {}) exceeds file size {}",
                name, offset, end, self.raw.len()
            )))
        } else {
            Ok(())
        }
    }
}

/// Compute n_blocks * block_bytes with overflow checking.
fn checked_dequant_size(numel: usize, block_size: usize, block_bytes: usize, name: &str) -> AzResult<(usize, usize)> {
    let n_blocks = (numel + block_size - 1) / block_size;
    let total_bytes = n_blocks.checked_mul(block_bytes).ok_or_else(|| AzError::ModelLoad(format!(
        "{}: dequant size overflow (n_blocks={}, block_bytes={})", name, n_blocks, block_bytes
    )))?;
    Ok((n_blocks, total_bytes))
}

/// Dequantize Q4_0 blocks to f32.
/// Q4_0 format: block_size=32, each block = f16 scale (2 bytes) + 16 bytes (32 × 4-bit) = 18 bytes.
/// Each byte holds two 4-bit unsigned values (low nibble first), offset by -8 to get signed range.
fn dequantize_q4_0(raw: &[u8], offset: usize, numel: usize) -> AzResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + BLOCK_SIZE / 2; // f16 scale + 16 bytes of nibbles

    let (n_blocks, total_bytes) = checked_dequant_size(numel, BLOCK_SIZE, BLOCK_BYTES, "Q4_0")?;

    if offset + total_bytes > raw.len() {
        return Err(AzError::ModelLoad(format!(
            "Q4_0: need {} bytes at offset {}, but file is {} bytes",
            total_bytes, offset, raw.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    let mut pos = offset;

    for _ in 0..n_blocks {
        let scale = f16_to_f32(u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap()));
        pos += 2;

        let remaining = BLOCK_SIZE.min(numel - out.len());
        for j in 0..BLOCK_SIZE / 2 {
            let byte = raw[pos + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            if j * 2 < remaining {
                out.push(lo as f32 * scale);
            }
            if j * 2 + 1 < remaining {
                out.push(hi as f32 * scale);
            }
        }
        pos += BLOCK_SIZE / 2;
    }

    Ok(out)
}

/// Dequantize Q4_1 blocks to f32.
/// Q4_1 format: block_size=32, each block = f16 scale (2) + f16 min (2) + 16 bytes = 20 bytes.
/// value = nibble * scale + min (no offset subtraction, min handles the bias).
fn dequantize_q4_1(raw: &[u8], offset: usize, numel: usize) -> AzResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 2 + BLOCK_SIZE / 2; // scale + min + nibbles

    let (n_blocks, total_bytes) = checked_dequant_size(numel, BLOCK_SIZE, BLOCK_BYTES, "Q4_1")?;

    if offset + total_bytes > raw.len() {
        return Err(AzError::ModelLoad(format!(
            "Q4_1: need {} bytes at offset {}, but file is {} bytes",
            total_bytes, offset, raw.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    let mut pos = offset;

    for _ in 0..n_blocks {
        let scale = f16_to_f32(u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap()));
        pos += 2;
        let min = f16_to_f32(u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap()));
        pos += 2;

        let remaining = BLOCK_SIZE.min(numel - out.len());
        for j in 0..BLOCK_SIZE / 2 {
            let byte = raw[pos + j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            if j * 2 < remaining {
                out.push(lo * scale + min);
            }
            if j * 2 + 1 < remaining {
                out.push(hi * scale + min);
            }
        }
        pos += BLOCK_SIZE / 2;
    }

    Ok(out)
}

/// Dequantize Q5_0 blocks to f32.
/// Q5_0 format: block_size=32, each block = f16 scale (2) + 4 bytes high bits + 16 bytes nibbles = 22 bytes.
/// 5-bit values: 4 bits from nibble + 1 high bit from bitfield, offset by -16.
fn dequantize_q5_0(raw: &[u8], offset: usize, numel: usize) -> AzResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 4 + BLOCK_SIZE / 2; // scale + high_bits + nibbles

    let (n_blocks, total_bytes) = checked_dequant_size(numel, BLOCK_SIZE, BLOCK_BYTES, "Q5_0")?;

    if offset + total_bytes > raw.len() {
        return Err(AzError::ModelLoad(format!(
            "Q5_0: need {} bytes at offset {}, but file is {} bytes",
            total_bytes, offset, raw.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    let mut pos = offset;

    for _ in 0..n_blocks {
        let scale = f16_to_f32(u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap()));
        pos += 2;

        let qh = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let remaining = BLOCK_SIZE.min(numel - out.len());
        for j in 0..BLOCK_SIZE / 2 {
            let byte = raw[pos + j];
            let lo4 = (byte & 0x0F) as i32;
            let hi4 = ((byte >> 4) & 0x0F) as i32;
            let lo_h = ((qh >> (j * 2)) & 1) as i32;
            let hi_h = ((qh >> (j * 2 + 1)) & 1) as i32;
            let lo5 = lo4 | (lo_h << 4);
            let hi5 = hi4 | (hi_h << 4);
            if j * 2 < remaining {
                out.push((lo5 - 16) as f32 * scale);
            }
            if j * 2 + 1 < remaining {
                out.push((hi5 - 16) as f32 * scale);
            }
        }
        pos += BLOCK_SIZE / 2;
    }

    Ok(out)
}

/// Dequantize Q5_1 blocks to f32.
/// Q5_1 format: block_size=32, each block = f16 scale (2) + f16 min (2) + 4 bytes high bits + 16 bytes = 24 bytes.
/// 5-bit values: 4 bits from nibble + 1 high bit, value = q5 * scale + min.
fn dequantize_q5_1(raw: &[u8], offset: usize, numel: usize) -> AzResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 2 + 4 + BLOCK_SIZE / 2; // scale + min + high_bits + nibbles

    let (n_blocks, total_bytes) = checked_dequant_size(numel, BLOCK_SIZE, BLOCK_BYTES, "Q5_1")?;

    if offset + total_bytes > raw.len() {
        return Err(AzError::ModelLoad(format!(
            "Q5_1: need {} bytes at offset {}, but file is {} bytes",
            total_bytes, offset, raw.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    let mut pos = offset;

    for _ in 0..n_blocks {
        let scale = f16_to_f32(u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap()));
        pos += 2;
        let min = f16_to_f32(u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap()));
        pos += 2;

        let qh = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let remaining = BLOCK_SIZE.min(numel - out.len());
        for j in 0..BLOCK_SIZE / 2 {
            let byte = raw[pos + j];
            let lo4 = (byte & 0x0F) as f32;
            let hi4 = ((byte >> 4) & 0x0F) as f32;
            let lo_h = ((qh >> (j * 2)) & 1) as f32;
            let hi_h = ((qh >> (j * 2 + 1)) & 1) as f32;
            let lo5 = lo4 + lo_h * 16.0;
            let hi5 = hi4 + hi_h * 16.0;
            if j * 2 < remaining {
                out.push(lo5 * scale + min);
            }
            if j * 2 + 1 < remaining {
                out.push(hi5 * scale + min);
            }
        }
        pos += BLOCK_SIZE / 2;
    }

    Ok(out)
}

/// Dequantize Q8_0 blocks to f32.
/// Q8_0 format: block_size=32, each block = f16 scale (2 bytes) + 32 × i8 (32 bytes) = 34 bytes.
fn dequantize_q8_0(raw: &[u8], offset: usize, numel: usize) -> AzResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + BLOCK_SIZE; // f16 scale + 32 i8 values

    let (n_blocks, total_bytes) = checked_dequant_size(numel, BLOCK_SIZE, BLOCK_BYTES, "Q8_0")?;

    if offset + total_bytes > raw.len() {
        return Err(AzError::ModelLoad(format!(
            "Q8_0: need {} bytes at offset {}, but file is {} bytes",
            total_bytes, offset, raw.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    let mut pos = offset;

    for _ in 0..n_blocks {
        let scale_bits = u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap());
        let scale = f16_to_f32(scale_bits);
        pos += 2;

        let count = BLOCK_SIZE.min(numel - out.len());
        for j in 0..count {
            let q = raw[pos + j] as i8;
            out.push(q as f32 * scale);
        }
        pos += BLOCK_SIZE;
    }

    Ok(out)
}

/// Unpack 6-bit scales and mins from the 12-byte K-quant scale packing.
/// Used by Q4_K and Q5_K. Returns (scales[8], mins[8]).
/// Layout: bytes 0-3 contain low 6 bits of sc[0-3], bytes 4-7 low 6 bits of mn[0-3],
/// bytes 8-11 contain high 2 bits of sc[4-7] and mn[4-7].
fn unpack_k4_scales(s: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for i in 0..4 {
        sc[i] = s[i] & 63;
        mn[i] = s[i + 4] & 63;
    }
    for i in 0..4 {
        sc[i + 4] = (s[i + 8] & 0x0F) | ((s[i] >> 6) << 4);
        mn[i + 4] = (s[i + 8] >> 4)    | ((s[i + 4] >> 6) << 4);
    }
    (sc, mn)
}

/// Dequantize Q4_K blocks to f32.
/// Q4_K: block_size=256, layout = f16 d (2) + f16 dmin (2) + scales[12] + qs[128] = 144 bytes.
/// 4 groups of 64 elements: first 32 use low nibble, second 32 use high nibble of same 32 bytes.
/// Each group of 32 elements has its own 6-bit scale and min (8 total per super-block).
fn dequantize_q4_k(raw: &[u8], offset: usize, numel: usize) -> AzResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let (n_blocks, total_bytes) = checked_dequant_size(numel, BLOCK_SIZE, BLOCK_BYTES, "Q4_K")?;
    if offset + total_bytes > raw.len() {
        return Err(AzError::ModelLoad(format!(
            "Q4_K: need {} bytes at offset {}, but file is {} bytes",
            total_bytes, offset, raw.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    let mut pos = offset;

    for _ in 0..n_blocks {
        let d = f16_to_f32(u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap()));
        let dmin = f16_to_f32(u16::from_le_bytes(raw[pos + 2..pos + 4].try_into().unwrap()));
        let (sc, mn) = unpack_k4_scales(&raw[pos + 4..pos + 16]);
        let qs = &raw[pos + 16..pos + 144];

        let mut is = 0usize; // scale index
        let mut q_off = 0usize; // offset into qs

        // 4 groups of 64 elements each
        for _ in 0..4 {
            let d1 = d * sc[is] as f32;
            let m1 = dmin * mn[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * mn[is + 1] as f32;

            // First 32: low nibble
            for l in 0..32 {
                if out.len() >= numel { break; }
                let v = (qs[q_off + l] & 0x0F) as f32;
                out.push(v * d1 - m1);
            }
            // Second 32: high nibble
            for l in 0..32 {
                if out.len() >= numel { break; }
                let v = ((qs[q_off + l] >> 4) & 0x0F) as f32;
                out.push(v * d2 - m2);
            }

            q_off += 32;
            is += 2;
        }

        pos += BLOCK_BYTES;
    }

    out.truncate(numel);
    Ok(out)
}

/// Dequantize Q5_K blocks to f32.
/// Q5_K: block_size=256, layout = f16 d (2) + f16 dmin (2) + scales[12] + qh[32] + qs[128] = 176 bytes.
/// Same as Q4_K but with extra high bit per element in qh (256 bits = 32 bytes).
fn dequantize_q5_k(raw: &[u8], offset: usize, numel: usize) -> AzResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 176;

    let (n_blocks, total_bytes) = checked_dequant_size(numel, BLOCK_SIZE, BLOCK_BYTES, "Q5_K")?;
    if offset + total_bytes > raw.len() {
        return Err(AzError::ModelLoad(format!(
            "Q5_K: need {} bytes at offset {}, but file is {} bytes",
            total_bytes, offset, raw.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    let mut pos = offset;

    for _ in 0..n_blocks {
        let d = f16_to_f32(u16::from_le_bytes(raw[pos..pos + 2].try_into().unwrap()));
        let dmin = f16_to_f32(u16::from_le_bytes(raw[pos + 2..pos + 4].try_into().unwrap()));
        let (sc, mn) = unpack_k4_scales(&raw[pos + 4..pos + 16]);
        let qh = &raw[pos + 16..pos + 48];
        let qs = &raw[pos + 48..pos + 176];

        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut elem = 0usize; // element counter within block

        for _ in 0..4 {
            let d1 = d * sc[is] as f32;
            let m1 = dmin * mn[is] as f32;
            let d2 = d * sc[is + 1] as f32;
            let m2 = dmin * mn[is + 1] as f32;

            for l in 0..32 {
                if out.len() >= numel { break; }
                let qh_bit = ((qh[elem / 8] >> (elem % 8)) & 1) as f32;
                let v = (qs[q_off + l] & 0x0F) as f32 + qh_bit * 16.0;
                out.push(v * d1 - m1);
                elem += 1;
            }
            for l in 0..32 {
                if out.len() >= numel { break; }
                let qh_bit = ((qh[elem / 8] >> (elem % 8)) & 1) as f32;
                let v = ((qs[q_off + l] >> 4) & 0x0F) as f32 + qh_bit * 16.0;
                out.push(v * d2 - m2);
                elem += 1;
            }

            q_off += 32;
            is += 2;
        }
        pos += BLOCK_BYTES;
    }

    out.truncate(numel);
    Ok(out)
}

/// Dequantize Q6_K blocks to f32.
/// Q6_K: block_size=256, layout = ql[128] + qh[64] + scales[16] + f16 d (2) = 210 bytes.
/// 6-bit values: 4 from ql + 2 from qh, with signed offset -32.
/// Processed in 2 halves of 128 elements, each half has 4 groups of 32.
fn dequantize_q6_k(raw: &[u8], offset: usize, numel: usize) -> AzResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 210;

    let (n_blocks, total_bytes) = checked_dequant_size(numel, BLOCK_SIZE, BLOCK_BYTES, "Q6_K")?;
    if offset + total_bytes > raw.len() {
        return Err(AzError::ModelLoad(format!(
            "Q6_K: need {} bytes at offset {}, but file is {} bytes",
            total_bytes, offset, raw.len()
        )));
    }

    let mut out = Vec::with_capacity(numel);
    let mut pos = offset;

    for _ in 0..n_blocks {
        let ql = &raw[pos..pos + 128];
        let qh = &raw[pos + 128..pos + 192];
        let scales = &raw[pos + 192..pos + 208];
        let d = f16_to_f32(u16::from_le_bytes(raw[pos + 208..pos + 210].try_into().unwrap()));

        // Process two 128-element halves
        for half in 0..2 {
            let ql_base = half * 64;
            let qh_base = half * 32;
            let sc_base = half * 8;

            // 4 groups of 32 within each half:
            // Group 0: ql[ql_base+l] low nibble + qh[qh_base+l] bits 0-1
            // Group 1: ql[ql_base+32+l] low nibble + qh[qh_base+l] bits 2-3
            // Group 2: ql[ql_base+l] high nibble + qh[qh_base+l] bits 4-5
            // Group 3: ql[ql_base+32+l] high nibble + qh[qh_base+l] bits 6-7
            // Scale mapping: group g, l in 0..31 → scales[sc_base + g*2 + l/16]

            for l in 0..32 {
                if out.len() >= numel { break; }
                let sc_idx = sc_base + l / 16;
                let sc_val = scales[sc_idx] as i8 as f32;
                let lo4 = (ql[ql_base + l] & 0x0F) as i32;
                let hi2 = ((qh[qh_base + l] >> 0) & 3) as i32;
                let q = lo4 | (hi2 << 4);
                out.push((q - 32) as f32 * d * sc_val);
            }
            for l in 0..32 {
                if out.len() >= numel { break; }
                let sc_idx = sc_base + 2 + l / 16;
                let sc_val = scales[sc_idx] as i8 as f32;
                let lo4 = (ql[ql_base + 32 + l] & 0x0F) as i32;
                let hi2 = ((qh[qh_base + l] >> 2) & 3) as i32;
                let q = lo4 | (hi2 << 4);
                out.push((q - 32) as f32 * d * sc_val);
            }
            for l in 0..32 {
                if out.len() >= numel { break; }
                let sc_idx = sc_base + 4 + l / 16;
                let sc_val = scales[sc_idx] as i8 as f32;
                let lo4 = ((ql[ql_base + l] >> 4) & 0x0F) as i32;
                let hi2 = ((qh[qh_base + l] >> 4) & 3) as i32;
                let q = lo4 | (hi2 << 4);
                out.push((q - 32) as f32 * d * sc_val);
            }
            for l in 0..32 {
                if out.len() >= numel { break; }
                let sc_idx = sc_base + 6 + l / 16;
                let sc_val = scales[sc_idx] as i8 as f32;
                let lo4 = ((ql[ql_base + 32 + l] >> 4) & 0x0F) as i32;
                let hi2 = ((qh[qh_base + l] >> 6) & 3) as i32;
                let q = lo4 | (hi2 << 4);
                out.push((q - 32) as f32 * d * sc_val);
            }
        }

        pos += BLOCK_BYTES;
    }

    out.truncate(numel);
    Ok(out)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut f = frac as f32 / 1024.0;
        f *= 2.0f32.powi(-14);
        if sign == 1 { -f } else { f }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::NAN
        }
    } else {
        let f32_exp = exp + 112;
        let f32_frac = frac << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac)
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn test_f16_conversions() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Build a minimal Q8_0 block: scale=1.0 (f16: 0x3C00), then 32 i8 values
        let mut data = vec![0u8; 34]; // 2 + 32
        data[0] = 0x00; data[1] = 0x3C; // f16 1.0 in LE
        data[2] = 1; // i8 = 1 → f32 = 1.0
        data[3] = 2; // i8 = 2 → f32 = 2.0
        data[4] = 0xFE; // i8 = -2 → f32 = -2.0

        let result = dequantize_q8_0(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 2.0).abs() < 1e-4);
        assert!((result[2] - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn test_dequantize_q4_0_basic() {
        // Q4_0 block: scale=1.0 (f16: 0x3C00), 16 bytes of nibble pairs
        let mut data = vec![0u8; 18]; // 2 + 16
        data[0] = 0x00; data[1] = 0x3C; // f16 1.0 in LE
        // byte[0] = 0x98: lo nibble = 8 → 8-8=0, hi nibble = 9 → 9-8=1
        data[2] = 0x98;
        // byte[1] = 0xBA: lo = 0xA = 10 → 10-8=2, hi = 0xB = 11 → 11-8=3
        data[3] = 0xBA;

        let result = dequantize_q4_0(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.0).abs() < 1e-4); // (8-8)*1.0
        assert!((result[1] - 1.0).abs() < 1e-4); // (9-8)*1.0
        assert!((result[2] - 2.0).abs() < 1e-4); // (10-8)*1.0
        assert!((result[3] - 3.0).abs() < 1e-4); // (11-8)*1.0
    }

    #[test]
    fn test_dequantize_q4_1_basic() {
        // Q4_1 block: scale=2.0 (f16: 0x4000), min=1.0 (f16: 0x3C00), 16 bytes
        let mut data = vec![0u8; 20]; // 2 + 2 + 16
        data[0] = 0x00; data[1] = 0x40; // f16 2.0
        data[2] = 0x00; data[3] = 0x3C; // f16 1.0
        // byte[0] = 0x10: lo = 0 → 0*2+1=1, hi = 1 → 1*2+1=3
        data[4] = 0x10;

        let result = dequantize_q4_1(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 1.0).abs() < 1e-4); // 0*2.0 + 1.0
        assert!((result[1] - 3.0).abs() < 1e-4); // 1*2.0 + 1.0
    }

    #[test]
    fn test_dequantize_q5_0_basic() {
        // Q5_0 block: scale=1.0 (f16: 0x3C00), 4 bytes high_bits, 16 bytes nibbles = 22 bytes
        let mut data = vec![0u8; 22]; // 2 + 4 + 16
        data[0] = 0x00; data[1] = 0x3C; // f16 1.0
        // high_bits = 0 (all high bits 0)
        data[2] = 0; data[3] = 0; data[4] = 0; data[5] = 0;
        // byte[0] nibbles: lo=8 → (8|0)-16 = -8, hi=9 → (9|0)-16 = -7
        data[6] = 0x98;

        let result = dequantize_q5_0(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - (-8.0)).abs() < 1e-4);
        assert!((result[1] - (-7.0)).abs() < 1e-4);

        // Now with high bit set: high_bits bit 0 = 1 → lo gets +16
        data[2] = 0x01; // bit 0 set
        let result2 = dequantize_q5_0(&data, 0, 32).unwrap();
        // lo nibble 8, high bit 1 → (8 | 16) - 16 = 8
        assert!((result2[0] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_gguf_magic() {
        let bytes = b"GGUF";
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(magic, GGUF_MAGIC);
    }

    #[test]
    fn test_cursor_read_string() {
        // GGUF string: u64 len + bytes
        let mut data = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes()); // len = 5
        data.extend_from_slice(b"hello");
        let mut cur = Cursor::new(&data);
        let s = cur.read_string().unwrap();
        assert_eq!(s, "hello");
        assert_eq!(cur.pos, 13); // 8 + 5
    }

    #[test]
    fn test_synthetic_gguf_header() {
        // Build a minimal valid GGUF file with 0 tensors and 1 metadata kv
        let mut data = Vec::new();
        // Magic
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        data.extend_from_slice(&3u32.to_le_bytes());
        // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes());
        // kv_count = 1
        data.extend_from_slice(&1u64.to_le_bytes());
        // KV: key = "general.architecture", type = string, value = "rwkv"
        let key = "general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        let val = "rwkv";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val.as_bytes());

        // Pad to alignment boundary so data_offset doesn't exceed file size
        while data.len() < GGUF_DEFAULT_ALIGNMENT * 3 {
            data.push(0);
        }

        // Write to temp file and parse
        let tmp = std::env::temp_dir().join("test_gguf_header.gguf");
        std::fs::write(&tmp, &data).unwrap();
        let gguf = GgufFile::open(&tmp).unwrap();
        assert_eq!(gguf.tensor_names().len(), 0);
        let arch = gguf.get_meta("general.architecture").unwrap();
        assert_eq!(arch.as_str().unwrap(), "rwkv");
        let _ = std::fs::remove_file(&tmp);
    }
}
