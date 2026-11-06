//! Redukt GPU Backend Plugin -- portable GPU compute via wgpu.
//!
//! Compiles to a shared library (cdylib) that Redukt loads at runtime.
//! Implements the Backend FFI v1 contract (4 exported functions).
//!
//! Current state: SKELETON -- returns uniform logits as placeholder.
//! Next steps: wgpu device init, compute shader for Mamba forward pass.

use std::sync::Mutex;

// --- FFI types (must match src/domain/backend_ffi.rs exactly) ---

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendResult {
    Ok = 0,
    InitFailed = 1,
    ForwardFailed = 2,
    VersionMismatch = 3,
    WeightsNotFound = 4,
}

const BACKEND_FFI_VERSION: u32 = 1;
const BACKEND_NAME_MAX: usize = 64;

#[repr(C)]
pub struct BackendInfo {
    pub ffi_version: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub name: [u8; BACKEND_NAME_MAX],
}

// --- Plugin state ---

struct PluginState {
    vocab_size: u32,
    _hidden_dim: u32,
    /// Placeholder logits buffer (uniform distribution).
    logits: Vec<f32>,
}

static PLUGIN: Mutex<Option<PluginState>> = Mutex::new(None);

fn make_name(s: &str) -> [u8; BACKEND_NAME_MAX] {
    let mut buf = [0u8; BACKEND_NAME_MAX];
    let bytes = s.as_bytes();
    let len = bytes.len().min(BACKEND_NAME_MAX - 1);
    buf[..len].copy_from_slice(&bytes[..len]);
    buf
}

// --- Exported FFI functions ---

/// Initialize the GPU backend. Currently a placeholder -- no actual GPU init.
#[no_mangle]
pub unsafe extern "C" fn redukt_backend_init(
    weights_path: *const u8,
    _device_id: i32,
    info_out: *mut BackendInfo,
) -> BackendResult {
    if weights_path.is_null() || info_out.is_null() {
        return BackendResult::InitFailed;
    }

    // Read weights path (null-terminated)
    let mut len = 0;
    while *weights_path.add(len) != 0 {
        len += 1;
    }
    let path_bytes = std::slice::from_raw_parts(weights_path, len);
    let path = std::str::from_utf8(path_bytes).unwrap_or("<invalid>");

    eprintln!("[redukt-gpu] init: weights={}, device=placeholder", path);

    // Placeholder: 256 vocab (byte-level), uniform logits
    let vocab_size = 256u32;
    let hidden_dim = 768u32;

    let state = PluginState {
        vocab_size,
        _hidden_dim: hidden_dim,
        logits: vec![0.0f32; vocab_size as usize],
    };

    let mut guard = PLUGIN.lock().unwrap();
    *guard = Some(state);

    *info_out = BackendInfo {
        ffi_version: BACKEND_FFI_VERSION,
        vocab_size,
        hidden_dim,
        name: make_name("wgpu-placeholder"),
    };

    BackendResult::Ok
}

/// Single-step forward pass. Currently returns uniform logits.
#[no_mangle]
pub unsafe extern "C" fn redukt_backend_forward(
    _token: u32,
    out_logits: *mut f32,
    vocab_size: u32,
) -> BackendResult {
    let guard = PLUGIN.lock().unwrap();
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return BackendResult::ForwardFailed,
    };

    if vocab_size != state.vocab_size {
        return BackendResult::ForwardFailed;
    }

    // Placeholder: copy uniform logits (all zeros = uniform after softmax)
    let dst = std::slice::from_raw_parts_mut(
        out_logits,
        vocab_size as usize,
    );
    dst.copy_from_slice(&state.logits);

    BackendResult::Ok
}

/// Reset recurrent state to zeros.
#[no_mangle]
pub unsafe extern "C" fn redukt_backend_reset_state() {
    let mut guard = PLUGIN.lock().unwrap();
    if let Some(state) = guard.as_mut() {
        state.logits.fill(0.0);
    }
}

/// Free all resources.
#[no_mangle]
pub unsafe extern "C" fn redukt_backend_free() {
    let mut guard = PLUGIN.lock().unwrap();
    *guard = None;
    eprintln!("[redukt-gpu] freed");
}
