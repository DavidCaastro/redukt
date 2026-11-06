//! FFI interface for external compute backends (R-072).
//!
//! Defines the C ABI contract that GPU/accelerator plugins must implement.
//! The core binary loads plugins at runtime via dlopen/LoadLibrary (R-073).
//! This module is domain-level: it defines the CONTRACT, not the implementation.


/// FFI interface version. Plugin must match this exactly.
pub const BACKEND_FFI_VERSION: u32 = 1;

/// Maximum backend name length (including null terminator).
pub const BACKEND_NAME_MAX: usize = 64;

/// Result codes for backend operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendResult {
    /// Operation succeeded.
    Ok = 0,
    /// Initialization failed (no GPU, bad weights, etc).
    InitFailed = 1,
    /// Forward pass failed (internal error).
    ForwardFailed = 2,
    /// Version mismatch between host and plugin.
    VersionMismatch = 3,
    /// Weights file not found or unreadable.
    WeightsNotFound = 4,
}

/// Information returned by the backend after initialization.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// FFI version the plugin was built against (must == BACKEND_FFI_VERSION).
    pub ffi_version: u32,
    /// Vocabulary size (number of logits produced per forward call).
    pub vocab_size: u32,
    /// Hidden dimension (d_model).
    pub hidden_dim: u32,
    /// Human-readable backend name, null-terminated UTF-8.
    pub name: [u8; BACKEND_NAME_MAX],
}

impl BackendInfo {
    /// Read the name as a Rust string slice (up to first null byte).
    pub fn name_str(&self) -> &str {
        let end = self.name.iter()
            .position(|&b| b == 0)
            .unwrap_or(BACKEND_NAME_MAX);
        std::str::from_utf8(&self.name[..end]).unwrap_or("unknown")
    }
}

/// Function pointer types for the backend vtable.
///
/// `backend_init`: Load model weights and initialize GPU resources.
///   - `weights_path`: null-terminated UTF-8 path to weights directory.
///   - `device_id`: GPU device index (0 = default).
///   - `info_out`: filled on success with backend metadata.
///   - Returns: BackendResult::Ok on success.
pub type BackendInitFn = unsafe extern "C" fn(
    weights_path: *const u8,
    device_id: i32,
    info_out: *mut BackendInfo,
) -> BackendResult;

/// `backend_forward`: Single-step forward pass (RNN mode).
///   - `token`: input token ID (0..vocab_size).
///   - `out_logits`: pre-allocated buffer of `vocab_size` f32 values.
///   - `vocab_size`: expected length of out_logits buffer.
///   - Returns: BackendResult::Ok on success.
pub type BackendForwardFn = unsafe extern "C" fn(
    token: u32,
    out_logits: *mut f32,
    vocab_size: u32,
) -> BackendResult;

/// `backend_reset_state`: Reset recurrent state to zeros.
///   Called between files in batch mode.
pub type BackendResetFn = unsafe extern "C" fn();

/// `backend_free`: Release all GPU resources and model memory.
///   Called once when the host process shuts down.
pub type BackendFreeFn = unsafe extern "C" fn();

/// Virtual function table for a loaded backend plugin.
/// All pointers must be non-null after successful load.
#[derive(Clone)]
pub struct BackendVTable {
    pub init: BackendInitFn,
    pub forward: BackendForwardFn,
    pub reset_state: BackendResetFn,
    pub free: BackendFreeFn,
}

/// Symbol names exported by plugins (used by the loader in R-073).
pub const SYM_INIT: &str = "redukt_backend_init";
pub const SYM_FORWARD: &str = "redukt_backend_forward";
pub const SYM_RESET: &str = "redukt_backend_reset_state";
pub const SYM_FREE: &str = "redukt_backend_free";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_result_repr() {
        assert_eq!(BackendResult::Ok as i32, 0);
        assert_eq!(BackendResult::InitFailed as i32, 1);
        assert_eq!(BackendResult::ForwardFailed as i32, 2);
        assert_eq!(BackendResult::VersionMismatch as i32, 3);
        assert_eq!(BackendResult::WeightsNotFound as i32, 4);
    }

    #[test]
    fn backend_info_name_str() {
        let mut info = BackendInfo {
            ffi_version: BACKEND_FFI_VERSION,
            vocab_size: 256,
            hidden_dim: 768,
            name: [0u8; BACKEND_NAME_MAX],
        };
        let label = b"wgpu-vulkan";
        info.name[..label.len()].copy_from_slice(label);
        assert_eq!(info.name_str(), "wgpu-vulkan");
    }

    #[test]
    fn backend_info_empty_name() {
        let info = BackendInfo {
            ffi_version: BACKEND_FFI_VERSION,
            vocab_size: 50280,
            hidden_dim: 768,
            name: [0u8; BACKEND_NAME_MAX],
        };
        assert_eq!(info.name_str(), "");
    }

    #[test]
    fn vtable_construction_with_dummy_fns() {
        unsafe extern "C" fn dummy_init(
            _p: *const u8, _d: i32, _i: *mut BackendInfo,
        ) -> BackendResult {
            BackendResult::Ok
        }
        unsafe extern "C" fn dummy_fwd(
            _t: u32, _o: *mut f32, _v: u32,
        ) -> BackendResult {
            BackendResult::Ok
        }
        unsafe extern "C" fn dummy_reset() {}
        unsafe extern "C" fn dummy_free() {}

        let vt = BackendVTable {
            init: dummy_init,
            forward: dummy_fwd,
            reset_state: dummy_reset,
            free: dummy_free,
        };

        let mut info = BackendInfo {
            ffi_version: 0, vocab_size: 0, hidden_dim: 0,
            name: [0u8; BACKEND_NAME_MAX],
        };
        let result = unsafe {
            (vt.init)(std::ptr::null(), 0, &mut info)
        };
        assert_eq!(result, BackendResult::Ok);
    }

    #[test]
    fn ffi_version_is_one() {
        assert_eq!(BACKEND_FFI_VERSION, 1);
    }

    #[test]
    fn symbol_names_valid() {
        assert!(SYM_INIT.starts_with("redukt_"));
        assert!(SYM_FORWARD.starts_with("redukt_"));
        assert!(SYM_RESET.starts_with("redukt_"));
        assert!(SYM_FREE.starts_with("redukt_"));
    }
}
