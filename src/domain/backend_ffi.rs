//! FFI interface for external compute backends (R-072).
//!
//! Defines the C ABI contract that GPU/accelerator plugins must implement.
//! The core binary loads plugins at runtime via dlopen/LoadLibrary (R-073).
//! This module is domain-level: it defines the CONTRACT, not the implementation.
//!
//! FFI v2 (R-096) is ADDITIVE: a plugin that exports `redukt_backend_abi_version()`
//! returning >= 2 plus the three batch symbols gains a multi-stream path.  The v1
//! symbols, the `BackendInfo` layout and `BACKEND_FFI_VERSION` are unchanged, so old
//! hosts keep working with new plugins and new hosts with old plugins.
//! Contract: docs/api/PLUGIN-FFI-V2.md.

/// FFI interface version. Plugin must match this exactly.
/// This is the BASE (v1) contract reported in `BackendInfo.ffi_version`; it is NOT
/// bumped by v2, which is detected through the optional `SYM_ABI_VERSION` symbol.
pub const BACKEND_FFI_VERSION: u32 = 1;

/// Minimum value returned by `redukt_backend_abi_version()` for the batch API (R-096).
pub const BACKEND_ABI_V2: u32 = 2;

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
    /// v2 only: bad batch arguments (slot out of range, duplicate slot, bad token,
    /// vocab mismatch, n_streams > max_streams).  State is left untouched.
    InvalidArgument = 5,
}

impl BackendResult {
    /// Map a raw `int32_t` returned by a v2 entry point.  v2 functions return a plain
    /// integer (never a Rust enum across the boundary) so an unexpected value from a
    /// foreign plugin cannot produce an invalid enum discriminant; unknown codes map
    /// to `ForwardFailed`.
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => Self::Ok,
            1 => Self::InitFailed,
            3 => Self::VersionMismatch,
            4 => Self::WeightsNotFound,
            5 => Self::InvalidArgument,
            _ => Self::ForwardFailed,
        }
    }
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

// ---------------------------------------------------------------------------
// FFI v2 — batched multi-stream forward (R-096). All symbols optional.
// ---------------------------------------------------------------------------

/// `redukt_backend_abi_version`: highest ABI revision implemented (>= 2 for batch).
pub type BackendAbiVersionFn = unsafe extern "C" fn() -> u32;

/// `redukt_backend_batch_init`: (re)allocate `max_streams` independent recurrent
/// states in device memory, all zeroed; `0` releases the pool.  Must be called after
/// `backend_init`.  Returns a `BackendResult` code as `i32`.
pub type BackendBatchInitFn = unsafe extern "C" fn(max_streams: u32) -> i32;

/// `redukt_backend_batch_reset`: zero the states of `slots[0..n_slots]`;
/// `slots == NULL` resets every slot.  Returns a `BackendResult` code as `i32`.
pub type BackendBatchResetFn = unsafe extern "C" fn(slots: *const u32, n_slots: u32) -> i32;

/// `redukt_backend_forward_batch`: one recurrent step for `n_streams` streams.
///   - `slots[b]`: state slot of batch position b (distinct, < max_streams).
///   - `tokens[b]`: input token of batch position b (< vocab_size).
///   - `out_logits`: caller-owned, `n_streams * vocab_size` f32, row-major
///     (row b = logits of batch position b); valid when the call returns.
///
/// Synchronous; the state of each listed slot advances exactly one token, every other
/// slot is untouched.  Returns a `BackendResult` code as `i32`.
pub type BackendForwardBatchFn = unsafe extern "C" fn(
    n_streams: u32,
    slots: *const u32,
    tokens: *const u32,
    out_logits: *mut f32,
    vocab_size: u32,
) -> i32;

/// Optional v2 function table; present only if every v2 symbol resolved and
/// `abi_version() >= BACKEND_ABI_V2`.
#[derive(Clone)]
pub struct BackendVTableV2 {
    pub abi_version: u32,
    pub batch_init: BackendBatchInitFn,
    pub batch_reset: BackendBatchResetFn,
    pub forward_batch: BackendForwardBatchFn,
}

pub const SYM_ABI_VERSION: &str = "redukt_backend_abi_version";
pub const SYM_BATCH_INIT: &str = "redukt_backend_batch_init";
pub const SYM_BATCH_RESET: &str = "redukt_backend_batch_reset";
pub const SYM_FORWARD_BATCH: &str = "redukt_backend_forward_batch";

/// Host-side validation of `forward_batch` arguments (mirrors the plugin checks so a
/// bad call is rejected before crossing the FFI boundary).  `seen` is a scratch
/// buffer of at least `max_streams` entries; it is cleared on entry.
pub fn validate_batch_args(
    slots: &[u32], tokens: &[u32], out_len: usize,
    vocab_size: u32, max_streams: u32, seen: &mut [bool],
) -> Result<(), BackendResult> {
    let n = slots.len();
    let shape_ok = tokens.len() == n
        && n <= max_streams as usize
        && seen.len() >= max_streams as usize
        && out_len == n * vocab_size as usize;
    if !shape_ok {
        return Err(BackendResult::InvalidArgument);
    }
    seen.iter_mut().for_each(|s| *s = false);
    for (&slot, &tok) in slots.iter().zip(tokens) {
        if slot >= max_streams || tok >= vocab_size || seen[slot as usize] {
            return Err(BackendResult::InvalidArgument);
        }
        seen[slot as usize] = true;
    }
    Ok(())
}

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
        assert_eq!(BackendResult::InvalidArgument as i32, 5);
    }

    #[test]
    fn from_code_roundtrip_and_unknown() {
        for r in [
            BackendResult::Ok, BackendResult::InitFailed, BackendResult::ForwardFailed,
            BackendResult::VersionMismatch, BackendResult::WeightsNotFound,
            BackendResult::InvalidArgument,
        ] {
            assert_eq!(BackendResult::from_code(r as i32), r);
        }
        assert_eq!(BackendResult::from_code(-1), BackendResult::ForwardFailed);
        assert_eq!(BackendResult::from_code(99), BackendResult::ForwardFailed);
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

    #[test]
    fn v2_symbols_and_version() {
        assert_eq!(BACKEND_ABI_V2, 2);
        for s in [SYM_ABI_VERSION, SYM_BATCH_INIT, SYM_BATCH_RESET, SYM_FORWARD_BATCH] {
            assert!(s.starts_with("redukt_backend_"));
        }
    }

    #[test]
    fn validate_batch_args_accepts_valid() {
        let mut seen = vec![false; 8];
        let r = validate_batch_args(&[3, 0, 7], &[1, 2, 255], 3 * 256, 256, 8, &mut seen);
        assert_eq!(r, Ok(()));
        // An empty batch is a valid no-op.
        assert_eq!(validate_batch_args(&[], &[], 0, 256, 8, &mut seen), Ok(()));
    }

    #[test]
    fn validate_batch_args_rejects_bad_input() {
        let mut seen = vec![false; 4];
        let bad = Err(BackendResult::InvalidArgument);
        // duplicate slot
        assert_eq!(validate_batch_args(&[1, 1], &[0, 0], 512, 256, 4, &mut seen), bad);
        // slot out of range
        assert_eq!(validate_batch_args(&[4], &[0], 256, 256, 4, &mut seen), bad);
        // token out of range
        assert_eq!(validate_batch_args(&[0], &[256], 256, 256, 4, &mut seen), bad);
        // slots/tokens length mismatch
        assert_eq!(validate_batch_args(&[0, 1], &[0], 512, 256, 4, &mut seen), bad);
        // wrong output size
        assert_eq!(validate_batch_args(&[0], &[0], 255, 256, 4, &mut seen), bad);
        // more streams than the pool
        let slots = [0, 1, 2, 3, 0];
        assert_eq!(validate_batch_args(&slots, &[0; 5], 5 * 256, 256, 4, &mut seen), bad);
    }
}
