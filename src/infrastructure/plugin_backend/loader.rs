//! Dynamic shared library loader for GPU backend plugins (R-073).
//!
//! Searches for `redukt-gpu.dll` (Windows) / `libredukt_gpu.so` (Linux) /
//! `libredukt_gpu.dylib` (macOS) next to the executable.  If found, loads
//! the 4 FFI symbols and builds a `BackendVTable`.  If not found, returns
//! `None`.  Uses raw OS FFI (LoadLibrary/dlopen) — zero crate dependencies.

use crate::domain::backend_ffi::{
    BackendVTable, BackendInitFn, BackendForwardFn,
    BackendResetFn, BackendFreeFn,
    BACKEND_FFI_VERSION, BackendInfo, BackendResult,
    SYM_INIT, SYM_FORWARD, SYM_RESET, SYM_FREE,
};

// ---------------------------------------------------------------------------
// Platform abstractions
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod platform {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    pub type LibHandle = *mut std::ffi::c_void;

    extern "system" {
        fn LoadLibraryW(name: *const u16) -> LibHandle;
        fn GetProcAddress(
            module: LibHandle,
            name: *const u8,
        ) -> *mut std::ffi::c_void;
        fn FreeLibrary(module: LibHandle) -> i32;
    }

    pub unsafe fn load_library(path: &str) -> LibHandle {
        let wide: Vec<u16> = OsStr::new(path)
            .encode_wide()
            .chain(Some(0))
            .collect();
        LoadLibraryW(wide.as_ptr())
    }

    pub unsafe fn get_symbol(
        handle: LibHandle,
        name: &str,
    ) -> *mut std::ffi::c_void {
        let mut buf = Vec::with_capacity(name.len() + 1);
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        GetProcAddress(handle, buf.as_ptr())
    }

    pub unsafe fn free_library(handle: LibHandle) {
        if !handle.is_null() {
            FreeLibrary(handle);
        }
    }

    pub fn is_null(handle: LibHandle) -> bool {
        handle.is_null()
    }
}

#[cfg(unix)]
mod platform {
    use std::ffi::CString;

    pub type LibHandle = *mut std::ffi::c_void;

    extern "C" {
        fn dlopen(filename: *const i8, flags: i32) -> LibHandle;
        fn dlsym(
            handle: LibHandle,
            symbol: *const i8,
        ) -> *mut std::ffi::c_void;
        fn dlclose(handle: LibHandle) -> i32;
    }

    const RTLD_NOW: i32 = 2;

    pub unsafe fn load_library(path: &str) -> LibHandle {
        let cstr = CString::new(path).unwrap_or_default();
        dlopen(cstr.as_ptr(), RTLD_NOW)
    }

    pub unsafe fn get_symbol(
        handle: LibHandle,
        name: &str,
    ) -> *mut std::ffi::c_void {
        let cstr = CString::new(name).unwrap_or_default();
        dlsym(handle, cstr.as_ptr())
    }

    pub unsafe fn free_library(handle: LibHandle) {
        if !handle.is_null() {
            dlclose(handle);
        }
    }

    pub fn is_null(handle: LibHandle) -> bool {
        handle.is_null()
    }
}

// ---------------------------------------------------------------------------
// Plugin library name
// ---------------------------------------------------------------------------

#[cfg(windows)]
const PLUGIN_FILENAME: &str = "redukt-gpu.dll";
#[cfg(target_os = "linux")]
const PLUGIN_FILENAME: &str = "libredukt_gpu.so";
#[cfg(target_os = "macos")]
const PLUGIN_FILENAME: &str = "libredukt_gpu.dylib";

// ---------------------------------------------------------------------------
// PluginLibrary
// ---------------------------------------------------------------------------

/// A loaded GPU backend plugin with resolved function pointers.
pub struct PluginLibrary {
    handle: platform::LibHandle,
    pub vtable: BackendVTable,
}

// SAFETY: The plugin library handle is process-global and the vtable
// function pointers are Sync by contract (single-threaded forward calls).
unsafe impl Send for PluginLibrary {}
unsafe impl Sync for PluginLibrary {}

impl PluginLibrary {
    /// Try to load the GPU plugin from next to the executable.
    /// Returns `None` silently if not found or load fails.
    pub fn try_load() -> Option<Self> {
        let exe = std::env::current_exe().ok()?;
        let dir = exe.parent()?;
        let path = dir.join(PLUGIN_FILENAME);
        if !path.exists() {
            return None;
        }
        let path_str = path.to_str()?;

        unsafe { Self::load_from(path_str) }
    }

    /// Load a plugin from an explicit path.  Returns `None` if any
    /// symbol is missing or the library cannot be opened.
    unsafe fn load_from(path: &str) -> Option<Self> {
        let handle = platform::load_library(path);
        if platform::is_null(handle) {
            return None;
        }

        let p_init = platform::get_symbol(handle, SYM_INIT);
        let p_fwd = platform::get_symbol(handle, SYM_FORWARD);
        let p_rst = platform::get_symbol(handle, SYM_RESET);
        let p_free = platform::get_symbol(handle, SYM_FREE);

        if p_init.is_null() || p_fwd.is_null()
            || p_rst.is_null() || p_free.is_null()
        {
            platform::free_library(handle);
            return None;
        }

        let vtable = BackendVTable {
            init: std::mem::transmute::<
                *mut std::ffi::c_void, BackendInitFn,
            >(p_init),
            forward: std::mem::transmute::<
                *mut std::ffi::c_void, BackendForwardFn,
            >(p_fwd),
            reset_state: std::mem::transmute::<
                *mut std::ffi::c_void, BackendResetFn,
            >(p_rst),
            free: std::mem::transmute::<
                *mut std::ffi::c_void, BackendFreeFn,
            >(p_free),
        };

        Some(Self { handle, vtable })
    }

    /// Initialize the backend.  Returns `BackendInfo` on success.
    pub fn init(
        &self,
        weights_path: &str,
        device_id: i32,
    ) -> Result<BackendInfo, BackendResult> {
        let mut buf = Vec::with_capacity(weights_path.len() + 1);
        buf.extend_from_slice(weights_path.as_bytes());
        buf.push(0);

        let mut info = BackendInfo {
            ffi_version: 0,
            vocab_size: 0,
            hidden_dim: 0,
            name: [0u8; crate::domain::backend_ffi::BACKEND_NAME_MAX],
        };

        let res = unsafe {
            (self.vtable.init)(buf.as_ptr(), device_id, &mut info)
        };
        if res != BackendResult::Ok {
            return Err(res);
        }
        if info.ffi_version != BACKEND_FFI_VERSION {
            return Err(BackendResult::VersionMismatch);
        }
        Ok(info)
    }
}

impl Drop for PluginLibrary {
    fn drop(&mut self) {
        unsafe { (self.vtable.free)(); }
        unsafe { platform::free_library(self.handle); }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_load_returns_none_when_no_plugin() {
        // No plugin .dll/.so exists next to the test binary.
        assert!(PluginLibrary::try_load().is_none());
    }

    #[test]
    fn plugin_filename_has_correct_extension() {
        #[cfg(windows)]
        assert!(PLUGIN_FILENAME.ends_with(".dll"));
        #[cfg(target_os = "linux")]
        assert!(PLUGIN_FILENAME.ends_with(".so"));
        #[cfg(target_os = "macos")]
        assert!(PLUGIN_FILENAME.ends_with(".dylib"));
    }

    #[test]
    fn backend_ffi_version_constant() {
        assert_eq!(BACKEND_FFI_VERSION, 1);
    }
}
