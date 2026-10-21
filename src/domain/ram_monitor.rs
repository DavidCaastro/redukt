//! RAM monitoring with abort-on-threshold.
//!
//! Uses Windows GlobalMemoryStatusEx to check physical RAM usage.
//! Provides a guard that aborts model loading if RAM exceeds a threshold.

/// RAM usage info in bytes.
#[allow(dead_code)]
pub struct RamInfo {
    pub total: u64,
    pub available: u64,
    pub used: u64,
    pub percent_used: f64,
}

impl std::fmt::Display for RamInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1} GB used / {:.1} GB total ({:.1}%)",
            self.used as f64 / (1024.0 * 1024.0 * 1024.0),
            self.total as f64 / (1024.0 * 1024.0 * 1024.0),
            self.percent_used)
    }
}

/// Query current physical RAM usage.
/// Returns None if the system call fails (non-Windows or API error).
pub fn get_ram_info() -> Option<RamInfo> {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct MEMORYSTATUSEX {
            dw_length: u32,
            dw_memory_load: u32,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }

        extern "system" {
            fn GlobalMemoryStatusEx(lp_buffer: *mut MEMORYSTATUSEX) -> i32;
        }

        let mut mem = MEMORYSTATUSEX {
            dw_length: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };

        let ok = unsafe { GlobalMemoryStatusEx(&mut mem) };
        if ok != 0 {
            let used = mem.ull_total_phys - mem.ull_avail_phys;
            Some(RamInfo {
                total: mem.ull_total_phys,
                available: mem.ull_avail_phys,
                used,
                percent_used: used as f64 / mem.ull_total_phys as f64 * 100.0,
            })
        } else {
            None
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

/// Check RAM and abort (exit 1) if usage exceeds the given percent threshold.
/// Prints a clear diagnostic message before aborting.
pub fn check_ram_or_abort(threshold_percent: f64, context: &str) {
    if let Some(info) = get_ram_info() {
        if info.percent_used > threshold_percent {
            eprintln!("[RAM ABORT] {} — RAM at {:.1}% ({:.1} GB / {:.1} GB), threshold {:.0}%",
                context,
                info.percent_used,
                info.used as f64 / (1024.0 * 1024.0 * 1024.0),
                info.total as f64 / (1024.0 * 1024.0 * 1024.0),
                threshold_percent);
            eprintln!("[RAM ABORT] Aborting to prevent system instability. Free RAM or use a smaller model.");
            std::process::exit(1);
        }
    }
}

/// Log current RAM usage to stderr.
pub fn log_ram(context: &str) {
    if let Some(info) = get_ram_info() {
        eprintln!("[ram] {} — {}", context, info);
    }
}
