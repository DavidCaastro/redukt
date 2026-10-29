//! Hardware governor — dynamic resource regulation (R-HW1).
//!
//! Monitors battery and RAM during compression/decompression.
//! Communicates via AtomicU8 → ThrottleLevel (zero overhead in hot path).
//!
//! Pre-flight: rejects dangerous conditions before starting.
//! Watchdog: polls every 2s, updates ThrottleLevel atomically.
//! Hot path: one atomic load per check (~1ns).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Throttle levels — higher values = more restriction.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThrottleLevel {
    Full = 0,
    Eco = 1,
    Abort = 2,
}

impl ThrottleLevel {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Full,
            1 => Self::Eco,
            _ => Self::Abort,
        }
    }
}

impl std::fmt::Display for ThrottleLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Eco => write!(f, "eco"),
            Self::Abort => write!(f, "abort"),
        }
    }
}

/// Battery/power information.
pub struct PowerInfo {
    pub ac_online: bool,
    pub battery_percent: u8,
    pub has_battery: bool,
}

/// Query battery/power status via kernel32 FFI.
#[cfg(target_os = "windows")]
pub fn get_power_info() -> Option<PowerInfo> {
    #[repr(C)]
    struct SystemPowerStatus {
        ac_line_status: u8,
        battery_flag: u8,
        battery_life_percent: u8,
        system_status_flag: u8,
        battery_life_time: u32,
        battery_full_life_time: u32,
    }

    extern "system" {
        fn GetSystemPowerStatus(lp: *mut SystemPowerStatus) -> i32;
    }

    let mut status = SystemPowerStatus {
        ac_line_status: 0, battery_flag: 0, battery_life_percent: 0,
        system_status_flag: 0, battery_life_time: 0, battery_full_life_time: 0,
    };

    let ok = unsafe { GetSystemPowerStatus(&mut status) };
    if ok != 0 {
        Some(PowerInfo {
            ac_online: status.ac_line_status == 1,
            battery_percent: status.battery_life_percent,
            has_battery: status.battery_flag & 128 == 0,
        })
    } else {
        None
    }
}

#[cfg(not(target_os = "windows"))]
pub fn get_power_info() -> Option<PowerInfo> {
    None
}

/// Hardware governor with watchdog thread.
pub struct HardwareGovernor {
    level: Arc<AtomicU8>,
    enabled: bool,
}

impl HardwareGovernor {
    pub fn new(enabled: bool) -> Self {
        Self {
            level: Arc::new(AtomicU8::new(ThrottleLevel::Full as u8)),
            enabled,
        }
    }

    /// Pre-flight checks. Returns Err if conditions are too dangerous.
    pub fn preflight(&self) -> Result<(), String> {
        if !self.enabled { return Ok(()); }

        // RAM check
        if let Some(info) = crate::domain::ram_monitor::get_ram_info() {
            if info.percent_used > 95.0 {
                return Err(format!(
                    "RAM at {:.1}% — too high to start safely", info.percent_used
                ));
            }
            eprintln!("[governor] RAM: {}", info);
        }

        // Battery check
        if let Some(power) = get_power_info() {
            if power.has_battery {
                if power.ac_online {
                    eprintln!("[governor] power: AC online, battery {}%",
                              power.battery_percent);
                } else if power.battery_percent == 255 {
                    eprintln!("[governor] power: battery level unknown (unplugged)");
                } else {
                    eprintln!("[governor] power: battery {}% (unplugged)",
                              power.battery_percent);
                    if power.battery_percent < 5 {
                        return Err(format!(
                            "battery at {}% — too low to proceed", power.battery_percent
                        ));
                    }
                    if power.battery_percent < 20 {
                        eprintln!("[governor] WARNING: battery low — eco mode active");
                        self.level.store(ThrottleLevel::Eco as u8, Ordering::Relaxed);
                    }
                }
            } else {
                eprintln!("[governor] power: no battery (desktop/AC-only)");
            }
        } else {
            eprintln!("[governor] power: sensor unavailable (non-Windows)");
        }

        Ok(())
    }

    /// Start background watchdog. Polls battery + RAM every 2 seconds.
    pub fn start_watchdog(&self) {
        if !self.enabled { return; }

        let level = Arc::clone(&self.level);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));

                let mut new_level = ThrottleLevel::Full;

                // RAM check
                if let Some(info) = crate::domain::ram_monitor::get_ram_info() {
                    if info.percent_used > 97.0 {
                        new_level = ThrottleLevel::Abort;
                    } else if info.percent_used > 95.0 {
                        new_level = ThrottleLevel::Eco;
                    }
                }

                // Battery (only escalates, never deescalates past RAM)
                if new_level != ThrottleLevel::Abort {
                    if let Some(power) = get_power_info() {
                        if power.has_battery && !power.ac_online
                            && power.battery_percent != 255
                        {
                            if power.battery_percent < 5 {
                                new_level = ThrottleLevel::Abort;
                            } else if power.battery_percent < 20
                                && (new_level as u8) < ThrottleLevel::Eco as u8
                            {
                                new_level = ThrottleLevel::Eco;
                            }
                        }
                    }
                }

                level.store(new_level as u8, Ordering::Relaxed);
            }
        });
    }

    /// Read current throttle level. O(1), ~1ns.
    pub fn check(&self) -> ThrottleLevel {
        if !self.enabled { return ThrottleLevel::Full; }
        ThrottleLevel::from_u8(self.level.load(Ordering::Relaxed))
    }

    /// Apply throttling. Called periodically from the hot loop.
    /// Returns Err on Abort (caller should gracefully shut down).
    pub fn apply(&self) -> Result<(), String> {
        match self.check() {
            ThrottleLevel::Full => Ok(()),
            ThrottleLevel::Eco => {
                std::thread::sleep(std::time::Duration::from_millis(2));
                Ok(())
            }
            ThrottleLevel::Abort => {
                Err("governor: aborting — battery or RAM critically low".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_level_roundtrip() {
        for v in 0..=2u8 {
            let level = ThrottleLevel::from_u8(v);
            assert_eq!(level as u8, v);
        }
        // Out of range maps to Abort
        assert_eq!(ThrottleLevel::from_u8(99), ThrottleLevel::Abort);
    }

    #[test]
    fn disabled_governor_always_full() {
        let gov = HardwareGovernor::new(false);
        assert_eq!(gov.check(), ThrottleLevel::Full);
        assert!(gov.preflight().is_ok());
        assert!(gov.apply().is_ok());
    }

    #[test]
    fn atomic_level_update() {
        let gov = HardwareGovernor::new(true);
        assert_eq!(gov.check(), ThrottleLevel::Full);

        gov.level.store(ThrottleLevel::Eco as u8, Ordering::Relaxed);
        assert_eq!(gov.check(), ThrottleLevel::Eco);

        gov.level.store(ThrottleLevel::Abort as u8, Ordering::Relaxed);
        assert_eq!(gov.check(), ThrottleLevel::Abort);
        assert!(gov.apply().is_err());
    }

    #[test]
    fn preflight_passes_normally() {
        let gov = HardwareGovernor::new(true);
        // On any normal system, preflight should pass
        // (RAM < 95%, battery > 5% or AC)
        assert!(gov.preflight().is_ok());
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", ThrottleLevel::Full), "full");
        assert_eq!(format!("{}", ThrottleLevel::Eco), "eco");
        assert_eq!(format!("{}", ThrottleLevel::Abort), "abort");
    }
}
