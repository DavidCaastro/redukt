//! Centralized error handling for Redukt — zero external dependencies.

use std::fmt;
use std::io;

/// Unified error type for all fallible operations.
#[derive(Debug)]
pub enum AzError {
    /// File I/O failure (read, write, create).
    Io { path: String, source: io::Error },
    /// CLI argument parsing or config parsing failure.
    Parse { context: String, value: String },
    /// Weight/model loading failure.
    ModelLoad(String),
}

impl fmt::Display for AzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AzError::Io { path, source } => write!(f, "I/O error on '{}': {}", path, source),
            AzError::Parse { context, value } => write!(f, "parse error in {}: '{}'", context, value),
            AzError::ModelLoad(msg) => write!(f, "model load error: {}", msg),
        }
    }
}

impl std::error::Error for AzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AzError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Convenience alias used throughout the codebase.
pub type AzResult<T> = Result<T, AzError>;

/// Read a file with structured error context.
pub fn read_file(path: &str) -> AzResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| AzError::Io { path: path.to_string(), source: e })
}

/// Write a file with structured error context.
pub fn write_file(path: &str, data: &[u8]) -> AzResult<()> {
    std::fs::write(path, data).map_err(|e| AzError::Io { path: path.to_string(), source: e })
}

/// Create a file for writing with structured error context.
pub fn create_file(path: &str) -> AzResult<std::fs::File> {
    std::fs::File::create(path).map_err(|e| AzError::Io { path: path.to_string(), source: e })
}

/// Parse a CLI argument value with structured error context.
pub fn parse_arg<T: std::str::FromStr>(value: &str, flag: &str) -> AzResult<T> {
    value.parse::<T>().map_err(|_| AzError::Parse {
        context: flag.to_string(),
        value: value.to_string(),
    })
}

/// Safely advance CLI arg index and return the next value.
/// Exits with a clear error if the flag is the last argument (no value follows).
pub fn next_arg<'a>(args: &'a [String], i: &mut usize, flag: &str) -> &'a str {
    *i += 1;
    if *i >= args.len() {
        eprintln!("error: {} requires a value", flag);
        std::process::exit(1);
    }
    &args[*i]
}

/// Maximum file size for `read_file_capped()` (2 GB).
/// Prevents accidental OOM from `std::fs::read()` on huge files.
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// Read a file with size check. Rejects files larger than MAX_FILE_SIZE.
pub fn read_file_capped(path: &str) -> AzResult<Vec<u8>> {
    let meta = std::fs::metadata(path)
        .map_err(|e| AzError::Io { path: path.to_string(), source: e })?;
    if meta.len() > MAX_FILE_SIZE {
        return Err(AzError::ModelLoad(format!(
            "'{}' is {} MB — exceeds {} MB limit",
            path, meta.len() / (1024 * 1024), MAX_FILE_SIZE / (1024 * 1024),
        )));
    }
    read_file(path)
}

/// Compare f32 values safely, treating NaN as less than everything.
/// Use this instead of `.partial_cmp().unwrap()` on floats.
pub fn safe_f32_cmp(a: &f32, b: &f32) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        // NaN handling: NaN sorts to the end (less than everything)
        if a.is_nan() && b.is_nan() {
            std::cmp::Ordering::Equal
        } else if a.is_nan() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    })
}
