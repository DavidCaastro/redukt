//! R-104 F3 — libm lint: no platform transcendental may feed the coded stream.
//!
//! `f32`/`f64` methods such as `exp`, `ln`, `powf` or `tanh` call the platform libm (UCRT, glibc,
//! musl, macOS libSystem), whose results differ by ulps between platforms. One differing ulp in a
//! probability changes the CDF and breaks decoding on another machine. Since R-104 F1 every
//! transcendental on the codec path goes through `domain::fmath` (pure Rust, numeric contract in
//! the AZTH header). This test keeps it that way: it scans the non-test source and fails on any
//! new libm call outside the audited exceptions below.
//!
//! Allowed without a marker:
//! - `.sqrt(` — IEEE 754 requires correct rounding, so it is identical everywhere.
//! - `contract.exp(` / `contract.ln(` — the `NumericContract` dispatch (fmath, or libm only when
//!   decoding a pre-R-104 archive that was encoded with libm).
//! - `.log2(` on a line that accumulates `ce_bits` — cross-entropy telemetry, printed, never coded.
//! - Any line carrying a `libm-ok:` comment with the reason.
//!
//! Runs without weights: `cargo test --test libm_lint`.

use std::fs;
use std::path::{Path, PathBuf};

/// Methods that resolve to the platform libm (or, for `powi`, to an unspecified-precision intrinsic).
const BANNED: &[&str] = &[
    "exp", "exp2", "exp_m1", "ln", "ln_1p", "log", "log2", "log10", "powf", "powi", "cbrt", "hypot",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
];

/// Files outside the coded-stream path (path suffix, reason).
const EXEMPT_FILES: &[(&str, &str)] = &[
    ("domain/fmath.rs", "the portable implementation; its legacy branch calls libm by contract 0"),
    ("domain/spectral.rs", "offline weight analysis (spectral profile), never runs during coding"),
    ("application/telemetry.rs", "reporting only"),
    ("application/benchmark.rs", "evaluation commands, print metrics, write no archive"),
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {:?}: {}", dir, e));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

/// Lines of `text` before the trailing `#[cfg(test)] mod …` block (test code may use libm freely).
fn non_test_lines(text: &str) -> Vec<(usize, &str)> {
    let lines: Vec<&str> = text.lines().collect();
    let end = (0..lines.len())
        .find(|&i| {
            lines[i].trim() == "#[cfg(test)]" && lines.get(i + 1).is_some_and(|n| n.trim_start().starts_with("mod "))
        })
        .unwrap_or(lines.len());
    lines[..end].iter().enumerate().map(|(i, l)| (i + 1, *l)).collect()
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Banned calls on one line: method form `x.exp(` and path form `f64::exp(`.
fn violations(line: &str) -> Vec<&'static str> {
    let code = line.split("//").next().unwrap_or("");
    let mut found = Vec::new();
    for &name in BANNED {
        let method = format!(".{}(", name);
        for (pos, _) in code.match_indices(&method) {
            let receiver: String = code[..pos].chars().rev().take_while(|&c| is_ident(c)).collect();
            let receiver: String = receiver.chars().rev().collect();
            let dispatch = receiver == "contract" && (name == "exp" || name == "ln");
            let telemetry = name == "log2" && code.contains("ce_bits");
            if !dispatch && !telemetry {
                found.push(name);
            }
        }
        for prefix in ["f32::", "f64::"] {
            if code.contains(&format!("{}{}(", prefix, name)) {
                found.push(name);
            }
        }
    }
    found
}

#[test]
fn no_libm_on_the_codec_path() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    assert!(files.len() > 10, "source tree not found under {:?}", src);

    let mut report = Vec::new();
    let mut scanned = 0usize;
    for file in &files {
        let unix = file.to_string_lossy().replace('\\', "/");
        if EXEMPT_FILES.iter().any(|(suffix, _)| unix.ends_with(suffix)) {
            continue;
        }
        let text = fs::read_to_string(file).unwrap_or_else(|e| panic!("read {:?}: {}", file, e));
        scanned += 1;
        for (number, line) in non_test_lines(&text) {
            if line.contains("libm-ok:") {
                continue;
            }
            for name in violations(line) {
                report.push(format!("{}:{}: `{}` -> {}", unix, number, name, line.trim()));
            }
        }
    }
    assert!(scanned > 10, "only {} files scanned", scanned);
    assert!(
        report.is_empty(),
        "libm call on the codec path (use domain::fmath, or add `// libm-ok: <reason>` after an audit):\n{}",
        report.join("\n")
    );
}

#[test]
fn lint_detects_what_it_must() {
    assert_eq!(violations("let y = x.exp();"), vec!["exp"]);
    assert_eq!(violations("let y = f64::powf(a, b);"), vec!["powf"]);
    assert_eq!(violations("let y = (a + b).tanh() * c.ln();"), vec!["ln", "tanh"]);
    assert!(violations("let e = contract.exp(l - max_l);").is_empty());
    assert!(violations("ce_bits += -prob.max(1e-30).log2();").is_empty());
    assert_eq!(violations("h -= p * p.log2();"), vec!["log2"]);
    assert!(violations("let s = (var + eps).sqrt();").is_empty());
    assert!(violations("// x.exp() in a comment").is_empty());
    assert!(violations("let y = fmath::exp_f64(x);").is_empty());
}

#[test]
fn test_modules_are_skipped() {
    let text = "fn a() {}\n#[cfg(test)]\nfn helper() {}\n#[cfg(test)]\nmod tests {\n    fn t() { 1f32.exp(); }\n}\n";
    let kept = non_test_lines(text);
    assert_eq!(kept.len(), 3);
    assert_eq!(kept[2].1, "fn helper() {}");
}
