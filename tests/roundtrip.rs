//! End-to-end roundtrip: compress -> decompress -> verify byte-identical.
//! Requires RWKV-7 weights in weights/rwkv7-0.1b/.
//!
//! Run with: cargo test --test roundtrip -- --ignored

use std::path::Path;
use std::process::Command;

fn redukt_bin() -> std::path::PathBuf {
    let mut path = std::env::current_exe().unwrap();
    // test binary is in target/<profile>/deps/, go up to target/<profile>/
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push(format!("redukt{}", std::env::consts::EXE_SUFFIX));
    path
}

#[test]
#[ignore] // requires weights (~400 MB) — run explicitly with --ignored
fn compress_decompress_roundtrip() {
    let bin = redukt_bin();
    if !bin.exists() {
        panic!("binary not found at {:?} — run `cargo build --release` first", bin);
    }

    if !Path::new("weights/rwkv7-0.1b/model.safetensors").exists() {
        eprintln!("SKIP: weights not present, run scripts/download-weights.sh first");
        return;
    }

    let tmp = std::env::temp_dir().join("redukt_roundtrip_test");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    let compressed = tmp.join("output.azth");
    let restored = tmp.join("restored.txt");

    let test_data = b"Hello, world! This is a roundtrip test for Redukt. \
        The quick brown fox jumps over the lazy dog. 0123456789 \
        Lossless compression must recover every single byte.";
    std::fs::write(&input, test_data).unwrap();

    // Compress
    let status = Command::new(&bin)
        .args([
            "compress",
            "--input", input.to_str().unwrap(),
            "--output", compressed.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run compress");
    assert!(status.success(), "compress exited with {}", status);
    assert!(compressed.exists(), "compressed file not created");

    // Decompress
    let status = Command::new(&bin)
        .args([
            "decompress",
            "--input", compressed.to_str().unwrap(),
            "--output", restored.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run decompress");
    assert!(status.success(), "decompress exited with {}", status);

    // Verify byte-identical
    let original = std::fs::read(&input).unwrap();
    let recovered = std::fs::read(&restored).unwrap();
    assert_eq!(
        original.len(), recovered.len(),
        "size mismatch: original={}, recovered={}", original.len(), recovered.len()
    );
    assert_eq!(original, recovered, "roundtrip FAILED: bytes differ");

    // Cleanup
    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("roundtrip OK: {} bytes compressed and recovered", original.len());
}
