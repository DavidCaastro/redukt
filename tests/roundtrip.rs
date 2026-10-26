//! End-to-end roundtrip and edge-case tests.
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

fn has_weights() -> bool {
    Path::new("weights/rwkv7-0.1b/model.safetensors").exists()
}

fn ensure_bin() -> std::path::PathBuf {
    let bin = redukt_bin();
    if !bin.exists() {
        panic!("binary not found at {:?} — run `cargo build` first", bin);
    }
    bin
}

/// Compress input_path → compressed_path, return exit status success.
fn compress(bin: &Path, input: &Path, output: &Path) -> bool {
    Command::new(bin)
        .args(["compress", "--input", input.to_str().unwrap(),
               "--output", output.to_str().unwrap()])
        .status()
        .expect("failed to run compress")
        .success()
}

/// Decompress compressed_path → output_path, return exit status success.
fn decompress(bin: &Path, input: &Path, output: &Path) -> bool {
    Command::new(bin)
        .args(["decompress", "--input", input.to_str().unwrap(),
               "--output", output.to_str().unwrap()])
        .status()
        .expect("failed to run decompress")
        .success()
}

// ---------------------------------------------------------------------------
// R-005: Edge case tests
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires weights
fn compress_decompress_roundtrip() {
    let bin = ensure_bin();
    if !has_weights() { eprintln!("SKIP: no weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_roundtrip_test");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    let compressed = tmp.join("output.azth");
    let restored = tmp.join("restored.txt");

    let test_data = b"Hello, world! This is a roundtrip test for Redukt. \
        The quick brown fox jumps over the lazy dog. 0123456789 \
        Lossless compression must recover every single byte.";
    std::fs::write(&input, test_data).unwrap();

    assert!(compress(&bin, &input, &compressed), "compress failed");
    assert!(compressed.exists(), "compressed file not created");

    assert!(decompress(&bin, &compressed, &restored), "decompress failed");

    let original = std::fs::read(&input).unwrap();
    let recovered = std::fs::read(&restored).unwrap();
    assert_eq!(original.len(), recovered.len(),
        "size mismatch: original={}, recovered={}", original.len(), recovered.len());
    assert_eq!(original, recovered, "roundtrip FAILED: bytes differ");

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("roundtrip OK: {} bytes", original.len());
}

#[test]
#[ignore] // requires weights
fn roundtrip_single_byte() {
    let bin = ensure_bin();
    if !has_weights() { eprintln!("SKIP: no weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_1byte");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("one.bin");
    let compressed = tmp.join("one.azth");
    let restored = tmp.join("one.restored");

    std::fs::write(&input, &[0x42]).unwrap(); // single byte: 'B'

    assert!(compress(&bin, &input, &compressed), "compress 1-byte failed");
    assert!(decompress(&bin, &compressed, &restored), "decompress 1-byte failed");

    let original = std::fs::read(&input).unwrap();
    let recovered = std::fs::read(&restored).unwrap();
    assert_eq!(original, recovered, "1-byte roundtrip FAILED");

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("1-byte roundtrip OK");
}

#[test]
#[ignore] // requires weights
fn roundtrip_determinism() {
    let bin = ensure_bin();
    if !has_weights() { eprintln!("SKIP: no weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_determinism");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    let compressed1 = tmp.join("run1.azth");
    let compressed2 = tmp.join("run2.azth");

    let test_data = b"Determinism test: compress this twice and compare.";
    std::fs::write(&input, test_data).unwrap();

    assert!(compress(&bin, &input, &compressed1), "compress run1 failed");
    assert!(compress(&bin, &input, &compressed2), "compress run2 failed");

    let bytes1 = std::fs::read(&compressed1).unwrap();
    let bytes2 = std::fs::read(&compressed2).unwrap();
    assert_eq!(bytes1, bytes2, "determinism FAILED: two compressions differ");

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("determinism OK: {} bytes identical", bytes1.len());
}

#[test]
fn decompress_rejects_empty_file() {
    let bin = ensure_bin();
    let tmp = std::env::temp_dir().join("redukt_test_empty");
    std::fs::create_dir_all(&tmp).unwrap();

    let empty = tmp.join("empty.azth");
    let output = tmp.join("empty.out");
    std::fs::write(&empty, &[]).unwrap();

    // Decompression of a 0-byte file must fail (not panic, not produce output)
    let ok = decompress(&bin, &empty, &output);
    assert!(!ok, "decompress should reject empty file");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn decompress_rejects_short_file() {
    let bin = ensure_bin();
    let tmp = std::env::temp_dir().join("redukt_test_short");
    std::fs::create_dir_all(&tmp).unwrap();

    let short = tmp.join("short.azth");
    let output = tmp.join("short.out");
    // Valid magic prefix but only 16 bytes total (< 28 byte header)
    std::fs::write(&short, b"AZTH\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();

    let ok = decompress(&bin, &short, &output);
    assert!(!ok, "decompress should reject file shorter than header");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn decompress_rejects_bad_magic() {
    let bin = ensure_bin();
    let tmp = std::env::temp_dir().join("redukt_test_magic");
    std::fs::create_dir_all(&tmp).unwrap();

    let bad = tmp.join("bad_magic.azth");
    let output = tmp.join("bad_magic.out");
    // Wrong magic bytes, correct size
    let mut data = vec![0u8; 32];
    data[..4].copy_from_slice(b"NOPE");
    std::fs::write(&bad, &data).unwrap();

    let ok = decompress(&bin, &bad, &output);
    assert!(!ok, "decompress should reject bad magic");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn decompress_rejects_unsupported_version() {
    let bin = ensure_bin();
    let tmp = std::env::temp_dir().join("redukt_test_version");
    std::fs::create_dir_all(&tmp).unwrap();

    let bad = tmp.join("v99.azth");
    let output = tmp.join("v99.out");
    let mut data = vec![0u8; 32];
    data[..4].copy_from_slice(b"AZTH");
    data[4] = 99; // unsupported version
    std::fs::write(&bad, &data).unwrap();

    let ok = decompress(&bin, &bad, &output);
    assert!(!ok, "decompress should reject unsupported version");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
#[ignore] // requires weights
fn decompress_rejects_corrupted_crc() {
    let bin = ensure_bin();
    if !has_weights() { eprintln!("SKIP: no weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_crc_corrupt");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    let compressed = tmp.join("compressed.azth");
    let restored = tmp.join("restored.txt");

    std::fs::write(&input, b"CRC corruption test data.").unwrap();
    assert!(compress(&bin, &input, &compressed), "compress failed");

    // Flip a bit in the compressed stream (not the CRC itself)
    let mut data = std::fs::read(&compressed).unwrap();
    let stream_offset = 28 + 2; // header=28, skip a couple bytes into stream
    if stream_offset < data.len() - 4 {
        data[stream_offset] ^= 0xFF; // corrupt stream
    }
    std::fs::write(&compressed, &data).unwrap();

    // Decompression should fail due to CRC mismatch
    let ok = decompress(&bin, &compressed, &restored);
    assert!(!ok, "decompress should fail on CRC mismatch after stream corruption");

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("CRC corruption detection OK");
}
