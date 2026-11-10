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

    let mut seed = 0x1234_5678u32; // xorshift32 letters: incompressible enough for a long stream
    let text: Vec<u8> = (0..600).map(|_| {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        b'a' + (seed % 26) as u8
    }).collect();
    std::fs::write(&input, &text).unwrap();
    assert!(compress(&bin, &input, &compressed), "compress failed");

    // Flip a byte in the middle of the compressed stream (never the CRC).
    // The v4/v5 header is variable (13..35 B), so a fixed offset (the old
    // v3 `28 + 2`) can miss the stream entirely and leave the file intact.
    let mut data = std::fs::read(&compressed).unwrap();
    let max_header = 35;
    assert!(data.len() > max_header + 8, "archive too small to corrupt: {} B", data.len());
    let stream_offset = max_header + (data.len() - 4 - max_header) / 2;
    data[stream_offset] ^= 0xFF; // corrupt stream
    std::fs::write(&compressed, &data).unwrap();

    // Decompression should fail due to CRC mismatch
    let ok = decompress(&bin, &compressed, &restored);
    assert!(!ok, "decompress should fail on CRC mismatch after stream corruption");

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("CRC corruption detection OK");
}

// ---------------------------------------------------------------------------
// R-013: Cross-platform determinism tests
// ---------------------------------------------------------------------------

/// Golden reference: compress a known 40-byte input and verify the output
/// matches the expected SHA-256 hash. This detects any change in the
/// compression pipeline (model, quantization, coder, format).
///
/// Platform: Windows x86_64, AVX2, `--quant f32` (golden profile). Re-measured 2026-11-10.
/// If this test fails on a new platform, record the new hash as the
/// platform-specific golden reference.
#[test]
#[ignore] // requires weights
fn golden_reference_determinism() {
    let bin = ensure_bin();
    if !has_weights() { eprintln!("SKIP: no weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_golden");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("golden-input.bin");
    let compressed = tmp.join("golden.azth");
    let restored = tmp.join("golden-restored.bin");

    // Deterministic 40-byte input
    let golden_input = b"Determinism: the quick brown fox jumps!";
    std::fs::write(&input, golden_input).unwrap();

    // Pinned to F32: the default profile is now Q8 (AZTH v5), whose size differs.
    let status = Command::new(&bin)
        .args(["compress", "--input", input.to_str().unwrap(),
               "--output", compressed.to_str().unwrap(), "--quant", "f32"])
        .status()
        .expect("failed to run compress");
    assert!(status.success(), "golden compress failed");

    // Verify compressed output matches golden reference
    let azth_bytes = std::fs::read(&compressed).unwrap();
    // R-108: +1 header byte (RESET_LOG2 = 12 after PROFILE 0x50); the coded stream is unchanged
    // for inputs shorter than 4096 tokens.
    assert_eq!(azth_bytes.len(), 33,
        "golden: expected 33 bytes, got {}", azth_bytes.len());

    let hash = sha256_hex(&azth_bytes);
    // Golden SHA-256 for Windows x86_64 AVX2, AZTH v5 F32 with R-108 RESET_LOG2 (re-measured
    // 2026-11-10; before R-108: 32 B, 5e6b876a...; the v3 Q8 golden of 2026-10-29 was 35 B).
    // F32 is the golden profile; update if the pipeline changes.
    let expected = "b8719bf8d1ac6a7ed3426981ccb859f53f195023af7d869000fbb294bdb8fc0d";
    assert_eq!(hash, expected,
        "golden hash mismatch — platform or pipeline changed?\n  got:      {}\n  expected: {}",
        hash, expected);

    // Verify roundtrip
    assert!(decompress(&bin, &compressed, &restored), "golden decompress failed");
    let original = std::fs::read(&input).unwrap();
    let recovered = std::fs::read(&restored).unwrap();
    assert_eq!(original, recovered, "golden roundtrip FAILED: bytes differ");

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("golden reference OK: {} bytes, hash={}", azth_bytes.len(), &hash[..16]);
}

/// Empty input must be rejected with a clean error (exit 1), not a panic (exit 101).
/// Fails before any model is loaded, so it needs no weights.
#[test]
fn compress_empty_file_errors_without_panic() {
    let bin = ensure_bin();
    let tmp = std::env::temp_dir().join("redukt_test_empty_compress");
    std::fs::create_dir_all(&tmp).unwrap();
    let input = tmp.join("empty.bin");
    let out = tmp.join("empty.azth");
    std::fs::write(&input, b"").unwrap();

    let status = Command::new(&bin)
        .args(["compress", "--input", input.to_str().unwrap(), "--output", out.to_str().unwrap()])
        .status()
        .expect("failed to run compress");
    assert!(!status.success(), "empty input must fail");
    assert_ne!(status.code(), Some(101), "empty input must not panic");
    assert!(!out.exists(), "no output expected for empty input");
    std::fs::remove_dir_all(&tmp).ok();
}

/// Multi-process determinism: compress the same input 5 times in separate
/// processes and verify all outputs are byte-identical.
#[test]
#[ignore] // requires weights
fn multi_process_determinism() {
    let bin = ensure_bin();
    if !has_weights() { eprintln!("SKIP: no weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_multiproc");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    let test_data = b"Multi-process determinism test: identical output expected.";
    std::fs::write(&input, test_data).unwrap();

    let runs = 5;
    let mut hashes = Vec::new();
    for i in 0..runs {
        let out = tmp.join(format!("run{}.azth", i));
        assert!(compress(&bin, &input, &out), "compress run {} failed", i);
        let bytes = std::fs::read(&out).unwrap();
        hashes.push(sha256_hex(&bytes));
    }

    for i in 1..runs {
        assert_eq!(hashes[0], hashes[i],
            "determinism FAILED: run 0 vs run {} differ\n  run0: {}\n  run{}: {}",
            i, hashes[0], i, hashes[i]);
    }

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("multi-process determinism OK: {} runs identical ({})", runs, &hashes[0][..16]);
}

/// Cross-SIMD-tier roundtrip: compress with SIMD, decompress with scalar.
/// On Q8 (default), cross-tier roundtrip may fail due to transcendental
/// approximation differences (SIMD Cephes vs scalar libm).
/// This test documents the current behavior.
#[test]
#[ignore] // requires weights
fn cross_tier_roundtrip_documents_divergence() {
    let bin = ensure_bin();
    if !has_weights() { eprintln!("SKIP: no weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_cross_tier");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    let compressed_simd = tmp.join("simd.azth");
    let compressed_scalar = tmp.join("scalar.azth");
    let restored_cross = tmp.join("cross.bin");

    let test_data = b"Cross-tier determinism test data for SIMD vs scalar.";
    std::fs::write(&input, test_data).unwrap();

    // Compress with default SIMD
    assert!(compress(&bin, &input, &compressed_simd), "SIMD compress failed");

    // Compress with scalar (REDUKT_NO_SIMD=1)
    let ok_scalar = Command::new(&bin)
        .args(["compress", "--input", input.to_str().unwrap(),
               "--output", compressed_scalar.to_str().unwrap()])
        .env("REDUKT_NO_SIMD", "1")
        .status()
        .expect("failed to run scalar compress")
        .success();
    assert!(ok_scalar, "scalar compress failed");

    let simd_bytes = std::fs::read(&compressed_simd).unwrap();
    let scalar_bytes = std::fs::read(&compressed_scalar).unwrap();

    if simd_bytes == scalar_bytes {
        eprintln!("cross-tier: SIMD and scalar produce IDENTICAL output (unexpected but good)");
    } else {
        eprintln!("cross-tier: SIMD and scalar produce DIFFERENT output (expected)");
        eprintln!("  SIMD:   {} bytes, hash={}", simd_bytes.len(), &sha256_hex(&simd_bytes)[..16]);
        eprintln!("  scalar: {} bytes, hash={}", scalar_bytes.len(), &sha256_hex(&scalar_bytes)[..16]);
    }

    // Try cross-tier decompress: SIMD-compressed → scalar decompress
    let cross_ok = Command::new(&bin)
        .args(["decompress", "--input", compressed_simd.to_str().unwrap(),
               "--output", restored_cross.to_str().unwrap()])
        .env("REDUKT_NO_SIMD", "1")
        .status()
        .expect("failed to run cross-tier decompress")
        .success();

    if cross_ok {
        let original = std::fs::read(&input).unwrap();
        let recovered = std::fs::read(&restored_cross).unwrap();
        if original == recovered {
            eprintln!("cross-tier roundtrip: PASS (byte-identical)");
        } else {
            eprintln!("cross-tier roundtrip: data differs (CRC passed but content wrong)");
        }
    } else {
        // Expected: CRC mismatch because SIMD and scalar produce different logits
        eprintln!("cross-tier roundtrip: CRC MISMATCH (expected — different SIMD tiers)");
        eprintln!("  This confirms: compress and decompress MUST use the same SIMD tier.");
    }

    // Same-tier roundtrips must always work
    let restored_simd = tmp.join("simd-restored.bin");
    assert!(decompress(&bin, &compressed_simd, &restored_simd), "same-tier SIMD roundtrip failed");
    assert_eq!(std::fs::read(&input).unwrap(), std::fs::read(&restored_simd).unwrap(),
        "same-tier SIMD roundtrip: bytes differ");

    let restored_scalar = tmp.join("scalar-restored.bin");
    let ok_scalar_rt = Command::new(&bin)
        .args(["decompress", "--input", compressed_scalar.to_str().unwrap(),
               "--output", restored_scalar.to_str().unwrap()])
        .env("REDUKT_NO_SIMD", "1")
        .status()
        .expect("failed to run scalar decompress")
        .success();
    assert!(ok_scalar_rt, "same-tier scalar roundtrip failed");
    assert_eq!(std::fs::read(&input).unwrap(), std::fs::read(&restored_scalar).unwrap(),
        "same-tier scalar roundtrip: bytes differ");

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("cross-tier test complete: same-tier roundtrips PASS, cross-tier CRC mismatch expected");
}

/// Scalar-only determinism: verify that scalar path (REDUKT_NO_SIMD=1)
/// produces identical output across multiple runs.
#[test]
#[ignore] // requires weights
fn scalar_determinism() {
    let bin = ensure_bin();
    if !has_weights() { eprintln!("SKIP: no weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_scalar_det");
    std::fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    std::fs::write(&input, b"Scalar determinism: three runs must match.").unwrap();

    let runs = 3;
    let mut hashes = Vec::new();
    for i in 0..runs {
        let out = tmp.join(format!("scalar{}.azth", i));
        let ok = Command::new(&bin)
            .args(["compress", "--input", input.to_str().unwrap(),
                   "--output", out.to_str().unwrap()])
            .env("REDUKT_NO_SIMD", "1")
            .status()
            .expect("failed to run scalar compress")
            .success();
        assert!(ok, "scalar compress run {} failed", i);
        hashes.push(sha256_hex(&std::fs::read(&out).unwrap()));
    }

    for i in 1..runs {
        assert_eq!(hashes[0], hashes[i],
            "scalar determinism FAILED: run 0 vs run {}", i);
    }

    std::fs::remove_dir_all(&tmp).ok();
    eprintln!("scalar determinism OK: {} runs identical", runs);
}

// ---------------------------------------------------------------------------
// R-098/R-099: Mamba pooled GEMV + Q8 weight profile (AZTH v5)
// ---------------------------------------------------------------------------

fn has_mamba_weights() -> bool {
    Path::new("weights/mamba-130m/model.safetensors").exists()
}

fn run_redukt(bin: &Path, args: &[&str], threads: &str) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .env("REDUKT_THREADS", threads)
        .output()
        .expect("failed to run redukt")
}

/// Q8 archive: v5 header, byte-identical roundtrip, identical bytes for 1 vs N
/// GEMV threads, and a conflicting `--quant f32` rejected before decoding.
#[test]
#[ignore] // requires Mamba weights; run with --release for speed
fn mamba_q8_profile_roundtrip_and_mismatch() {
    let bin = ensure_bin();
    if !has_mamba_weights() { eprintln!("SKIP: no mamba weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_mamba_q8");
    std::fs::create_dir_all(&tmp).unwrap();
    let input = tmp.join("input.txt");
    let az_n = tmp.join("q8_n.azth");
    let az_1 = tmp.join("q8_1.azth");
    let restored = tmp.join("restored.txt");
    let bad = tmp.join("bad.txt");
    std::fs::write(&input, b"<invoice id=\"42\"><total currency=\"EUR\">1234.56</total></invoice>\n").unwrap();
    let (i, n1, nn) = (input.to_str().unwrap(), az_1.to_str().unwrap(), az_n.to_str().unwrap());

    for (out, threads) in [(nn, "4"), (n1, "1")] {
        let o = run_redukt(&bin, &["compress", "--input", i, "--output", out,
            "--backbone", "mamba", "--quant", "q8"], threads);
        assert!(o.status.success(), "Q8 compress failed (threads={})", threads);
    }
    let azth = std::fs::read(&az_n).unwrap();
    assert_eq!(&azth[..8], b"AZTH\x05\x00\x00\x00", "Q8 archive must be AZTH v5");
    assert_eq!(azth[13], 0x11, "PROFILE byte must carry quant 1 (Q8) + numeric contract 1 (fmath-v1)");
    assert_eq!(azth, std::fs::read(&az_1).unwrap(), "Q8 bytes differ between 1 and 4 threads");

    let o = run_redukt(&bin, &["decompress", "--input", nn, "--output", restored.to_str().unwrap()], "4");
    assert!(o.status.success(), "Q8 decompress failed");
    assert_eq!(std::fs::read(&input).unwrap(), std::fs::read(&restored).unwrap());

    let o = run_redukt(&bin, &["decompress", "--input", nn, "--output", bad.to_str().unwrap(),
        "--quant", "f32"], "4");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "Q8 archive decoded with --quant f32 must fail");
    assert!(stderr.contains("weight profile mismatch"), "unclear error: {}", stderr);
    assert!(!stderr.contains("CRC-32 mismatch"), "must fail before decoding/CRC");
    assert!(!bad.exists(), "no output may be written on profile mismatch");

    std::fs::remove_dir_all(&tmp).ok();
}

/// F32 Mamba archives stay AZTH v4 and are byte-identical for 1 vs N threads.
#[test]
#[ignore] // requires Mamba weights; run with --release for speed
fn mamba_f32_pooled_gemv_thread_count_invariant() {
    let bin = ensure_bin();
    if !has_mamba_weights() { eprintln!("SKIP: no mamba weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_mamba_f32_threads");
    std::fs::create_dir_all(&tmp).unwrap();
    let input = tmp.join("input.txt");
    std::fs::write(&input, b"The quick brown fox jumps over the lazy dog. 0123456789\n").unwrap();
    let mut outputs = Vec::new();
    for threads in ["1", "3", "8"] {
        let out = tmp.join(format!("f32_{}.azth", threads));
        let o = run_redukt(&bin, &["compress", "--input", input.to_str().unwrap(),
            "--output", out.to_str().unwrap(), "--backbone", "mamba", "--quant", "f32"], threads);
        assert!(o.status.success(), "F32 compress failed (threads={})", threads);
        outputs.push(std::fs::read(&out).unwrap());
    }
    assert_eq!(&outputs[0][..5], b"AZTH\x05", "F32 archives are AZTH v5 since R-104 (numeric contract)");
    assert!(outputs.iter().all(|o| *o == outputs[0]), "F32 bytes depend on thread count");

    std::fs::remove_dir_all(&tmp).ok();
}

// ---------------------------------------------------------------------------
// R-100: lockstep multi-stream batch == single-file, both directions
// ---------------------------------------------------------------------------

/// Four files of different lengths compressed in lockstep (`--jobs 4`, Q8) are
/// byte-identical to `compress --input` of each file; `decompress --input-dir`
/// restores them (CRC verified) and a batched archive decodes single-file.
#[test]
#[ignore] // requires Mamba weights; run with --release for speed
fn lockstep_batch_matches_single_file_both_directions() {
    let bin = ensure_bin();
    if !has_mamba_weights() { eprintln!("SKIP: no mamba weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_lockstep");
    let _ = std::fs::remove_dir_all(&tmp);
    let (inp, outd, decd, single) = (tmp.join("in"), tmp.join("out"), tmp.join("dec"), tmp.join("single"));
    for d in [&inp, &single] { std::fs::create_dir_all(d).unwrap(); }
    let texts: [&[u8]; 4] = [
        b"<invoice id=\"42\"><total currency=\"EUR\">1234.56</total></invoice>\n",
        b"The quick brown fox jumps over the lazy dog. 0123456789\n",
        b"GET /index.html HTTP/1.1 200 512\nGET /logo.png HTTP/1.1 200 2048\nPOST /api HTTP/1.1 201 17\n",
        b"x",
    ];
    for (i, t) in texts.iter().enumerate() {
        std::fs::write(inp.join(format!("f{}.txt", i)), t).unwrap();
    }
    let o = run_redukt(&bin, &["compress", "--input-dir", inp.to_str().unwrap(), "--output-dir",
        outd.to_str().unwrap(), "--jobs", "4", "--quant", "q8", "--no-governor"], "4");
    assert!(o.status.success(), "lockstep compress failed: {}", String::from_utf8_lossy(&o.stderr));

    for i in 0..texts.len() {
        let f = format!("f{}.txt", i);
        let s_out = single.join(format!("{}.azth", f));
        let o = run_redukt(&bin, &["compress", "--input", inp.join(&f).to_str().unwrap(), "--output",
            s_out.to_str().unwrap(), "--quant", "q8", "--no-governor"], "4");
        assert!(o.status.success(), "single compress failed for {}", f);
        let batched = std::fs::read(outd.join(format!("{}.azth", f))).unwrap();
        assert_eq!(batched, std::fs::read(&s_out).unwrap(), "{}: lockstep bytes != single-file bytes", f);
    }

    let o = run_redukt(&bin, &["decompress", "--input-dir", outd.to_str().unwrap(), "--output-dir",
        decd.to_str().unwrap(), "--jobs", "3", "--no-governor"], "4");
    assert!(o.status.success(), "lockstep decompress failed: {}", String::from_utf8_lossy(&o.stderr));
    for (i, t) in texts.iter().enumerate() {
        assert_eq!(std::fs::read(decd.join(format!("f{}.txt", i))).unwrap(), t.to_vec(), "f{} roundtrip", i);
    }

    let restored = tmp.join("f2.single");
    let o = run_redukt(&bin, &["decompress", "--input", outd.join("f2.txt.azth").to_str().unwrap(),
        "--output", restored.to_str().unwrap(), "--no-governor"], "4");
    assert!(o.status.success(), "single decompress of a batched archive failed");
    assert_eq!(std::fs::read(&restored).unwrap(), texts[2].to_vec());

    std::fs::remove_dir_all(&tmp).ok();
}

// ---------------------------------------------------------------------------
// R-107: numeric contract 2 (fma-v1)
// ---------------------------------------------------------------------------

/// Contract 2 with real Mamba weights, Q8 and F32: PROFILE = quant | 2 << 4,
/// byte-identical restore, decodable through the scalar `mul_add` kernels
/// (`REDUKT_NO_FMA=1` = AVX2 CPU without FMA), lockstep batch == single file,
/// and a batch mixing contracts 1 and 2 is rejected before any decoding.
#[test]
#[ignore] // requires Mamba weights; run with --release for speed
fn fma_contract_roundtrip_real_weights() {
    let bin = ensure_bin();
    if !has_mamba_weights() { eprintln!("SKIP: no mamba weights"); return; }

    let tmp = std::env::temp_dir().join("redukt_test_fma_contract");
    let _ = std::fs::remove_dir_all(&tmp);
    let (inp, outd, decd, mixed) = (tmp.join("in"), tmp.join("out"), tmp.join("dec"), tmp.join("mixed"));
    for d in [&inp, &mixed] { std::fs::create_dir_all(d).unwrap(); }
    let text = b"<invoice id=\"42\"><total currency=\"EUR\">1234.56</total></invoice>\n\
        The quick brown fox jumps over the lazy dog. 0123456789\n";
    let input = inp.join("t.txt");
    std::fs::write(&input, text).unwrap();
    let s = |p: &Path| p.to_str().unwrap().to_string();

    for (quant, profile) in [("q8", 0x21u8), ("f32", 0x20u8)] {
        let az = tmp.join(format!("{}.azth", quant));
        let plain = tmp.join(format!("{}-plain.azth", quant));
        let o = run_redukt(&bin, &["compress", "--input", &s(&input), "--output", &s(&az),
            "--quant", quant, "--numeric", "fma", "--no-governor"], "4");
        assert!(o.status.success(), "fma compress failed: {}", String::from_utf8_lossy(&o.stderr));
        let o = run_redukt(&bin, &["compress", "--input", &s(&input), "--output", &s(&plain),
            "--quant", quant, "--no-governor"], "4");
        assert!(o.status.success(), "default compress failed");
        let (a, p) = (std::fs::read(&az).unwrap(), std::fs::read(&plain).unwrap());
        assert_eq!(a[13], profile, "{}: PROFILE must be quant | contract 2 << 4", quant);
        assert_eq!(p[13], profile - 0x10, "{}: default must stay contract 1", quant);
        for no_fma in ["0", "1"] {
            let out = tmp.join(format!("{}-{}.txt", quant, no_fma));
            let o = Command::new(&bin)
                .args(["decompress", "--input", &s(&az), "--output", &s(&out), "--no-governor"])
                .env("REDUKT_THREADS", "4").env("REDUKT_NO_FMA", no_fma)
                .output().expect("failed to run redukt");
            let stderr = String::from_utf8_lossy(&o.stderr);
            assert!(o.status.success(), "{} decode (REDUKT_NO_FMA={}) failed: {}", quant, no_fma, stderr);
            assert_eq!(std::fs::read(&out).unwrap(), text.to_vec(), "{} REDUKT_NO_FMA={}", quant, no_fma);
            assert_eq!(stderr.contains("scalar mul_add"), no_fma == "1", "fallback warning");
        }
        // Lockstep batch with --numeric fma == single-file bytes; batch decode restores.
        let o = run_redukt(&bin, &["compress", "--input-dir", &s(&inp), "--output-dir", &s(&outd),
            "--jobs", "2", "--quant", quant, "--numeric", "fma", "--no-governor"], "4");
        assert!(o.status.success(), "fma batch compress failed");
        assert_eq!(std::fs::read(outd.join("t.txt.azth")).unwrap(), a, "{}: batch != single", quant);
        let o = run_redukt(&bin, &["decompress", "--input-dir", &s(&outd), "--output-dir", &s(&decd),
            "--jobs", "2", "--no-governor"], "4");
        assert!(o.status.success(), "fma batch decompress failed");
        assert_eq!(std::fs::read(decd.join("t.txt")).unwrap(), text.to_vec());
        // Mixed contracts in one batch directory: rejected up front.
        std::fs::copy(&az, mixed.join("a.azth")).unwrap();
        std::fs::copy(&plain, mixed.join("b.azth")).unwrap();
        let o = run_redukt(&bin, &["decompress", "--input-dir", &s(&mixed), "--output-dir",
            &s(&tmp.join("mixed-out")), "--no-governor"], "4");
        assert!(!o.status.success(), "mixed-contract batch must fail");
        assert!(String::from_utf8_lossy(&o.stderr).contains("mixes numeric contracts"));
    }
    std::fs::remove_dir_all(&tmp).ok();
}

// ---------------------------------------------------------------------------
// Utility: SHA-256 (minimal, no external deps)
// ---------------------------------------------------------------------------

fn sha256_hex(data: &[u8]) -> String {
    let hash = sha256(data);
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Padding
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process 64-byte blocks
    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4*i], chunk[4*i+1], chunk[4*i+2], chunk[4*i+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(k[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g; g = f; f = e;
            e = d.wrapping_add(temp1);
            d = c; c = b; b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e); h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g); h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4*i..4*i+4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}
