//! Benchmark command — self-contained compression comparison tool.
//!
//! Performs ACTUAL compression (not just cross-entropy estimation) and compares
//! against gzip-9, brotli-11, zstd-19. All BPB values are measured identically:
//! compressed_size * 8 / original_size.
//!
//! With --verify: decompresses and verifies byte-identical roundtrip.
//!
//! Usage:
//!   redukt benchmark --input <FILE_OR_DIR> --output report.json [options]

use crate::domain::error::{AzError, AzResult, next_arg, parse_arg, read_file};
use crate::domain::tensor::{softmax, Tensor};
use crate::domain::ngram::TokenNgram;
use crate::domain::bias_head::BiasHead;
use crate::domain::mixer::AdaptiveMixer;
use crate::domain::coder::{Cdf, RangeEncoder};
use crate::domain::crc32;
use crate::application::codec::{cdf_from_logits, CRC_SIZE, build_v4_header};
use crate::application::model_loader;
use crate::infrastructure::rwkv7::model::{Rwkv7Model, Rwkv7State};
use crate::infrastructure::rwkv7::tokenizer::WorldTokenizer;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Redukt actual compression result for a single file.
struct ReduktResult {
    compressed_bpb: f64,
    ce_bpb: f64,
    compressed_size: usize,
    coder_efficiency_pct: f64,
    tokens: usize,
    bytes_per_tok: f64,
    tok_per_sec: f64,
    bytes_per_sec: f64,
    time_s: f64,
    roundtrip_ok: Option<bool>,
}

/// Competitor compressor result for a single file.
struct CompetitorResult {
    compressed_size: usize,
    bpb: f64,
}

/// Full benchmark result for one file.
struct FileResult {
    name: String,
    domain: String,
    bytes: usize,
    crc32: u32,
    redukt: ReduktResult,
    gzip9: Option<CompetitorResult>,
    brotli11: Option<CompetitorResult>,
    zstd19: Option<CompetitorResult>,
}

pub fn cmd_benchmark(args: &[String]) -> AzResult<()> {
    let mut input_path = String::new();
    let mut output_path = String::new();
    let mut weights_dir = "weights/rwkv7-0.1b".to_string();
    let mut max_bytes: usize = 10240;
    let mut quant_str = String::new();
    let mut emb_surgery = String::new();
    let mut cost_per_gb: f64 = 0.023;
    let mut retention_years: f64 = 5.0;
    let mut verbose = false;
    let mut verify = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input_path = next_arg(args, &mut i, "--input")?.to_string();
            }
            "--output" => {
                output_path = next_arg(args, &mut i, "--output")?.to_string();
            }
            "--weights" => {
                weights_dir = next_arg(args, &mut i, "--weights")?.to_string();
            }
            "--bytes" => {
                max_bytes = parse_arg(next_arg(args, &mut i, "--bytes")?, "--bytes")?;
            }
            "--quant" => {
                quant_str = next_arg(args, &mut i, "--quant")?.to_string();
            }
            "--emb-surgery" => {
                emb_surgery = next_arg(args, &mut i, "--emb-surgery")?.to_string();
            }
            "--cost-per-gb" => {
                cost_per_gb = parse_arg(
                    next_arg(args, &mut i, "--cost-per-gb")?,
                    "--cost-per-gb",
                )?;
            }
            "--retention-years" => {
                retention_years = parse_arg(
                    next_arg(args, &mut i, "--retention-years")?,
                    "--retention-years",
                )?;
            }
            "--verbose" | "-v" => verbose = true,
            "--verify" => verify = true,
            _ => {}
        }
        i += 1;
    }

    if input_path.is_empty() {
        return Err(AzError::Parse {
            context: "--input".into(),
            value: "(missing)".into(),
        });
    }
    if output_path.is_empty() {
        return Err(AzError::Parse {
            context: "--output".into(),
            value: "(missing)".into(),
        });
    }

    let files = discover_files(&input_path, max_bytes)?;
    if files.is_empty() {
        return Err(AzError::Io {
            path: input_path.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no files found at input path",
            ),
        });
    }

    eprintln!(
        "[benchmark] files: {}, window: {} bytes, verify: {}",
        files.len(),
        max_bytes,
        if verify { "YES" } else { "NO" }
    );

    // Load model once for all files
    eprintln!("[benchmark] loading model from {} ...", weights_dir);
    let quant_override = if quant_str.is_empty() {
        None
    } else {
        model_loader::parse_quant_flag(&quant_str)
    };
    let (mut model, tokenizer) =
        model_loader::load_model_and_tokenizer_with_quant(&weights_dir, quant_override);

    if !emb_surgery.is_empty() {
        model.embedding_surgery(&emb_surgery);
    }

    eprintln!(
        "[benchmark] model loaded (vocab={})",
        model.config.vocab_size
    );

    // Check competitor availability
    let has_gzip = check_available("gzip");
    let has_brotli = check_available("brotli");
    let has_zstd = check_available("zstd");

    eprintln!(
        "[benchmark] competitors: gzip={}, brotli={}, zstd={}",
        avail_str(has_gzip),
        avail_str(has_brotli),
        avail_str(has_zstd),
    );
    eprintln!();

    let total = files.len();
    let mut results: Vec<FileResult> = Vec::with_capacity(total);

    for (idx, (path, data)) in files.iter().enumerate() {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let domain = classify_domain(&name);
        let crc = crc32::crc32(data);

        eprintln!(
            "[{}/{}] {} ({}, {} bytes, CRC {:08X})",
            idx + 1,
            total,
            name,
            domain,
            data.len(),
            crc
        );

        // Redukt ACTUAL compression
        let (azth, mut redukt) = compress_and_measure(data, &model, &tokenizer, &emb_surgery);
        eprintln!(
            "  redukt:    {:.4} BPB (actual) | {:.4} BPB (ce) | {:.1}% coder eff | {:.1} B/s | {:.1}s",
            redukt.compressed_bpb,
            redukt.ce_bpb,
            redukt.coder_efficiency_pct,
            redukt.bytes_per_sec,
            redukt.time_s
        );
        eprintln!(
            "             {} bytes -> {} bytes (ratio {:.4})",
            data.len(),
            redukt.compressed_size,
            redukt.compressed_size as f64 / data.len() as f64
        );

        // Optional roundtrip verification
        if verify {
            eprint!("  verify:    decompressing ...");
            let rt_ok = verify_roundtrip(&azth, data, &model, &tokenizer);
            redukt.roundtrip_ok = Some(rt_ok);
            if rt_ok {
                eprintln!("\r  verify:    PASS (byte-identical roundtrip)           ");
            } else {
                eprintln!("\r  verify:    **FAIL** — decompressed output differs!   ");
            }
        }

        // Competitors (all use same metric: compressed_size * 8 / input_size)
        let gzip9 = if has_gzip {
            compress_pipe(data, "gzip", &["-9"])
        } else {
            None
        };
        if let Some(ref g) = gzip9 {
            eprintln!(
                "  gzip-9:    {:.4} BPB ({} bytes)",
                g.bpb, g.compressed_size
            );
        }

        let brotli11 = if has_brotli {
            compress_pipe(data, "brotli", &["-q", "11", "-c"])
        } else {
            None
        };
        if let Some(ref b) = brotli11 {
            eprintln!(
                "  brotli-11: {:.4} BPB ({} bytes)",
                b.bpb, b.compressed_size
            );
        }

        let zstd19 = if has_zstd {
            compress_pipe(data, "zstd", &["-19", "-c", "--no-progress"])
        } else {
            None
        };
        if let Some(ref z) = zstd19 {
            eprintln!(
                "  zstd-19:   {:.4} BPB ({} bytes)",
                z.bpb, z.compressed_size
            );
        }

        // Delta vs best competitor (using compressed_bpb — same metric)
        let ref_result = brotli11.as_ref().or(gzip9.as_ref());
        if let Some(ref r) = ref_result {
            let delta = ((redukt.compressed_bpb - r.bpb) / r.bpb) * 100.0;
            let ref_name = if brotli11.is_some() { "brotli" } else { "gzip" };
            eprintln!("  vs {}:  {:+.1}%", ref_name, delta);
        }

        results.push(FileResult {
            name,
            domain,
            bytes: data.len(),
            crc32: crc,
            redukt,
            gzip9,
            brotli11,
            zstd19,
        });
        eprintln!();
    }

    // Write JSON report
    let json = format_json(
        &results,
        &input_path,
        max_bytes,
        &weights_dir,
        cost_per_gb,
        retention_years,
        verify,
    );
    crate::domain::error::write_file(&output_path, json.as_bytes())?;

    // Print human-readable summary
    print_summary(&results, verbose);

    eprintln!("[benchmark] report written to: {}", output_path);
    Ok(())
}

// --- Actual compression + measurement ---

fn compress_and_measure(
    data: &[u8],
    model: &Rwkv7Model,
    tokenizer: &WorldTokenizer,
    emb_surgery: &str,
) -> (Vec<u8>, ReduktResult) {
    let v = model.config.vocab_size;
    let tokens = tokenizer.encode(data);
    let total_bytes = data.len();

    let mut ngram = TokenNgram::new(4, v, 0.5);
    let mut bias = BiasHead::new(v, 0.30);
    let mut mixer = AdaptiveMixer::new(0.01);
    let mut enc = RangeEncoder::new();
    let mut state = Rwkv7State::new(&model.config);
    let mut scratch = model.create_scratch();
    let mut logits = vec![0.0f32; v];
    let mut ce_bits = 0.0f64;

    let t_start = Instant::now();

    for t in 0..tokens.len() {
        let tok = tokens[t] as usize;

        if t == 0 {
            let cdf = Cdf::uniform(v);
            enc.encode_symbol(tok, &cdf);
            let prob = 1.0 / v as f64;
            ce_bits += -prob.log2();
        } else {
            // Uses None for entropy gating — matches cmd_compress exactly
            let ng_bias = ngram.compute_bias(None);
            let b_vec = bias.bias_vector();
            let ensemble = mixer.combine(&logits, &ng_bias, b_vec);

            let cdf = cdf_from_logits(&ensemble);
            enc.encode_symbol(tok, &cdf);

            let tensor = Tensor::from_data(ensemble, vec![v]);
            let probs = softmax(&tensor);
            let prob = probs.data[tok] as f64;
            ce_bits += -prob.max(1e-30).log2();

            mixer.update(
                &probs.data,
                &ngram.compute_bias(None),
                bias.bias_vector(),
                tok,
            );
            bias.update(&probs.data, tok);
        }

        ngram.observe(tok as u32);
        model.forward_into(tok, &mut state, &mut scratch, &mut logits);

        if t > 0 && (t % 500 == 0 || t == tokens.len() - 1) {
            let elapsed = t_start.elapsed().as_secs_f64();
            eprint!(
                "\r  redukt:    {:.1}% | {:.1} tok/s   ",
                100.0 * t as f64 / tokens.len() as f64,
                t as f64 / elapsed
            );
        }
    }
    eprint!("\r                                                        \r");

    let compressed = enc.finish();
    let elapsed = t_start.elapsed().as_secs_f64();

    // Build AZTH v4 (identical to cmd_compress output)
    let input_crc = crc32::crc32(data);
    let hdr = build_v4_header(total_bytes as u32, 0.30, 0.50, 0.01, &emb_surgery, false, 0, 0, false);
    let mut azth = Vec::with_capacity(hdr.len() + compressed.len() + CRC_SIZE);
    azth.extend_from_slice(&hdr);
    azth.extend_from_slice(&compressed);
    azth.extend_from_slice(&input_crc.to_le_bytes());

    let tok_per_sec = tokens.len() as f64 / elapsed;
    let bytes_per_tok = total_bytes as f64 / tokens.len().max(1) as f64;
    let bytes_per_sec = tok_per_sec * bytes_per_tok;
    let ce_bpb = ce_bits / total_bytes as f64;
    let compressed_bpb = (azth.len() as f64 * 8.0) / total_bytes as f64;
    let efficiency = if azth.len() > 0 {
        (ce_bits / (azth.len() as f64 * 8.0)) * 100.0
    } else {
        0.0
    };

    let result = ReduktResult {
        compressed_bpb,
        ce_bpb,
        compressed_size: azth.len(),
        coder_efficiency_pct: efficiency,
        tokens: tokens.len(),
        bytes_per_tok,
        tok_per_sec,
        bytes_per_sec,
        time_s: elapsed,
        roundtrip_ok: None,
    };

    (azth, result)
}

/// Decompress AZTH and verify byte-identical output.
fn verify_roundtrip(
    azth: &[u8],
    original: &[u8],
    model: &Rwkv7Model,
    tokenizer: &WorldTokenizer,
) -> bool {
    match crate::application::api::decompress_bytes(azth, model, tokenizer) {
        Ok(restored) => restored == original,
        Err(e) => {
            eprintln!("\n  verify:    decompress error: {}", e);
            false
        }
    }
}

/// Compress data by piping through an external compressor.
fn compress_pipe(data: &[u8], cmd: &str, args: &[&str]) -> Option<CompetitorResult> {
    use std::process::{Command, Stdio};

    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(data);
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }

    let compressed_size = output.stdout.len();
    let bpb = (compressed_size as f64 * 8.0) / data.len().max(1) as f64;
    Some(CompetitorResult {
        compressed_size,
        bpb,
    })
}

fn check_available(cmd: &str) -> bool {
    use std::process::{Command, Stdio};
    Command::new(cmd)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn avail_str(available: bool) -> &'static str {
    if available {
        "YES"
    } else {
        "NO"
    }
}

// --- File discovery ---

fn discover_files(input: &str, max_bytes: usize) -> AzResult<Vec<(PathBuf, Vec<u8>)>> {
    let path = Path::new(input);
    let mut files = Vec::new();

    if path.is_file() {
        let data = read_file(input)?;
        let n = if max_bytes > 0 {
            max_bytes.min(data.len())
        } else {
            data.len()
        };
        if n > 0 {
            files.push((path.to_path_buf(), data[..n].to_vec()));
        }
    } else if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| AzError::Io {
                path: input.to_string(),
                source: e,
            })?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && !p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .starts_with('.')
            })
            .collect();
        entries.sort();

        for p in entries {
            let path_str = p.to_string_lossy().to_string();
            match read_file(&path_str) {
                Ok(data) => {
                    let n = if max_bytes > 0 {
                        max_bytes.min(data.len())
                    } else {
                        data.len()
                    };
                    if n > 0 {
                        files.push((p, data[..n].to_vec()));
                    }
                }
                Err(e) => eprintln!("[benchmark] skip {}: {}", path_str, e),
            }
        }
    } else {
        return Err(AzError::Io {
            path: input.to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "path not found"),
        });
    }

    Ok(files)
}

// --- Domain classification ---

fn classify_domain(name: &str) -> String {
    match name {
        "enwik8" => "text/encyclopedia",
        "dickens" => "text/literature",
        "webster" => "text/dictionary",
        "reymont" => "text/polish",
        "samba" => "code/c-source",
        "mozilla" => "binary/executable",
        "ooffice" => "binary/office",
        "mr" => "medical/mri",
        "x-ray" => "medical/xray",
        "nci" => "scientific/chemistry",
        "sao" => "scientific/astronomy",
        "osdb" => "database/mysql",
        "xml" => "markup/xml",
        "apache-access.log" => "logs/apache",
        "hdfs-audit.log" => "logs/hdfs",
        "linux-syslog.log" => "logs/syslog",
        "openssh-auth.log" => "logs/ssh",
        "ubl-invoice-batch.xml" => "e-invoicing/ubl",
        "facturae-batch.xml" => "e-invoicing/facturae",
        _ => {
            if name.ends_with(".log") {
                "logs/other"
            } else if name.ends_with(".xml") {
                "markup/xml"
            } else if name.ends_with(".json") {
                "data/json"
            } else if name.ends_with(".csv") {
                "data/csv"
            } else {
                "unknown"
            }
        }
    }
    .to_string()
}

// --- Hardware info ---

fn get_cpu_name() -> String {
    use std::process::Command;

    // Windows: wmic
    if let Ok(output) = Command::new("wmic")
        .args(["cpu", "get", "Name", "/value"])
        .output()
    {
        let s = String::from_utf8_lossy(&output.stdout);
        for line in s.lines() {
            if let Some(name) = line.strip_prefix("Name=") {
                let trimmed = name.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }

    // Linux: /proc/cpuinfo
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in info.lines() {
            if line.starts_with("model name") {
                if let Some((_key, val)) = line.split_once(':') {
                    let trimmed = val.trim();
                    if !trimmed.is_empty() {
                        return trimmed.to_string();
                    }
                }
            }
        }
    }

    "unknown".to_string()
}

fn git_short_sha() -> String {
    use std::process::Command;
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

// --- JSON output ---

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => out.push(c),
        }
    }
    out
}

fn format_json(
    results: &[FileResult],
    input_path: &str,
    window_bytes: usize,
    weights_dir: &str,
    cost_per_gb: f64,
    retention_years: f64,
    verify: bool,
) -> String {
    let mut s = String::with_capacity(8192);
    let sha = git_short_sha();
    let cpu = get_cpu_name();
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);

    s.push_str("{\n");

    // Metadata
    s.push_str("  \"metadata\": {\n");
    s.push_str("    \"generator\": \"redukt benchmark\",\n");
    s.push_str(&format!(
        "    \"git_commit\": \"{}\",\n",
        json_escape(&sha)
    ));
    s.push_str(&format!(
        "    \"input\": \"{}\",\n",
        json_escape(input_path)
    ));
    s.push_str(&format!("    \"window_bytes\": {},\n", window_bytes));
    s.push_str(&format!(
        "    \"weights\": \"{}\",\n",
        json_escape(weights_dir)
    ));
    s.push_str(&format!("    \"roundtrip_verified\": {},\n", verify));
    s.push_str(&format!(
        "    \"os\": \"{}\",\n",
        std::env::consts::OS
    ));
    s.push_str(&format!(
        "    \"arch\": \"{}\",\n",
        std::env::consts::ARCH
    ));
    s.push_str(&format!(
        "    \"cpu\": \"{}\",\n",
        json_escape(&cpu)
    ));
    s.push_str(&format!("    \"logical_cpus\": {},\n", logical_cpus));
    s.push_str(&format!(
        "    \"cost_per_gb_month\": {},\n",
        cost_per_gb
    ));
    s.push_str(&format!(
        "    \"retention_years\": {},\n",
        retention_years
    ));
    s.push_str("    \"methodology\": \"All BPB values = compressed_size * 8 / original_size. \
        Redukt BPB is from actual AZTH file output, not cross-entropy estimate. \
        Competitors compressed via stdin pipe. Apples-to-apples comparison.\"\n");
    s.push_str("  },\n");

    // Files array
    s.push_str("  \"files\": [\n");
    for (i, r) in results.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!(
            "      \"name\": \"{}\",\n",
            json_escape(&r.name)
        ));
        s.push_str(&format!(
            "      \"domain\": \"{}\",\n",
            json_escape(&r.domain)
        ));
        s.push_str(&format!("      \"bytes\": {},\n", r.bytes));
        s.push_str(&format!("      \"crc32\": \"{:08X}\",\n", r.crc32));

        // Redukt
        s.push_str("      \"redukt\": {\n");
        s.push_str(&format!(
            "        \"compressed_size\": {},\n",
            r.redukt.compressed_size
        ));
        s.push_str(&format!(
            "        \"compressed_bpb\": {:.6},\n",
            r.redukt.compressed_bpb
        ));
        s.push_str(&format!(
            "        \"ce_bpb\": {:.6},\n",
            r.redukt.ce_bpb
        ));
        s.push_str(&format!(
            "        \"ratio\": {:.6},\n",
            r.redukt.compressed_size as f64 / r.bytes.max(1) as f64
        ));
        s.push_str(&format!(
            "        \"coder_efficiency_pct\": {:.2},\n",
            r.redukt.coder_efficiency_pct
        ));
        s.push_str(&format!(
            "        \"tokens\": {},\n",
            r.redukt.tokens
        ));
        s.push_str(&format!(
            "        \"bytes_per_tok\": {:.2},\n",
            r.redukt.bytes_per_tok
        ));
        s.push_str(&format!(
            "        \"tok_per_sec\": {:.1},\n",
            r.redukt.tok_per_sec
        ));
        s.push_str(&format!(
            "        \"bytes_per_sec\": {:.1},\n",
            r.redukt.bytes_per_sec
        ));
        s.push_str(&format!(
            "        \"time_s\": {:.3},\n",
            r.redukt.time_s
        ));
        match r.redukt.roundtrip_ok {
            Some(true) => s.push_str("        \"roundtrip\": \"PASS\"\n"),
            Some(false) => s.push_str("        \"roundtrip\": \"FAIL\"\n"),
            None => s.push_str("        \"roundtrip\": null\n"),
        }
        s.push_str("      },\n");

        // Competitors
        format_competitor(&mut s, "gzip9", &r.gzip9, r.bytes);
        format_competitor(&mut s, "brotli11", &r.brotli11, r.bytes);
        format_competitor_last(&mut s, "zstd19", &r.zstd19, r.bytes);

        s.push_str("    }");
        if i < results.len() - 1 {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ],\n");

    // Summary
    write_summary_json(&mut s, results, cost_per_gb, retention_years);

    s.push_str("}\n");
    s
}

fn format_competitor(
    s: &mut String,
    name: &str,
    result: &Option<CompetitorResult>,
    input_bytes: usize,
) {
    match result {
        Some(r) => {
            s.push_str(&format!("      \"{}\": {{\n", name));
            s.push_str(&format!(
                "        \"compressed_size\": {},\n",
                r.compressed_size
            ));
            s.push_str(&format!("        \"bpb\": {:.6},\n", r.bpb));
            s.push_str(&format!(
                "        \"ratio\": {:.6}\n",
                r.compressed_size as f64 / input_bytes.max(1) as f64
            ));
            s.push_str("      },\n");
        }
        None => {
            s.push_str(&format!("      \"{}\": null,\n", name));
        }
    }
}

fn format_competitor_last(
    s: &mut String,
    name: &str,
    result: &Option<CompetitorResult>,
    input_bytes: usize,
) {
    match result {
        Some(r) => {
            s.push_str(&format!("      \"{}\": {{\n", name));
            s.push_str(&format!(
                "        \"compressed_size\": {},\n",
                r.compressed_size
            ));
            s.push_str(&format!("        \"bpb\": {:.6},\n", r.bpb));
            s.push_str(&format!(
                "        \"ratio\": {:.6}\n",
                r.compressed_size as f64 / input_bytes.max(1) as f64
            ));
            s.push_str("      }\n");
        }
        None => {
            s.push_str(&format!("      \"{}\": null\n", name));
        }
    }
}

fn write_summary_json(
    s: &mut String,
    results: &[FileResult],
    cost_per_gb: f64,
    retention_years: f64,
) {
    let n = results.len();
    let mean_compressed: f64 =
        results.iter().map(|r| r.redukt.compressed_bpb).sum::<f64>() / n as f64;
    let mean_ce: f64 = results.iter().map(|r| r.redukt.ce_bpb).sum::<f64>() / n as f64;
    let mean_throughput: f64 =
        results.iter().map(|r| r.redukt.bytes_per_sec).sum::<f64>() / n as f64;

    let (wins_gzip, mean_gzip) = competitor_stats(results, |r| &r.gzip9);
    let (wins_brotli, mean_brotli) = competitor_stats(results, |r| &r.brotli11);
    let (wins_zstd, mean_zstd) = competitor_stats(results, |r| &r.zstd19);

    s.push_str("  \"summary\": {\n");
    s.push_str(&format!("    \"total_files\": {},\n", n));
    s.push_str(&format!(
        "    \"mean_redukt_compressed_bpb\": {:.6},\n",
        mean_compressed
    ));
    s.push_str(&format!(
        "    \"mean_redukt_ce_bpb\": {:.6},\n",
        mean_ce
    ));
    s.push_str(&format!(
        "    \"mean_throughput_bps\": {:.1},\n",
        mean_throughput
    ));

    if mean_gzip > 0.0 {
        s.push_str(&format!("    \"mean_gzip9_bpb\": {:.6},\n", mean_gzip));
        s.push_str(&format!("    \"redukt_wins_vs_gzip9\": {},\n", wins_gzip));
    }
    if mean_brotli > 0.0 {
        s.push_str(&format!(
            "    \"mean_brotli11_bpb\": {:.6},\n",
            mean_brotli
        ));
        s.push_str(&format!(
            "    \"redukt_wins_vs_brotli11\": {},\n",
            wins_brotli
        ));
    }
    if mean_zstd > 0.0 {
        s.push_str(&format!("    \"mean_zstd19_bpb\": {:.6},\n", mean_zstd));
        s.push_str(&format!(
            "    \"redukt_wins_vs_zstd19\": {},\n",
            wins_zstd
        ));
    }

    // Storage savings (using compressed_bpb — honest metric)
    let ref_bpb = if mean_brotli > 0.0 {
        mean_brotli
    } else if mean_gzip > 0.0 {
        mean_gzip
    } else {
        8.0
    };
    let ref_name = if mean_brotli > 0.0 {
        "brotli11"
    } else if mean_gzip > 0.0 {
        "gzip9"
    } else {
        "uncompressed"
    };

    let redukt_ratio = mean_compressed / 8.0;
    let ref_ratio = ref_bpb / 8.0;
    let savings_pct = if ref_ratio > 0.0 {
        (1.0 - redukt_ratio / ref_ratio) * 100.0
    } else {
        0.0
    };

    s.push_str("    \"storage_projection\": {\n");
    s.push_str(&format!(
        "      \"reference\": \"{}\",\n",
        ref_name
    ));
    s.push_str(&format!(
        "      \"redukt_ratio\": {:.6},\n",
        redukt_ratio
    ));
    s.push_str(&format!(
        "      \"reference_ratio\": {:.6},\n",
        ref_ratio
    ));
    s.push_str(&format!(
        "      \"savings_vs_reference_pct\": {:.1},\n",
        savings_pct
    ));
    s.push_str(&format!(
        "      \"cost_per_gb_month\": {},\n",
        cost_per_gb
    ));
    s.push_str(&format!(
        "      \"retention_years\": {}\n",
        retention_years
    ));
    s.push_str("    }\n");
    s.push_str("  }\n");
}

fn competitor_stats<F>(results: &[FileResult], getter: F) -> (usize, f64)
where
    F: Fn(&FileResult) -> &Option<CompetitorResult>,
{
    let available: Vec<(&FileResult, &CompetitorResult)> = results
        .iter()
        .filter_map(|r| getter(r).as_ref().map(|c| (r, c)))
        .collect();

    if available.is_empty() {
        return (0, 0.0);
    }

    // Wins using compressed_bpb (actual file size, not cross-entropy)
    let wins = available
        .iter()
        .filter(|(r, c)| r.redukt.compressed_bpb < c.bpb)
        .count();
    let mean = available.iter().map(|(_, c)| c.bpb).sum::<f64>() / available.len() as f64;

    (wins, mean)
}

// --- Human-readable summary ---

fn print_summary(results: &[FileResult], verbose: bool) {
    eprintln!("================================================================");
    eprintln!("  REDUKT BENCHMARK SUMMARY");
    eprintln!("  (all BPB = compressed_size * 8 / original_size)");
    eprintln!("================================================================");
    eprintln!();

    let has_brotli = results.iter().any(|r| r.brotli11.is_some());
    let has_gzip = results.iter().any(|r| r.gzip9.is_some());
    let has_zstd = results.iter().any(|r| r.zstd19.is_some());
    let has_roundtrip = results.iter().any(|r| r.redukt.roundtrip_ok.is_some());

    // Header
    eprint!("  {:<24} {:>6} {:>8}", "File", "Bytes", "Redukt");
    if has_gzip {
        eprint!(" {:>8}", "gzip-9");
    }
    if has_brotli {
        eprint!(" {:>8}", "brotli");
    }
    if has_zstd {
        eprint!(" {:>8}", "zstd-19");
    }
    if has_brotli {
        eprint!(" {:>10}", "vs brotli");
    } else if has_gzip {
        eprint!(" {:>10}", "vs gzip");
    }
    if has_roundtrip {
        eprint!(" {:>5}", "RT");
    }
    eprintln!();

    eprint!("  {:<24} {:>6} {:>8}", "----", "-----", "------");
    if has_gzip {
        eprint!(" {:>8}", "------");
    }
    if has_brotli {
        eprint!(" {:>8}", "------");
    }
    if has_zstd {
        eprint!(" {:>8}", "-------");
    }
    if has_brotli || has_gzip {
        eprint!(" {:>10}", "---------");
    }
    if has_roundtrip {
        eprint!(" {:>5}", "----");
    }
    eprintln!();

    for r in results {
        eprint!(
            "  {:<24} {:>6} {:>8.4}",
            truncate(&r.name, 24),
            r.bytes,
            r.redukt.compressed_bpb
        );
        if has_gzip {
            match &r.gzip9 {
                Some(g) => eprint!(" {:>8.4}", g.bpb),
                None => eprint!(" {:>8}", "-"),
            }
        }
        if has_brotli {
            match &r.brotli11 {
                Some(b) => eprint!(" {:>8.4}", b.bpb),
                None => eprint!(" {:>8}", "-"),
            }
        }
        if has_zstd {
            match &r.zstd19 {
                Some(z) => eprint!(" {:>8.4}", z.bpb),
                None => eprint!(" {:>8}", "-"),
            }
        }
        // Delta vs best competitor (using compressed_bpb)
        let ref_bpb = r
            .brotli11
            .as_ref()
            .map(|b| b.bpb)
            .or_else(|| r.gzip9.as_ref().map(|g| g.bpb));
        if let Some(rb) = ref_bpb {
            let delta = ((r.redukt.compressed_bpb - rb) / rb) * 100.0;
            let marker = if delta < 0.0 { "<<" } else { "" };
            eprint!(" {:>+8.1}% {}", delta, marker);
        }
        if has_roundtrip {
            match r.redukt.roundtrip_ok {
                Some(true) => eprint!("  OK"),
                Some(false) => eprint!(" FAIL"),
                None => eprint!("    -"),
            }
        }
        eprintln!();
    }

    eprintln!();

    // Aggregates
    let n = results.len();
    let mean_compressed: f64 =
        results.iter().map(|r| r.redukt.compressed_bpb).sum::<f64>() / n as f64;
    let mean_ce: f64 =
        results.iter().map(|r| r.redukt.ce_bpb).sum::<f64>() / n as f64;
    let mean_throughput: f64 =
        results.iter().map(|r| r.redukt.bytes_per_sec).sum::<f64>() / n as f64;
    let mean_efficiency: f64 =
        results.iter().map(|r| r.redukt.coder_efficiency_pct).sum::<f64>() / n as f64;

    eprintln!("  Mean compressed BPB: {:.4}", mean_compressed);
    eprintln!(
        "  Mean cross-ent BPB: {:.4} (coder efficiency: {:.1}%)",
        mean_ce, mean_efficiency
    );
    eprintln!("  Mean throughput:     {:.1} B/s", mean_throughput);

    if has_brotli {
        let (wins, mean) = competitor_stats(results, |r| &r.brotli11);
        let avail = results.iter().filter(|r| r.brotli11.is_some()).count();
        eprintln!(
            "  Wins vs brotli-11:   {}/{} (mean brotli: {:.4})",
            wins, avail, mean
        );
    }
    if has_gzip {
        let (wins, mean) = competitor_stats(results, |r| &r.gzip9);
        let avail = results.iter().filter(|r| r.gzip9.is_some()).count();
        eprintln!(
            "  Wins vs gzip-9:      {}/{} (mean gzip: {:.4})",
            wins, avail, mean
        );
    }
    if has_zstd {
        let (wins, mean) = competitor_stats(results, |r| &r.zstd19);
        let avail = results.iter().filter(|r| r.zstd19.is_some()).count();
        eprintln!(
            "  Wins vs zstd-19:     {}/{} (mean zstd: {:.4})",
            wins, avail, mean
        );
    }

    if verbose {
        eprintln!();
        eprintln!("  Per-file detail:");
        for r in results {
            eprintln!(
                "    {:<24} ce={:.4} comp={:.4} eff={:.1}% | {:.1} tok/s | {:.1} B/s",
                truncate(&r.name, 24),
                r.redukt.ce_bpb,
                r.redukt.compressed_bpb,
                r.redukt.coder_efficiency_pct,
                r.redukt.tok_per_sec,
                r.redukt.bytes_per_sec
            );
        }
    }

    eprintln!();
    eprintln!("================================================================");
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escape_handles_special_chars() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape("a\"b"), "a\\\"b");
        assert_eq!(json_escape("a\\b"), "a\\\\b");
        assert_eq!(json_escape("a\nb"), "a\\nb");
        assert_eq!(json_escape("a\tb"), "a\\tb");
    }

    #[test]
    fn classify_known_domains() {
        assert_eq!(classify_domain("enwik8"), "text/encyclopedia");
        assert_eq!(classify_domain("apache-access.log"), "logs/apache");
        assert_eq!(
            classify_domain("facturae-batch.xml"),
            "e-invoicing/facturae"
        );
        assert_eq!(classify_domain("x-ray"), "medical/xray");
    }

    #[test]
    fn classify_unknown_by_extension() {
        assert_eq!(classify_domain("my-data.log"), "logs/other");
        assert_eq!(classify_domain("config.xml"), "markup/xml");
        assert_eq!(classify_domain("data.json"), "data/json");
        assert_eq!(classify_domain("random.bin"), "unknown");
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn truncate_long_string() {
        assert_eq!(truncate("abcdefghij", 7), "abcd...");
    }

    #[test]
    fn competitor_stats_empty() {
        let results: Vec<FileResult> = vec![];
        let (wins, mean) = competitor_stats(&results, |r| &r.gzip9);
        assert_eq!(wins, 0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn discover_files_missing_path() {
        let result = discover_files("/nonexistent/path/xyz", 1024);
        assert!(result.is_err());
    }

    #[test]
    fn compress_pipe_missing_command() {
        let data = b"test data for compression";
        let result = compress_pipe(data, "nonexistent_compressor_xyz", &["-9"]);
        assert!(result.is_none());
    }

    #[test]
    fn compress_pipe_gzip_works() {
        let data = b"hello world hello world hello world hello world";
        if !check_available("gzip") {
            eprintln!("gzip not available, skipping");
            return;
        }
        let result = compress_pipe(data, "gzip", &["-9"]);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.compressed_size < data.len());
        assert!(r.bpb < 8.0);
        assert!(r.bpb > 0.0);
    }
}
