mod domain;
mod infrastructure;
mod application;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "compress" => application::codec::cmd_compress(&args[2..]),
        "decompress" => application::codec::cmd_decompress(&args[2..]),
        "hybrid-eval" => application::codec::cmd_hybrid_eval(&args[2..]),
        "baseline" => application::codec::cmd_baseline(&args[2..]),
        "cm-eval" => application::codec::cmd_cm_eval(&args[2..]),
        "benchmark" => application::benchmark::cmd_benchmark(&args[2..]),
        "api" => { application::api::cmd_api(&args[2..]); Ok(()) }
        "--help" | "-h" | "help" => { print_usage(); Ok(()) }
        other => {
            eprintln!("error: unknown command '{}'", other);
            eprintln!();
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!("redukt — Neural compression engine (RWKV-7 + Context Mixing + Arithmetic Coding)");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  compress    Compress a file to AZTH format");
    eprintln!("  decompress  Decompress an AZTH file to original");
    eprintln!("  hybrid-eval Evaluate BPB with full ensemble (no output file)");
    eprintln!("  baseline    Evaluate BPB with RWKV-only (no ensemble)");
    eprintln!("  benchmark   Evaluate compression vs gzip/brotli/zstd (JSON report)");
    eprintln!("  cm-eval     Spectral analysis of weight matrices");
    eprintln!("  api         Start HTTP API server");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  redukt compress    --input <FILE> --output <FILE> [options]");
    eprintln!("  redukt compress    --input-dir <DIR> --output-dir <DIR> --jobs <N> [options]");
    eprintln!("  redukt decompress  --input <FILE> --output <FILE> [--weights <DIR>]");
    eprintln!("  redukt hybrid-eval --input <FILE> [options] [--log <FILE>]");
    eprintln!("  redukt baseline    --input <FILE> [--weights <DIR>] [--bytes <N>] [--log <FILE>]");
    eprintln!("  redukt benchmark   --input <FILE_OR_DIR> --output <JSON> [options]");
    eprintln!("  redukt cm-eval     --output <FILE> [--weights <DIR>]");
    eprintln!("  redukt api         [--host <ADDR>] [--port <PORT>] [--weights <DIR>]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --input <FILE>       Input file path");
    eprintln!("  --output <FILE>      Output file path");
    eprintln!("  --weights <DIR>      Weights directory or .gguf file (default: weights/rwkv7-0.1b)");
    eprintln!("  --bytes <N>          Max bytes to process (0 = all)");
    eprintln!("  --lr <F>             Bias head learning rate (default: 0.30)");
    eprintln!("  --ngram-scale <F>    N-gram logit bias scale (default: 0.5)");
    eprintln!("  --mix-eta <F>        Adaptive mixer learning rate (default: 0.01)");
    eprintln!("  --tau <F>            Surprise-modulated lr decay (0 = static, default: 0)");
    eprintln!("  --emb-surgery <M>    Embedding surgery method (analyze|norm|center|spread)");
    eprintln!("  --log <FILE>         Structured .jsonl log output");
    eprintln!("  --quant <LEVEL>      Force quantization: f32, q8, q4, auto (default: auto)");
    eprintln!("  --input-dir <DIR>    Batch mode: input directory (compress all files)");
    eprintln!("  --output-dir <DIR>   Batch mode: output directory (required with --input-dir)");
    eprintln!("  --jobs <N>           Batch mode: parallel worker threads (default: 1)");
    eprintln!("  --svd-rank <K>       SVD rank for TimeMix projections (0=off, e.g. 256/384)");
    eprintln!("  --verbose, -v        Show per-phase timing breakdown");
}
