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
        "mamba-eval" => application::codec::cmd_mamba_eval(&args[2..]),
        "api" => application::api::cmd_api(&args[2..]),
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
    eprintln!("  mamba-eval  Evaluate BPB with Mamba SSM backbone (no ensemble)");
    eprintln!("  api         Start HTTP API server");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  redukt compress    --input <FILE> --output <FILE> [options]");
    eprintln!("  redukt compress    --input-dir <DIR> --output-dir <DIR> --jobs <N> [options]");
    eprintln!("  redukt decompress  --input <FILE> --output <FILE> [--weights <DIR>]");
    eprintln!("  redukt decompress  --input-dir <DIR> --output-dir <DIR> [--jobs <N>]");
    eprintln!("  redukt hybrid-eval --input <FILE> [options] [--log <FILE>]");
    eprintln!("  redukt baseline    --input <FILE> [--weights <DIR>] [--bytes <N>] [--log <FILE>]");
    eprintln!("  redukt benchmark   --input <FILE_OR_DIR> --output <JSON> [options]");
    eprintln!("  redukt cm-eval     --output <FILE> [--weights <DIR>]");
    eprintln!("  redukt api         [--bind <ADDR>] [--port <PORT>] [--weights <DIR>] [--backbone <B>] [--quant <L>]");
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
    eprintln!("  --quant <LEVEL>      RWKV: f32, q8, q4, auto (default: auto = q8, a fixed choice). The RWKV level");
    eprintln!("                       is not stored in the archive: decompress needs the same --quant unless it");
    eprintln!("                       was the default.");
    eprintln!("                       Mamba: auto|q8 (default: weight-only Q8, AZTH v5) or f32 (AZTH v4).");
    eprintln!("                       f32 gives ~1% better BPB and is the recommended profile for batches of");
    eprintln!("                       >=16 streams (--input-dir), where aggregate throughput is equal; q8 is ~3.3x");
    eprintln!("                       faster for single files (+0.6..1.2% BPB). decompress ignores this flag:");
    eprintln!("                       the profile always comes from the archive header (a conflicting value fails).");
    eprintln!("  --backbone <B>       rwkv, mamba, mambabyte, plugin, auto (default: auto = route by content);");
    eprintln!("                       any other value is an error (also for --quant, --numeric, --e8e9, --shuffle,");
    eprintln!("                       --delta)");
    eprintln!("                       api: also a comma list (e.g. mamba,rwkv) to load several models at start-up");
    eprintln!("  --bind <ADDR>        api: listen address (default: 0.0.0.0; --host is an alias)");
    eprintln!("  --port <PORT>        api: listen port (default: 8080)");
    eprintln!("  --streaming          Decompress: write output progressively (lower memory)");
    eprintln!("  --e8e9 <MODE>        E8E9 x86 preprocessing: auto (default), on, off");
    eprintln!("  --shuffle <MODE>     Byte-plane shuffle: auto (default), on, on:N, off");
    eprintln!("  --delta <MODE>       Delta coding: auto (default), on, on:N, off");
    eprintln!("  --no-governor        Disable hardware governor (battery/RAM monitoring)");
    eprintln!("  --input-dir <DIR>    Batch mode: compress every file / decompress every .azth in DIR");
    eprintln!("  --output-dir <DIR>   Batch mode: output directory (required with --input-dir)");
    eprintln!("  --jobs <N>           Batch mode: files advanced in lockstep per step (default: 8).");
    eprintln!("                       Streams share each weight fetch (GEMV->GEMM); output per file is");
    eprintln!("                       byte-identical to single-file mode. Memory ~6 MB per stream (Mamba).");
    eprintln!("                       Worker threads are set by REDUKT_THREADS, not by --jobs.");
    eprintln!("  --reset-every <N>    Compress (Mamba family): zero the backbone state every N tokens (R-108;");
    eprintln!("                       0 = never, else a power of two 256..1048576; default 4096). Stored in the");
    eprintln!("                       header, so decompress always follows the archive. Mamba-130M degenerates");
    eprintln!("                       beyond ~16K tokens of context; the reset keeps whole-file BPB near 10 KB.");
    eprintln!("  --verbose, -v        Show per-phase timing breakdown");
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  REDUKT_THREADS=N     Mamba GEMV/SSM worker threads incl. caller (default: all logical CPUs)");
    eprintln!("  REDUKT_SPIN_US=N     Worker spin before parking, microseconds (default: 100)");
    eprintln!("  REDUKT_NO_SIMD=1     Force scalar kernels (slower; same bytes as the AVX2 kernels since R-104 F2)");
}
