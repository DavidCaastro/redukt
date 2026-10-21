mod domain;
mod infrastructure;
mod application;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "compress" => {
            application::codec::cmd_compress(&args[2..]);
        }
        "decompress" => {
            application::codec::cmd_decompress(&args[2..]);
        }
        "--help" | "-h" | "help" => {
            print_usage();
        }
        other => {
            eprintln!("error: unknown command '{}'", other);
            eprintln!();
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("redukt — Neural compression engine (RWKV-7 + Context Mixing + Arithmetic Coding)");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  redukt compress   --input <FILE> --output <FILE> [--weights <DIR>] [--bytes <N>]");
    eprintln!("  redukt decompress --input <FILE> --output <FILE> [--weights <DIR>]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --input <FILE>    Input file path");
    eprintln!("  --output <FILE>   Output file path");
    eprintln!("  --weights <DIR>   Weights directory or .gguf file (default: weights/rwkv7-0.1b)");
    eprintln!("  --bytes <N>       Max bytes to compress (0 = all)");
    eprintln!("  --lr <F>          Bias head learning rate (default: 0.30)");
    eprintln!("  --ngram-scale <F> N-gram logit bias scale (default: 0.5)");
    eprintln!("  --mix-eta <F>     Adaptive mixer learning rate (default: 0.01)");
}
