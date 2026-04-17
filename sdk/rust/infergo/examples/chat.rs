//! Interactive chat example using the infergo Rust SDK.
//!
//! Usage:
//!     INFERGO_LIB_DIR=/path/to/build cargo run --example chat -- /path/to/model.gguf
//!
//! Environment variables:
//!     INFERGO_LIB_DIR  — directory containing libinfer_api.so
//!
//! The model must be a GGUF file (e.g. llama3-8b-q4.gguf).

use std::env;
use std::io::{self, BufRead, Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [--gpu-layers N] [--ctx-size N]", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let n_gpu_layers = find_arg(&args, "--gpu-layers").unwrap_or(99);
    let ctx_size = find_arg(&args, "--ctx-size").unwrap_or(4096);

    eprintln!("Loading model: {}", model_path);
    let llm = match infergo::Llm::new(model_path, n_gpu_layers, ctx_size, 1, 512) {
        Ok(llm) => llm,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("Model loaded (vocab_size={})", llm.vocab_size());

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\n> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "quit" || prompt == "exit" {
            break;
        }

        // Tokenize the prompt
        let tokens = match llm.tokenize(prompt, true) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Tokenize error: {}", e);
                continue;
            }
        };

        // Stream tokens to stdout
        let callback: infergo::TokenCallback = Box::new(|_token, piece| {
            print!("{}", piece);
            io::stdout().flush().unwrap();
            true
        });

        match llm.generate(&tokens, 512, 0.7, 0.9, None, Some(callback)) {
            Ok(result) => {
                println!();
                eprintln!("[generated {} tokens]", result.n_tokens);
            }
            Err(e) => {
                eprintln!("\nGeneration error: {}", e);
            }
        }
    }
}

fn find_arg(args: &[String], flag: &str) -> Option<i32> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
