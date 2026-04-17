//! Embedding + vector search example using the infergo Rust SDK.
//!
//! Usage:
//!     INFERGO_LIB_DIR=/path/to/build cargo run --example embed -- \
//!         --model /path/to/embedding.onnx \
//!         --tokenizer /path/to/tokenizer.json
//!
//! This example:
//! 1. Loads an ONNX embedding model
//! 2. Embeds several documents into a VectorDB
//! 3. Indexes the same documents in a BM25 index
//! 4. Runs vector search, BM25 search, and hybrid search

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_path = find_str_arg(&args, "--model")
        .unwrap_or_else(|| die("--model <path> is required"));
    let tokenizer_path = find_str_arg(&args, "--tokenizer")
        .unwrap_or_else(|| die("--tokenizer <path> is required"));
    let provider = find_str_arg(&args, "--provider").unwrap_or_else(|| "cpu".to_string());

    // --- Load embedding model ---
    eprintln!("Loading embedding model: {}", model_path);
    let embedder = match infergo::Embedding::new(&model_path, &tokenizer_path, &provider, 0, 1024)
    {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load embedding model: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("Embedding model loaded.");

    // --- Sample documents ---
    let documents = [
        "Rust is a systems programming language focused on safety and performance.",
        "Python is great for machine learning and data science workflows.",
        "Go excels at building concurrent network services and CLI tools.",
        "JavaScript powers the web with both frontend and backend capabilities.",
        "CUDA enables massively parallel computation on NVIDIA GPUs.",
    ];

    // --- Embed and insert into VectorDB ---
    eprintln!("Embedding {} documents...", documents.len());
    let dim = {
        let v = embedder.embed(documents[0]).expect("embed failed");
        v.len() as i32
    };
    eprintln!("Embedding dimension: {}", dim);

    let vdb = infergo::VectorDb::new(dim, 16, 200).expect("vectordb create failed");
    let bm25 = infergo::Bm25::new(1.2, 0.75).expect("bm25 create failed");

    for (i, doc) in documents.iter().enumerate() {
        let vec = embedder.embed(doc).expect("embed failed");
        vdb.insert(i as i64, &vec, Some(doc)).expect("insert failed");
        bm25.insert(i as i64, doc).expect("bm25 insert failed");
    }
    eprintln!(
        "Indexed {} vectors, {} BM25 documents.",
        vdb.size(),
        bm25.size()
    );

    // --- Vector search ---
    let query = "GPU parallel computing";
    eprintln!("\nQuery: \"{}\"", query);

    let query_vec = embedder.embed(query).expect("embed query failed");
    let vec_results = vdb.search(&query_vec, 3, 100, None).expect("search failed");

    println!("\n=== Vector Search Results ===");
    for r in &vec_results {
        println!("  id={} distance={:.4} -> {}", r.id, r.distance, documents[r.id as usize]);
    }

    // --- BM25 search ---
    let bm25_results = bm25.search(query, 3).expect("bm25 search failed");

    println!("\n=== BM25 Search Results ===");
    for r in &bm25_results {
        println!("  id={} score={:.4} -> {}", r.id, r.score, documents[r.id as usize]);
    }

    // --- Hybrid search ---
    let hybrid_results =
        infergo::hybrid_search(&vec_results, &bm25_results, 0.5, 3).expect("hybrid failed");

    println!("\n=== Hybrid Search Results (alpha=0.5) ===");
    for r in &hybrid_results {
        println!("  id={} score={:.4} -> {}", r.id, r.score, documents[r.id as usize]);
    }

    eprintln!("\nDone.");
}

fn find_str_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn die(msg: &str) -> ! {
    eprintln!("Error: {}", msg);
    eprintln!("Usage: embed --model <path.onnx> --tokenizer <path.json> [--provider cpu|cuda]");
    std::process::exit(1);
}
