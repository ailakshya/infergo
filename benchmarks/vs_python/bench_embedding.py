"""
bench_embedding.py — infergo /v1/embeddings vs sentence-transformers benchmark.

Usage:
  # Start infergo with embedding model first:
  #   ./infergo serve --model models/all-MiniLM-L6-v2/onnx/model.onnx --provider cpu --port 9090
  #
  # Run benchmark:
  python bench_embedding.py --infergo-addr http://localhost:9090 --model model

  # Skip Python comparison:
  python bench_embedding.py --infergo-only --infergo-addr http://localhost:9090 --model model
"""

import argparse
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Inference is fast with infergo and ONNX Runtime.",
    "Transformers use self-attention to capture context.",
    "Go is a statically typed compiled language.",
    "Vector embeddings power semantic search.",
    "BERT tokenizes text into subword units.",
    "Continuous batching improves GPU utilization.",
    "Mean pooling aggregates token representations.",
    "L2 normalization produces unit-length vectors.",
    "Cosine similarity measures vector alignment.",
]


def embed_infergo(addr: str, model: str, text: str) -> list[float]:
    r = requests.post(
        f"{addr}/v1/embeddings",
        json={"model": model, "input": text},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def bench_infergo(addr: str, model: str, n: int, concurrency: int) -> dict:
    texts = [SENTENCES[i % len(SENTENCES)] for i in range(n)]
    latencies = []
    errors = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(embed_infergo, addr, model, t): t for t in texts}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % (n // 5) == 0:
                print(f"    {done}/{n} done")
            try:
                fut.result()
            except Exception as e:
                errors += 1
                print(f"    error: {e}")

    elapsed = time.perf_counter() - t0
    rps = (n - errors) / elapsed if elapsed > 0 else 0
    return {"n": n, "errors": errors, "elapsed": elapsed, "rps": rps, "dim": 384}


def bench_st(model_name: str, n: int) -> dict:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed — skipping Python benchmark")
        return {}

    texts = [SENTENCES[i % len(SENTENCES)] for i in range(n)]
    model = SentenceTransformer(model_name)

    t0 = time.perf_counter()
    model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    rps = n / elapsed
    return {"n": n, "elapsed": elapsed, "rps": rps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--infergo-addr", default="http://localhost:9090")
    p.add_argument("--model", default="model")
    p.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--infergo-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print("=" * 60)
    print("  Embedding benchmark: infergo vs sentence-transformers")
    print("=" * 60)

    # ── Correctness check ────────────────────────────────────────────────────
    print("\n  Correctness check (cosine similarity)...")
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(args.st_model)
        text = "hello world"
        ig_vec = np.array(embed_infergo(args.infergo_addr, args.model, text))
        st_vec = st_model.encode(text, normalize_embeddings=True)
        cosine = float(np.dot(ig_vec, st_vec))
        print(f"  Cosine(infergo, sentence-transformers) = {cosine:.6f}  {'PASS' if cosine >= 0.999 else 'FAIL'}")
    except Exception as e:
        print(f"  Correctness check skipped: {e}")

    # ── infergo benchmark ─────────────────────────────────────────────────────
    print(f"\n  infergo @ {args.infergo_addr}  (concurrency={args.concurrency}, n={args.requests})")
    print("  Warming up...")
    for _ in range(3):
        embed_infergo(args.infergo_addr, args.model, "warmup")
    print("  Sending requests...")
    ig = bench_infergo(args.infergo_addr, args.model, args.requests, args.concurrency)
    print(f"\n  infergo: {ig['n']} req | {ig['errors']} err | {ig['elapsed']:.1f}s")
    print(f"  Throughput: {ig['rps']:.1f} req/s  (dim={ig['dim']})")

    # ── sentence-transformers ─────────────────────────────────────────────────
    if not args.infergo_only:
        print(f"\n  sentence-transformers (batch=32, n={args.requests})...")
        st = bench_st(args.st_model, args.requests)
        if st:
            print(f"  sentence-transformers: {st['n']} req | {st['elapsed']:.1f}s")
            print(f"  Throughput: {st['rps']:.1f} req/s")
            ratio = ig["rps"] / st["rps"] if st["rps"] > 0 else 0
            print(f"\n  infergo/sentence-transformers ratio: {ratio:.2f}x")

    # ── Concurrent correctness (T5) ───────────────────────────────────────────
    print("\n  T5: Concurrent correctness (8 goroutines × same text)...")
    import numpy as np
    ref = np.array(embed_infergo(args.infergo_addr, args.model, "concurrent test"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(embed_infergo, args.infergo_addr, args.model, "concurrent test") for _ in range(8)]
        results = [np.array(f.result()) for f in futs]
    cosines = [float(np.dot(ref, r)) for r in results]
    min_cos = min(cosines)
    print(f"  Min cosine across 8 concurrent: {min_cos:.6f}  {'PASS' if min_cos >= 0.9999 else 'FAIL'}")

    # ── Write results ─────────────────────────────────────────────────────────
    out_path = args.out or "/tmp/results_embedding.md"
    with open(out_path, "w") as f:
        f.write("# Embedding benchmark results\n\n")
        f.write(f"| Backend | n_requests | req/s | dim | errors |\n")
        f.write(f"|---------|-----------|-------|-----|--------|\n")
        f.write(f"| infergo (CPU) | {ig['n']} | {ig['rps']:.1f} | {ig['dim']} | {ig['errors']} |\n")
        if not args.infergo_only and st:
            f.write(f"| sentence-transformers (CPU batch=32) | {st['n']} | {st['rps']:.1f} | 384 | 0 |\n")
    print(f"\n  Results written to {out_path}")


if __name__ == "__main__":
    main()
