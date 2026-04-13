#!/usr/bin/env python3
"""Benchmark: TOON vs JSON vs Plain text structured output."""

import json
import time
import statistics
import requests

PORT = 9191
URL = f"http://localhost:{PORT}/v1/chat/completions"
N = 50  # requests per mode
MAX_TOKENS = 64

def bench_mode(mode_name, response_format=None, prompt="Hi", sys_prompt=None):
    """Run N requests with the given response_format and return latencies."""
    latencies = []
    tokens_list = []
    outputs = []

    for i in range(N):
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": "llm",
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }
        if response_format:
            body["response_format"] = response_format

        t0 = time.perf_counter()
        r = requests.post(URL, json=body, timeout=30)
        t1 = time.perf_counter()

        if r.status_code != 200:
            print(f"  [{mode_name}] req {i}: HTTP {r.status_code}")
            continue

        data = r.json()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        content = data["choices"][0]["message"]["content"]
        tok_count = data.get("usage", {}).get("completion_tokens", len(content.split()))
        tokens_list.append(tok_count)
        if i == 0:
            outputs.append(content[:200])

    if not latencies:
        return None

    avg_tokens = statistics.mean(tokens_list)
    result = {
        "mode": mode_name,
        "n": len(latencies),
        "avg_ms": round(statistics.mean(latencies), 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "std_ms": round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0,
        "avg_tokens": round(avg_tokens, 1),
        "tok_per_sec": round(avg_tokens / (statistics.mean(latencies) / 1000), 1),
        "ms_per_tok": round(statistics.mean(latencies) / avg_tokens, 1),
        "rps": round(len(latencies) / (sum(latencies) / 1000), 2),
        "sample": outputs[0] if outputs else "",
    }
    return result

def main():
    print(f"=== TOON vs JSON vs Plain — {N} requests each ===\n")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        requests.post(URL, json={
            "model": "llm",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
        }, timeout=10)

    # 1. Plain text (no grammar)
    prompt_plain = "Return a person with name, age, city, occupation, hobbies (list of 3)."
    print("\n[1/3] Plain text (no grammar constraint)...")
    plain = bench_mode("plain", prompt=prompt_plain)

    # 2. JSON mode
    prompt_json = "Return a JSON object with fields: name, age, city, occupation, hobbies (array of 3 strings)."
    print("[2/3] JSON mode (JSON GBNF grammar)...")
    json_mode = bench_mode("json", {"type": "json_object"}, prompt=prompt_json,
                           sys_prompt="You are a helpful assistant that outputs valid JSON only.")

    # 3. TOON mode
    prompt_toon = "Return person data with fields: name, age, city, occupation, hobbies (list of 3)."
    print("[3/3] TOON mode (TOON GBNF grammar)...")
    toon_mode = bench_mode("toon", {"type": "toon"}, prompt=prompt_toon,
                           sys_prompt="Output TOON format only. TOON uses key:value pairs separated by |. Nested objects use (). Arrays use []. Example: name:Alice|age:25|hobbies:[reading,coding]|addr:(city:NYC|zip:10001)")

    # Results
    print("\n" + "=" * 80)
    print(f"{'Mode':<12} {'Avg ms':>8} {'P50 ms':>8} {'P99 ms':>8} {'Tokens':>8} {'ms/tok':>8} {'tok/s':>8} {'RPS':>6}")
    print("-" * 88)
    for r in [plain, json_mode, toon_mode]:
        if r:
            print(f"{r['mode']:<12} {r['avg_ms']:>8.1f} {r['p50_ms']:>8.1f} {r['p99_ms']:>8.1f} {r['avg_tokens']:>8.1f} {r['ms_per_tok']:>8.1f} {r['tok_per_sec']:>8.1f} {r['rps']:>6.2f}")

    if json_mode and toon_mode:
        speedup = json_mode["avg_ms"] / toon_mode["avg_ms"]
        token_saving = (1 - toon_mode["avg_tokens"] / json_mode["avg_tokens"]) * 100
        print(f"\nTOON vs JSON:")
        print(f"  Latency:  {json_mode['avg_ms']:.0f}ms → {toon_mode['avg_ms']:.0f}ms ({speedup:.2f}x faster)")
        print(f"  Tokens:   {json_mode['avg_tokens']:.0f} → {toon_mode['avg_tokens']:.0f} ({token_saving:.0f}% fewer)")
        print(f"  tok/s:    {json_mode['tok_per_sec']:.0f} → {toon_mode['tok_per_sec']:.0f}")

    # Sample outputs
    print("\n--- Sample outputs ---")
    for r in [plain, json_mode, toon_mode]:
        if r:
            print(f"\n[{r['mode']}]:")
            print(f"  {r['sample']}")

    # Save results
    results = {"plain": plain, "json": json_mode, "toon": toon_mode}
    with open("benchmarks/toon_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmarks/toon_results.json")

if __name__ == "__main__":
    main()
