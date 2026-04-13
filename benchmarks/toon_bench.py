#!/usr/bin/env python3
"""Benchmark: TOON vs JSON (strict & lazy) vs Plain text structured output."""

import json
import time
import statistics
import requests

PORT = 9191
URL = f"http://localhost:{PORT}/v1/chat/completions"
N = 30  # requests per mode
MAX_TOKENS = 32

# JSON GBNF grammar (same as server's built-in, used for strict mode)
JSON_GRAMMAR = r"""root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^\\"\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (([eE] [-+]? [0-9]+))? ws

ws ::= ([ \t\n] ws)?
"""


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
        r = requests.post(URL, json=body, timeout=60)
        t1 = time.perf_counter()

        if r.status_code != 200:
            print(f"  [{mode_name}] req {i}: HTTP {r.status_code}")
            time.sleep(0.5)  # let GC reclaim KV slots
            continue
        time.sleep(0.05)  # small delay between requests for KV cleanup

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
    print(f"=== TOON vs JSON (strict & lazy) vs Plain — {N} requests each ===\n")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        requests.post(URL, json={
            "model": "llm",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
        }, timeout=10)

    prompt_structured = "Return a person object with fields: name, age, city, occupation, hobbies (list of 3)."
    sys_json = "You are a helpful assistant that outputs valid JSON only. No markdown, no explanation."
    sys_toon = "Output TOON format only. TOON uses key:value pairs separated by |. Nested objects use (). Arrays use []. Example: name:Alice|age:25|hobbies:[reading,coding]|addr:(city:NYC|zip:10001). No other text."

    # 1. Plain text (no grammar)
    print("\n[1/4] Plain text (no grammar constraint)...")
    plain = bench_mode("plain", prompt=prompt_structured)

    # 2. JSON lazy (built-in json_object mode — lazy grammar triggers on { or [)
    print("[2/4] JSON lazy (triggers on {{ or [, preamble free)...")
    json_lazy = bench_mode("json_lazy", {"type": "json_object"},
                           prompt=prompt_structured, sys_prompt=sys_json)

    # 3. JSON strict (same grammar, strict enforcement from token 1)
    # Uses "grammar" type which bypasses lazy detection
    print("[3/4] JSON strict (grammar enforced from token 1)...")
    json_strict = bench_mode("json_strict", {"type": "grammar", "grammar": JSON_GRAMMAR},
                             prompt=prompt_structured, sys_prompt=sys_json)

    # 4. TOON strict (always strict)
    print("[4/4] TOON strict (grammar enforced from token 1)...")
    toon = bench_mode("toon", {"type": "toon"},
                      prompt=prompt_structured, sys_prompt=sys_toon)

    # Results
    modes = [plain, json_lazy, json_strict, toon]
    print("\n" + "=" * 96)
    print(f"{'Mode':<14} {'Avg ms':>8} {'P50 ms':>8} {'P99 ms':>8} {'Tokens':>8} {'ms/tok':>8} {'tok/s':>8} {'RPS':>6}")
    print("-" * 96)
    for r in modes:
        if r:
            print(f"{r['mode']:<14} {r['avg_ms']:>8.1f} {r['p50_ms']:>8.1f} {r['p99_ms']:>8.1f} {r['avg_tokens']:>8.1f} {r['ms_per_tok']:>8.1f} {r['tok_per_sec']:>8.1f} {r['rps']:>6.2f}")

    # Fair comparison: JSON strict vs TOON strict (both enforced from token 1)
    if json_strict and toon:
        speedup = json_strict["avg_ms"] / toon["avg_ms"]
        tok_speedup = json_strict["ms_per_tok"] / toon["ms_per_tok"]
        token_saving = (1 - toon["avg_tokens"] / json_strict["avg_tokens"]) * 100
        print(f"\n=== Fair comparison: JSON strict vs TOON strict ===")
        print(f"  Latency:    {json_strict['avg_ms']:.0f}ms → {toon['avg_ms']:.0f}ms ({speedup:.2f}x)")
        print(f"  ms/tok:     {json_strict['ms_per_tok']:.1f} → {toon['ms_per_tok']:.1f} ({tok_speedup:.2f}x faster grammar)")
        print(f"  Tokens:     {json_strict['avg_tokens']:.0f} → {toon['avg_tokens']:.0f} ({token_saving:.0f}% fewer)")
        print(f"  Throughput: {json_strict['tok_per_sec']:.0f} → {toon['tok_per_sec']:.0f} tok/s")

    # Sample outputs
    print("\n--- Sample outputs ---")
    for r in modes:
        if r:
            print(f"\n[{r['mode']}]:")
            print(f"  {r['sample']}")

    # Save results
    results = {r["mode"]: r for r in modes if r}
    with open("benchmarks/toon_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmarks/toon_results.json")


if __name__ == "__main__":
    main()
