#!/usr/bin/env python3
"""Head-to-head: infergo vs LM Studio — can we beat it?"""

import json, os, subprocess, time, statistics, requests, concurrent.futures

MODEL = "/tmp/qwen2.5-coder-1.5b-q4.gguf"
N = 30
MAX_TOKENS = 64
PROMPT = "Return a JSON object with fields: name, age, city, occupation, hobbies (array of 3)."
SYS = "You are a helpful assistant. Output valid JSON only."

INFERGO = os.path.expanduser("~/cgo/infergo")
LMS = os.path.expanduser("~/.lmstudio/bin/lms")
LD = ":".join([os.path.expanduser(p) for p in [
    "~/cgo/build/cpp/api", "~/cgo/build/cpp/onnx", "~/cgo/build/cpp/tokenizer",
    "~/yolo-env/lib/python3.12/site-packages/torch/lib"
]] + ["/usr/local/cuda/lib64"])


def kill_all():
    for p in ["port 9191", "llmster"]:
        subprocess.run(f"pkill -9 -f '{p}'", shell=True, capture_output=True)
    time.sleep(3)


def wait(url, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except: pass
        time.sleep(1)
    return False


def do_request(url, model_id):
    body = {"model": model_id, "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": PROMPT}
    ], "max_tokens": MAX_TOKENS, "temperature": 0.7}
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=60)
    t1 = time.perf_counter()
    if r.status_code != 200: return None, 0, ""
    d = r.json()
    return (t1-t0)*1000, d.get("usage",{}).get("completion_tokens",0), d["choices"][0]["message"]["content"]


def bench(name, url, model_id, c=1):
    # Warmup
    for _ in range(3):
        try: do_request(url, model_id)
        except: pass

    lats, toks, errs, sample = [], [], 0, ""
    t0 = time.perf_counter()

    if c == 1:
        for _ in range(N):
            ms, tok, txt = do_request(url, model_id)
            if ms is None: errs += 1; continue
            lats.append(ms); toks.append(tok)
            if not sample: sample = txt[:120]
    else:
        with concurrent.futures.ThreadPoolExecutor(c) as pool:
            futs = [pool.submit(do_request, url, model_id) for _ in range(N)]
            for f in concurrent.futures.as_completed(futs):
                ms, tok, txt = f.result()
                if ms is None: errs += 1; continue
                lats.append(ms); toks.append(tok)
                if not sample: sample = txt[:120]

    wall = time.perf_counter() - t0
    if not lats: return None

    avg_ms = statistics.mean(lats)
    avg_tok = statistics.mean(toks)
    return {
        "name": f"{name} c={c}", "avg": round(avg_ms,1),
        "p50": round(statistics.median(lats),1),
        "min": round(min(lats),1), "max": round(max(lats),1),
        "tok": round(avg_tok,1), "mpt": round(avg_ms/avg_tok,1) if avg_tok else 0,
        "rps": round(len(lats)/wall,2), "tps": round(sum(toks)/wall,1),
        "err": errs, "n": len(lats), "sample": sample
    }


def main():
    print("=" * 90)
    print("HEAD-TO-HEAD: infergo vs LM Studio")
    print(f"Model: Qwen 1.5B Q4 | N={N} | max_tokens={MAX_TOKENS}")
    print("=" * 90)

    results = []

    for c in [1, 4, 8]:
        # infergo
        kill_all()
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = LD + ":" + env.get("LD_LIBRARY_PATH", "")
        p = subprocess.Popen([INFERGO, "serve", "--model", f"llm:{MODEL}",
            "--provider", "cuda", "--port", "9191", "--grpc-port", "0",
            "--max-seqs", "8", "--ctx-size", "8192"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if wait("http://localhost:9191/health/live"):
            r = bench("infergo", "http://localhost:9191", "llm", c)
            if r: results.append(r)
        p.kill(); p.wait()

        # LM Studio
        kill_all()
        subprocess.run([LMS, "daemon", "up"], capture_output=True, timeout=30)
        time.sleep(3)
        subprocess.run([LMS, "server", "start", "--port", "1234"], capture_output=True, timeout=15)
        time.sleep(2)
        subprocess.run([LMS, "load", "qwen2.5-coder-1.5b", "--gpu", "max", "-y"], capture_output=True, timeout=60)
        time.sleep(3)
        if wait("http://localhost:1234/v1/models", 30):
            r = bench("lm-studio", "http://localhost:1234", "qwen2.5-coder-1.5b", c)
            if r: results.append(r)
        subprocess.run([LMS, "unload", "--all", "-y"], capture_output=True)
        subprocess.run([LMS, "server", "stop"], capture_output=True)

    # Print
    print(f"\n{'Engine':<20} {'Avg':>7} {'P50':>7} {'Tok':>5} {'ms/t':>6} {'RPS':>6} {'Tot t/s':>8} {'Err':>4}")
    print("-" * 75)
    for r in results:
        winner = ""
        # Find matching opponent
        for r2 in results:
            if r2["name"].split(" c=")[1] == r["name"].split(" c=")[1] and r2["name"] != r["name"]:
                if r["avg"] < r2["avg"]: winner = " ***"
        print(f"{r['name']:<20} {r['avg']:>7.1f} {r['p50']:>7.1f} {r['tok']:>5.1f} {r['mpt']:>6.1f} {r['rps']:>6.2f} {r['tps']:>8.1f} {r['err']:>4}{winner}")

    # Save
    with open("benchmarks/head2head_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to benchmarks/head2head_results.json")


if __name__ == "__main__":
    main()
