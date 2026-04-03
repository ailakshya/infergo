# P50 Latency vs Concurrency Sweep — infergo CUDA

*Date: 2026-04-03*  
*Model: llama3-8b-q4.gguf, GPU: RTX 5070 Ti, max_tokens=128*

## Raw results

| Concurrency | req/s | tok/s | Mean ms | P50 ms | P95 ms | P99 ms | Errors |
|---|---|---|---|---|---|---|---|
| 1 | 1.2 | 144 | 807 | 884 | 896 | 903 | 0 |
| 2 | 1.2 | 138 | 1670 | 1895 | 1974 | 1985 | 0 |
| 4 | 1.6 | 187 | 2299 | 2579 | 2610 | 2611 | 0 |

## Analysis: is the "collectively" argument correct?

At concurrency=1 the server handles one request at a time, so P50=884ms = true single-request latency.

If 4 users arrive simultaneously at a **sequential** server (one at a time, no batching):
- User 1: waits 884ms
- User 2: queues → waits 1768ms
- User 3: queues → waits 2652ms
- User 4: queues → waits 3536ms
- **Average user wait: 2210ms**

With **continuous batching** (concurrency=4, real data):
- All 4 users processed together → each waits 2579ms
- **Average user wait: 2579ms**

## Verdict

| Metric | Sequential (simulated c=4) | Continuous batching (c=4) | Delta |
|---|---|---|---|
| User 1 (best case) | 884ms | 2579ms | **+192% worse** |
| User 4 (worst case) | 3536ms | 2579ms | **-27% better** |
| Average user wait | 2210ms | 2579ms | +17% worse |
| Total throughput | 144 tok/s | 187 tok/s | **+30% better** |
| Wall-clock for 20 req | 16.15s | 12.43s | **-23% faster** |

**The "collectively users aren't worse off" argument is partially wrong:**
- Tail users (the ones who would have waited 3-4× at a sequential server) **do** benefit from batching.
- Head users (the ones who arrived first) are penalized — they wait for the batch to fill.
- Average user P50 is ~17% higher with batching.
- However, overall throughput is +30% and total wall-clock time is -23% — the system finishes the work faster.

## Correct framing

Continuous batching is the right choice for **throughput-optimized serving** (maximize tokens/s, requests/s). It is not optimal for **latency-optimized serving** (minimize P50 for each individual user). 

The real fix for P50 is OPT-22 (PagedAttention) — which reduces per-step generation time, bringing down the absolute cost of each batched step, so all users benefit from batching without the latency penalty.
