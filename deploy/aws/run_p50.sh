#!/usr/bin/env bash
# deploy/aws/run_p50.sh
# P50 latency target test on p3.2xlarge (1× V100-32GB).
# Target: P50 ≤ 600ms at concurrency=4.
# Run ON the p50 instance after setup_instance.sh completes.

set -euo pipefail

INFERGO="$HOME/cgo/infergo"
MODEL="$HOME/models/llama3-8b-q4.gguf"
RESULTS="$HOME/p50_results.md"
PORT=9090
CONCURRENT=4
REQUESTS=40

wait_ready() {
  for i in $(seq 1 30); do
    curl -sf "http://localhost:$PORT/health/ready" &>/dev/null && return 0
    sleep 2
  done
  echo "ERROR: server not ready after 60s"; return 1
}

run_single() {
  local start end ms
  start=$(date +%s%3N)
  curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Explain what a transformer neural network is in two sentences."}],"max_tokens":64,"stream":false}' \
    > /dev/null
  end=$(date +%s%3N)
  echo $((end - start))
}

{
echo "# P50 Latency Test — p3.2xlarge (V100-32GB)"
echo ""
echo "*Date: $(date -u +%Y-%m-%d) | Instance: p3.2xlarge | Model: llama3-8b-q4.gguf*"
echo "*Concurrency: $CONCURRENT | Requests: $REQUESTS | max_tokens: 64*"
echo ""
} > "$RESULTS"

echo "==> Starting infergo on p3.2xlarge (V100)"
$INFERGO serve --model "$MODEL" --n-gpu-layers 99 --port $PORT &
SERVER_PID=$!
wait_ready

# ── Warmup ────────────────────────────────────────────────────────────────
echo "==> Warmup (5 requests)"
for i in $(seq 1 5); do run_single > /dev/null; done

# ── Concurrency=1 baseline ────────────────────────────────────────────────
echo "==> c=1 ($(($REQUESTS/4)) requests)"
LATENCIES_C1=()
for i in $(seq 1 $(($REQUESTS/4))); do
  LATENCIES_C1+=($(run_single))
done
P50_C1=$(printf '%s\n' "${LATENCIES_C1[@]}" | sort -n | awk "NR==int(${#LATENCIES_C1[@]}*0.5)+1")
P99_C1=$(printf '%s\n' "${LATENCIES_C1[@]}" | sort -n | awk "NR==int(${#LATENCIES_C1[@]}*0.99)+1")
echo "  c=1: P50=${P50_C1}ms P99=${P99_C1}ms"

# ── Concurrency=4 ─────────────────────────────────────────────────────────
echo "==> c=4 ($REQUESTS requests, $CONCURRENT concurrent)"
TMPDIR=$(mktemp -d)

send_batch() {
  for i in $(seq 1 $(($REQUESTS / $CONCURRENT))); do
    run_single >> "$TMPDIR/lat_$1.txt"
  done
}

for c in $(seq 1 $CONCURRENT); do
  send_batch $c &
done
wait

cat "$TMPDIR"/lat_*.txt > "$TMPDIR/all.txt"
TOTAL=$(wc -l < "$TMPDIR/all.txt")
P50_C4=$(sort -n "$TMPDIR/all.txt" | awk "NR==int($TOTAL*0.5)+1")
P99_C4=$(sort -n "$TMPDIR/all.txt" | awk "NR==int($TOTAL*0.99)+1")
echo "  c=4: P50=${P50_C4}ms P99=${P99_C4}ms ($TOTAL samples)"
rm -rf "$TMPDIR"

# ── Target check ──────────────────────────────────────────────────────────
TARGET=600
if [[ $P50_C4 -le $TARGET ]]; then
  TARGET_RESULT="PASS — P50=${P50_C4}ms ≤ ${TARGET}ms target"
else
  TARGET_RESULT="FAIL — P50=${P50_C4}ms > ${TARGET}ms target (${P50_C4}ms on V100 vs 1059ms on RTX 5070 Ti)"
fi

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true

{
echo "## Results"
echo ""
echo "| Concurrency | P50 ms | P99 ms | Target |"
echo "|---|---|---|---|"
echo "| 1 | $P50_C1 | $P99_C1 | — |"
echo "| 4 | $P50_C4 | $P99_C4 | ≤ 600ms |"
echo ""
echo "**P50 c=4 target: $TARGET_RESULT**"
echo ""
echo "## Hardware comparison"
echo ""
echo "| GPU | P50 c=4 |"
echo "|---|---|"
echo "| RTX 5070 Ti 16GB (gpu_dev) | 1059ms |"
echo "| V100 32GB (p3.2xlarge) | ${P50_C4}ms |"
} >> "$RESULTS"

echo ""
echo "==> Results written to $RESULTS"
cat "$RESULTS"
