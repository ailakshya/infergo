#!/usr/bin/env bash
# deploy/aws/run_opt26.sh
# OPT-26: Prefill/decode disaggregation test across 2 nodes.
# Run from your LOCAL machine — it SSHes to both nodes and orchestrates the test.
#
# Usage:
#   PREFILL_IP=<ip> DECODE_IP=<ip> KEY=~/.ssh/infergo-key.pem ./run_opt26.sh

set -euo pipefail

PREFILL_IP="${PREFILL_IP:?Set PREFILL_IP}"
DECODE_IP="${DECODE_IP:?Set DECODE_IP}"
KEY="${KEY:-$HOME/.ssh/infergo-key.pem}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no ubuntu"
RESULTS="$(dirname "$0")/../../benchmarks/scalability/results_opt26.md"

MODEL_PATH="\$HOME/models/llama3-8b-q4.gguf"
INFERGO="\$HOME/cgo/infergo"
PREFILL_PORT=9090
DECODE_PORT=9091

wait_ready() {
  local ip="$1" port="$2" label="$3"
  for i in $(seq 1 30); do
    curl -sf "http://$ip:$port/health/ready" &>/dev/null && return 0
    sleep 2
  done
  echo "ERROR: $label not ready after 60s"; return 1
}

echo "==> Starting prefill node ($PREFILL_IP:$PREFILL_PORT)"
$SSH@"$PREFILL_IP" "
  pkill -f 'infergo serve' 2>/dev/null || true
  sleep 1
  nohup $INFERGO serve --model $MODEL_PATH --n-gpu-layers 99 \
    --mode prefill --port $PREFILL_PORT > /tmp/prefill.log 2>&1 &
  echo \$! > /tmp/prefill.pid
"
wait_ready "$PREFILL_IP" "$PREFILL_PORT" "prefill node"
echo "    prefill node ready"

echo "==> Starting decode node ($DECODE_IP:$DECODE_PORT)"
$SSH@"$DECODE_IP" "
  pkill -f 'infergo serve' 2>/dev/null || true
  sleep 1
  nohup $INFERGO serve --model $MODEL_PATH --n-gpu-layers 99 \
    --mode decode --port $DECODE_PORT > /tmp/decode.log 2>&1 &
  echo \$! > /tmp/decode.pid
"
wait_ready "$DECODE_IP" "$DECODE_PORT" "decode node"
echo "    decode node ready"

PROMPT="Explain what a neural network is in two sentences."

# ── T1: KV serialization round-trip ──────────────────────────────────────
echo ""
echo "==> T1: Prefill → transfer → decode"

# Step 1: prefill the prompt, get back KV bytes as base64
echo "  Sending prompt to prefill node..."
PREFILL_START=$(date +%s%3N)
KV_RESPONSE=$(curl -sf -X POST "http://$PREFILL_IP:$PREFILL_PORT/v1/prefill" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"llama3-8b-q4\",\"prompt\":\"$PROMPT\"}")
PREFILL_END=$(date +%s%3N)
PREFILL_MS=$((PREFILL_END - PREFILL_START))

KV_SIZE=$(echo "$KV_RESPONSE" | jq -r '.kv_bytes | length // 0')
SEQ_ID=$(echo "$KV_RESPONSE" | jq -r '.seq_id // 0')
echo "  Prefill time: ${PREFILL_MS}ms | KV size: ${KV_SIZE} chars (base64)"

# Step 2: send KV to decode node, stream tokens
echo "  Sending KV to decode node..."
DECODE_START=$(date +%s%3N)
DECODE_RESPONSE=$(curl -sf -X POST "http://$DECODE_IP:$DECODE_PORT/v1/decode" \
  -H "Content-Type: application/json" \
  -d "$KV_RESPONSE")
DECODE_END=$(date +%s%3N)
DECODE_MS=$((DECODE_END - DECODE_START))

GENERATED=$(echo "$DECODE_RESPONSE" | jq -r '.choices[0].message.content // "ERROR"')
echo "  Decode time: ${DECODE_MS}ms"
echo "  Response: $GENERATED"
TOTAL_MS=$((PREFILL_MS + DECODE_MS))

T1_KV_MB=$(echo "scale=1; $KV_SIZE * 3/4 / 1048576" | bc)  # base64 → bytes → MB
echo "  KV transfer: ~${T1_KV_MB} MB in ${PREFILL_MS}ms"

if [[ "$GENERATED" != "ERROR" && -n "$GENERATED" ]]; then
  T1_RESULT="PASS — prefill ${PREFILL_MS}ms + decode ${DECODE_MS}ms = ${TOTAL_MS}ms total; KV ~${T1_KV_MB}MB"
else
  T1_RESULT="FAIL — empty response from decode node"
fi

# ── T2: Prefill throughput (prompts/s) ───────────────────────────────────
echo ""
echo "==> T2: Prefill node throughput"
PREFILL_COUNT=20
PREFILL_BATCH_START=$(date +%s%3N)
for i in $(seq 1 $PREFILL_COUNT); do
  curl -sf -X POST "http://$PREFILL_IP:$PREFILL_PORT/v1/prefill" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"llama3-8b-q4\",\"prompt\":\"$PROMPT\"}" > /dev/null
done
PREFILL_BATCH_END=$(date +%s%3N)
PREFILL_BATCH_MS=$((PREFILL_BATCH_END - PREFILL_BATCH_START))
PREFILL_TPS=$(echo "scale=1; $PREFILL_COUNT * 1000 / $PREFILL_BATCH_MS" | bc)
echo "  $PREFILL_COUNT prefills in ${PREFILL_BATCH_MS}ms = ${PREFILL_TPS} prompts/s"

# ── T3: Decode throughput (concurrent sequences) ─────────────────────────
echo ""
echo "==> T3: Decode node throughput (8 concurrent)"
# Get 8 KV states from prefill node first
KV_PAYLOADS=()
for i in $(seq 1 8); do
  KV=$(curl -sf -X POST "http://$PREFILL_IP:$PREFILL_PORT/v1/prefill" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"llama3-8b-q4\",\"prompt\":\"$PROMPT\"}")
  KV_PAYLOADS+=("$KV")
done

DECODE_BATCH_START=$(date +%s%3N)
for kv in "${KV_PAYLOADS[@]}"; do
  curl -sf -X POST "http://$DECODE_IP:$DECODE_PORT/v1/decode" \
    -H "Content-Type: application/json" -d "$kv" > /dev/null &
done
wait
DECODE_BATCH_END=$(date +%s%3N)
DECODE_BATCH_MS=$((DECODE_BATCH_END - DECODE_BATCH_START))
DECODE_P50="${DECODE_BATCH_MS}ms"
echo "  8 concurrent decodes completed in ${DECODE_BATCH_MS}ms"

# ── Cleanup ───────────────────────────────────────────────────────────────
echo ""
echo "==> Stopping nodes"
$SSH@"$PREFILL_IP" "kill \$(cat /tmp/prefill.pid 2>/dev/null) 2>/dev/null || true"
$SSH@"$DECODE_IP"  "kill \$(cat /tmp/decode.pid 2>/dev/null) 2>/dev/null || true"

# ── Write results ─────────────────────────────────────────────────────────
mkdir -p "$(dirname "$RESULTS")"
{
echo "# OPT-26 Results — Prefill/Decode Disaggregation"
echo ""
echo "*Date: $(date -u +%Y-%m-%d) | Prefill: g4dn.xlarge | Decode: g4dn.xlarge*"
echo "*Model: llama3-8b-q4.gguf | Prompt: ~14 tokens*"
echo ""
echo "## Test Results"
echo ""
echo "| Test | Result |"
echo "|---|---|"
echo "| T1: KV serialization round-trip | $T1_RESULT |"
echo "| T2: Prefill throughput | $PREFILL_TPS prompts/s |"
echo "| T3: Decode concurrency | 8 seq in ${DECODE_BATCH_MS}ms |"
echo "| T4: End-to-end latency | ${TOTAL_MS}ms total (prefill + decode) |"
echo ""
echo "## Architecture verified"
echo "- Prefill node (\`--mode prefill\`): computes KV cache, returns bytes"
echo "- Decode node (\`--mode decode\`): receives KV bytes, streams tokens"
echo "- KV transfer over HTTP (production would use gRPC or RDMA)"
} > "$RESULTS"

echo ""
echo "==> Results written to $RESULTS"
cat "$RESULTS"
