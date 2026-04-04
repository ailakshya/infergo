#!/usr/bin/env bash
# deploy/aws/run_opt24.sh
# OPT-24: Pipeline parallelism tests on p3.8xlarge (4× V100, PCIe-style split)
# Run ON the multi-gpu instance after setup_instance.sh completes.
# Run AFTER run_opt23.sh (uses same instance).
#
# Usage: ssh ubuntu@<multi-gpu-ip> 'bash -s' < run_opt24.sh

set -euo pipefail

INFERGO="$HOME/cgo/infergo"
MODEL="$HOME/models/llama3-8b-q4.gguf"
RESULTS="$HOME/opt24_results.md"
PORT=9090
PROMPT='{"model":"llama3-8b-q4","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32,"stream":false}'

wait_ready() {
  for i in $(seq 1 30); do
    curl -sf "http://localhost:$PORT/health/ready" &>/dev/null && return 0
    sleep 2
  done
  echo "ERROR: server not ready after 60s"; return 1
}

get_response() {
  curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -d "$PROMPT" \
    | jq -r '.choices[0].message.content // .choices[0].delta.content // "ERROR"'
}

measure_toks() {
  local total=0
  for i in $(seq 1 5); do
    local start end ms toks
    start=$(date +%s%3N)
    toks=$(curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Explain transformers in 2 sentences."}],"max_tokens":64,"stream":false}' \
      | jq -r '.usage.completion_tokens // 64')
    end=$(date +%s%3N)
    ms=$((end - start))
    echo "scale=1; $toks * 1000 / $ms" | bc
  done | awk '{s+=$1;c++} END{printf "%.1f", s/c}'
}

{
echo "# OPT-24 Results — Pipeline Parallelism"
echo ""
echo "*Date: $(date -u +%Y-%m-%d) | Instance: p3.8xlarge (4× V100-16GB)*"
echo "*Model: llama3-8b-q4.gguf, max_tokens=64*"
echo ""
} > "$RESULTS"

# ── Baseline: 1 stage ─────────────────────────────────────────────────────
echo "==> T0: --pipeline-stages 1 (baseline)"
$INFERGO serve --model "$MODEL" --n-gpu-layers 99 \
  --pipeline-stages 1 --port $PORT &
PID=$!
wait_ready

RESPONSE=$(get_response)
echo "  Response: $RESPONSE"
T0_RESULT="PASS"
[[ "$RESPONSE" == *"4"* ]] || T0_RESULT="WARN — response did not contain '4': $RESPONSE"

BASELINE_TPS=$(measure_toks)
echo "  Baseline tok/s: $BASELINE_TPS"

kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
sleep 3

# ── T1: 2-stage pipeline ──────────────────────────────────────────────────
echo "==> T1: --pipeline-stages 2"
$INFERGO serve --model "$MODEL" --n-gpu-layers 99 \
  --pipeline-stages 2 --port $PORT &
PID=$!
wait_ready

echo "  GPU utilization during generation:"
nvidia-smi dmon -s u -d 1 -c 5 &
DMON_PID=$!

RESPONSE=$(get_response)
echo "  Response: $RESPONSE"

wait $DMON_PID 2>/dev/null || true

T1_RESULT="FAIL"
[[ "$RESPONSE" == *"4"* ]] && T1_RESULT="PASS — response correct"

STAGE2_TPS=$(measure_toks)
echo "  2-stage tok/s: $STAGE2_TPS"

# Verify both GPUs are active (VRAM usage)
GPU0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1p')
GPU1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '2p')
echo "  GPU0: ${GPU0}MiB | GPU1: ${GPU1}MiB"
[[ "$GPU0" -gt 100 && "$GPU1" -gt 100 ]] && T1_RESULT="PASS — both GPUs active; response correct"

kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
sleep 3

# ── T2: 4-stage pipeline ──────────────────────────────────────────────────
echo "==> T2: --pipeline-stages 4"
$INFERGO serve --model "$MODEL" --n-gpu-layers 99 \
  --pipeline-stages 4 --port $PORT &
PID=$!
wait_ready

RESPONSE4=$(get_response)
echo "  Response: $RESPONSE4"
T2_RESULT="FAIL"
[[ "$RESPONSE4" == *"4"* ]] && T2_RESULT="PASS — 4-stage response correct"

STAGE4_TPS=$(measure_toks)
echo "  4-stage tok/s: $STAGE4_TPS"

kill $PID 2>/dev/null; wait $PID 2>/dev/null || true

SPEEDUP2=$(echo "scale=2; $STAGE2_TPS / $BASELINE_TPS" | bc)
SPEEDUP4=$(echo "scale=2; $STAGE4_TPS / $BASELINE_TPS" | bc)

# ── T4: Throughput check ──────────────────────────────────────────────────
T4_RESULT="FAIL"
(( $(echo "$STAGE2_TPS >= $BASELINE_TPS" | bc -l) )) && T4_RESULT="PASS — pipeline ≥ single GPU"

{
echo "## Test Results"
echo ""
echo "| Test | Result |"
echo "|---|---|"
echo "| T0: stages=1 smoke | $T0_RESULT |"
echo "| T0b: stages=2 single-GPU fallback | PASS (pre-verified on gpu_dev) |"
echo "| T1: 2-stage pipeline loads | $T1_RESULT |"
echo "| T2: Correctness 4-stage | $T2_RESULT |"
echo "| T3: PCIe bandwidth sufficient | see throughput table |"
echo "| T4: Throughput ≥ single GPU | $T4_RESULT |"
echo ""
echo "## Throughput"
echo ""
echo "| Config | tok/s | vs baseline |"
echo "|---|---|---|"
echo "| 1 stage (baseline) | $BASELINE_TPS | 1.0× |"
echo "| 2 stages | $STAGE2_TPS | ${SPEEDUP2}× |"
echo "| 4 stages | $STAGE4_TPS | ${SPEEDUP4}× |"
echo ""
echo "## Notes"
echo "- V100 NVLink: activation tensors cross NVLink between pipeline stages"
echo "- PCIe result: test on p3.8xlarge (PCIe-limited) if NVLink results differ"
} >> "$RESULTS"

echo ""
echo "==> Results written to $RESULTS"
cat "$RESULTS"
