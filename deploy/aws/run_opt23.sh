#!/usr/bin/env bash
# deploy/aws/run_opt23.sh
# OPT-23: Tensor parallelism tests on p3.8xlarge (4× V100 NVLink)
# Run ON the multi-gpu instance after setup_instance.sh completes.
#
# Usage: ssh ubuntu@<multi-gpu-ip> 'bash -s' < run_opt23.sh

set -euo pipefail

INFERGO="$HOME/cgo/infergo"
MODEL="$HOME/models/llama3-8b-q4.gguf"
RESULTS="$HOME/opt23_results.md"
PORT=9090

check_model() {
  if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found at $MODEL"
    echo "Run: $INFERGO pull bartowski/Meta-Llama-3-8B-Instruct-GGUF --quant Q4_K_M --out $MODEL"
    exit 1
  fi
}

wait_ready() {
  local port="${1:-$PORT}"
  for i in $(seq 1 30); do
    curl -sf "http://localhost:$port/health/ready" &>/dev/null && return 0
    sleep 2
  done
  echo "ERROR: server not ready after 60s"
  return 1
}

measure_toks() {
  local label="$1" port="${2:-$PORT}"
  local total=0
  for i in $(seq 1 5); do
    local start end ms toks
    start=$(date +%s%3N)
    toks=$(curl -sf -X POST "http://localhost:$port/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Write 3 sentences about space."}],"max_tokens":64,"stream":false}' \
      | jq -r '.usage.completion_tokens // 64')
    end=$(date +%s%3N)
    ms=$((end - start))
    local tps=$(echo "scale=1; $toks * 1000 / $ms" | bc)
    echo "    run $i: $tps tok/s"
    total=$(echo "scale=1; $total + $tps" | bc)
  done
  echo "scale=1; $total / 5" | bc
}

# ── Verify GPU count ─────────────────────────────────────────────────────────
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "==> GPUs available: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

{
echo "# OPT-23 Results — Tensor Parallelism"
echo ""
echo "*Date: $(date -u +%Y-%m-%d) | Instance: p3.8xlarge (4× V100-16GB NVLink)*"
echo "*Model: llama3-8b-q4.gguf, max_tokens=64*"
echo ""
} > "$RESULTS"

# ── T4: Single GPU baseline (no --tensor-split) ───────────────────────────
echo ""
echo "==> T4: Baseline — single GPU"
check_model
$INFERGO serve --model "$MODEL" --n-gpu-layers 99 --port $PORT &
SERVER_PID=$!
wait_ready $PORT

echo "  measuring baseline tok/s..."
BASELINE_TPS=$(measure_toks "baseline" $PORT)
echo "  Baseline tok/s: $BASELINE_TPS"

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
sleep 3

# ── T1: Load across 2 GPUs ────────────────────────────────────────────────
echo ""
echo "==> T1: --tensor-split 0.5,0.5 (2 GPUs)"
$INFERGO serve --model "$MODEL" --n-gpu-layers 99 \
  --tensor-split 0.5,0.5 --port $PORT &
SERVER_PID=$!
wait_ready $PORT

echo "  VRAM usage per GPU:"
nvidia-smi --query-gpu=index,memory.used --format=csv

# OPT-23-T1 PASS if both GPUs show VRAM > 0
GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1p')
GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '2p')
echo "  GPU0: ${GPU0_MEM}MiB | GPU1: ${GPU1_MEM}MiB"

if [[ "$GPU0_MEM" -gt 100 && "$GPU1_MEM" -gt 100 ]]; then
  T1_RESULT="PASS — both GPUs show VRAM usage (${GPU0_MEM}MiB / ${GPU1_MEM}MiB)"
else
  T1_RESULT="FAIL — GPU1 VRAM=${GPU1_MEM}MiB (expected >100MiB)"
fi

# OPT-23-T3: Throughput with 2 GPUs
echo "  measuring 2-GPU tok/s..."
SPLIT2_TPS=$(measure_toks "2-gpu" $PORT)
echo "  2-GPU tok/s: $SPLIT2_TPS"
SPEEDUP=$(echo "scale=2; $SPLIT2_TPS / $BASELINE_TPS" | bc)
echo "  Speedup: ${SPEEDUP}× (target ≥ 1.6×)"

if (( $(echo "$SPEEDUP >= 1.6" | bc -l) )); then
  T3_RESULT="PASS — ${SPEEDUP}× speedup ≥ 1.6× target"
else
  T3_RESULT="FAIL — ${SPEEDUP}× speedup < 1.6× target"
fi

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
sleep 3

# ── T1 with 4 GPUs ────────────────────────────────────────────────────────
echo ""
echo "==> T1: --tensor-split 0.25,0.25,0.25,0.25 (4 GPUs)"
$INFERGO serve --model "$MODEL" --n-gpu-layers 99 \
  --tensor-split 0.25,0.25,0.25,0.25 --port $PORT &
SERVER_PID=$!
wait_ready $PORT

echo "  VRAM per GPU:"
nvidia-smi --query-gpu=index,memory.used --format=csv

SPLIT4_TPS=$(measure_toks "4-gpu" $PORT)
echo "  4-GPU tok/s: $SPLIT4_TPS"
SPEEDUP4=$(echo "scale=2; $SPLIT4_TPS / $BASELINE_TPS" | bc)

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true

# ── Write results ─────────────────────────────────────────────────────────
{
echo "## Test Results"
echo ""
echo "| Test | Result |"
echo "|---|---|"
echo "| T1: 2-GPU load | $T1_RESULT |"
echo "| T3: 2-GPU throughput | $T3_RESULT |"
echo "| T4: Single GPU fallback | PASS (pre-verified on gpu_dev) |"
echo "| T5: Concurrent requests | pending (run bench_scale.py) |"
echo ""
echo "## Throughput"
echo ""
echo "| Config | tok/s | Speedup |"
echo "|---|---|---|"
echo "| 1 GPU (baseline) | $BASELINE_TPS | 1.0× |"
echo "| 2 GPU (--tensor-split 0.5,0.5) | $SPLIT2_TPS | ${SPEEDUP}× |"
echo "| 4 GPU (--tensor-split 0.25,0.25,0.25,0.25) | $SPLIT4_TPS | ${SPEEDUP4}× |"
echo ""
echo "## Notes"
echo "- Instance: p3.8xlarge (V100 NVLink — full bandwidth between GPUs)"
echo "- Model: llama3-8b-q4.gguf (Q4_K_M, 4.6 GB)"
} >> "$RESULTS"

echo ""
echo "==> Results written to $RESULTS"
cat "$RESULTS"
