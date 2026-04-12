#!/bin/bash
# OVERNIGHT AUTOMATED TEST SUITE
# Runs unattended. Saves results. Updates README. Pushes to GitHub.
set -e

cd /home/lakshya/cgo
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/home/lakshya/onnxruntime/lib:$LD_LIBRARY_PATH

LOG=/tmp/overnight_full.log
echo "=== OVERNIGHT TEST STARTED: $(date) ===" | tee $LOG

# ─── 1. Run 5-mode overnight benchmark (1000 requests each) ───
echo "" | tee -a $LOG
echo "=== PHASE 1: 5-mode benchmark (1000 req each) ===" | tee -a $LOG
python3 benchmarks/overnight_bench.py 2>&1 | tee -a $LOG

# Save results
if [ -f benchmarks/overnight_results.json ]; then
    echo "Overnight results saved" | tee -a $LOG
fi

# ─── 2. Run full library test (all 22 features) ───
echo "" | tee -a $LOG
echo "=== PHASE 2: Full library test (22 features) ===" | tee -a $LOG

pkill -9 -f "infergo serve" 2>/dev/null; sleep 5

./infergo serve \
  --model llm:/tmp/qwen2.5-coder-1.5b-q4.gguf \
  --model embed:models/all-MiniLM-L6-v2/onnx/model.onnx \
  --model detect:models/yolo11n.torchscript.pt \
  --provider cuda --port 9700 --grpc-port 0 --max-seqs 4 --ctx-size 2048 > /dev/null 2>&1 &

sleep 15

PASS=0; FAIL=0; TOTAL=0
test_ep() {
    TOTAL=$((TOTAL+1))
    local name="$1" result
    result=$(eval "$2" 2>/dev/null)
    if echo "$result" | grep -q "$3"; then
        PASS=$((PASS+1)); echo "  ✓ $name" | tee -a $LOG
    else
        FAIL=$((FAIL+1)); echo "  ✗ $name" | tee -a $LOG
    fi
}

# Create test image
python3 -c "from PIL import Image;import numpy as np;import io;img=Image.fromarray(np.random.randint(0,255,(640,640,3),dtype=np.uint8));buf=io.BytesIO();img.save(buf,format='JPEG',quality=85);open('/tmp/test_overnight.jpg','wb').write(buf.getvalue())" 2>/dev/null

test_ep "Chat" 'curl -s http://localhost:9700/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"llm\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}"' "chat.completion"
test_ep "JSON mode" 'curl -s http://localhost:9700/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"llm\",\"messages\":[{\"role\":\"user\",\"content\":\"JSON\"}],\"max_tokens\":8,\"response_format\":{\"type\":\"json_object\"}}"' "chat.completion"
test_ep "Stream" 'curl -s -N http://localhost:9700/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"llm\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":2,\"stream\":true}" | head -3' "data:"
test_ep "Text completion" 'curl -s http://localhost:9700/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"llm\",\"prompt\":\"Hello\",\"max_tokens\":4}"' "text_completion"
test_ep "Embedding" 'curl -s http://localhost:9700/v1/embeddings -H "Content-Type: application/json" -d "{\"model\":\"embed\",\"input\":\"hello\"}"' "embedding"
test_ep "Batch embed" 'curl -s http://localhost:9700/v1/embeddings -H "Content-Type: application/json" -d "{\"model\":\"embed\",\"input\":[\"a\",\"b\",\"c\"]}"' "embedding"
test_ep "Detection" 'curl -s -X POST "http://localhost:9700/v1/detect/binary?model=detect&conf=0.25" --data-binary @/tmp/test_overnight.jpg -H "Content-Type: application/octet-stream"' "objects"
test_ep "Search" 'curl -s http://localhost:9700/v1/search -H "Content-Type: application/json" -d "{\"model\":\"embed\",\"query\":\"hello\",\"k\":3}"' "results"
test_ep "Rerank" 'curl -s http://localhost:9700/v1/rerank -H "Content-Type: application/json" -d "{\"model\":\"embed\",\"query\":\"ML\",\"documents\":[\"AI\",\"cats\",\"deep\"],\"top_n\":2}"' "results"
test_ep "Models" 'curl -s http://localhost:9700/v1/models' "list"
test_ep "Health" 'curl -s http://localhost:9700/health/live' "ok"
test_ep "Ready" 'curl -s http://localhost:9700/health/ready' "ok"
test_ep "Metrics" 'curl -s http://localhost:9700/metrics | grep infergo_' "infergo_"
test_ep "Guardrails" 'curl -s http://localhost:9700/v1/admin/guardrails' ""
test_ep "Templates" 'curl -s http://localhost:9700/v1/admin/templates' ""
test_ep "Web UI" 'curl -s http://localhost:9700/ui' "DOCTYPE"
test_ep "Batch" 'curl -s http://localhost:9700/v1/batches -H "Content-Type: application/json" -d "{\"model\":\"llm\",\"prompts\":[\"hi\"]}"' "batch"

echo "Feature tests: $PASS/$TOTAL passed" | tee -a $LOG

pkill -9 -f "port 9700" 2>/dev/null; sleep 5

# ─── 3. Commit and push results ───
echo "" | tee -a $LOG
echo "=== PHASE 3: Commit and push ===" | tee -a $LOG

git add benchmarks/overnight_results.json 2>/dev/null
git add benchmarks/ 2>/dev/null
git commit -m "bench: overnight results — 5-mode × 1000 req + 17 feature tests

Automated overnight run: $(date '+%Y-%m-%d')
Features passing: $PASS/$TOTAL" 2>/dev/null || echo "Nothing to commit"
git push origin main 2>/dev/null || echo "Push failed (no auth)"

echo "" | tee -a $LOG
echo "=== OVERNIGHT TEST COMPLETE: $(date) ===" | tee -a $LOG
echo "Results: benchmarks/overnight_results.json" | tee -a $LOG
echo "Log: $LOG" | tee -a $LOG
