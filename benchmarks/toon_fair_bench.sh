#!/bin/bash
# Fair TOON vs JSON benchmark — restarts server between modes
set -e

export LD_LIBRARY_PATH=/home/lakshya/cgo/build/cpp/api:/home/lakshya/cgo/build/cpp/onnx:/home/lakshya/cgo/build/cpp/tokenizer:/home/lakshya/yolo-env/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

PORT=9191
MODEL=/tmp/qwen2.5-coder-1.5b-q4.gguf
BINARY=/home/lakshya/cgo/infergo
N=20
MAX_TOK=128

start_server() {
    pkill -9 -f "port $PORT" 2>/dev/null || true
    sleep 3
    $BINARY serve --model llm:$MODEL --provider cuda --port $PORT --grpc-port 0 --max-seqs 4 --ctx-size 4096 &
    disown
    sleep 15
    curl -s http://localhost:$PORT/health/live >/dev/null 2>&1 || { echo "FATAL: server didn't start"; exit 1; }
    echo "Server ready"
}

bench() {
    local mode=$1 body=$2
    echo ""
    echo "=== $mode ($N requests, max_tokens=$MAX_TOK) ==="

    local total=0 count=0 min=999999 max=0 tokens_total=0

    for i in $(seq 1 $N); do
        local t0=$(date +%s%N)
        local resp=$(curl -s -w "\n%{http_code}" http://localhost:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" -d "$body" 2>/dev/null)
        local t1=$(date +%s%N)

        local http_code=$(echo "$resp" | tail -1)
        if [ "$http_code" != "200" ]; then
            echo "  req $i: HTTP $http_code"
            continue
        fi

        local ms=$(( (t1 - t0) / 1000000 ))
        local tok=$(echo "$resp" | head -1 | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")

        total=$((total + ms))
        tokens_total=$((tokens_total + tok))
        count=$((count + 1))
        [ $ms -lt $min ] && min=$ms
        [ $ms -gt $max ] && max=$ms
    done

    if [ $count -gt 0 ]; then
        local avg=$((total / count))
        local avg_tok=$((tokens_total / count))
        local ms_per_tok=0
        [ $avg_tok -gt 0 ] && ms_per_tok=$((avg / avg_tok))
        echo "  Results: $count/$N ok, avg=${avg}ms, min=${min}ms, max=${max}ms, avg_tok=$avg_tok, ms/tok=$ms_per_tok"

        # Sample output
        local sample=$(curl -s http://localhost:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" -d "$body" 2>/dev/null | \
            python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:120])" 2>/dev/null || echo "N/A")
        echo "  Sample: $sample"
    else
        echo "  All requests failed!"
    fi
}

SYS_JSON='You output valid JSON only. No markdown.'
SYS_TOON='Output TOON format only. key:value|key:value. Arrays: [a,b]. Nested: (k:v|k:v). Example: name:Alice|age:25|hobbies:[reading,coding]'
PROMPT='Return a person with name, age, city, occupation, hobbies (3 items).'

echo "===== TOON vs JSON Fair Benchmark ====="
echo "Model: $MODEL"
echo "N=$N, max_tokens=$MAX_TOK"
echo ""

# Mode 1: Plain text
start_server
bench "Plain (no grammar)" \
    "{\"model\":\"llm\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOK,\"temperature\":0.7}"

# Mode 2: JSON lazy
start_server
bench "JSON lazy" \
    "{\"model\":\"llm\",\"messages\":[{\"role\":\"system\",\"content\":\"$SYS_JSON\"},{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOK,\"temperature\":0.7,\"response_format\":{\"type\":\"json_object\"}}"

# Mode 3: JSON strict (via custom grammar type — bypasses lazy detection)
JSON_GRAMMAR='root ::= object\nvalue ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\nobject ::= \"{\" ws (string \":\" ws value (\",\" ws string \":\" ws value)*)? \"}\" ws\narray ::= \"[\" ws (value (\",\" ws value)*)? \"]\" ws\nstring ::= \"\\\"\" ([^\\\\\"\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F]{4}))* \"\\\"\" ws\nnumber ::= (\"-\"? ([0-9] | [1-9] [0-9]*)) (\".\" [0-9]+)? (([eE] [-+]? [0-9]+))? ws\nws ::= ([ \\t\\n] ws)?'

start_server
bench "JSON strict" \
    "{\"model\":\"llm\",\"messages\":[{\"role\":\"system\",\"content\":\"$SYS_JSON\"},{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOK,\"temperature\":0.7,\"response_format\":{\"type\":\"grammar\",\"grammar\":\"$JSON_GRAMMAR\"}}"

# Mode 4: TOON strict
start_server
bench "TOON strict" \
    "{\"model\":\"llm\",\"messages\":[{\"role\":\"system\",\"content\":\"$SYS_TOON\"},{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOK,\"temperature\":0.7,\"response_format\":{\"type\":\"toon\"}}"

# Cleanup
pkill -9 -f "port $PORT" 2>/dev/null || true
echo ""
echo "===== Benchmark complete ====="
