// Prompt cache benchmark: first request (cache miss) vs subsequent (cache hit)
package main

import (
	"fmt"
	"os"
	"time"

	"github.com/ailakshya/infergo/llm"
)

const modelPath = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

func main() {
	m, err := llm.Load(modelPath, 99, 4096, 4, 512)
	if err != nil { fmt.Fprintf(os.Stderr, "Load: %v\n", err); os.Exit(1) }
	defer m.Close()

	prompt := "<|system|>\nYou are a helpful, concise assistant. Answer questions directly and briefly.</s>\n<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n"
	tokens, _ := m.Tokenize(prompt, false, 512)

	fmt.Printf("Model:  TinyLlama 1.1B Q4_K_M (CUDA)\n")
	fmt.Printf("Prompt: %d tokens | Max gen: 32\n\n", len(tokens))

	// First call: cache miss (prefill runs)
	start := time.Now()
	text1, gen1, _ := m.GenerateC(tokens, 32, 0.8, 0.9, "")
	miss := time.Since(start)

	fmt.Printf("Cache MISS (1st request):\n")
	fmt.Printf("  Time: %.2f ms | Tokens: %d | Output: %.50s\n\n", float64(miss.Microseconds())/1000, gen1, text1)

	// Subsequent calls: cache hit (prefill skipped)
	runs := 20
	var hitTimes []float64
	var lastText string
	for i := 0; i < runs; i++ {
		start := time.Now()
		text, _, _ := m.GenerateC(tokens, 32, 0.8, 0.9, "")
		elapsed := float64(time.Since(start).Microseconds()) / 1000.0
		hitTimes = append(hitTimes, elapsed)
		lastText = text
	}

	hitAvg := 0.0
	hitMin := hitTimes[0]
	for _, t := range hitTimes {
		hitAvg += t
		if t < hitMin { hitMin = t }
	}
	hitAvg /= float64(runs)

	fmt.Printf("Cache HIT (%d requests, same prompt):\n", runs)
	fmt.Printf("  Avg: %.2f ms | Min: %.2f ms | Output: %.50s\n\n", hitAvg, hitMin, lastText)

	missMs := float64(miss.Microseconds()) / 1000
	saved := missMs - hitAvg
	fmt.Printf("=== RESULT ===\n")
	fmt.Printf("Cache miss: %.2f ms (prefill + generate)\n", missMs)
	fmt.Printf("Cache hit:  %.2f ms (generate only)\n", hitAvg)
	fmt.Printf("Prefill saved: %.2f ms (%.0f%%)\n", saved, saved/missMs*100)
}
