package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/ailakshya/infergo/llm"
)

// benchCmd implements `infergo bench model.gguf` — quick model benchmarking. OPT-69.
func benchCmd(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "Usage: infergo bench <model.gguf> [--output results.json]")
		os.Exit(1)
	}

	modelPath := args[0]
	outputPath := ""
	for i := 1; i < len(args); i++ {
		if args[i] == "--output" && i+1 < len(args) {
			outputPath = args[i+1]
			i++
		}
	}

	fmt.Printf("Benchmarking: %s\n", modelPath)

	// Load model
	t0 := time.Now()
	m, err := llm.Load(modelPath, -1, 4096, 1, 2048)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()
	loadMs := time.Since(t0).Milliseconds()
	fmt.Printf("  Load time: %dms\n", loadMs)

	// Tokenize test prompt
	prompt := "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"
	tokens, err := m.Tokenize(prompt, true, 4096)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Tokenize failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Prompt tokens: %d\n", len(tokens))

	// Warmup
	fmt.Print("  Warming up...")
	m.GenerateC(tokens, 8, 0.7, 0.9, "")
	fmt.Println(" done")

	// Benchmark
	N := 20
	maxTokens := 32
	fmt.Printf("  Running %d requests (max_tokens=%d)...\n", N, maxTokens)

	var latencies []float64
	var tokCounts []int
	for i := 0; i < N; i++ {
		t0 := time.Now()
		_, genToks, err := m.GenerateC(tokens, maxTokens, 0.7, 0.9, "")
		ms := float64(time.Since(t0).Milliseconds())
		if err != nil {
			fmt.Printf("  req %d: ERROR %v\n", i, err)
			continue
		}
		latencies = append(latencies, ms)
		tokCounts = append(tokCounts, genToks)
	}

	if len(latencies) == 0 {
		fmt.Fprintln(os.Stderr, "All requests failed!")
		os.Exit(1)
	}

	sort.Float64s(latencies)
	avg := 0.0
	totalTok := 0
	for i, l := range latencies {
		avg += l
		totalTok += tokCounts[i]
	}
	avg /= float64(len(latencies))
	avgTok := float64(totalTok) / float64(len(latencies))

	results := map[string]interface{}{
		"model":     modelPath,
		"load_ms":   loadMs,
		"n":         len(latencies),
		"max_tokens": maxTokens,
		"avg_ms":    fmt.Sprintf("%.1f", avg),
		"p50_ms":    fmt.Sprintf("%.1f", latencies[len(latencies)/2]),
		"min_ms":    fmt.Sprintf("%.1f", latencies[0]),
		"max_ms":    fmt.Sprintf("%.1f", latencies[len(latencies)-1]),
		"avg_tokens": fmt.Sprintf("%.1f", avgTok),
		"ms_per_tok": fmt.Sprintf("%.2f", avg/avgTok),
		"tok_per_sec": fmt.Sprintf("%.1f", avgTok/(avg/1000)),
	}

	fmt.Printf("\n=== Results ===\n")
	fmt.Printf("  Avg: %sms  P50: %sms  Min: %sms\n", results["avg_ms"], results["p50_ms"], results["min_ms"])
	fmt.Printf("  ms/tok: %s  tok/s: %s\n", results["ms_per_tok"], results["tok_per_sec"])

	if outputPath != "" {
		f, err := os.Create(outputPath)
		if err == nil {
			json.NewEncoder(f).Encode(results)
			f.Close()
			fmt.Printf("  Saved to %s\n", outputPath)
		}
	}
}
