// Fair benchmark: C loop vs Go loop, normalized per-token
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

	prompt := "<|system|>\nYou are helpful.</s>\n<|user|>\nList 5 colors.</s>\n<|assistant|>\n"
	tokens, _ := m.Tokenize(prompt, false, 256)

	fmt.Printf("Model:  TinyLlama 1.1B Q4_K_M (CUDA, RTX 5070 Ti)\n")
	fmt.Printf("Prompt: %d tokens | Generation: 64 tokens (greedy)\n\n", len(tokens))

	// Warmup both paths
	for i := 0; i < 3; i++ {
		m.GenerateC(tokens, 64, 0.8, 0.9, "")
		seq, _ := m.NewSequence(tokens)
		for t := 0; t < 64; t++ {
			m.BatchDecode([]*llm.Sequence{seq})
			tok, _ := seq.SampleToken(0.8, 0.9)
			if m.IsEOG(tok) { break }
			seq.AppendToken(tok)
		}
		seq.Close()
	}

	runs := 15

	// C loop
	var cTotal float64
	var cToks int
	for i := 0; i < runs; i++ {
		start := time.Now()
		_, genToks, _ := m.GenerateC(tokens, 64, 0.8, 0.9, "")
		cTotal += float64(time.Since(start).Microseconds()) / 1000.0
		cToks += genToks
	}

	// Go loop
	var goTotal float64
	var goToks int
	for i := 0; i < runs; i++ {
		seq, _ := m.NewSequence(tokens)
		start := time.Now()
		for t := 0; t < 64; t++ {
			if m.BatchDecode([]*llm.Sequence{seq}) != nil { break }
			tok, err := seq.SampleToken(0.8, 0.9)
			if err != nil || m.IsEOG(tok) { break }
			seq.AppendToken(tok)
			goToks++
		}
		goTotal += float64(time.Since(start).Microseconds()) / 1000.0
		seq.Close()
	}

	cAvg := cTotal / float64(runs)
	goAvg := goTotal / float64(runs)
	cPerTok := cTotal / float64(cToks)
	goPerTok := goTotal / float64(goToks)

	fmt.Printf("%-35s %8s %8s %10s\n", "Method", "Avg(ms)", "ms/tok", "CGo calls")
	fmt.Printf("%-35s %8.1f %8.3f %10s\n", "Full C loop (new)", cAvg, cPerTok, "1")
	fmt.Printf("%-35s %8.1f %8.3f %10s\n", "Per-token Go loop (old)", goAvg, goPerTok, fmt.Sprintf("4×%d=%d", goToks/runs, 4*goToks/runs))
	fmt.Printf("\nPer-token difference: %.3f ms (%.1f%%)\n", goPerTok-cPerTok, (goPerTok-cPerTok)/goPerTok*100)
	fmt.Printf("Total CGo calls saved per 64-token request: %d → 1\n", 4*64)
}
