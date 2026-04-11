// Grammar-constrained sampling benchmark.
// Compares: standard sampling vs grammar (zero-copy) vs grammar (logits-copy).
//
// Usage: go run benchmarks/grammar_bench.go
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/ailakshya/infergo/llm"
	"github.com/ailakshya/infergo/server"
)

const modelPath = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
const maxTokens = 64
const warmupRuns = 2
const benchRuns = 10

func main() {
	if _, err := os.Stat(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "Model not found: %s\n", modelPath)
		os.Exit(1)
	}

	m, err := llm.Load(modelPath, 99, 2048, 4, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Load: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	prompt := "Generate a JSON object with fields: name (string), age (number), active (boolean):"
	tokens, err := m.Tokenize(prompt, true, 256)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Tokenize: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Model:       %s\n", modelPath)
	fmt.Printf("Prompt:      %d tokens\n", len(tokens))
	fmt.Printf("Max tokens:  %d\n", maxTokens)
	fmt.Printf("Warmup:      %d runs\n", warmupRuns)
	fmt.Printf("Bench:       %d runs\n", benchRuns)
	fmt.Println()

	// ─── Benchmark 1: Standard Go-side sampling (no grammar) ─────────────
	fmt.Println("=== Standard sampling (Go-side, no grammar) ===")
	standardResults := benchStandard(m, tokens)
	printResults(standardResults)

	// ─── Benchmark 2: Grammar sampling via zero-copy SampleSeq ───────────
	fmt.Println("=== Grammar sampling (zero-copy SampleSeq) ===")
	grammarZCResults := benchGrammarZeroCopy(m, tokens)
	printResults(grammarZCResults)

	// ─── Benchmark 3: Grammar sampling via Logits+Sample (old path) ──────
	fmt.Println("=== Grammar sampling (logits-copy Sample) ===")
	grammarCopyResults := benchGrammarCopy(m, tokens)
	printResults(grammarCopyResults)

	// ─── Summary ─────────────────────────────────────────────────────────
	fmt.Println("=== Summary ===")
	fmt.Printf("%-35s %8s %8s %8s %8s\n", "Method", "Avg(ms)", "Min(ms)", "Max(ms)", "Tok/s")
	printSummaryRow("Standard (Go, no grammar)", standardResults)
	printSummaryRow("Grammar (zero-copy SampleSeq)", grammarZCResults)
	printSummaryRow("Grammar (logits-copy Sample)", grammarCopyResults)

	if len(grammarZCResults) > 0 && len(grammarCopyResults) > 0 {
		zcAvg := avg(grammarZCResults)
		cpAvg := avg(grammarCopyResults)
		if zcAvg > 0 {
			fmt.Printf("\nZero-copy speedup over logits-copy: %.1fx\n", cpAvg/zcAvg)
		}
	}
}

type runResult struct {
	durationMs float64
	tokens     int
	output     string
	validJSON  bool
}

func benchStandard(m *llm.Model, promptTokens []int32) []runResult {
	var results []runResult
	for i := 0; i < warmupRuns+benchRuns; i++ {
		seq, err := m.NewSequence(promptTokens)
		if err != nil {
			fmt.Fprintf(os.Stderr, "NewSequence: %v\n", err)
			continue
		}

		start := time.Now()
		var output []byte
		for t := 0; t < maxTokens; t++ {
			if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
				break
			}
			tok, err := seq.SampleToken(0.8, 0.9)
			if err != nil || m.IsEOG(tok) {
				break
			}
			piece, _ := m.TokenToPiece(tok)
			output = append(output, piece...)
			seq.AppendToken(tok)
		}
		elapsed := time.Since(start)
		seq.Close()

		if i >= warmupRuns {
			results = append(results, runResult{
				durationMs: float64(elapsed.Microseconds()) / 1000.0,
				tokens:     len(output),
				output:     string(output),
				validJSON:  json.Valid(output),
			})
		}
	}
	return results
}

func benchGrammarZeroCopy(m *llm.Model, promptTokens []int32) []runResult {
	var results []runResult
	for i := 0; i < warmupRuns+benchRuns; i++ {
		seq, err := m.NewSequence(promptTokens)
		if err != nil {
			continue
		}
		smpl, err := llm.NewGrammarSampler(m, server.JSONGrammar, "root", 0.8, 0.9, 0, 0)
		if err != nil {
			seq.Close()
			continue
		}

		start := time.Now()
		var output []byte
		for t := 0; t < maxTokens; t++ {
			if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
				break
			}
			tok, err := smpl.SampleSeq(seq) // ZERO-COPY
			if err != nil || m.IsEOG(tok) {
				break
			}
			piece, _ := m.TokenToPiece(tok)
			output = append(output, piece...)
			seq.AppendToken(tok)
		}
		elapsed := time.Since(start)
		smpl.Close()
		seq.Close()

		if i >= warmupRuns {
			results = append(results, runResult{
				durationMs: float64(elapsed.Microseconds()) / 1000.0,
				tokens:     len(output),
				output:     string(output),
				validJSON:  json.Valid(output),
			})
		}
	}
	return results
}

func benchGrammarCopy(m *llm.Model, promptTokens []int32) []runResult {
	var results []runResult
	for i := 0; i < warmupRuns+benchRuns; i++ {
		seq, err := m.NewSequence(promptTokens)
		if err != nil {
			continue
		}
		smpl, err := llm.NewGrammarSampler(m, server.JSONGrammar, "root", 0.8, 0.9, 0, 0)
		if err != nil {
			seq.Close()
			continue
		}

		start := time.Now()
		var output []byte
		for t := 0; t < maxTokens; t++ {
			if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
				break
			}
			logits, err := seq.Logits() // COPY logits to Go
			if err != nil {
				break
			}
			tok, err := smpl.Sample(logits) // COPY logits back to C++
			if err != nil || m.IsEOG(tok) {
				break
			}
			piece, _ := m.TokenToPiece(tok)
			output = append(output, piece...)
			seq.AppendToken(tok)
		}
		elapsed := time.Since(start)
		smpl.Close()
		seq.Close()

		if i >= warmupRuns {
			results = append(results, runResult{
				durationMs: float64(elapsed.Microseconds()) / 1000.0,
				tokens:     len(output),
				output:     string(output),
				validJSON:  json.Valid(output),
			})
		}
	}
	return results
}

func printResults(results []runResult) {
	for i, r := range results {
		jsonTag := ""
		if r.validJSON {
			jsonTag = " [VALID JSON]"
		}
		fmt.Printf("  run %2d: %7.2f ms, %d bytes%s\n", i+1, r.durationMs, r.tokens, jsonTag)
	}
	if len(results) > 0 {
		fmt.Printf("  sample output: %.80s\n", results[0].output)
	}
	fmt.Println()
}

func printSummaryRow(name string, results []runResult) {
	if len(results) == 0 {
		return
	}
	a := avg(results)
	mn := minD(results)
	mx := maxD(results)
	// Estimate tokens per second from average bytes and time
	avgBytes := 0.0
	for _, r := range results {
		avgBytes += float64(r.tokens)
	}
	avgBytes /= float64(len(results))
	tps := 0.0
	if a > 0 {
		tps = avgBytes / (a / 1000.0)
	}
	fmt.Printf("%-35s %8.2f %8.2f %8.2f %8.0f\n", name, a, mn, mx, tps)
}

func avg(r []runResult) float64 {
	if len(r) == 0 {
		return 0
	}
	var sum float64
	for _, v := range r {
		sum += v.durationMs
	}
	return sum / float64(len(r))
}

func minD(r []runResult) float64 {
	if len(r) == 0 {
		return 0
	}
	m := r[0].durationMs
	for _, v := range r[1:] {
		if v.durationMs < m {
			m = v.durationMs
		}
	}
	return m
}

func maxD(r []runResult) float64 {
	if len(r) == 0 {
		return 0
	}
	m := r[0].durationMs
	for _, v := range r[1:] {
		if v.durationMs > m {
			m = v.durationMs
		}
	}
	return m
}
