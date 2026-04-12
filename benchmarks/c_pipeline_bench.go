// Benchmark: Full C embedding pipeline vs old Go pipeline
package main

import (
	"fmt"
	"os"
	"time"
	"unsafe"

	"github.com/ailakshya/infergo/llm"
	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/tensor"
	"github.com/ailakshya/infergo/tokenizer"
)

func main() {
	modelPath := os.ExpandEnv("${HOME}/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")
	tokPath := os.ExpandEnv("${HOME}/cgo/models/all-MiniLM-L6-v2/tokenizer.json")

	sess, _ := onnx.NewSession("cuda", 0)
	defer sess.Close()
	sess.Load(modelPath)

	tok, _ := tokenizer.Load(tokPath)
	defer tok.Close()

	texts := []string{"The quick brown fox.", "Machine learning.", "Go is compiled."}
	const runs = 30
	const warmup = 10

	fmt.Println("Embedding pipeline benchmark: C vs Go")
	fmt.Printf("Texts: %d | Runs: %d | CUDA\n\n", len(texts), runs)

	// ── NEW: Full C pipeline (one CGo call) ──
	fmt.Println("=== Full C Pipeline (1 CGo call) ===")
	sessPtr := sess.Handle()
	tokPtr := tok.Handle()

	for i := 0; i < warmup; i++ {
		llm.EmbedBatchPipeline(sessPtr, tokPtr, texts, 384)
	}

	var cTimes []float64
	for i := 0; i < runs; i++ {
		start := time.Now()
		_, err := llm.EmbedBatchPipeline(sessPtr, tokPtr, texts, 384)
		elapsed := float64(time.Since(start).Microseconds()) / 1000
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			continue
		}
		cTimes = append(cTimes, elapsed)
	}
	cAvg := avg(cTimes)
	fmt.Printf("  Avg: %.2fms | Min: %.2fms\n", cAvg, mn(cTimes))

	// ── OLD: Go pipeline (tokenize in Go, N CGo calls) ──
	fmt.Println("\n=== Go Pipeline (tokenize + alloc + CGo + pool + normalize) ===")

	// Warmup
	for i := 0; i < warmup; i++ {
		goEmbed(sess, tok, texts)
	}

	var goTimes []float64
	for i := 0; i < runs; i++ {
		start := time.Now()
		goEmbed(sess, tok, texts)
		goTimes = append(goTimes, float64(time.Since(start).Microseconds())/1000)
	}
	goAvg := avg(goTimes)
	fmt.Printf("  Avg: %.2fms | Min: %.2fms\n", goAvg, mn(goTimes))

	fmt.Printf("\n=== RESULT ===\n")
	fmt.Printf("  C pipeline:  %.2fms\n", cAvg)
	fmt.Printf("  Go pipeline: %.2fms\n", goAvg)
	if cAvg < goAvg {
		fmt.Printf("  C is %.1fx faster\n", goAvg/cAvg)
	} else {
		fmt.Printf("  Go is %.1fx faster\n", cAvg/goAvg)
	}
}

func goEmbed(sess *onnx.Session, tok *tokenizer.Tokenizer, texts []string) {
	maxLen := 0
	encs := make([]tokenizer.Encoding, len(texts))
	for i, t := range texts {
		enc, _ := tok.Encode(t, true, 128)
		encs[i] = enc
		if len(enc.IDs) > maxLen {
			maxLen = len(enc.IDs)
		}
	}

	n := len(texts)
	shape := []int{n, maxLen}
	ids, _ := tensor.NewTensorCPU(shape, tensor.Int64)
	defer ids.Free()
	mask, _ := tensor.NewTensorCPU(shape, tensor.Int64)
	defer mask.Free()
	tt, _ := tensor.NewTensorCPU(shape, tensor.Int64)
	defer tt.Free()

	ip := (*[1 << 20]int64)(ids.DataPtr())[:n*maxLen]
	mp := (*[1 << 20]int64)(mask.DataPtr())[:n*maxLen]
	for i, enc := range encs {
		for j := 0; j < maxLen; j++ {
			if j < len(enc.IDs) {
				ip[i*maxLen+j] = int64(enc.IDs[j])
				mp[i*maxLen+j] = int64(enc.AttentionMask[j])
			}
		}
	}

	outputs, _ := sess.Run([]*tensor.Tensor{ids, mask, tt})
	for _, o := range outputs {
		o.Free()
	}
}

func avg(t []float64) float64 { s := 0.0; for _, v := range t { s += v }; return s / float64(len(t)) }
func mn(t []float64) float64  { m := t[0]; for _, v := range t[1:] { if v < m { m = v } }; return m }

// Need to expose session/tokenizer handles
var _ = unsafe.Pointer(nil)
