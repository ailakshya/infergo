// In-process CUDA embedding benchmark — no HTTP
package main

import (
	"fmt"
	"os"
	"time"
	"unsafe"

	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/tensor"
	"github.com/ailakshya/infergo/tokenizer"
)

func main() {
	modelPath := os.ExpandEnv("${HOME}/cgo/models/all-MiniLM-L6-v2/onnx/model.onnx")
	tokPath := os.ExpandEnv("${HOME}/cgo/models/all-MiniLM-L6-v2/tokenizer.json")

	sess, err := onnx.NewSession("cuda", 0)
	if err != nil { fmt.Println("Session:", err); os.Exit(1) }
	defer sess.Close()
	if err := sess.Load(modelPath); err != nil { fmt.Println("Load:", err); os.Exit(1) }

	tok, err := tokenizer.Load(tokPath)
	if err != nil { fmt.Println("Tokenizer:", err); os.Exit(1) }
	defer tok.Close()

	texts := []string{"The quick brown fox.", "Machine learning transforms.", "Go is compiled."}
	const runs = 30
	const warmup = 10

	encodings := make([]tokenizer.Encoding, len(texts))
	maxLen := 0
	for i, t := range texts {
		enc, _ := tok.Encode(t, true, 128)
		encodings[i] = enc
		if len(enc.IDs) > maxLen { maxLen = len(enc.IDs) }
	}

	n := len(texts)
	fmt.Printf("Embedding: %d texts, maxLen=%d, CUDA, runs=%d\n\n", n, maxLen, runs)

	embedOnce := func() {
		shape := []int{n, maxLen}
		ids, _ := tensor.NewTensorCPU(shape, tensor.Int64)
		defer ids.Free()
		mask, _ := tensor.NewTensorCPU(shape, tensor.Int64)
		defer mask.Free()
		ttids, _ := tensor.NewTensorCPU(shape, tensor.Int64)
		defer ttids.Free()

		idsPtr := (*[1 << 20]int64)(ids.DataPtr())[:n*maxLen]
		maskPtr := (*[1 << 20]int64)(mask.DataPtr())[:n*maxLen]

		for i, enc := range encodings {
			for j := 0; j < maxLen; j++ {
				if j < len(enc.IDs) {
					idsPtr[i*maxLen+j] = int64(enc.IDs[j])
					maskPtr[i*maxLen+j] = int64(enc.AttentionMask[j])
				}
			}
		}

		outputs, err := sess.Run([]*tensor.Tensor{ids, mask, ttids})
		if err != nil { return }
		for _, o := range outputs { o.Free() }
	}
	_ = unsafe.Pointer(nil) // keep import

	for i := 0; i < warmup; i++ { embedOnce() }

	var times []float64
	for i := 0; i < runs; i++ {
		start := time.Now()
		embedOnce()
		times = append(times, float64(time.Since(start).Microseconds())/1000)
	}

	avg, mn := 0.0, times[0]
	for _, t := range times { avg += t; if t < mn { mn = t } }
	avg /= float64(len(times))

	fmt.Printf("infergo ONNX CUDA (in-process): avg=%.1fms | min=%.1fms\n", avg, mn)
	fmt.Printf("Python sentence-transformers:   avg=2.1ms (reference)\n")
	if avg < 2.1 {
		fmt.Printf("→ infergo %.1fx FASTER\n", 2.1/avg)
	} else {
		fmt.Printf("→ gap=%.1fms (%.1fx)\n", avg-2.1, avg/2.1)
	}
}
