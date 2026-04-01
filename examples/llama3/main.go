// examples/llama3/main.go
//
// LLaMA generation example — loads a GGUF model and streams generated text
// to stdout one token at a time.
//
// Usage:
//
//	go run . -model /path/to/model.gguf -prompt "Once upon a time"
//	go run . -model /path/to/model.gguf -prompt "Tell me a joke" -max-tokens 200 -temp 0.8
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/ailakshya/infergo/llm"
)

func main() {
	modelPath := flag.String("model", "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "path to GGUF model file")
	prompt    := flag.String("prompt", "Once upon a time", "text prompt to continue")
	maxTokens := flag.Int("max-tokens", 200, "maximum number of tokens to generate")
	temp      := flag.Float64("temp", 0.8, "sampling temperature (0 = greedy)")
	topP      := flag.Float64("top-p", 0.9, "top-p nucleus sampling threshold (0 = disabled)")
	gpuLayers := flag.Int("gpu-layers", 99, "transformer layers to offload to GPU")
	ctxSize   := flag.Int("ctx-size", 2048, "KV cache context size in tokens")
	flag.Parse()

	// Load model
	m, err := llm.Load(*modelPath, *gpuLayers, *ctxSize, 1, 512)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	// Tokenize prompt (with BOS)
	tokens, err := m.Tokenize(*prompt, true, *ctxSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenize error: %v\n", err)
		os.Exit(1)
	}

	// Create sequence
	seq, err := m.NewSequence(tokens)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sequence error: %v\n", err)
		os.Exit(1)
	}
	defer seq.Close()

	// Echo prompt text and stream generated tokens
	fmt.Print(*prompt)

	generated := 0
	for !seq.IsDone() && generated < *maxTokens {
		if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
			fmt.Fprintf(os.Stderr, "\nbatch decode error: %v\n", err)
			os.Exit(1)
		}

		tok, err := seq.SampleToken(float32(*temp), float32(*topP))
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nsample error: %v\n", err)
			os.Exit(1)
		}

		if m.IsEOG(tok) {
			break
		}

		piece, err := m.TokenToPiece(tok)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\ntoken_to_piece error: %v\n", err)
			os.Exit(1)
		}

		fmt.Print(piece)
		seq.AppendToken(tok)
		generated++
	}

	fmt.Println()
}
