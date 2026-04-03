package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/tensor"
	"github.com/ailakshya/infergo/tokenizer"
)

// embeddingAdapter wraps an ONNX session + HuggingFace tokenizer to implement
// server.EmbeddingModel. The pipeline is:
//
//  1. Tokenize input text → input_ids, attention_mask (token_type_ids = zeros)
//  2. Run ONNX session → last_hidden_state [1, seqLen, hiddenDim]
//  3. Masked mean pool → [hiddenDim]
//  4. L2 normalize → unit-norm embedding vector
type embeddingAdapter struct {
	sess *onnx.Session
	tok  *tokenizer.Tokenizer
}

// Close releases the ONNX session and tokenizer.
// Implements server.Model.
func (a *embeddingAdapter) Close() {
	a.sess.Close()
	a.tok.Close()
}

// Embed tokenizes input, runs the ONNX model, mean-pools the last hidden state
// with the attention mask, and returns a unit-norm embedding vector.
// Implements server.EmbeddingModel.
func (a *embeddingAdapter) Embed(_ context.Context, input string) ([]float32, error) {
	// 1. Tokenize — max 512 tokens (standard BERT limit)
	enc, err := a.tok.Encode(input, true, 512)
	if err != nil {
		return nil, fmt.Errorf("embed: tokenize: %w", err)
	}
	seqLen := len(enc.IDs)
	if seqLen == 0 {
		return nil, errors.New("embed: empty token sequence")
	}

	// 2. Allocate tensors: input_ids, attention_mask, token_type_ids — all [1, seqLen] int64
	shape := []int{1, seqLen}
	inputIDs, err := tensor.NewTensorCPU(shape, tensor.Int64)
	if err != nil {
		return nil, fmt.Errorf("embed: alloc input_ids: %w", err)
	}
	defer inputIDs.Free()

	attMask, err := tensor.NewTensorCPU(shape, tensor.Int64)
	if err != nil {
		return nil, fmt.Errorf("embed: alloc attention_mask: %w", err)
	}
	defer attMask.Free()

	tokenTypeIDs, err := tensor.NewTensorCPU(shape, tensor.Int64)
	if err != nil {
		return nil, fmt.Errorf("embed: alloc token_type_ids: %w", err)
	}
	defer tokenTypeIDs.Free()

	// Zero-initialize tokenTypeIDs (malloc, not calloc)
	zeros := make([]int64, seqLen)
	if err := tokenTypeIDs.CopyFrom(unsafe.Pointer(&zeros[0]), seqLen*8); err != nil {
		return nil, fmt.Errorf("embed: zero token_type_ids: %w", err)
	}

	// Fill input_ids and attention_mask
	idsPtr := unsafe.Slice((*int64)(inputIDs.DataPtr()), seqLen)
	maskPtr := unsafe.Slice((*int64)(attMask.DataPtr()), seqLen)
	for i, id := range enc.IDs {
		idsPtr[i] = int64(id)
		maskPtr[i] = int64(enc.AttentionMask[i])
	}

	// 3. Run ONNX — expects [input_ids, attention_mask, token_type_ids]
	outputs, err := a.sess.Run([]*tensor.Tensor{inputIDs, attMask, tokenTypeIDs})
	if err != nil {
		return nil, fmt.Errorf("embed: onnx run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Free()
		}
	}()
	if len(outputs) == 0 {
		return nil, errors.New("embed: no outputs from model")
	}

	// 4. Mean pool last_hidden_state [1, seqLen, hiddenDim]
	hidden := outputs[0]
	shp := hidden.Shape()
	if len(shp) != 3 {
		return nil, fmt.Errorf("embed: expected 3-dim output, got shape %v", shp)
	}
	hiddenDim := shp[2] // e.g. 384 for all-MiniLM-L6-v2

	nElem := hidden.NElements() // 1 * seqLen * hiddenDim
	data := unsafe.Slice((*float32)(hidden.DataPtr()), nElem)

	pooled := make([]float32, hiddenDim)
	var maskSum float32
	for i := 0; i < seqLen; i++ {
		m := float32(enc.AttentionMask[i])
		maskSum += m
		row := data[i*hiddenDim : (i+1)*hiddenDim]
		for j, v := range row {
			pooled[j] += v * m
		}
	}
	if maskSum > 0 {
		for j := range pooled {
			pooled[j] /= maskSum
		}
	}

	// 5. L2 normalize
	var norm float32
	for _, v := range pooled {
		norm += v * v
	}
	if norm > 0 {
		norm = float32(math.Sqrt(float64(norm)))
		for j := range pooled {
			pooled[j] /= norm
		}
	}

	return pooled, nil
}

// findTokenizerJSON searches for tokenizer.json starting at dir and walking up
// to maxDepth parent directories. Returns the path if found, "" otherwise.
func findTokenizerJSON(dir string, maxDepth int) string {
	cur := dir
	for i := 0; i <= maxDepth; i++ {
		candidate := filepath.Join(cur, "tokenizer.json")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break // reached filesystem root
		}
		cur = parent
	}
	return ""
}

// loadEmbedding creates an embeddingAdapter from an ONNX model path.
// It searches for tokenizer.json near the model file (up to 2 dirs up).
func loadEmbedding(modelPath, provider string) (*embeddingAdapter, error) {
	sess, err := onnx.NewSession(provider, 0)
	if err != nil {
		return nil, fmt.Errorf("loadEmbedding: new session: %w", err)
	}
	if err := sess.Load(modelPath); err != nil {
		sess.Close()
		return nil, fmt.Errorf("loadEmbedding: load model: %w", err)
	}

	tokPath := findTokenizerJSON(filepath.Dir(modelPath), 2)
	if tokPath == "" {
		sess.Close()
		return nil, fmt.Errorf("loadEmbedding: tokenizer.json not found near %s", modelPath)
	}

	tok, err := tokenizer.Load(tokPath)
	if err != nil {
		sess.Close()
		return nil, fmt.Errorf("loadEmbedding: load tokenizer: %w", err)
	}

	return &embeddingAdapter{sess: sess, tok: tok}, nil
}
