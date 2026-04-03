package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"github.com/ailakshya/infergo/onnx"
	"github.com/ailakshya/infergo/server"
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

// embedBatch runs a batched ONNX inference on N texts in one call.
// Input tensors: [N, maxSeqLen] int64 (padded with zeros).
// Output: N unit-norm embedding vectors.
func (a *embeddingAdapter) embedBatch(texts []string) ([][]float32, error) {
	n := len(texts)
	if n == 0 {
		return nil, nil
	}

	// 1. Tokenize all texts, track maxSeqLen.
	encs := make([]tokenizer.Encoding, n)
	maxSeqLen := 0
	for i, text := range texts {
		enc, err := a.tok.Encode(text, true, 512)
		if err != nil {
			return nil, fmt.Errorf("embedBatch: tokenize[%d]: %w", i, err)
		}
		encs[i] = enc
		if len(enc.IDs) > maxSeqLen {
			maxSeqLen = len(enc.IDs)
		}
	}
	if maxSeqLen == 0 {
		return nil, errors.New("embedBatch: all sequences are empty")
	}

	// 2. Allocate [N, maxSeqLen] int64 tensors.
	shape := []int{n, maxSeqLen}
	totalElems := n * maxSeqLen

	inputIDs, err := tensor.NewTensorCPU(shape, tensor.Int64)
	if err != nil {
		return nil, fmt.Errorf("embedBatch: alloc input_ids: %w", err)
	}
	defer inputIDs.Free()

	attMask, err := tensor.NewTensorCPU(shape, tensor.Int64)
	if err != nil {
		return nil, fmt.Errorf("embedBatch: alloc attention_mask: %w", err)
	}
	defer attMask.Free()

	tokenTypeIDs, err := tensor.NewTensorCPU(shape, tensor.Int64)
	if err != nil {
		return nil, fmt.Errorf("embedBatch: alloc token_type_ids: %w", err)
	}
	defer tokenTypeIDs.Free()

	// 3. Zero-initialize all three tensors (tensor uses malloc, not calloc).
	zeros := make([]int64, totalElems)
	byteLen := totalElems * 8
	if err := inputIDs.CopyFrom(unsafe.Pointer(&zeros[0]), byteLen); err != nil {
		return nil, fmt.Errorf("embedBatch: zero input_ids: %w", err)
	}
	if err := attMask.CopyFrom(unsafe.Pointer(&zeros[0]), byteLen); err != nil {
		return nil, fmt.Errorf("embedBatch: zero attention_mask: %w", err)
	}
	if err := tokenTypeIDs.CopyFrom(unsafe.Pointer(&zeros[0]), byteLen); err != nil {
		return nil, fmt.Errorf("embedBatch: zero token_type_ids: %w", err)
	}

	// 4. Fill input_ids and attention_mask row by row; padding stays 0.
	idsPtr := unsafe.Slice((*int64)(inputIDs.DataPtr()), totalElems)
	maskPtr := unsafe.Slice((*int64)(attMask.DataPtr()), totalElems)
	for i, enc := range encs {
		rowStart := i * maxSeqLen
		for j, id := range enc.IDs {
			idsPtr[rowStart+j] = int64(id)
			maskPtr[rowStart+j] = int64(enc.AttentionMask[j])
		}
	}

	// 5. Run ONNX — output[0] is [N, maxSeqLen, hiddenDim].
	outputs, err := a.sess.Run([]*tensor.Tensor{inputIDs, attMask, tokenTypeIDs})
	if err != nil {
		return nil, fmt.Errorf("embedBatch: onnx run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Free()
		}
	}()
	if len(outputs) == 0 {
		return nil, errors.New("embedBatch: no outputs from model")
	}

	hidden := outputs[0]
	shp := hidden.Shape()
	if len(shp) != 3 {
		return nil, fmt.Errorf("embedBatch: expected 3-dim output, got shape %v", shp)
	}
	hiddenDim := shp[2]
	data := unsafe.Slice((*float32)(hidden.DataPtr()), hidden.NElements())

	// 6. For each sequence: masked mean pool then L2 normalize.
	results := make([][]float32, n)
	for i, enc := range encs {
		seqLen := len(enc.IDs)
		pooled := make([]float32, hiddenDim)
		var maskSum float32
		rowBase := i * maxSeqLen * hiddenDim
		for j := 0; j < seqLen; j++ {
			m := float32(enc.AttentionMask[j])
			maskSum += m
			tokenBase := rowBase + j*hiddenDim
			for k := 0; k < hiddenDim; k++ {
				pooled[k] += data[tokenBase+k] * m
			}
		}
		if maskSum > 0 {
			for k := range pooled {
				pooled[k] /= maskSum
			}
		}

		// L2 normalize
		var norm float32
		for _, v := range pooled {
			norm += v * v
		}
		if norm > 0 {
			norm = float32(math.Sqrt(float64(norm)))
			for k := range pooled {
				pooled[k] /= norm
			}
		}
		results[i] = pooled
	}

	return results, nil
}

// ─── Batcher types ────────────────────────────────────────────────────────────

type embedResult struct {
	vec []float32
	err error
}

type embedBatchItem struct {
	text string
	out  chan<- embedResult
}

// embeddingBatcher collects concurrent Embed requests and dispatches them to
// embedBatch in groups, reducing ONNX call overhead.
type embeddingBatcher struct {
	adapter  *embeddingAdapter
	queue    chan embedBatchItem
	maxBatch int
	maxWait  time.Duration
}

// newEmbeddingBatcher creates a batcher wrapping the given adapter and starts
// the background dispatch goroutine.
func newEmbeddingBatcher(a *embeddingAdapter) *embeddingBatcher {
	b := &embeddingBatcher{
		adapter:  a,
		queue:    make(chan embedBatchItem, 256),
		maxBatch: 32,
		maxWait:  0, // opportunistic: drain queue immediately, no deliberate wait
	}
	go b.run()
	return b
}

// Close shuts down the underlying adapter. The queue goroutine will drain
// naturally once the process exits.
func (b *embeddingBatcher) Close() {
	b.adapter.Close()
}

// Embed satisfies server.EmbeddingModel. It enqueues the request and waits for
// the batcher to process it (or for ctx to be cancelled).
func (b *embeddingBatcher) Embed(ctx context.Context, text string) ([]float32, error) {
	out := make(chan embedResult, 1)
	item := embedBatchItem{text: text, out: out}

	select {
	case b.queue <- item:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	select {
	case res := <-out:
		return res.vec, res.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// run is the background goroutine that collects items and calls embedBatch.
func (b *embeddingBatcher) run() {
	for {
		// Block until at least one item arrives.
		first, ok := <-b.queue
		if !ok {
			return
		}

		items := []embedBatchItem{first}

		// Non-blocking drain: pick up any requests already queued.
		// Zero wait — if nothing is queued right now, fire immediately.
		// This gives batch size > 1 when requests arrive concurrently,
		// with zero added latency for serial requests.
	collect:
		for len(items) < b.maxBatch {
			select {
			case item, ok := <-b.queue:
				if !ok {
					break collect
				}
				items = append(items, item)
			default:
				break collect
			}
		}

		// Build text slice and run batched inference.
		texts := make([]string, len(items))
		for i, it := range items {
			texts[i] = it.text
		}

		vecs, err := b.adapter.embedBatch(texts)

		// Dispatch results back to callers.
		for i, it := range items {
			if err != nil {
				it.out <- embedResult{err: err}
			} else {
				it.out <- embedResult{vec: vecs[i]}
			}
		}
	}
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

// loadEmbedding creates an embedding model from an ONNX model path.
// It searches for tokenizer.json near the model file (up to 2 dirs up).
//
// Provider selection:
//   - "cpu": returns *embeddingAdapter directly. ONNX Runtime is thread-safe —
//     concurrent Run() calls across goroutines already saturate CPU cores, so a
//     batcher goroutine would only serialize work and hurt throughput.
//   - "cuda" / "tensorrt": returns *embeddingBatcher. GPU kernels benefit from
//     batched [N, seqLen] calls — fewer launches, better tensor-core utilization.
func loadEmbedding(modelPath, provider string) (server.EmbeddingModel, error) {
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

	adapter := &embeddingAdapter{sess: sess, tok: tok}
	if provider == "cuda" || provider == "tensorrt" {
		return newEmbeddingBatcher(adapter), nil
	}
	return adapter, nil
}
