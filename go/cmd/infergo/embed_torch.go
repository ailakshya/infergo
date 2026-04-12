package main

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/ailakshya/infergo/server"
	"github.com/ailakshya/infergo/tensor"
	"github.com/ailakshya/infergo/tokenizer"
	"github.com/ailakshya/infergo/torch"
)

// torchEmbeddingAdapter wraps a TorchScript model for embedding.
// Uses PyTorch's CUDA kernels instead of ONNX Runtime — 0.3ms faster for batch.
type torchEmbeddingAdapter struct {
	sess *torch.Session
	tok  *tokenizer.Tokenizer
}

var _ server.EmbeddingModel = (*torchEmbeddingAdapter)(nil)

func (a *torchEmbeddingAdapter) Close() {
	a.sess.Close()
	a.tok.Close()
}

func (a *torchEmbeddingAdapter) Embed(_ context.Context, input string) ([]float32, error) {
	vecs, err := a.embedBatch([]string{input})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

func (a *torchEmbeddingAdapter) EmbedBatch(_ context.Context, inputs []string) ([][]float32, error) {
	return a.embedBatch(inputs)
}

func (a *torchEmbeddingAdapter) embedBatch(texts []string) ([][]float32, error) {
	n := len(texts)
	if n == 0 {
		return nil, errors.New("empty input")
	}

	// Tokenize
	maxLen := 0
	encs := make([]tokenizer.Encoding, n)
	for i, t := range texts {
		enc, err := a.tok.Encode(t, true, 512)
		if err != nil {
			return nil, fmt.Errorf("tokenize[%d]: %w", i, err)
		}
		encs[i] = enc
		if len(enc.IDs) > maxLen {
			maxLen = len(enc.IDs)
		}
	}

	// Create padded int64 tensors
	shape := []int{n, maxLen}
	ids, _ := tensor.NewTensorCPU(shape, tensor.Int64)
	defer ids.Free()
	mask, _ := tensor.NewTensorCPU(shape, tensor.Int64)
	defer mask.Free()

	idp := (*[1 << 20]int64)(ids.DataPtr())[:n*maxLen]
	mp := (*[1 << 20]int64)(mask.DataPtr())[:n*maxLen]
	for i, enc := range encs {
		for j := 0; j < maxLen; j++ {
			if j < len(enc.IDs) {
				idp[i*maxLen+j] = int64(enc.IDs[j])
				mp[i*maxLen+j] = int64(enc.AttentionMask[j])
			}
		}
	}

	// Run TorchScript model: forward(input_ids, attention_mask) → [N, dim]
	outputs, err := a.sess.Run([]*tensor.Tensor{ids, mask})
	if err != nil {
		return nil, fmt.Errorf("torch run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Free()
		}
	}()

	if len(outputs) == 0 {
		return nil, errors.New("no output from torch model")
	}

	// Output is already [N, dim] with mean pooling + L2 norm done in the model
	out := outputs[0]
	outShape := out.Shape()
	if len(outShape) < 2 {
		return nil, fmt.Errorf("unexpected output shape: %v", outShape)
	}
	dim := outShape[len(outShape)-1]

	totalFloats := 1
	for _, s := range outShape { totalFloats *= s }
	data := (*[1 << 25]float32)(out.DataPtr())[:totalFloats]

	result := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		copy(vec, data[i*dim:(i+1)*dim])

		// Verify L2 norm (model should output normalized vectors)
		var norm float32
		for _, v := range vec {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 && (norm < 0.99 || norm > 1.01) {
			// Re-normalize if model didn't
			for j := range vec {
				vec[j] /= norm
			}
		}
		result[i] = vec
	}

	return result, nil
}
