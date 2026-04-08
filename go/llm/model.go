// Package llm provides a Go wrapper for the infergo LLM C API.
// It loads GGUF models via llama.cpp and exposes multi-sequence
// batch inference for use in generation pipelines.
package llm

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp/api -linfer_api -Wl,-rpath,${SRCDIR}/../../build/cpp/api

#include "infer_api.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Model wraps an InferLLM handle (opaque C pointer).
// Always call Close when done to release C resources.
type Model struct {
	ptr       C.InferLLM
	vocabSize int
}

// lastError returns the thread-local C error string as a Go error.
func lastError() error {
	msg := C.infer_last_error_string()
	if msg == nil || C.GoString(msg) == "" {
		return errors.New("unknown C error")
	}
	return errors.New(C.GoString(msg))
}

// Load creates an LLM engine from a GGUF model file.
// nGPULayers: transformer layers to offload to GPU (use 999 for all).
// ctxSize: total KV cache token budget across all sequences.
// nSeqMax: max number of concurrent sequences.
// nBatch: max tokens per decode call.
func Load(path string, nGPULayers, ctxSize, nSeqMax, nBatch int) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ptr := C.infer_llm_create(cPath, C.int(nGPULayers), C.int(ctxSize), C.int(nSeqMax), C.int(nBatch))
	if ptr == nil {
		return nil, fmt.Errorf("llm: load %q failed: %w", path, lastError())
	}
	m := &Model{
		ptr:       ptr,
		vocabSize: int(C.infer_llm_vocab_size(ptr)),
	}
	runtime.SetFinalizer(m, (*Model).Close)
	return m, nil
}

// LoadSplit creates an LLM engine with tensor parallelism across multiple GPUs.
// tensorSplit: fractions per GPU summing to 1.0 (e.g. []float32{0.5, 0.5}); nil = single GPU.
// All other parameters have the same meaning as Load.
func LoadSplit(path string, nGPULayers, ctxSize, nSeqMax, nBatch int, tensorSplit []float32) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var splitPtr *C.float
	nSplit := C.int(0)
	if len(tensorSplit) > 0 {
		splitPtr = (*C.float)(unsafe.Pointer(&tensorSplit[0]))
		nSplit = C.int(len(tensorSplit))
	}

	ptr := C.infer_llm_create_split(cPath, C.int(nGPULayers), C.int(ctxSize), C.int(nSeqMax), C.int(nBatch), splitPtr, nSplit)
	if ptr == nil {
		return nil, fmt.Errorf("llm: load split %q failed: %w", path, lastError())
	}
	m := &Model{
		ptr:       ptr,
		vocabSize: int(C.infer_llm_vocab_size(ptr)),
	}
	runtime.SetFinalizer(m, (*Model).Close)
	return m, nil
}

// LoadPipeline creates an LLM engine with pipeline parallelism across n_stages GPUs.
// Layers are distributed evenly using LLAMA_SPLIT_MODE_LAYER.
// n_stages=1 is equivalent to Load() (single GPU, no split).
func LoadPipeline(path string, nGPULayers, ctxSize, nSeqMax, nBatch, nStages int) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ptr := C.infer_llm_create_pipeline(cPath, C.int(nGPULayers), C.int(ctxSize), C.int(nSeqMax), C.int(nBatch), C.int(nStages))
	if ptr == nil {
		return nil, fmt.Errorf("llm: load pipeline %q failed: %w", path, lastError())
	}
	m := &Model{
		ptr:       ptr,
		vocabSize: int(C.infer_llm_vocab_size(ptr)),
	}
	runtime.SetFinalizer(m, (*Model).Close)
	return m, nil
}

// Close destroys the LLM engine and frees all C resources.
// Safe to call multiple times.
func (m *Model) Close() {
	if m.ptr == nil {
		return
	}
	C.infer_llm_destroy(m.ptr)
	m.ptr = nil
	runtime.SetFinalizer(m, nil)
}

// VocabSize returns the vocabulary size of the loaded model.
func (m *Model) VocabSize() int { return m.vocabSize }

// BOS returns the beginning-of-sequence token ID.
func (m *Model) BOS() int32 {
	if m.ptr == nil {
		return -1
	}
	return int32(C.infer_llm_bos(m.ptr))
}

// EOS returns the end-of-sequence token ID.
func (m *Model) EOS() int32 {
	if m.ptr == nil {
		return -1
	}
	return int32(C.infer_llm_eos(m.ptr))
}

// IsEOG returns true if the token is an end-of-generation token (EOS or EOT).
func (m *Model) IsEOG(token int32) bool {
	if m.ptr == nil {
		return false
	}
	return C.infer_llm_is_eog(m.ptr, C.int(token)) != 0
}

// KVPagesFree returns the number of free KV cache pages.
func (m *Model) KVPagesFree() int {
	if m.ptr == nil {
		return 0
	}
	return int(C.infer_llm_kv_pages_free(m.ptr))
}

// KVPagesTotal returns total KV cache pages (= ctx_size / page_size).
func (m *Model) KVPagesTotal() int {
	if m.ptr == nil {
		return 0
	}
	return int(C.infer_llm_kv_pages_total(m.ptr))
}

// KVPageSize returns the page size in tokens (16).
func (m *Model) KVPageSize() int {
	if m.ptr == nil {
		return 0
	}
	return int(C.infer_llm_kv_page_size(m.ptr))
}

// NewSequence creates a new inference sequence with the given prompt tokens.
// The returned Sequence holds a KV cache slot; always call Close when done.
// Returns an error if the KV slot pool is exhausted.
func (m *Model) NewSequence(tokens []int32) (*Sequence, error) {
	if m.ptr == nil {
		return nil, errors.New("llm: NewSequence called on closed model")
	}
	if len(tokens) == 0 {
		return nil, errors.New("llm: NewSequence requires at least one token")
	}

	cTokens := make([]C.int, len(tokens))
	for i, t := range tokens {
		cTokens[i] = C.int(t)
	}

	ptr := C.infer_seq_create(m.ptr, &cTokens[0], C.int(len(cTokens)))
	if ptr == nil {
		return nil, fmt.Errorf("llm: NewSequence failed: %w", lastError())
	}

	seq := &Sequence{ptr: ptr, model: m}
	runtime.SetFinalizer(seq, (*Sequence).Close)
	return seq, nil
}

// Tokenize converts text to token IDs using the model's built-in vocabulary.
// addBOS: prepend the BOS token when true.
// maxTokens: hard cap on output length (defaults to 4096 if <= 0).
func (m *Model) Tokenize(text string, addBOS bool, maxTokens int) ([]int32, error) {
	if m.ptr == nil {
		return nil, errors.New("llm: Tokenize called on closed model")
	}
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	addBOSInt := C.int(0)
	if addBOS {
		addBOSInt = 1
	}

	ids := make([]C.int, maxTokens)
	n := C.infer_llm_tokenize(m.ptr, cText, addBOSInt, &ids[0], C.int(maxTokens))
	if n < 0 {
		return nil, fmt.Errorf("llm: Tokenize failed: %w", lastError())
	}

	out := make([]int32, int(n))
	for i := range out {
		out[i] = int32(ids[i])
	}
	return out, nil
}

// TokenToPiece converts a single token ID to its string piece.
// Useful for streaming token-by-token output during generation.
func (m *Model) TokenToPiece(id int32) (string, error) {
	if m.ptr == nil {
		return "", errors.New("llm: TokenToPiece called on closed model")
	}
	var buf [256]C.char
	rc := C.infer_llm_token_to_piece(m.ptr, C.int(id), &buf[0], C.int(len(buf)))
	if rc < 0 {
		return "", fmt.Errorf("llm: TokenToPiece(%d) failed: %w", id, lastError())
	}
	return C.GoString(&buf[0]), nil
}

// BatchDecode runs one decode step for all provided sequences in a single
// forward pass through the model. After this call, each sequence's logits
// are available via seq.Logits() or seq.SampleToken().
func (m *Model) BatchDecode(seqs []*Sequence) error {
	if m.ptr == nil {
		return errors.New("llm: BatchDecode called on closed model")
	}
	if len(seqs) == 0 {
		return nil
	}

	cSeqs := make([]C.InferSeq, len(seqs))
	for i, s := range seqs {
		if s == nil || s.ptr == nil {
			return fmt.Errorf("llm: seqs[%d] is nil or closed", i)
		}
		cSeqs[i] = s.ptr
	}

	rc := C.infer_llm_batch_decode(m.ptr, &cSeqs[0], C.int(len(cSeqs)))
	if rc != C.INFER_OK {
		return fmt.Errorf("llm: BatchDecode failed: %w", lastError())
	}
	return nil
}

// SerializeKV serializes the KV cache for a sequence slot to bytes.
// seqID is the slot ID returned by seq.SlotID().
// Returns the serialized bytes, or an error if the context is not loaded or
// the sequence is empty / invalid.
func (m *Model) SerializeKV(seqID int) ([]byte, error) {
	if m.ptr == nil {
		return nil, errors.New("llm: SerializeKV called on closed model")
	}

	// First call: query required size.
	needed := C.infer_llm_kv_serialize(m.ptr, C.int(seqID), nil, 0)
	if needed <= 0 {
		return nil, fmt.Errorf("llm: SerializeKV(%d) size query failed: %w", seqID, lastError())
	}

	// Second call: fill the buffer.
	buf := make([]byte, int(needed))
	written := C.infer_llm_kv_serialize(m.ptr, C.int(seqID),
		(*C.uint8_t)(unsafe.Pointer(&buf[0])), C.int(needed))
	if written <= 0 {
		return nil, fmt.Errorf("llm: SerializeKV(%d) failed: %w", seqID, lastError())
	}

	return buf[:int(written)], nil
}

// DeserializeKV deserializes KV cache bytes into a sequence slot.
// seqID is the destination slot; data must have been produced by SerializeKV.
func (m *Model) DeserializeKV(seqID int, data []byte) error {
	if m.ptr == nil {
		return errors.New("llm: DeserializeKV called on closed model")
	}
	if len(data) == 0 {
		return errors.New("llm: DeserializeKV: data is empty")
	}

	rc := C.infer_llm_kv_deserialize(m.ptr, C.int(seqID),
		(*C.uint8_t)(unsafe.Pointer(&data[0])), C.int(len(data)))
	if rc != 0 {
		return fmt.Errorf("llm: DeserializeKV(%d) failed: %w", seqID, lastError())
	}
	return nil
}
