package llm

/*
#include "infer_api.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// BatchRequest represents one request in a batch generation.
type BatchRequest struct {
	PromptTokens []int32
	MaxTokens    int
}

// BatchResult holds the output of one request from batch generation.
type BatchResult struct {
	Text      string
	GenTokens int
}

// GenerateBatch runs N requests simultaneously using continuous batching.
// All sequences share one llama_decode call per step.
func (m *Model) GenerateBatch(requests []BatchRequest, temperature, topP float32, grammar string) ([]BatchResult, error) {
	if m.ptr == nil {
		return nil, errors.New("llm: GenerateBatch on closed model")
	}
	n := len(requests)
	if n == 0 {
		return nil, errors.New("llm: GenerateBatch: no requests")
	}

	// Flatten all tokens + build offsets
	totalTokens := 0
	for _, r := range requests {
		totalTokens += len(r.PromptTokens)
	}

	allTokens := make([]C.int, totalTokens)
	offsets := make([]C.int, n+1)
	idx := 0
	for i, r := range requests {
		offsets[i] = C.int(idx)
		for _, t := range r.PromptTokens {
			allTokens[idx] = C.int(t)
			idx++
		}
	}
	offsets[n] = C.int(idx)

	// Max tokens (use first request's)
	maxTok := requests[0].MaxTokens
	if maxTok <= 0 {
		maxTok = 256
	}

	// Grammar
	var cGrammar *C.char
	if grammar != "" {
		cGrammar = C.CString(grammar)
		defer C.free(unsafe.Pointer(cGrammar))
	}

	// Allocate output buffers
	const maxText = 8192
	outBufs := make([][]byte, n)
	outPtrs := make([]*C.char, n)
	outToks := make([]C.int, n)
	for i := range outBufs {
		outBufs[i] = make([]byte, maxText)
		outPtrs[i] = (*C.char)(unsafe.Pointer(&outBufs[i][0]))
	}

	rc := C.infer_llm_generate_batch(
		m.ptr,
		C.int(n),
		&allTokens[0],
		&offsets[0],
		C.int(maxTok),
		C.float(temperature),
		C.float(topP),
		cGrammar,
		&outPtrs[0],
		C.int(maxText),
		&outToks[0],
	)
	if rc != 0 {
		return nil, fmt.Errorf("llm: GenerateBatch failed: %w", lastError())
	}

	results := make([]BatchResult, n)
	for i := range results {
		results[i] = BatchResult{
			Text:      C.GoString(outPtrs[i]),
			GenTokens: int(outToks[i]),
		}
	}
	return results, nil
}
