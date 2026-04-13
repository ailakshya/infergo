package llm

/*
#include "infer_api.h"
#include <stdlib.h>
#include <string.h>

// Helper: allocate array of char* in C memory, each pointing to a C buffer.
static char** alloc_string_array(int n, int buf_size) {
    char** arr = (char**)malloc(sizeof(char*) * n);
    for (int i = 0; i < n; i++) {
        arr[i] = (char*)calloc(buf_size, 1);
    }
    return arr;
}

static void free_string_array(char** arr, int n) {
    for (int i = 0; i < n; i++) free(arr[i]);
    free(arr);
}

static char* string_array_get(char** arr, int i) {
    return arr[i];
}
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

	// Flatten all tokens + build offsets (in C memory)
	totalTokens := 0
	for _, r := range requests {
		totalTokens += len(r.PromptTokens)
	}

	cAllTokens := (*C.int)(C.malloc(C.size_t(totalTokens) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cAllTokens))
	cOffsets := (*C.int)(C.malloc(C.size_t(n+1) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cOffsets))

	allTokensSlice := unsafe.Slice(cAllTokens, totalTokens)
	offsetsSlice := unsafe.Slice(cOffsets, n+1)

	idx := 0
	for i, r := range requests {
		offsetsSlice[i] = C.int(idx)
		for _, t := range r.PromptTokens {
			allTokensSlice[idx] = C.int(t)
			idx++
		}
	}
	offsetsSlice[n] = C.int(idx)

	// Max tokens
	maxTok := requests[0].MaxTokens
	if maxTok <= 0 {
		maxTok = 256
	}

	// Grammar (in C memory)
	var cGrammar *C.char
	if grammar != "" {
		cGrammar = C.CString(grammar)
		defer C.free(unsafe.Pointer(cGrammar))
	}

	// Allocate output buffers in C memory (avoids GC moving pointers)
	const maxText = 8192
	cOutTexts := C.alloc_string_array(C.int(n), C.int(maxText))
	defer C.free_string_array(cOutTexts, C.int(n))

	cOutToks := (*C.int)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cOutToks))

	rc := C.infer_llm_generate_batch(
		m.ptr,
		C.int(n),
		cAllTokens,
		cOffsets,
		C.int(maxTok),
		C.float(temperature),
		C.float(topP),
		cGrammar,
		cOutTexts,
		C.int(maxText),
		cOutToks,
	)
	if rc != 0 {
		return nil, fmt.Errorf("llm: GenerateBatch failed: %w", lastError())
	}

	outToksSlice := unsafe.Slice(cOutToks, n)
	results := make([]BatchResult, n)
	for i := range results {
		results[i] = BatchResult{
			Text:      C.GoString(C.string_array_get(cOutTexts, C.int(i))),
			GenTokens: int(outToksSlice[i]),
		}
	}
	return results, nil
}
