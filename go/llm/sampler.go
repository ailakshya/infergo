package llm

/*
#include "infer_api.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Sampler wraps an InferSampler handle — a llama.cpp sampler chain with
// GBNF grammar constraint. Each Sampler maintains grammar state and must
// be used for a single generation sequence. Always call Close when done.
type Sampler struct {
	ptr C.InferSampler
}

// NewGrammarSampler creates a grammar-constrained sampler chain.
// gbnf: the GBNF grammar string (e.g. "root ::= ...").
// root: the root rule name (empty string defaults to "root").
// temperature: sampling temperature (0 = greedy).
// topP: nucleus sampling (1.0 = disabled).
// topK: top-k filter (0 = disabled).
// seed: random seed (0 = random).
func NewGrammarSampler(model *Model, gbnf, root string, temperature, topP float32, topK int, seed uint32) (*Sampler, error) {
	if model == nil || model.ptr == nil {
		return nil, errors.New("llm: NewGrammarSampler called on nil model")
	}
	if gbnf == "" {
		return nil, errors.New("llm: NewGrammarSampler: grammar must not be empty")
	}

	cGbnf := C.CString(gbnf)
	defer C.free(unsafe.Pointer(cGbnf))

	var cRoot *C.char
	if root != "" {
		cRoot = C.CString(root)
		defer C.free(unsafe.Pointer(cRoot))
	}

	ptr := C.infer_sampler_create(
		model.ptr,
		cGbnf,
		cRoot,
		C.float(temperature),
		C.float(topP),
		C.int(topK),
		C.uint32_t(seed),
	)
	if ptr == nil {
		return nil, fmt.Errorf("llm: NewGrammarSampler failed: %w", lastError())
	}

	s := &Sampler{ptr: ptr}
	runtime.SetFinalizer(s, (*Sampler).Close)
	return s, nil
}

// Sample samples one token from raw logits using the grammar-constrained
// sampler chain. The grammar state advances automatically.
func (s *Sampler) Sample(logits []float32) (int32, error) {
	if s.ptr == nil {
		return -1, errors.New("llm: Sample called on closed sampler")
	}
	if len(logits) == 0 {
		return -1, errors.New("llm: Sample: empty logits")
	}

	tok := C.infer_sampler_sample(s.ptr, (*C.float)(&logits[0]), C.int(len(logits)))
	if tok < 0 {
		return -1, fmt.Errorf("llm: Sample failed: %w", lastError())
	}
	return int32(tok), nil
}

// Close frees the sampler chain. Safe to call multiple times.
func (s *Sampler) Close() {
	if s.ptr == nil {
		return
	}
	C.infer_sampler_free(s.ptr)
	s.ptr = nil
	runtime.SetFinalizer(s, nil)
}
