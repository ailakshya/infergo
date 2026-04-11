package llm

/*
#include "infer_api.h"
#include <stdlib.h>

// Go callback trampoline — defined in speculative_cb.go via //export
extern int goSpecTokenCallback(int token, const char* piece, void* user_data);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// SpeculativeDecoder wraps a C++ speculative decoding engine.
// It loads a draft model and runs the full draft-verify-accept loop in C++.
type SpeculativeDecoder struct {
	ptr C.InferSpeculative
}

// SpeculativeStats holds generation statistics.
type SpeculativeStats struct {
	Predicted int // total tokens generated
	Drafted   int // total tokens drafted
	Accepted  int // draft tokens accepted by target
}

// AcceptRate returns the fraction of drafted tokens that were accepted.
func (s SpeculativeStats) AcceptRate() float64 {
	if s.Drafted == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Drafted)
}

// Speedup returns the effective speedup ratio.
// Each speculative step produces 1 + accept_rate tokens on average.
func (s SpeculativeStats) Speedup() float64 {
	if s.Predicted == 0 || s.Drafted == 0 {
		return 1.0
	}
	return 1.0 + s.AcceptRate()
}

// NewSpeculativeDecoder creates a speculative decoder with a draft model.
// draftPath: path to the draft GGUF model (must share vocab with target).
// nGPULayers: GPU layers for the draft model.
// nDraft: number of tokens to draft per step.
func NewSpeculativeDecoder(model *Model, draftPath string, nGPULayers, nDraft int) (*SpeculativeDecoder, error) {
	if model == nil || model.ptr == nil {
		return nil, errors.New("llm: NewSpeculativeDecoder: nil model")
	}
	if draftPath == "" {
		return nil, errors.New("llm: NewSpeculativeDecoder: empty draft path")
	}
	if nDraft <= 0 {
		nDraft = 5
	}

	cPath := C.CString(draftPath)
	defer C.free(unsafe.Pointer(cPath))

	ptr := C.infer_speculative_create(model.ptr, cPath, C.int(nGPULayers), C.int(nDraft))
	if ptr == nil {
		return nil, fmt.Errorf("llm: speculative create failed: %w", lastError())
	}

	sd := &SpeculativeDecoder{ptr: ptr}
	runtime.SetFinalizer(sd, (*SpeculativeDecoder).Close)
	return sd, nil
}

// Generate runs the full speculative generation loop in C++.
// Returns the generated text, token count, and speculative stats.
func (sd *SpeculativeDecoder) Generate(promptTokens []int32, maxTokens int, temperature float32) (string, SpeculativeStats, error) {
	if sd.ptr == nil {
		return "", SpeculativeStats{}, errors.New("llm: Generate on closed speculative decoder")
	}

	cTokens := make([]C.int, len(promptTokens))
	for i, t := range promptTokens {
		cTokens[i] = C.int(t)
	}

	const maxText = 32768
	textBuf := make([]byte, maxText)

	var nPredict, nDrafted, nAccepted C.int

	rc := C.infer_speculative_generate(
		sd.ptr,
		&cTokens[0], C.int(len(cTokens)),
		C.int(maxTokens),
		C.float(temperature),
		nil, nil, // no streaming callback for sync generate
		(*C.char)(unsafe.Pointer(&textBuf[0])), C.int(maxText),
		&nPredict, &nDrafted, &nAccepted,
	)
	if rc != 0 {
		return "", SpeculativeStats{}, fmt.Errorf("llm: speculative generate failed: %w", lastError())
	}

	text := C.GoString((*C.char)(unsafe.Pointer(&textBuf[0])))
	stats := SpeculativeStats{
		Predicted: int(nPredict),
		Drafted:   int(nDrafted),
		Accepted:  int(nAccepted),
	}
	return text, stats, nil
}

// Close frees the speculative decoder. Safe to call multiple times.
func (sd *SpeculativeDecoder) Close() {
	if sd.ptr == nil {
		return
	}
	C.infer_speculative_free(sd.ptr)
	sd.ptr = nil
	runtime.SetFinalizer(sd, nil)
}
