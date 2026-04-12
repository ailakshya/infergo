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

// GenerateC runs the full generation loop in C++ — one CGo call.
// Eliminates all per-token CGo overhead (BatchDecode/SampleToken/AppendToken).
// grammar: GBNF grammar string (empty = no constraint).
// Returns generated text and token count.
func (m *Model) GenerateC(promptTokens []int32, maxTokens int, temperature, topP float32, grammar string) (string, int, error) {
	if m.ptr == nil {
		return "", 0, errors.New("llm: GenerateC on closed model")
	}
	if len(promptTokens) == 0 {
		return "", 0, errors.New("llm: GenerateC: empty prompt")
	}

	cTokens := make([]C.int, len(promptTokens))
	for i, t := range promptTokens {
		cTokens[i] = C.int(t)
	}

	var cGrammar *C.char
	if grammar != "" {
		cGrammar = C.CString(grammar)
		defer C.free(unsafe.Pointer(cGrammar))
	}

	const maxText = 65536
	textBuf := make([]byte, maxText)
	var genTokens C.int

	rc := C.infer_llm_generate(
		m.ptr,
		&cTokens[0], C.int(len(cTokens)),
		C.int(maxTokens),
		C.float(temperature),
		C.float(topP),
		cGrammar,
		nil, nil, // no streaming callback
		(*C.char)(unsafe.Pointer(&textBuf[0])), C.int(maxText),
		&genTokens,
	)
	if rc != 0 {
		return "", 0, fmt.Errorf("llm: GenerateC failed: %w", lastError())
	}

	text := C.GoString((*C.char)(unsafe.Pointer(&textBuf[0])))
	return text, int(genTokens), nil
}
