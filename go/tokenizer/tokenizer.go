// Package tokenizer provides a Go wrapper for the infergo tokenizer C API.
// It loads HuggingFace tokenizer.json files via the Rust tokenizers library
// and exposes encode/decode operations for use in inference pipelines.
package tokenizer

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp/api -linfer_api -L${SRCDIR}/../../build/cpp/tokenizer -linfer_tokenizer_cpp -L${SRCDIR}/../../tokenizer-rs/target/release -linfer_tokenizer -Wl,-rpath,${SRCDIR}/../../build/cpp/api -Wl,-rpath,${SRCDIR}/../../build/cpp/tokenizer -Wl,-rpath,${SRCDIR}/../../tokenizer-rs/target/release

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

// Encoding holds the result of a tokenizer encode call.
type Encoding struct {
	IDs           []int32
	AttentionMask []int32
}

// Tokenizer wraps an InferTokenizer (opaque C pointer).
// Always call Close when done to release C resources.
type Tokenizer struct {
	ptr C.InferTokenizer
}

// lastError returns the thread-local C error string as a Go error.
func lastError() error {
	msg := C.infer_last_error_string()
	if msg == nil || C.GoString(msg) == "" {
		return errors.New("unknown C error")
	}
	return errors.New(C.GoString(msg))
}

// Load opens a HuggingFace tokenizer.json file and returns a ready Tokenizer.
// Returns an error if the file is missing or invalid.
func Load(path string) (*Tokenizer, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ptr := C.infer_tokenizer_load(cPath)
	if ptr == nil {
		return nil, fmt.Errorf("tokenizer: load %q failed: %w", path, lastError())
	}
	t := &Tokenizer{ptr: ptr}
	runtime.SetFinalizer(t, (*Tokenizer).Close)
	return t, nil
}

// Close destroys the tokenizer and frees all C resources.
// Safe to call multiple times.
func (t *Tokenizer) Close() {
	if t.ptr == nil {
		return
	}
	C.infer_tokenizer_destroy(t.ptr)
	t.ptr = nil
	runtime.SetFinalizer(t, nil)
}

// Encode tokenizes text and returns token IDs and an attention mask.
// addSpecialTokens: prepend/append BOS/EOS when true.
// maxTokens: hard cap on output length (default 4096 if <= 0).
func (t *Tokenizer) Encode(text string, addSpecialTokens bool, maxTokens int) (Encoding, error) {
	if t.ptr == nil {
		return Encoding{}, errors.New("tokenizer: Encode called on closed tokenizer")
	}
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	ids := make([]C.int, maxTokens)
	mask := make([]C.int, maxTokens)

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	addSp := C.int(0)
	if addSpecialTokens {
		addSp = 1
	}

	n := C.infer_tokenizer_encode(
		t.ptr,
		cText,
		addSp,
		&ids[0],
		&mask[0],
		C.int(maxTokens),
	)
	if n < 0 {
		return Encoding{}, fmt.Errorf("tokenizer: Encode failed: %w", lastError())
	}

	count := int(n)
	enc := Encoding{
		IDs:           make([]int32, count),
		AttentionMask: make([]int32, count),
	}
	for i := 0; i < count; i++ {
		enc.IDs[i] = int32(ids[i])
		enc.AttentionMask[i] = int32(mask[i])
	}
	return enc, nil
}

// Decode converts token IDs back to a string.
// skipSpecialTokens: omit BOS/EOS/PAD tokens from the output when true.
func (t *Tokenizer) Decode(ids []int32, skipSpecialTokens bool) (string, error) {
	if t.ptr == nil {
		return "", errors.New("tokenizer: Decode called on closed tokenizer")
	}
	if len(ids) == 0 {
		return "", nil
	}

	// 8 bytes per token + slack
	bufSize := len(ids)*8 + 64
	buf := make([]C.char, bufSize)

	cIDs := make([]C.int, len(ids))
	for i, id := range ids {
		cIDs[i] = C.int(id)
	}

	skipSp := C.int(0)
	if skipSpecialTokens {
		skipSp = 1
	}

	rc := C.infer_tokenizer_decode(
		t.ptr,
		&cIDs[0],
		C.int(len(cIDs)),
		skipSp,
		&buf[0],
		C.int(bufSize),
	)
	if rc != 0 {
		return "", fmt.Errorf("tokenizer: Decode failed: %w", lastError())
	}
	return C.GoString(&buf[0]), nil
}

// DecodeToken decodes a single token ID to its string piece.
// Useful for streaming token-by-token output.
func (t *Tokenizer) DecodeToken(id int32) (string, error) {
	if t.ptr == nil {
		return "", errors.New("tokenizer: DecodeToken called on closed tokenizer")
	}
	var buf [256]C.char
	rc := C.infer_tokenizer_decode_token(
		t.ptr,
		C.int(id),
		&buf[0],
		C.int(len(buf)),
	)
	if rc != 0 {
		return "", fmt.Errorf("tokenizer: DecodeToken(%d) failed: %w", id, lastError())
	}
	return C.GoString(&buf[0]), nil
}

// VocabSize returns the number of tokens in the vocabulary.
func (t *Tokenizer) VocabSize() int {
	if t.ptr == nil {
		return 0
	}
	return int(C.infer_tokenizer_vocab_size(t.ptr))
}
