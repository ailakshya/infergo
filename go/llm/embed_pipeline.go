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

// EmbedPipeline runs the full embedding in C++: tokenize → ONNX → pool → normalize.
// One CGo call. Zero Go-side compute.
func EmbedPipeline(session, tokenizer unsafe.Pointer, text string, maxDim int) ([]float32, error) {
	if session == nil || tokenizer == nil {
		return nil, errors.New("llm: EmbedPipeline: nil session or tokenizer")
	}
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	vec := make([]float32, maxDim)
	dim := C.infer_embed_pipeline(
		C.InferSession(session),
		C.InferTokenizer(tokenizer),
		cText,
		(*C.float)(&vec[0]),
		C.int(maxDim),
	)
	if dim < 0 {
		return nil, fmt.Errorf("llm: EmbedPipeline failed: %w", lastError())
	}
	return vec[:int(dim)], nil
}

// EmbedBatchPipeline embeds N texts in one C++ call.
func EmbedBatchPipeline(session, tokenizer unsafe.Pointer, texts []string, maxDim int) ([][]float32, error) {
	if session == nil || tokenizer == nil || len(texts) == 0 {
		return nil, errors.New("llm: EmbedBatchPipeline: invalid args")
	}

	n := len(texts)
	cTexts := make([]*C.char, n)
	for i, t := range texts {
		cTexts[i] = C.CString(t)
	}
	defer func() {
		for _, p := range cTexts {
			C.free(unsafe.Pointer(p))
		}
	}()

	flat := make([]float32, n*maxDim)
	dim := C.infer_embed_batch_pipeline(
		C.InferSession(session),
		C.InferTokenizer(tokenizer),
		(**C.char)(unsafe.Pointer(&cTexts[0])),
		C.int(n),
		(*C.float)(&flat[0]),
		C.int(maxDim),
	)
	if dim < 0 {
		return nil, fmt.Errorf("llm: EmbedBatchPipeline failed: %w", lastError())
	}

	d := int(dim)
	result := make([][]float32, n)
	for i := 0; i < n; i++ {
		result[i] = make([]float32, d)
		copy(result[i], flat[i*maxDim:i*maxDim+d])
	}
	return result, nil
}

// RerankPipeline reranks documents by query — all in C++.
func RerankPipeline(session, tokenizer unsafe.Pointer, query string, documents []string) ([]float32, []int, error) {
	if session == nil || tokenizer == nil || len(documents) == 0 {
		return nil, nil, errors.New("llm: RerankPipeline: invalid args")
	}

	n := len(documents)
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	cDocs := make([]*C.char, n)
	for i, d := range documents {
		cDocs[i] = C.CString(d)
	}
	defer func() {
		for _, p := range cDocs {
			C.free(unsafe.Pointer(p))
		}
	}()

	scores := make([]C.float, n)
	indices := make([]C.int, n)

	rc := C.infer_rerank_pipeline(
		C.InferSession(session),
		C.InferTokenizer(tokenizer),
		cQuery,
		(**C.char)(unsafe.Pointer(&cDocs[0])),
		C.int(n),
		&scores[0],
		&indices[0],
		C.int(n),
	)
	if rc < 0 {
		return nil, nil, fmt.Errorf("llm: RerankPipeline failed")
	}

	outScores := make([]float32, int(rc))
	outIndices := make([]int, int(rc))
	for i := 0; i < int(rc); i++ {
		outScores[i] = float32(scores[i])
		outIndices[i] = int(indices[i])
	}
	return outScores, outIndices, nil
}
