package llm

/*
#include "infer_api.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// BM25Index wraps a BM25 full-text search index backed by C++.
type BM25Index struct {
	ptr C.InferBM25
}

// NewBM25Index creates a BM25 full-text search index.
// k1: term frequency saturation (typical 1.2).
// b:  document length normalization (typical 0.75).
func NewBM25Index(k1, b float32) (*BM25Index, error) {
	ptr := C.infer_bm25_create(C.float(k1), C.float(b))
	if ptr == nil {
		return nil, errors.New("llm: failed to create BM25 index")
	}
	idx := &BM25Index{ptr: ptr}
	runtime.SetFinalizer(idx, (*BM25Index).Close)
	return idx, nil
}

// Insert adds a document with the given ID and text to the BM25 index.
func (idx *BM25Index) Insert(id int64, text string) error {
	if idx.ptr == nil {
		return errors.New("llm: Insert on closed BM25 index")
	}
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	C.infer_bm25_insert(idx.ptr, C.int64_t(id), cText)
	return nil
}

// Remove deletes a document from the BM25 index.
func (idx *BM25Index) Remove(id int64) error {
	if idx.ptr == nil {
		return errors.New("llm: Remove on closed BM25 index")
	}
	C.infer_bm25_remove(idx.ptr, C.int64_t(id))
	return nil
}

// BM25Result is one BM25 search hit.
type BM25Result struct {
	ID    int64
	Score float32
}

// Search finds the top-k documents matching the query by BM25 score.
func (idx *BM25Index) Search(query string, k int) ([]BM25Result, error) {
	if idx.ptr == nil {
		return nil, errors.New("llm: Search on closed BM25 index")
	}
	if k <= 0 {
		k = 10
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	ids := make([]C.int64_t, k)
	scores := make([]C.float, k)

	n := C.infer_bm25_search(idx.ptr, cQuery, C.int(k),
		&ids[0], &scores[0], C.int(k))
	if n < 0 {
		return nil, errors.New("llm: BM25 Search failed")
	}

	results := make([]BM25Result, int(n))
	for i := 0; i < int(n); i++ {
		results[i] = BM25Result{
			ID:    int64(ids[i]),
			Score: float32(scores[i]),
		}
	}
	return results, nil
}

// Size returns the number of documents in the BM25 index.
func (idx *BM25Index) Size() int {
	if idx.ptr == nil {
		return 0
	}
	return int(C.infer_bm25_size(idx.ptr))
}

// Save persists the BM25 index to a file.
func (idx *BM25Index) Save(path string) error {
	if idx.ptr == nil {
		return errors.New("llm: Save on closed BM25 index")
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if C.infer_bm25_save(idx.ptr, cPath) != 0 {
		return errors.New("llm: BM25 Save failed")
	}
	return nil
}

// Load loads a BM25 index from a file.
func (idx *BM25Index) Load(path string) error {
	if idx.ptr == nil {
		return errors.New("llm: Load on closed BM25 index")
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if C.infer_bm25_load(idx.ptr, cPath) != 0 {
		return errors.New("llm: BM25 Load failed")
	}
	return nil
}

// Close frees the BM25 index. Safe to call multiple times.
func (idx *BM25Index) Close() {
	if idx.ptr == nil {
		return
	}
	C.infer_bm25_free(idx.ptr)
	idx.ptr = nil
	runtime.SetFinalizer(idx, nil)
}
