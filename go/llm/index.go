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

// VectorIndex wraps an HNSW vector similarity search index.
type VectorIndex struct {
	ptr C.InferIndex
	dim int
}

// NewVectorIndex creates an HNSW index for vectors of the given dimension.
func NewVectorIndex(dim, M, efConstruction int) (*VectorIndex, error) {
	if dim <= 0 {
		return nil, errors.New("llm: dim must be > 0")
	}
	ptr := C.infer_index_create(C.int(dim), C.int(M), C.int(efConstruction))
	if ptr == nil {
		return nil, errors.New("llm: failed to create vector index")
	}
	idx := &VectorIndex{ptr: ptr, dim: dim}
	runtime.SetFinalizer(idx, (*VectorIndex).Close)
	return idx, nil
}

// Insert adds a vector with an ID and optional metadata.
func (idx *VectorIndex) Insert(id int64, vec []float32, metadata string) error {
	if idx.ptr == nil {
		return errors.New("llm: Insert on closed index")
	}
	if len(vec) != idx.dim {
		return errors.New("llm: vector dimension mismatch")
	}
	var cMeta *C.char
	if metadata != "" {
		cMeta = C.CString(metadata)
		defer C.free(unsafe.Pointer(cMeta))
	}
	rc := C.infer_index_insert(idx.ptr, C.int64_t(id), (*C.float)(&vec[0]), cMeta)
	if rc != 0 {
		return errors.New("llm: Insert failed")
	}
	return nil
}

// SearchResult is one search hit.
type SearchResult struct {
	ID       int64
	Distance float32 // cosine distance (0 = identical, 1 = orthogonal)
}

// Search finds the k nearest neighbors to the query vector.
func (idx *VectorIndex) Search(query []float32, k, efSearch int) ([]SearchResult, error) {
	if idx.ptr == nil {
		return nil, errors.New("llm: Search on closed index")
	}
	if len(query) != idx.dim {
		return nil, errors.New("llm: query dimension mismatch")
	}
	if k <= 0 {
		k = 10
	}

	ids := make([]C.int64_t, k)
	dists := make([]C.float, k)

	n := C.infer_index_search(idx.ptr, (*C.float)(&query[0]),
		C.int(k), C.int(efSearch),
		&ids[0], &dists[0], C.int(k))
	if n < 0 {
		return nil, errors.New("llm: Search failed")
	}

	results := make([]SearchResult, int(n))
	for i := 0; i < int(n); i++ {
		results[i] = SearchResult{
			ID:       int64(ids[i]),
			Distance: float32(dists[i]),
		}
	}
	return results, nil
}

// Size returns the number of vectors in the index.
func (idx *VectorIndex) Size() int {
	if idx.ptr == nil {
		return 0
	}
	return int(C.infer_index_size(idx.ptr))
}

// Close frees the index. Safe to call multiple times.
func (idx *VectorIndex) Close() {
	if idx.ptr == nil {
		return
	}
	C.infer_index_free(idx.ptr)
	idx.ptr = nil
	runtime.SetFinalizer(idx, nil)
}
