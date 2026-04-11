// Package cuda provides Go bindings for CUDA GPU utilities.
// All functions gracefully return zero values when CUDA is unavailable.
package cuda

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp/api -linfer_api -Wl,-rpath,${SRCDIR}/../../build/cpp/api

#include "infer_api.h"
*/
import "C"

// VRAMFree returns free GPU memory in bytes.
// Returns 0 if CUDA is not available.
func VRAMFree() uint64 {
	return uint64(C.infer_cuda_vram_free())
}

// VRAMTotal returns total GPU memory in bytes.
// Returns 0 if CUDA is not available.
func VRAMTotal() uint64 {
	return uint64(C.infer_cuda_vram_total())
}

// VRAMUsedPct returns GPU memory usage as a percentage (0-100).
// Returns 0 if CUDA is not available.
func VRAMUsedPct() int {
	return int(C.infer_cuda_vram_used_pct())
}
