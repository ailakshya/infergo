package llm

/*
#include "infer_api.h"
#include <stdlib.h>
*/
import "C"

import "unsafe"

// TOONGrammar returns the GBNF grammar for TOON format from C++.
func TOONGrammar() string {
	return C.GoString(C.infer_toon_grammar())
}

// TOONToJSON converts a TOON string to JSON using the C++ parser.
func TOONToJSON(toon string) string {
	if len(toon) == 0 {
		return "{}"
	}
	cToon := C.CString(toon)
	defer C.free(unsafe.Pointer(cToon))

	// Allocate output buffer (TOON→JSON expands ~2x)
	maxOut := C.int(len(toon)*3 + 64)
	buf := (*C.char)(C.malloc(C.size_t(maxOut)))
	defer C.free(unsafe.Pointer(buf))

	n := C.infer_toon_to_json(cToon, C.int(len(toon)), buf, maxOut)
	if n < 0 {
		return "{}"
	}
	return C.GoStringN(buf, n)
}

// JSONToTOON converts a JSON string to TOON using the C++ converter.
func JSONToTOON(json string) string {
	if len(json) == 0 {
		return ""
	}
	cJSON := C.CString(json)
	defer C.free(unsafe.Pointer(cJSON))

	maxOut := C.int(len(json) + 64)
	buf := (*C.char)(C.malloc(C.size_t(maxOut)))
	defer C.free(unsafe.Pointer(buf))

	n := C.infer_json_to_toon(cJSON, C.int(len(json)), buf, maxOut)
	if n < 0 {
		return ""
	}
	return C.GoStringN(buf, n)
}
