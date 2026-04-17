// CInfergo shim — re-exports the infergo C API for Swift consumption.
// The actual header lives in the infergo build tree; we include it via
// a search-path flag (-I) set in Package.swift's cSettings.

#ifndef CINFERGO_SHIM_H
#define CINFERGO_SHIM_H

#include "infer_api.h"

#endif // CINFERGO_SHIM_H
