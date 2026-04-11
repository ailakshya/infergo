// cpp/cuda/vram_monitor.hpp
// VRAM monitoring via CUDA runtime API — no subprocess overhead.
#pragma once

#include <cstddef>

namespace infergo {

// Returns free VRAM in bytes. Returns 0 if CUDA is not available.
size_t cuda_vram_free();

// Returns total VRAM in bytes. Returns 0 if CUDA is not available.
size_t cuda_vram_total();

// Returns VRAM usage as a percentage (0-100). Returns 0 if CUDA not available.
int cuda_vram_used_pct();

} // namespace infergo
