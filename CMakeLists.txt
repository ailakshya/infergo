cmake_minimum_required(VERSION 3.20)

# Auto-detect vcpkg toolchain before project() — RULE 16
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file"
    )
endif()

project(infergo
    VERSION 0.1.0
    DESCRIPTION "Production-grade AI inference runtime for Go"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Required for shared libraries linked by CGo — RULE 17
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default build type — RULE 18
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Debug|Release)" FORCE)
endif()

# Global compiler warnings — RULE 14: -Wall -Wextra -Werror, no warnings allowed
# Scoped to CXX only — nvcc does not accept GCC/Clang warning flags
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
)

# Build-type-specific flags — RULE 18
# Scoped to CXX only — nvcc handles its own optimisation flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    )
    # Restrict to CXX link steps — CUDA separable-compilation device-link
    # uses gcc and chokes on -fsanitize if passed unconditionally
    add_link_options($<$<LINK_LANGUAGE:CXX>:-fsanitize=address,undefined>)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
endif()

# Optional CUDA — T-04 requires graceful degradation when unavailable
# CMAKE_CUDA_ARCHITECTURES must be set before enable_language(CUDA).
# Default: "native" (auto-detect from installed GPU).
# Override: cmake -DCMAKE_CUDA_ARCHITECTURES=120
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found — GPU builds enabled (arch=${CMAKE_CUDA_ARCHITECTURES})")
    set(INFER_CUDA_AVAILABLE TRUE)
else()
    message(STATUS "CUDA not found — building CPU-only")
    set(INFER_CUDA_AVAILABLE FALSE)
endif()

# Google Test — installed via vcpkg (RULE 16)
find_package(GTest REQUIRED CONFIG)

# ONNX Runtime — prebuilt release, located by cmake/FindONNXRuntime.cmake
# Override root: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(ONNXRuntime REQUIRED)
message(STATUS "ONNX Runtime ${ONNXRuntime_VERSION} found at ${ONNXRuntime_LIBRARIES}")

enable_testing()

# Libraries — add more subdirectories as tasks progress
add_subdirectory(cpp/tensor)
add_subdirectory(cpp/onnx)
add_subdirectory(cpp/tokenizer)

# ─── llama.cpp (vendor/llama.cpp) ────────────────────────────────────────────
# Build as static library so CGo links a single .a instead of many .so files.
# GGML_CUDA mirrors the presence of CUDA in this build.
# Disable everything we don't need to keep build times short.
# NOTE: must come before cpp/api so infer_llm target exists when api is configured.
if(EXISTS "${CMAKE_SOURCE_DIR}/vendor/llama.cpp/CMakeLists.txt")
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
    if(INFER_CUDA_AVAILABLE)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(vendor/llama.cpp)
    message(STATUS "llama.cpp submodule found — building libllama.a (CUDA=${INFER_CUDA_AVAILABLE})")
    add_subdirectory(cpp/llm)
else()
    message(WARNING "vendor/llama.cpp not found — run: git submodule update --init vendor/llama.cpp")
endif()

# cpp/api depends on infer_llm (if available) — must come after llama.cpp block
add_subdirectory(cpp/api)
