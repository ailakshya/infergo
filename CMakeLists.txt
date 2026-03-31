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

enable_testing()

# Libraries — add more subdirectories as tasks progress
add_subdirectory(cpp/tensor)
add_subdirectory(cpp/api)
