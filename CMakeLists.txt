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
add_compile_options(-Wall -Wextra -Werror)

# Build-type-specific flags — RULE 18
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()

# Optional CUDA — T-04 requires graceful degradation when unavailable
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found — GPU builds enabled")
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
