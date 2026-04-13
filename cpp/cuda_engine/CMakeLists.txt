# Custom CUDA Inference Engine (experimental)
# Fused kernels for maximum throughput on Blackwell GPUs

if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA not found — skipping cuda_engine")
    return()
endif()

add_library(cuda_engine STATIC kernels.cu engine.cu)

target_include_directories(cuda_engine
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(cuda_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "120"
)

target_compile_options(cuda_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# Benchmark
add_executable(cuda_engine_bench bench.cu)

target_link_libraries(cuda_engine_bench
    PRIVATE cuda_engine cublas cudart
)

set_target_properties(cuda_engine_bench PROPERTIES
    CUDA_ARCHITECTURES "120"
)
