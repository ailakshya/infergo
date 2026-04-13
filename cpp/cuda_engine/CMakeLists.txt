# Custom CUDA Inference Engine (experimental)
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA not found — skipping cuda_engine")
    return()
endif()

add_library(cuda_engine STATIC kernels.cu engine.cu gguf_loader.cu)

target_include_directories(cuda_engine
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           ${CMAKE_SOURCE_DIR}/cpp/include
)

set_target_properties(cuda_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "120"
)

target_compile_options(cuda_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# Benchmark: custom engine vs llama.cpp
add_executable(cuda_engine_bench bench.cu)

target_include_directories(cuda_engine_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
)

target_link_libraries(cuda_engine_bench
    PRIVATE cuda_engine infer_api cublas cudart
)

set_target_properties(cuda_engine_bench PROPERTIES
    CUDA_ARCHITECTURES "120"
)

# Verify output correctness
add_executable(cuda_engine_verify verify.cu)
target_include_directories(cuda_engine_verify PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cuda_engine_verify PRIVATE cuda_engine infer_api cublas cudart)
set_target_properties(cuda_engine_verify PROPERTIES CUDA_ARCHITECTURES "120")
