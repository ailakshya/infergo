set(TENSOR_SOURCES
    tensor.cpp
)

if(INFER_CUDA_AVAILABLE)
    list(APPEND TENSOR_SOURCES tensor_cuda.cu)
    set_source_files_properties(tensor_cuda.cu PROPERTIES LANGUAGE CUDA)
endif()

# libinfer_tensor — shared library, linked by the Go CGo wrapper — RULE 17
add_library(infer_tensor SHARED ${TENSOR_SOURCES})

target_include_directories(infer_tensor
    PUBLIC
        ${CMAKE_SOURCE_DIR}/cpp/include    # exposes infer_api.h to consumers
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}        # internal headers (tensor.hpp)
)

if(INFER_CUDA_AVAILABLE)
    target_link_libraries(infer_tensor PRIVATE CUDA::cudart)
    target_compile_definitions(infer_tensor PUBLIC INFER_CUDA_AVAILABLE=1)

    set_target_properties(infer_tensor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# tensor_test — RULE 15: C++ tests before Go wrapper
add_executable(tensor_test
    tensor_test.cpp
)

target_link_libraries(tensor_test
    PRIVATE
        infer_tensor
        GTest::gtest_main
)

target_include_directories(tensor_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(tensor_test)
