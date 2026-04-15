# cpp/postprocess/CMakeLists.txt
# Postprocessing library — classify (softmax + top-k), NMS.
# Pure C++ — no external dependencies beyond infer_tensor.
# When CUDA is available, also builds the GPU NMS kernel (OPT-36).

add_library(infer_postprocess STATIC
    classify.cpp
    nms.cpp
    normalize_embedding.cpp
)

target_include_directories(infer_postprocess
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
            ${CMAKE_SOURCE_DIR}/cpp/tensor
            ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(infer_postprocess
    PUBLIC infer_tensor
)

# ─── GPU NMS kernel (OPT-36) ─────────────────────────────────────────────────
# Builds only when CUDA is available. Provides infer_nms_cuda() C API.

if(INFER_CUDA_AVAILABLE)
    add_library(infer_nms_cuda STATIC nms_cuda.cu)

    set_source_files_properties(nms_cuda.cu PROPERTIES LANGUAGE CUDA)

    target_include_directories(infer_nms_cuda
        PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
                ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(infer_nms_cuda
        PRIVATE CUDA::cudart
    )

    target_compile_definitions(infer_nms_cuda PUBLIC INFER_CUDA_AVAILABLE=1)

    # PIC required for linking into shared libinfer_api.so
    set_target_properties(infer_nms_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Link GPU NMS into the postprocess library so consumers get it automatically
    target_link_libraries(infer_postprocess PUBLIC infer_nms_cuda)
endif()

# ─── postprocess_test ─────────────────────────────────────────────────────────

add_executable(postprocess_test
    classify_test.cpp
    nms_test.cpp
    normalize_embedding_test.cpp
)

target_include_directories(postprocess_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/cpp/tensor
)

target_link_libraries(postprocess_test
    PRIVATE infer_postprocess GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(postprocess_test)

# ─── nms_cuda_test (OPT-36) ──────────────────────────────────────────────────

if(INFER_CUDA_AVAILABLE)
    add_executable(nms_cuda_test nms_cuda_test.cu)

    set_source_files_properties(nms_cuda_test.cu PROPERTIES LANGUAGE CUDA)

    target_include_directories(nms_cuda_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_SOURCE_DIR}/cpp/include
    )

    target_link_libraries(nms_cuda_test
        PRIVATE infer_nms_cuda CUDA::cudart GTest::gtest_main
    )

    target_compile_definitions(nms_cuda_test PRIVATE INFER_CUDA_AVAILABLE=1)

    gtest_discover_tests(nms_cuda_test)
endif()
