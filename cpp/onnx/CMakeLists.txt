add_library(infer_onnx SHARED onnx_session.cpp)

target_include_directories(infer_onnx
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp
            ${CMAKE_SOURCE_DIR}/cpp/onnx
)

target_link_libraries(infer_onnx
    PUBLIC  ONNXRuntime::ONNXRuntime
    PRIVATE infer_tensor
)

# Propagate rpath so the .so can find libonnxruntime.so at runtime
set_target_properties(infer_onnx PROPERTIES
    INSTALL_RPATH "${ONNXRuntime_LIBRARIES}"
    BUILD_RPATH   "${CMAKE_SOURCE_DIR}/../onnxruntime/lib"
)

if(INFER_CUDA_AVAILABLE)
    target_compile_definitions(infer_onnx PUBLIC INFER_CUDA_AVAILABLE=1)
endif()

# ─── onnx_test ────────────────────────────────────────────────────────────────

add_executable(onnx_test onnx_test.cpp)

target_include_directories(onnx_test
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp
            ${CMAKE_SOURCE_DIR}/cpp/onnx
)

target_link_libraries(onnx_test
    PRIVATE
        infer_onnx
        infer_tensor
        GTest::gtest_main
)

if(INFER_CUDA_AVAILABLE)
    target_compile_definitions(onnx_test PRIVATE INFER_CUDA_AVAILABLE=1)
endif()

include(GoogleTest)
gtest_discover_tests(onnx_test)
