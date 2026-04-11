add_library(infer_api SHARED api.cpp)

target_include_directories(infer_api
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp
)

if(TARGET infer_llm)
    target_include_directories(infer_api PRIVATE
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
    )
    target_link_libraries(infer_api PRIVATE infer_tensor infer_onnx infer_tokenizer infer_llm)
    target_compile_definitions(infer_api PRIVATE INFER_LLM_AVAILABLE=1)
else()
    target_link_libraries(infer_api PRIVATE infer_tensor infer_onnx infer_tokenizer)
endif()

if(TARGET infer_preprocess)
    target_link_libraries(infer_api PRIVATE infer_preprocess)
    target_compile_definitions(infer_api PRIVATE INFER_PREPROCESS_AVAILABLE=1)
endif()

if(TARGET infer_torch)
    target_link_libraries(infer_api PRIVATE infer_torch)
    target_compile_definitions(infer_api PRIVATE INFER_TORCH_AVAILABLE=1)
endif()

target_link_libraries(infer_api PRIVATE infer_postprocess)

if(TARGET infer_video)
    target_link_libraries(infer_api PRIVATE infer_video)
    target_compile_definitions(infer_api PRIVATE INFER_VIDEO_AVAILABLE=1)
    # Propagate OpenCV availability for resized decoder in api.cpp
    find_package(OpenCV QUIET COMPONENTS core imgproc)
    if(OpenCV_FOUND)
        target_include_directories(infer_api PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(infer_api PRIVATE ${OpenCV_LIBS})
        target_compile_definitions(infer_api PRIVATE INFER_OPENCV_AVAILABLE=1)
    endif()
endif()

if(TARGET infer_cuda_monitor)
    target_link_libraries(infer_api PRIVATE infer_cuda_monitor)
endif()

set_target_properties(infer_api PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/tokenizer-rs/target/release"
)

if(INFER_CUDA_AVAILABLE)
    target_compile_definitions(infer_api PUBLIC INFER_CUDA_AVAILABLE=1)
endif()

# ─── api_test ─────────────────────────────────────────────────────────────────

add_executable(api_test api_test.cpp)

target_include_directories(api_test
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp/include
)

target_link_libraries(api_test
    PRIVATE
        infer_api
        GTest::gtest_main
)

if(INFER_CUDA_AVAILABLE)
    target_compile_definitions(api_test PRIVATE INFER_CUDA_AVAILABLE=1)
endif()

if(TARGET infer_llm)
    target_compile_definitions(api_test PRIVATE INFER_LLM_AVAILABLE=1)
endif()

if(TARGET infer_preprocess)
    target_compile_definitions(api_test PRIVATE INFER_PREPROCESS_AVAILABLE=1)
endif()

if(TARGET infer_torch)
    target_compile_definitions(api_test PRIVATE INFER_TORCH_AVAILABLE=1)
endif()

if(TARGET infer_video)
    target_compile_definitions(api_test PRIVATE INFER_VIDEO_AVAILABLE=1)
endif()

include(GoogleTest)
gtest_discover_tests(api_test)
