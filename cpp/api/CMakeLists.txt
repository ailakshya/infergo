add_library(infer_api SHARED api.cpp)

target_include_directories(infer_api
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp
)

target_link_libraries(infer_api PRIVATE infer_tensor)

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

include(GoogleTest)
gtest_discover_tests(api_test)
