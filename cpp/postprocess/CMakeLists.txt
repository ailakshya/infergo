# cpp/postprocess/CMakeLists.txt
# Postprocessing library — classify (softmax + top-k), NMS.
# Pure C++ — no external dependencies beyond infer_tensor.

add_library(infer_postprocess STATIC
    classify.cpp
)

target_include_directories(infer_postprocess
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
            ${CMAKE_SOURCE_DIR}/cpp/tensor
            ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(infer_postprocess
    PUBLIC infer_tensor
)

# ─── postprocess_test ─────────────────────────────────────────────────────────

add_executable(postprocess_test
    classify_test.cpp
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
