# cpp/llm/CMakeLists.txt
# LLM engine — wraps llama.cpp for multi-sequence batch inference.

# ─── libinfer_llm ─────────────────────────────────────────────────────────────

add_library(infer_llm STATIC kv_cache.cpp)

target_include_directories(infer_llm
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
            ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp
)

target_link_libraries(infer_llm PUBLIC llama)

# ─── kv_cache_test ───────────────────────────────────────────────────────────

add_executable(kv_cache_test kv_cache_test.cpp)

target_include_directories(kv_cache_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kv_cache_test
    PRIVATE infer_llm GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(kv_cache_test)
