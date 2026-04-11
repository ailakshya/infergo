# cpp/llm/CMakeLists.txt
# LLM engine — wraps llama.cpp for multi-sequence batch inference.

# ─── libinfer_llm ─────────────────────────────────────────────────────────────

add_library(infer_llm STATIC kv_cache.cpp kv_paged.cpp llm_engine.cpp sampler.cpp infer_sequence.cpp speculative.cpp)

target_include_directories(infer_llm
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
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

# ─── kv_paged_test ───────────────────────────────────────────────────────────

add_executable(kv_paged_test kv_paged_test.cpp)

target_include_directories(kv_paged_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kv_paged_test
    PRIVATE infer_llm GTest::gtest_main
)

gtest_discover_tests(kv_paged_test)

# ─── llm_test ─────────────────────────────────────────────────────────────────

add_executable(llm_test llm_test.cpp)

target_include_directories(llm_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
)

target_link_libraries(llm_test
    PRIVATE infer_llm GTest::gtest_main
)

gtest_discover_tests(llm_test)

# ─── sampler_test ─────────────────────────────────────────────────────────────

add_executable(sampler_test sampler_test.cpp)

target_include_directories(sampler_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sampler_test
    PRIVATE infer_llm GTest::gtest_main
)

gtest_discover_tests(sampler_test)

# ─── infer_sequence_test ──────────────────────────────────────────────────────

add_executable(infer_sequence_test infer_sequence_test.cpp)

target_include_directories(infer_sequence_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
)

target_link_libraries(infer_sequence_test
    PRIVATE infer_llm GTest::gtest_main
)

gtest_discover_tests(infer_sequence_test)

# ─── kv_serialize_test (OPT-26) ───────────────────────────────────────────────

add_executable(kv_serialize_test kv_serialize_test.cpp)

target_include_directories(kv_serialize_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
)

target_link_libraries(kv_serialize_test
    PRIVATE infer_llm GTest::gtest_main
)

gtest_discover_tests(kv_serialize_test)
