# cpp/tokenizer/CMakeLists.txt
# Builds the tokenizer FFI test against libinfer_tokenizer.so (Rust).
#
# The Rust library is built separately from tokenizer-rs/ in the repo root.
# Location: ${CMAKE_SOURCE_DIR}/tokenizer-rs/target/release/libinfer_tokenizer.so
#
# To rebuild the Rust .so:
#   cd tokenizer-rs && cargo build --release

set(TOKENIZER_RS_LIB
    "${CMAKE_SOURCE_DIR}/tokenizer-rs/target/release/libinfer_tokenizer.so"
    CACHE FILEPATH "Path to libinfer_tokenizer.so built from tokenizer-rs/"
)

if(NOT EXISTS "${TOKENIZER_RS_LIB}")
    message(WARNING
        "libinfer_tokenizer.so not found at ${TOKENIZER_RS_LIB}. "
        "Run: cd tokenizer-rs && cargo build --release"
    )
    return()
endif()

set(TOKENIZER_RS_DIR "${CMAKE_SOURCE_DIR}/tokenizer-rs/target/release")

# Imported target for the Rust .so
add_library(infer_tokenizer_rs SHARED IMPORTED GLOBAL)
set_target_properties(infer_tokenizer_rs PROPERTIES
    IMPORTED_LOCATION             "${TOKENIZER_RS_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/cpp/include"
)

# ─── libinfer_tokenizer (C++ wrapper) ────────────────────────────────────────
# OUTPUT_NAME is "infer_tokenizer_cpp" to avoid collision with the Rust
# libinfer_tokenizer.so which lives in a different directory.

add_library(infer_tokenizer SHARED tokenizer.cpp)

set_target_properties(infer_tokenizer PROPERTIES
    OUTPUT_NAME "infer_tokenizer_cpp"
    BUILD_RPATH "${TOKENIZER_RS_DIR}"
)

target_include_directories(infer_tokenizer
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
            ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp
)

target_link_libraries(infer_tokenizer PUBLIC "${TOKENIZER_RS_LIB}")

# ─── tokenizer_ffi_test ───────────────────────────────────────────────────────

add_executable(tokenizer_ffi_test tokenizer_ffi_test.cpp)

target_include_directories(tokenizer_ffi_test
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp/include
)

target_link_libraries(tokenizer_ffi_test
    PRIVATE infer_tokenizer_rs GTest::gtest_main
)

set_target_properties(tokenizer_ffi_test PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/tokenizer-rs/target/release"
)

# ─── tokenizer_test (C++ wrapper tests) ──────────────────────────────────────

add_executable(tokenizer_test tokenizer_test.cpp)

target_include_directories(tokenizer_test
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp/include
            ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tokenizer_test
    PRIVATE infer_tokenizer GTest::gtest_main
)

set_target_properties(tokenizer_test PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/tokenizer-rs/target/release:${CMAKE_CURRENT_BINARY_DIR}"
)

include(GoogleTest)
gtest_discover_tests(tokenizer_ffi_test)
gtest_discover_tests(tokenizer_test)
