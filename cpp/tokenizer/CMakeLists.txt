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

# Imported target for the Rust .so
add_library(infer_tokenizer_rs SHARED IMPORTED)
set_target_properties(infer_tokenizer_rs PROPERTIES
    IMPORTED_LOCATION             "${TOKENIZER_RS_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/cpp/include"
)

# ─── tokenizer_ffi_test ───────────────────────────────────────────────────────

add_executable(tokenizer_ffi_test tokenizer_ffi_test.cpp)

target_include_directories(tokenizer_ffi_test
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp/include
)

target_link_libraries(tokenizer_ffi_test
    PRIVATE
        infer_tokenizer_rs
        GTest::gtest_main
)

# Bake rpath so the test binary finds libinfer_tokenizer.so at runtime
set_target_properties(tokenizer_ffi_test PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/tokenizer-rs/target/release"
)

include(GoogleTest)
gtest_discover_tests(tokenizer_ffi_test)
