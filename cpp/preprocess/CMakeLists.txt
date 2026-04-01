# cpp/preprocess/CMakeLists.txt
# Image preprocessing library — decode, letterbox, normalize, batch stack.
# Requires OpenCV 4, found via vcpkg or system install.

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
message(STATUS "OpenCV ${OpenCV_VERSION} found")

# ─── libinfer_preprocess ──────────────────────────────────────────────────────

add_library(infer_preprocess STATIC
    decode_image.cpp
    letterbox.cpp
    normalize.cpp
    stack_batch.cpp
)

target_include_directories(infer_preprocess
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
            ${CMAKE_SOURCE_DIR}/cpp/tensor
            ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(infer_preprocess
    PUBLIC  infer_tensor
    PRIVATE ${OpenCV_LIBS}
)

# ─── preprocess_test ──────────────────────────────────────────────────────────

add_executable(preprocess_test
    decode_image_test.cpp
    letterbox_test.cpp
    normalize_test.cpp
    stack_batch_test.cpp
)

target_include_directories(preprocess_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/cpp/tensor
            ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(preprocess_test
    PRIVATE infer_preprocess GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(preprocess_test)
