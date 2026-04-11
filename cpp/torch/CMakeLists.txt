# cpp/torch/CMakeLists.txt
# Builds the infer_torch static library (TorchScript inference backend).
# Only added when find_package(Torch) succeeds in the top-level CMakeLists.

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)

add_library(infer_torch STATIC
    torch_session.cpp
    gpu_preprocess.cpp
)

target_include_directories(infer_torch
    PUBLIC  ${TORCH_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp/tensor
            ${CMAKE_SOURCE_DIR}/cpp
)

target_link_libraries(infer_torch
    PUBLIC  ${TORCH_LIBRARIES}
    PRIVATE infer_tensor
)

# Link OpenCV for JPEG decode in gpu_preprocess.cpp.
# PUBLIC because infer_torch is a static library: its consumers (infer_api)
# must resolve the OpenCV symbols when linking the final shared library.
if(OpenCV_FOUND)
    target_include_directories(infer_torch PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(infer_torch PUBLIC ${OpenCV_LIBS})
    target_compile_definitions(infer_torch PRIVATE INFER_OPENCV_AVAILABLE=1)
endif()

# Suppress warnings from libtorch headers (they are not our code)
target_compile_options(infer_torch PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>
    $<$<COMPILE_LANGUAGE:CXX>:-w>
)

# Propagate torch C++ flags (e.g. -D_GLIBCXX_USE_CXX11_ABI)
set_property(TARGET infer_torch PROPERTY CXX_STANDARD 17)
if(TORCH_CXX_FLAGS)
    target_compile_options(infer_torch PUBLIC ${TORCH_CXX_FLAGS})
endif()

# ─── torch_test ──────────────────────────────────────────────────────────────

add_executable(torch_test torch_session_test.cpp)

target_include_directories(torch_test
    PRIVATE ${CMAKE_SOURCE_DIR}/cpp
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(torch_test
    PRIVATE
        infer_torch
        infer_tensor
        ${TORCH_LIBRARIES}
        GTest::gtest_main
)

# Suppress warnings from libtorch headers in test code
target_compile_options(torch_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>
    $<$<COMPILE_LANGUAGE:CXX>:-w>
)

if(TORCH_CXX_FLAGS)
    target_compile_options(torch_test PRIVATE ${TORCH_CXX_FLAGS})
endif()

include(GoogleTest)
gtest_discover_tests(torch_test)

# ─── gpu_preprocess_test ─────────────────────────────────────────────────────

if(OpenCV_FOUND)
    add_executable(gpu_preprocess_test gpu_preprocess_test.cpp)

    target_include_directories(gpu_preprocess_test
        PRIVATE ${CMAKE_SOURCE_DIR}/cpp
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${TORCH_INCLUDE_DIRS}
                ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(gpu_preprocess_test
        PRIVATE
            infer_torch
            infer_tensor
            ${TORCH_LIBRARIES}
            ${OpenCV_LIBS}
            GTest::gtest_main
    )

    target_compile_options(gpu_preprocess_test PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>
        $<$<COMPILE_LANGUAGE:CXX>:-w>
    )

    if(TORCH_CXX_FLAGS)
        target_compile_options(gpu_preprocess_test PRIVATE ${TORCH_CXX_FLAGS})
    endif()

    gtest_discover_tests(gpu_preprocess_test)
endif()
