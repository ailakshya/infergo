# cpp/video/CMakeLists.txt
# Hardware-accelerated video decode/encode via FFmpeg + NVDEC/NVENC.

# Set FFmpeg paths directly (vcpkg toolchain overrides find_library)
set(AVCODEC_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(AVCODEC_LIB "/usr/lib/x86_64-linux-gnu/libavcodec.so")
set(AVFORMAT_LIB "/usr/lib/x86_64-linux-gnu/libavformat.so")
set(AVUTIL_LIB "/usr/lib/x86_64-linux-gnu/libavutil.so")
set(SWSCALE_LIB "/usr/lib/x86_64-linux-gnu/libswscale.so")

message(STATUS "FFmpeg avcodec: ${AVCODEC_LIB}")
message(STATUS "FFmpeg avformat: ${AVFORMAT_LIB}")

add_library(infer_video STATIC decoder.cpp encoder.cpp frame_annotator.cpp)

target_include_directories(infer_video
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC  ${AVCODEC_INCLUDE_DIR}
)

target_link_libraries(infer_video PUBLIC ${AVCODEC_LIB} ${AVFORMAT_LIB} ${AVUTIL_LIB} ${SWSCALE_LIB})

# TurboJPEG for fast JPEG encoding in frame_annotator
find_library(TURBOJPEG_LIB turbojpeg)
if(TURBOJPEG_LIB)
    target_link_libraries(infer_video PRIVATE ${TURBOJPEG_LIB})
    target_compile_definitions(infer_video PRIVATE INFER_TURBOJPEG_AVAILABLE=1)
    message(STATUS "TurboJPEG found: ${TURBOJPEG_LIB}")
else()
    message(STATUS "TurboJPEG not found — frame_annotator JPEG encoding disabled")
endif()

# OpenCV for high-quality resize in frame_annotator
if(OpenCV_FOUND)
    target_link_libraries(infer_video PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(infer_video PRIVATE INFER_OPENCV_AVAILABLE=1)
    target_include_directories(infer_video PRIVATE ${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV available for frame_annotator resize")
endif()

# NVDEC/NVENC runtime libs
if(INFER_CUDA_AVAILABLE)
    if(EXISTS "/usr/lib/x86_64-linux-gnu/libnvcuvid.so")
        target_link_libraries(infer_video PRIVATE "/usr/lib/x86_64-linux-gnu/libnvcuvid.so")
        message(STATUS "nvcuvid found")
    endif()
    if(EXISTS "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so")
        target_link_libraries(infer_video PRIVATE "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so")
        message(STATUS "nvidia-encode found")
    endif()
endif()

# ─── video_test (GTest) ─────────────────────────────────────────────────────
if(TARGET GTest::gtest_main)
    add_executable(video_test decoder_test.cpp)
    target_link_libraries(video_test PRIVATE infer_video GTest::gtest_main)
    if(INFER_CUDA_AVAILABLE)
        target_compile_definitions(video_test PRIVATE INFER_CUDA_AVAILABLE=1)
    endif()
    include(GoogleTest)
    gtest_discover_tests(video_test)
endif()

# ─── frame_annotator_test (GTest) ──────────────────────────────────────────
if(TARGET GTest::gtest_main)
    add_executable(frame_annotator_test frame_annotator_test.cpp)
    target_link_libraries(frame_annotator_test PRIVATE infer_video GTest::gtest_main)
    if(TURBOJPEG_LIB)
        target_compile_definitions(frame_annotator_test PRIVATE INFER_TURBOJPEG_AVAILABLE=1)
    endif()
    if(OpenCV_FOUND)
        target_compile_definitions(frame_annotator_test PRIVATE INFER_OPENCV_AVAILABLE=1)
    endif()
    include(GoogleTest)
    gtest_discover_tests(frame_annotator_test)
endif()
