// cpp/video/decoder_test.cpp
// Tests for VideoDecoder and VideoEncoder.

#include "decoder.hpp"
#include "encoder.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <vector>

static const char* kTestVideo = "/tmp/test_1080p.mp4";
static const char* kTestOutput = "/tmp/test_output.mp4";

static bool file_exists(const char* path) {
    struct stat st{};
    return stat(path, &st) == 0 && st.st_size > 0;
}

// ---------------------------------------------------------------------------
// Decoder tests
// ---------------------------------------------------------------------------

TEST(VideoDecoder, TestOpenFile) {
    if (!file_exists(kTestVideo)) GTEST_SKIP() << "test video not found";

    infergo::VideoDecoder dec(kTestVideo, true);
    ASSERT_TRUE(dec.is_open()) << "decoder failed to open";
    EXPECT_EQ(dec.width(), 1920);
    EXPECT_EQ(dec.height(), 1080);
    EXPECT_GT(dec.fps(), 0.0);
    dec.close();
}

TEST(VideoDecoder, TestDecodeFrames) {
    if (!file_exists(kTestVideo)) GTEST_SKIP() << "test video not found";

    infergo::VideoDecoder dec(kTestVideo, true);
    ASSERT_TRUE(dec.is_open());

    for (int i = 0; i < 30; i++) {
        uint8_t* data = nullptr;
        infergo::FrameInfo info{};
        bool ok = dec.next_frame(&data, &info);
        ASSERT_TRUE(ok) << "failed at frame " << i;
        EXPECT_NE(data, nullptr);
        EXPECT_EQ(info.width, 1920);
        EXPECT_EQ(info.height, 1080);
        EXPECT_EQ(info.frame_number, i);
    }
    dec.close();
}

TEST(VideoDecoder, TestHWAccel) {
    if (!file_exists(kTestVideo)) GTEST_SKIP() << "test video not found";

    infergo::VideoDecoder dec(kTestVideo, true);
    ASSERT_TRUE(dec.is_open());

#ifdef INFER_CUDA_AVAILABLE
    // When CUDA is available, we expect NVDEC to be used.
    EXPECT_TRUE(dec.is_hw_accelerated()) << "NVDEC not active despite CUDA being available";
#else
    // CPU-only build: hw accel should be false.
    EXPECT_FALSE(dec.is_hw_accelerated());
#endif

    // Read one frame to verify decode works either way.
    uint8_t* data = nullptr;
    infergo::FrameInfo info{};
    EXPECT_TRUE(dec.next_frame(&data, &info));
    EXPECT_NE(data, nullptr);
    dec.close();
}

TEST(VideoDecoder, TestDecodeToEnd) {
    if (!file_exists(kTestVideo)) GTEST_SKIP() << "test video not found";

    infergo::VideoDecoder dec(kTestVideo, true);
    ASSERT_TRUE(dec.is_open());

    int count = 0;
    while (true) {
        uint8_t* data = nullptr;
        infergo::FrameInfo info{};
        if (!dec.next_frame(&data, &info)) break;
        count++;
    }
    // 10s @ 30fps = ~300 frames (allow some margin for variable-rate).
    EXPECT_GT(count, 200) << "too few frames decoded";
    EXPECT_LT(count, 500) << "unexpectedly many frames";
    fprintf(stderr, "[TestDecodeToEnd] decoded %d frames total\n", count);
    dec.close();
}

TEST(VideoDecoder, TestCPUFallback) {
    if (!file_exists(kTestVideo)) GTEST_SKIP() << "test video not found";

    // Explicitly request no hw accel.
    infergo::VideoDecoder dec(kTestVideo, false);
    ASSERT_TRUE(dec.is_open());
    EXPECT_FALSE(dec.is_hw_accelerated());

    uint8_t* data = nullptr;
    infergo::FrameInfo info{};
    EXPECT_TRUE(dec.next_frame(&data, &info));
    EXPECT_NE(data, nullptr);
    dec.close();
}

// ---------------------------------------------------------------------------
// Encoder tests
// ---------------------------------------------------------------------------

TEST(VideoEncoder, TestEncodeFile) {
    if (!file_exists(kTestVideo)) GTEST_SKIP() << "test video not found";

    // Decode 30 frames, then re-encode them.
    infergo::VideoDecoder dec(kTestVideo, true);
    ASSERT_TRUE(dec.is_open());

    int w = dec.width();
    int h = dec.height();
    int fps = static_cast<int>(dec.fps());
    if (fps <= 0) fps = 30;

    // Remove old output if it exists.
    std::remove(kTestOutput);

    infergo::VideoEncoder enc(kTestOutput, w, h, fps, "h264_nvenc");
    ASSERT_TRUE(enc.is_open()) << "encoder failed to open";

    int written = 0;
    for (int i = 0; i < 30; i++) {
        uint8_t* data = nullptr;
        infergo::FrameInfo info{};
        if (!dec.next_frame(&data, &info)) break;
        ASSERT_TRUE(enc.write_frame(data)) << "encode failed at frame " << i;
        written++;
    }
    EXPECT_EQ(written, 30);

    enc.close();
    dec.close();

    EXPECT_TRUE(file_exists(kTestOutput)) << "output file not created";
    fprintf(stderr, "[TestEncodeFile] wrote %d frames to %s (hw=%s)\n",
            written, kTestOutput, enc.is_hw_accelerated() ? "NVENC" : "CPU");
}

TEST(VideoEncoder, TestEncodeReopenDecode) {
    if (!file_exists(kTestOutput)) GTEST_SKIP() << "output from TestEncodeFile not found";

    // Verify the encoded file is decodable.
    infergo::VideoDecoder dec(kTestOutput, false);
    ASSERT_TRUE(dec.is_open());
    EXPECT_EQ(dec.width(), 1920);
    EXPECT_EQ(dec.height(), 1080);

    int count = 0;
    while (true) {
        uint8_t* data = nullptr;
        infergo::FrameInfo info{};
        if (!dec.next_frame(&data, &info)) break;
        count++;
    }
    EXPECT_GE(count, 28) << "encoded file has too few decodable frames";
    fprintf(stderr, "[TestEncodeReopenDecode] decoded %d frames from output\n", count);
    dec.close();
}

// ---------------------------------------------------------------------------
// Benchmarks (GTest style — print throughput manually)
// ---------------------------------------------------------------------------

TEST(VideoDecoder, BenchmarkDecode1080p) {
    if (!file_exists(kTestVideo)) GTEST_SKIP() << "test video not found";

    infergo::VideoDecoder dec(kTestVideo, true);
    ASSERT_TRUE(dec.is_open());

    auto start = std::chrono::steady_clock::now();
    int count = 0;
    while (true) {
        uint8_t* data = nullptr;
        infergo::FrameInfo info{};
        if (!dec.next_frame(&data, &info)) break;
        count++;
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();
    double fps_val = (elapsed_s > 0.0) ? static_cast<double>(count) / elapsed_s : 0.0;

    fprintf(stderr, "[BenchmarkDecode1080p] %d frames in %.3f s = %.1f FPS (hw=%s)\n",
            count, elapsed_s, fps_val, dec.is_hw_accelerated() ? "NVDEC" : "CPU");
    dec.close();
}

TEST(VideoEncoder, BenchmarkEncode1080p) {
    if (!file_exists(kTestVideo)) GTEST_SKIP() << "test video not found";

    // Pre-decode all frames into memory.
    infergo::VideoDecoder dec(kTestVideo, true);
    ASSERT_TRUE(dec.is_open());

    int w = dec.width();
    int h = dec.height();
    int fps = static_cast<int>(dec.fps());
    if (fps <= 0) fps = 30;
    int frame_bytes = w * h * 3;

    std::vector<std::vector<uint8_t>> frames;
    while (true) {
        uint8_t* data = nullptr;
        infergo::FrameInfo info{};
        if (!dec.next_frame(&data, &info)) break;
        frames.emplace_back(data, data + frame_bytes);
    }
    dec.close();
    ASSERT_GT(frames.size(), 0u);

    const char* bench_output = "/tmp/test_bench_encode.mp4";
    std::remove(bench_output);

    infergo::VideoEncoder enc(bench_output, w, h, fps, "h264_nvenc");
    ASSERT_TRUE(enc.is_open());

    auto start = std::chrono::steady_clock::now();
    for (auto& f : frames) {
        enc.write_frame(f.data());
    }
    enc.close();
    auto end = std::chrono::steady_clock::now();

    double elapsed_s = std::chrono::duration<double>(end - start).count();
    double fps_val = (elapsed_s > 0.0) ? static_cast<double>(frames.size()) / elapsed_s : 0.0;

    fprintf(stderr, "[BenchmarkEncode1080p] %zu frames in %.3f s = %.1f FPS (hw=%s)\n",
            frames.size(), elapsed_s, fps_val, enc.is_hw_accelerated() ? "NVENC" : "CPU");

    std::remove(bench_output);
}
