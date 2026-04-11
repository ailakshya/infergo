// cpp/torch/gpu_preprocess_test.cpp
// Tests for the GPU-resident detection pipeline.

#include "gpu_preprocess.hpp"
#include "torch_session.hpp"
#include "../include/infer_api.h"   // InferBox

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Helper: check if a file exists.
bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Helper: read a file into a byte vector.
std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    auto size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// Path to the YOLO model used for end-to-end tests.
constexpr const char* kModelPath = "models/yolo11n.torchscript.pt";

} // anonymous namespace

// ─── TestUploadImage ────────────────────────────────────────────────────────

TEST(GPUPreprocessTest, TestUploadImage) {
    // Create fake 4x3 RGB image
    const int H = 4, W = 3;
    std::vector<uint8_t> pixels(H * W * 3);
    for (int i = 0; i < H * W * 3; ++i) {
        pixels[i] = static_cast<uint8_t>(i % 256);
    }

    auto device = torch::kCPU;
    auto t = infergo::torch_upload_image(pixels.data(), H, W, device);

    // Verify shape [H, W, 3]
    ASSERT_EQ(t.dim(), 3);
    EXPECT_EQ(t.size(0), H);
    EXPECT_EQ(t.size(1), W);
    EXPECT_EQ(t.size(2), 3);

    // Verify dtype
    EXPECT_EQ(t.dtype(), torch::kUInt8);

    // Verify device
    EXPECT_TRUE(t.device().is_cpu());

    // Verify data integrity — first few values
    auto acc = t.accessor<uint8_t, 3>();
    EXPECT_EQ(acc[0][0][0], 0);
    EXPECT_EQ(acc[0][0][1], 1);
    EXPECT_EQ(acc[0][0][2], 2);
    EXPECT_EQ(acc[0][1][0], 3);
}

TEST(GPUPreprocessTest, TestUploadImageCUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    const int H = 480, W = 640;
    std::vector<uint8_t> pixels(H * W * 3, 128);

    auto device = torch::Device(torch::kCUDA, 0);
    auto t = infergo::torch_upload_image(pixels.data(), H, W, device);

    ASSERT_EQ(t.dim(), 3);
    EXPECT_EQ(t.size(0), H);
    EXPECT_EQ(t.size(1), W);
    EXPECT_EQ(t.size(2), 3);
    EXPECT_EQ(t.dtype(), torch::kUInt8);
    EXPECT_TRUE(t.device().is_cuda());
}

TEST(GPUPreprocessTest, TestUploadImageNull) {
    EXPECT_THROW(
        infergo::torch_upload_image(nullptr, 10, 10, torch::kCPU),
        std::invalid_argument);
}

// ─── TestLetterboxGPU ───────────────────────────────────────────────────────

TEST(GPUPreprocessTest, TestLetterboxGPU) {
    // Create a 480x640 uint8 image on CPU
    auto src = torch::full({480, 640, 3}, 200, torch::kUInt8);
    auto result = infergo::torch_letterbox_gpu(src, 640, 640);

    // Verify output shape
    ASSERT_EQ(result.dim(), 3);
    EXPECT_EQ(result.size(0), 640);
    EXPECT_EQ(result.size(1), 640);
    EXPECT_EQ(result.size(2), 3);
    EXPECT_EQ(result.dtype(), torch::kUInt8);

    // For a 480x640 image -> 640x640 letterbox:
    // scale = min(640/640, 640/480) = 1.0 horizontally, ~1.333 vertically => min = 1.0
    // new_w = 640, new_h = 480, pad_x = 0, pad_y = 80
    // Top pad rows (0..79) should be 114 (padding value)
    auto acc = result.accessor<uint8_t, 3>();
    EXPECT_EQ(acc[0][0][0], 114);   // top-left padding pixel
    EXPECT_EQ(acc[79][0][0], 114);  // last row of top padding

    // Bottom pad rows (560..639) should be 114
    EXPECT_EQ(acc[560][0][0], 114);
    EXPECT_EQ(acc[639][0][0], 114);

    // Centre area should contain image data (close to 200)
    uint8_t centre_val = acc[320][320][0];
    EXPECT_GE(centre_val, 190);
    EXPECT_LE(centre_val, 210);
}

TEST(GPUPreprocessTest, TestLetterboxGPUSquare) {
    // Square input: no padding needed
    auto src = torch::full({640, 640, 3}, 100, torch::kUInt8);
    auto result = infergo::torch_letterbox_gpu(src, 640, 640);

    ASSERT_EQ(result.size(0), 640);
    ASSERT_EQ(result.size(1), 640);
    ASSERT_EQ(result.size(2), 3);

    // All pixels should be ~100 (no padding)
    float mean = result.to(torch::kFloat32).mean().item<float>();
    EXPECT_NEAR(mean, 100.0f, 2.0f);
}

TEST(GPUPreprocessTest, TestLetterboxGPUBadShape) {
    auto bad = torch::zeros({10, 10});  // 2D, not 3D
    EXPECT_THROW(
        infergo::torch_letterbox_gpu(bad, 640, 640),
        std::invalid_argument);
}

// ─── TestNormalizeGPU ───────────────────────────────────────────────────────

TEST(GPUPreprocessTest, TestNormalizeGPU) {
    // Create a [640,640,3] uint8 tensor with known values
    auto src = torch::full({640, 640, 3}, 128, torch::kUInt8);
    auto result = infergo::torch_normalize_gpu(src);

    // Verify shape [1,3,640,640]
    ASSERT_EQ(result.dim(), 4);
    EXPECT_EQ(result.size(0), 1);
    EXPECT_EQ(result.size(1), 3);
    EXPECT_EQ(result.size(2), 640);
    EXPECT_EQ(result.size(3), 640);

    // Verify dtype float32
    EXPECT_EQ(result.dtype(), torch::kFloat32);

    // Verify range [0,1] — 128/255 ~ 0.502
    float val = result[0][0][0][0].item<float>();
    EXPECT_NEAR(val, 128.0f / 255.0f, 1e-4f);
}

TEST(GPUPreprocessTest, TestNormalizeGPUZeros) {
    auto src = torch::zeros({100, 100, 3}, torch::kUInt8);
    auto result = infergo::torch_normalize_gpu(src);

    float min_val = result.min().item<float>();
    float max_val = result.max().item<float>();
    EXPECT_FLOAT_EQ(min_val, 0.0f);
    EXPECT_FLOAT_EQ(max_val, 0.0f);
}

TEST(GPUPreprocessTest, TestNormalizeGPUMax) {
    auto src = torch::full({100, 100, 3}, 255, torch::kUInt8);
    auto result = infergo::torch_normalize_gpu(src);

    float max_val = result.max().item<float>();
    EXPECT_NEAR(max_val, 1.0f, 1e-4f);
}

TEST(GPUPreprocessTest, TestNormalizeGPUBadShape) {
    auto bad = torch::zeros({10, 10}, torch::kUInt8);
    EXPECT_THROW(
        infergo::torch_normalize_gpu(bad),
        std::invalid_argument);
}

// ─── TestNMSGPU ─────────────────────────────────────────────────────────────

TEST(GPUPreprocessTest, TestNMSGPU) {
    // Create a mock YOLO output [1,84,8400]
    // Place one confident detection at anchor 0 and a weak one at anchor 1
    auto output = torch::zeros({1, 84, 8400});

    // Anchor 0: cx=320, cy=320, w=100, h=100, class 0 score = 0.95
    output[0][0][0] = 320.0f;  // cx
    output[0][1][0] = 320.0f;  // cy
    output[0][2][0] = 100.0f;  // w
    output[0][3][0] = 100.0f;  // h
    output[0][4][0] = 0.95f;   // class 0 score

    // Anchor 1: overlapping box, should be suppressed
    output[0][0][1] = 325.0f;
    output[0][1][1] = 325.0f;
    output[0][2][1] = 100.0f;
    output[0][3][1] = 100.0f;
    output[0][4][1] = 0.80f;   // class 0 score

    // Anchor 2: non-overlapping detection, different class
    output[0][0][2] = 100.0f;
    output[0][1][2] = 100.0f;
    output[0][2][2] = 50.0f;
    output[0][3][2] = 50.0f;
    output[0][5][2] = 0.90f;   // class 1 score

    auto dets = infergo::torch_nms_gpu(
        output, /*conf_thresh=*/0.5f, /*iou_thresh=*/0.45f,
        /*scale=*/1.0f, /*pad_x=*/0.0f, /*pad_y=*/0.0f,
        /*orig_w=*/640, /*orig_h=*/640);

    // Should get 2 detections: anchor 0 (class 0) and anchor 2 (class 1)
    // Anchor 1 should be suppressed by anchor 0 (same class, high IoU)
    ASSERT_EQ(dets.size(), 2u);

    // Check the high-confidence detection
    bool found_class0 = false, found_class1 = false;
    for (const auto& d : dets) {
        if (d.class_id == 0) {
            EXPECT_NEAR(d.confidence, 0.95f, 0.01f);
            found_class0 = true;
        } else if (d.class_id == 1) {
            EXPECT_NEAR(d.confidence, 0.90f, 0.01f);
            found_class1 = true;
        }
    }
    EXPECT_TRUE(found_class0);
    EXPECT_TRUE(found_class1);
}

TEST(GPUPreprocessTest, TestNMSGPUEmpty) {
    // All scores below threshold -> no detections
    auto output = torch::zeros({1, 84, 8400});
    auto dets = infergo::torch_nms_gpu(
        output, 0.5f, 0.45f,
        1.0f, 0.0f, 0.0f, 640, 640);
    EXPECT_TRUE(dets.empty());
}

TEST(GPUPreprocessTest, TestNMSGPURescale) {
    // Verify box rescaling from letterbox to original coords
    auto output = torch::zeros({1, 84, 8400});

    // Place a detection at (320, 320) with size 100x100 in letterbox space
    output[0][0][0] = 320.0f;
    output[0][1][0] = 320.0f;
    output[0][2][0] = 100.0f;
    output[0][3][0] = 100.0f;
    output[0][4][0] = 0.95f;

    // Simulate letterbox: original 1920x1080, target 640x640
    // scale = min(640/1920, 640/1080) = 0.3333
    // pad_x = (640 - 1920*0.3333)/2 = (640 - 640)/2 = 0
    // pad_y = (640 - 1080*0.3333)/2 = (640 - 360)/2 = 140
    float scale = 640.0f / 1920.0f;
    float pad_x = 0.0f;
    float pad_y = (640.0f - 1080.0f * scale) / 2.0f;

    auto dets = infergo::torch_nms_gpu(
        output, 0.5f, 0.45f,
        scale, pad_x, pad_y, 1920, 1080);

    ASSERT_EQ(dets.size(), 1u);

    // Verify the box was rescaled
    // Original letterbox coords: x1=270, y1=270, x2=370, y2=370
    // Rescaled: x1 = (270 - 0) / 0.3333 = 810
    //           y1 = (270 - 140) / 0.3333 = 390
    auto& d = dets[0];
    EXPECT_NEAR(d.x1, (270.0f - pad_x) / scale, 2.0f);
    EXPECT_NEAR(d.y1, (270.0f - pad_y) / scale, 2.0f);
    EXPECT_NEAR(d.x2, (370.0f - pad_x) / scale, 2.0f);
    EXPECT_NEAR(d.y2, (370.0f - pad_y) / scale, 2.0f);
}

TEST(GPUPreprocessTest, TestNMSGPUBadShape) {
    auto bad = torch::zeros({1, 10, 100});  // Not [1,84,*]
    EXPECT_THROW(
        infergo::torch_nms_gpu(bad, 0.5f, 0.45f, 1.0f, 0.0f, 0.0f, 640, 640),
        std::invalid_argument);
}

// ─── End-to-end tests (require model and test images) ───────────────────────

class GPUDetectTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!file_exists(kModelPath)) {
            GTEST_SKIP() << "Model not found: " << kModelPath;
        }
        // Use CPU for tests; CUDA is tested separately
        sess_ = std::make_unique<infergo::TorchSession>("cpu", 0);
        sess_->load_model(kModelPath);
    }

    std::unique_ptr<infergo::TorchSession> sess_;
};

TEST_F(GPUDetectTest, TestDetectGPUNoDetections) {
    // Random noise JPEG — should produce 0 or very few detections
    // Create a minimal valid JPEG-like input: solid grey image
    // We'll create it via OpenCV (which is linked)
    // Actually, we encode a solid grey image to JPEG in memory
    // Use a 100x100 solid grey pixel block
    const int H = 100, W = 100;
    // Create JPEG data: raw pixel block won't work, we need actual JPEG.
    // Use torch_upload_image -> torch_letterbox_gpu -> torch_normalize_gpu
    // to test the individual pipeline stages, then test detect_gpu with
    // a real JPEG when we have test images.

    // Instead, create a synthetic JPEG using OpenCV
    std::vector<uint8_t> grey_pixels(H * W * 3, 128);
    cv::Mat grey(H, W, CV_8UC3, grey_pixels.data());
    std::vector<uint8_t> jpeg_buf;
    cv::imencode(".jpg", grey, jpeg_buf);

    auto dets = infergo::torch_detect_gpu(
        *sess_, jpeg_buf.data(), static_cast<int>(jpeg_buf.size()),
        0.5f, 0.45f);

    // Solid grey image: expect 0 detections (or very few false positives)
    EXPECT_LE(dets.size(), 2u);
}

TEST_F(GPUDetectTest, TestDetectGPUInvalidJPEG) {
    // Invalid JPEG data should throw
    uint8_t garbage[] = {0x00, 0x01, 0x02, 0x03};
    EXPECT_THROW(
        infergo::torch_detect_gpu(*sess_, garbage, 4, 0.5f, 0.45f),
        std::runtime_error);
}

TEST_F(GPUDetectTest, TestDetectGPUNullData) {
    EXPECT_THROW(
        infergo::torch_detect_gpu(*sess_, nullptr, 0, 0.5f, 0.45f),
        std::invalid_argument);
}

// ─── CUDA end-to-end test ───────────────────────────────────────────────────

TEST(GPUDetectCUDATest, TestDetectGPUCUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    if (!file_exists(kModelPath)) {
        GTEST_SKIP() << "Model not found: " << kModelPath;
    }

    infergo::TorchSession sess("cuda", 0);
    sess.load_model(kModelPath);

    // Create a solid colour JPEG
    const int H = 480, W = 640;
    std::vector<uint8_t> pixels(H * W * 3, 128);
    cv::Mat img(H, W, CV_8UC3, pixels.data());
    std::vector<uint8_t> jpeg_buf;
    cv::imencode(".jpg", img, jpeg_buf);

    auto dets = infergo::torch_detect_gpu(
        sess, jpeg_buf.data(), static_cast<int>(jpeg_buf.size()),
        0.5f, 0.45f);

    // Solid grey: expect few or no detections
    EXPECT_LE(dets.size(), 5u);
}

// ─── Latency benchmark ─────────────────────────────────────────────────────

TEST(GPUDetectBenchmark, TestDetectGPULatency) {
    if (!file_exists(kModelPath)) {
        GTEST_SKIP() << "Model not found: " << kModelPath;
    }

    const std::string provider = torch::cuda::is_available() ? "cuda" : "cpu";
    infergo::TorchSession sess(provider, 0);
    sess.load_model(kModelPath);

    // Create a 640x480 test image as JPEG
    const int H = 480, W = 640;
    std::vector<uint8_t> pixels(H * W * 3, 100);
    cv::Mat img(H, W, CV_8UC3, pixels.data());
    std::vector<uint8_t> jpeg_buf;
    cv::imencode(".jpg", img, jpeg_buf);

    // Warm up
    for (int i = 0; i < 3; ++i) {
        infergo::torch_detect_gpu(
            sess, jpeg_buf.data(), static_cast<int>(jpeg_buf.size()),
            0.5f, 0.45f);
    }

    // Benchmark WITHOUT sync (pipelined — overlaps GPU work)
    constexpr int N = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        infergo::torch_detect_gpu(
            sess, jpeg_buf.data(), static_cast<int>(jpeg_buf.size()),
            0.5f, 0.45f);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms_pipelined = std::chrono::duration<double, std::milli>(end - start).count();

    // Benchmark WITH sync (true per-image latency — same as Go measures)
    if (torch::cuda::is_available()) torch::cuda::synchronize();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        infergo::torch_detect_gpu(
            sess, jpeg_buf.data(), static_cast<int>(jpeg_buf.size()),
            0.5f, 0.45f);
        if (torch::cuda::is_available()) torch::cuda::synchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_synced = std::chrono::duration<double, std::milli>(end - start).count();

    std::printf("[BENCHMARK] torch_detect_gpu PIPELINED (%s): %.1f ms/image (avg over %d)\n",
                provider.c_str(), ms_pipelined / N, N);
    std::printf("[BENCHMARK] torch_detect_gpu SYNCED    (%s): %.1f ms/image (avg over %d)\n",
                provider.c_str(), ms_synced / N, N);
    double ms = ms_synced;

    // Sanity check: should be less than 5000ms per image even on CPU
    EXPECT_LT(ms / N, 5000.0);
}

// ─── Benchmark: zero-copy raw path vs original raw path ────────────────────

TEST(GPUDetectBenchmark, TestDetectRawOptimized) {
    if (!file_exists(kModelPath)) {
        GTEST_SKIP() << "Model not found: " << kModelPath;
    }

    const std::string provider = torch::cuda::is_available() ? "cuda" : "cpu";
    infergo::TorchSession sess(provider, 0);
    sess.load_model(kModelPath);

    // Create a 640x480 test image as raw RGB pixels (no JPEG)
    const int H = 480, W = 640;
    std::vector<uint8_t> rgb(H * W * 3, 100);

    // Warm up both paths
    for (int i = 0; i < 3; ++i) {
        infergo::torch_detect_gpu_raw(sess, rgb.data(), W, H, 0.5f, 0.45f);
    }

    constexpr int N = 50;

    // --- Benchmark original path (returns std::vector<Detection>) ---
    if (torch::cuda::is_available()) torch::cuda::synchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto dets = infergo::torch_detect_gpu_raw(
            sess, rgb.data(), W, H, 0.5f, 0.45f);
        if (torch::cuda::is_available()) torch::cuda::synchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms_original = std::chrono::duration<double, std::milli>(end - start).count();

    // --- Benchmark zero-copy path (writes directly to InferBox buffer) ---
    InferBox out_boxes[100];
    char error_buf[256];

    // Warm up zero-copy path
    for (int i = 0; i < 3; ++i) {
        infergo::torch_detect_gpu_raw_into(
            sess, rgb.data(), W, H, 0.5f, 0.45f,
            out_boxes, 100, error_buf, sizeof(error_buf));
    }

    if (torch::cuda::is_available()) torch::cuda::synchronize();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int count = infergo::torch_detect_gpu_raw_into(
            sess, rgb.data(), W, H, 0.5f, 0.45f,
            out_boxes, 100, error_buf, sizeof(error_buf));
        (void)count;
        if (torch::cuda::is_available()) torch::cuda::synchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_optimized = std::chrono::duration<double, std::milli>(end - start).count();

    std::printf("[BENCHMARK] torch_detect_gpu_raw          (%s): %.2f ms/image (avg over %d)\n",
                provider.c_str(), ms_original / N, N);
    std::printf("[BENCHMARK] torch_detect_gpu_raw_into     (%s): %.2f ms/image (avg over %d)\n",
                provider.c_str(), ms_optimized / N, N);
    std::printf("[BENCHMARK] Improvement: %.2f ms saved (%.1f%%)\n",
                (ms_original - ms_optimized) / N,
                100.0 * (ms_original - ms_optimized) / ms_original);

    // Sanity check: should be less than 5000ms per image
    EXPECT_LT(ms_optimized / N, 5000.0);
}

// ─── Correctness: zero-copy path produces same results as original ─────────

TEST(GPUDetectBenchmark, TestDetectRawIntoCorrectness) {
    if (!file_exists(kModelPath)) {
        GTEST_SKIP() << "Model not found: " << kModelPath;
    }

    const std::string provider = torch::cuda::is_available() ? "cuda" : "cpu";
    infergo::TorchSession sess(provider, 0);
    sess.load_model(kModelPath);

    // Create a test image with some actual content (gradient, not solid grey)
    const int H = 480, W = 640;
    std::vector<uint8_t> rgb(H * W * 3);
    for (int i = 0; i < H * W * 3; ++i) {
        rgb[i] = static_cast<uint8_t>(i % 256);
    }

    // Run original path
    auto dets_original = infergo::torch_detect_gpu_raw(
        sess, rgb.data(), W, H, 0.25f, 0.45f);

    // Run zero-copy path
    InferBox out_boxes[300];
    char error_buf[256] = {0};
    int count = infergo::torch_detect_gpu_raw_into(
        sess, rgb.data(), W, H, 0.25f, 0.45f,
        out_boxes, 300, error_buf, sizeof(error_buf));

    // Both should return the same number of detections
    ASSERT_GE(count, 0) << "Error: " << error_buf;
    ASSERT_EQ(count, static_cast<int>(dets_original.size()));

    // Compare each detection
    for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(out_boxes[i].x1, dets_original[i].x1, 0.01f) << "Mismatch at detection " << i;
        EXPECT_NEAR(out_boxes[i].y1, dets_original[i].y1, 0.01f) << "Mismatch at detection " << i;
        EXPECT_NEAR(out_boxes[i].x2, dets_original[i].x2, 0.01f) << "Mismatch at detection " << i;
        EXPECT_NEAR(out_boxes[i].y2, dets_original[i].y2, 0.01f) << "Mismatch at detection " << i;
        EXPECT_EQ(out_boxes[i].class_idx, dets_original[i].class_id) << "Mismatch at detection " << i;
        EXPECT_NEAR(out_boxes[i].confidence, dets_original[i].confidence, 0.01f) << "Mismatch at detection " << i;
    }
}

// ─── Test: torch_nms_gpu_into produces same results as torch_nms_gpu ───────

TEST(GPUPreprocessTest, TestNMSGPUInto) {
    // Create same mock YOLO output as TestNMSGPU
    auto output = torch::zeros({1, 84, 8400});

    // Anchor 0: cx=320, cy=320, w=100, h=100, class 0 score = 0.95
    output[0][0][0] = 320.0f;
    output[0][1][0] = 320.0f;
    output[0][2][0] = 100.0f;
    output[0][3][0] = 100.0f;
    output[0][4][0] = 0.95f;

    // Anchor 1: overlapping box, should be suppressed
    output[0][0][1] = 325.0f;
    output[0][1][1] = 325.0f;
    output[0][2][1] = 100.0f;
    output[0][3][1] = 100.0f;
    output[0][4][1] = 0.80f;

    // Anchor 2: non-overlapping detection, different class
    output[0][0][2] = 100.0f;
    output[0][1][2] = 100.0f;
    output[0][2][2] = 50.0f;
    output[0][3][2] = 50.0f;
    output[0][5][2] = 0.90f;

    // Run original path
    auto dets = infergo::torch_nms_gpu(
        output, 0.5f, 0.45f, 1.0f, 0.0f, 0.0f, 640, 640);

    // Run zero-copy path
    InferBox out_boxes[10];
    char error_buf[256] = {0};
    int count = infergo::torch_nms_gpu_into(
        output, 0.5f, 0.45f, 1.0f, 0.0f, 0.0f, 640, 640,
        out_boxes, 10, error_buf, sizeof(error_buf));

    ASSERT_GE(count, 0) << "Error: " << error_buf;
    ASSERT_EQ(count, static_cast<int>(dets.size()));
    EXPECT_EQ(count, 2);

    // Both paths should produce the same detections
    for (int i = 0; i < count; ++i) {
        EXPECT_NEAR(out_boxes[i].x1, dets[i].x1, 0.01f);
        EXPECT_NEAR(out_boxes[i].y1, dets[i].y1, 0.01f);
        EXPECT_NEAR(out_boxes[i].x2, dets[i].x2, 0.01f);
        EXPECT_NEAR(out_boxes[i].y2, dets[i].y2, 0.01f);
        EXPECT_EQ(out_boxes[i].class_idx, dets[i].class_id);
        EXPECT_NEAR(out_boxes[i].confidence, dets[i].confidence, 0.01f);
    }
}

// ─── Test: torch_nms_gpu_into error handling ───────────────────────────────

TEST(GPUPreprocessTest, TestNMSGPUIntoBadShape) {
    auto bad = torch::zeros({1, 3, 100});  // Not [1,5+,*]
    InferBox out_boxes[10];
    char error_buf[256] = {0};
    int count = infergo::torch_nms_gpu_into(
        bad, 0.5f, 0.45f, 1.0f, 0.0f, 0.0f, 640, 640,
        out_boxes, 10, error_buf, sizeof(error_buf));
    EXPECT_EQ(count, -1);
    EXPECT_NE(std::string(error_buf).find("expected"), std::string::npos);
}

// ─── Test: torch_nms_gpu_into respects max_boxes limit ─────────────────────

TEST(GPUPreprocessTest, TestNMSGPUIntoMaxBoxes) {
    auto output = torch::zeros({1, 84, 8400});

    // Place 3 non-overlapping detections
    for (int i = 0; i < 3; ++i) {
        float cx = 100.0f + static_cast<float>(i) * 200.0f;
        output[0][0][i] = cx;
        output[0][1][i] = 320.0f;
        output[0][2][i] = 50.0f;
        output[0][3][i] = 50.0f;
        output[0][4][i] = 0.9f - static_cast<float>(i) * 0.1f;
    }

    // Request max_boxes=2, should only get 2
    InferBox out_boxes[2];
    char error_buf[256] = {0};
    int count = infergo::torch_nms_gpu_into(
        output, 0.5f, 0.45f, 1.0f, 0.0f, 0.0f, 640, 640,
        out_boxes, 2, error_buf, sizeof(error_buf));

    EXPECT_EQ(count, 2);
}
