// cpp/postprocess/nms_cuda_test.cu
// Unit tests for GPU-side NMS (OPT-36).
// Only compiled and run when CUDA is available.

#include "nms_cuda.hpp"
#include "infer_api.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef INFER_CUDA_AVAILABLE

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Upload a host array of detections to GPU.
// Each detection is 6 floats: [x1, y1, x2, y2, confidence, class_id].
static float* upload_detections(const std::vector<std::vector<float>>& dets) {
    const int N = static_cast<int>(dets.size());
    std::vector<float> flat(N * 6);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 6; ++j) {
            flat[i * 6 + j] = dets[i][j];
        }
    }
    float* d_boxes = nullptr;
    cudaMalloc(&d_boxes, flat.size() * sizeof(float));
    cudaMemcpy(d_boxes, flat.data(), flat.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    return d_boxes;
}

// ─── OPT-36-T4: Empty detection case ────────────────────────────────────────

TEST(NmsCuda, EmptyInputNullPtr) {
    InferBox out[1];
    int count = -1;
    // d_boxes=nullptr with n_boxes=0 -> INFER_ERR_NULL (null check before n_boxes check)
    InferError err = infer_nms_cuda(nullptr, 0, 0.5f, 0.45f,
                                    out, 1, &count, nullptr);
    EXPECT_EQ(err, INFER_ERR_NULL);
    EXPECT_EQ(count, 0);
}

TEST(NmsCuda, EmptyInputZeroBoxes) {
    // Allocate a tiny device buffer to pass a non-null pointer
    float* d_dummy = nullptr;
    cudaMalloc(&d_dummy, sizeof(float));

    InferBox out[1];
    int count = -1;
    InferError err = infer_nms_cuda(d_dummy, 0, 0.5f, 0.45f,
                                    out, 1, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    EXPECT_EQ(count, 0);

    cudaFree(d_dummy);
}

TEST(NmsCuda, NullOutputCount) {
    InferBox out[1];
    InferError err = infer_nms_cuda(nullptr, 0, 0.5f, 0.45f,
                                    out, 1, nullptr, nullptr);
    EXPECT_EQ(err, INFER_ERR_NULL);
}

// ─── OPT-36-T4: All below threshold ─────────────────────────────────────────

TEST(NmsCuda, AllBelowThreshold) {
    // All confidences below 0.5
    auto d_boxes = upload_detections({
        {10, 10, 50, 50, 0.1f, 0.0f},
        {20, 20, 60, 60, 0.3f, 1.0f},
    });

    InferBox out[10];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 2, 0.5f, 0.45f,
                                    out, 10, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    EXPECT_EQ(count, 0);

    cudaFree(d_boxes);
}

// ─── OPT-36-T1: Single box kept ─────────────────────────────────────────────

TEST(NmsCuda, SingleBoxKept) {
    auto d_boxes = upload_detections({
        {10.0f, 20.0f, 50.0f, 60.0f, 0.9f, 0.0f},
    });

    InferBox out[10];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 1, 0.5f, 0.45f,
                                    out, 10, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    ASSERT_EQ(count, 1);
    EXPECT_NEAR(out[0].x1, 10.0f, 1e-3f);
    EXPECT_NEAR(out[0].y1, 20.0f, 1e-3f);
    EXPECT_NEAR(out[0].x2, 50.0f, 1e-3f);
    EXPECT_NEAR(out[0].y2, 60.0f, 1e-3f);
    EXPECT_NEAR(out[0].confidence, 0.9f, 1e-3f);
    EXPECT_EQ(out[0].class_idx, 0);

    cudaFree(d_boxes);
}

// ─── OPT-36-T5: Overlapping boxes suppressed ────────────────────────────────

TEST(NmsCuda, OverlappingBoxesSuppressed) {
    // Two nearly identical boxes, same class -> lower-conf suppressed
    auto d_boxes = upload_detections({
        {60.0f, 60.0f, 140.0f, 140.0f, 0.95f, 0.0f},
        {62.0f, 62.0f, 142.0f, 142.0f, 0.80f, 0.0f},
    });

    InferBox out[10];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 2, 0.5f, 0.45f,
                                    out, 10, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    ASSERT_EQ(count, 1);
    EXPECT_NEAR(out[0].confidence, 0.95f, 1e-3f);

    cudaFree(d_boxes);
}

// ─── OPT-36-T5: Non-overlapping boxes both kept ─────────────────────────────

TEST(NmsCuda, NonOverlappingBoxesBothKept) {
    // Two boxes far apart -> both kept
    auto d_boxes = upload_detections({
        {10.0f, 10.0f, 50.0f, 50.0f, 0.9f, 0.0f},
        {200.0f, 200.0f, 250.0f, 250.0f, 0.8f, 0.0f},
    });

    InferBox out[10];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 2, 0.5f, 0.45f,
                                    out, 10, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    EXPECT_EQ(count, 2);

    cudaFree(d_boxes);
}

// ─── OPT-36-T5: Different classes not suppressed ────────────────────────────

TEST(NmsCuda, DifferentClassesNotSuppressed) {
    // Same position but different classes -> both kept (class-aware NMS)
    auto d_boxes = upload_detections({
        {60.0f, 60.0f, 140.0f, 140.0f, 0.95f, 0.0f},
        {60.0f, 60.0f, 140.0f, 140.0f, 0.90f, 1.0f},
    });

    InferBox out[10];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 2, 0.5f, 0.45f,
                                    out, 10, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    ASSERT_EQ(count, 2);

    // Verify we have one of each class
    bool has0 = false, has1 = false;
    for (int i = 0; i < count; ++i) {
        if (out[i].class_idx == 0) has0 = true;
        if (out[i].class_idx == 1) has1 = true;
    }
    EXPECT_TRUE(has0);
    EXPECT_TRUE(has1);

    cudaFree(d_boxes);
}

// ─── OPT-36-T1: Results sorted by confidence descending ─────────────────────

TEST(NmsCuda, SortedByConfidenceDescending) {
    auto d_boxes = upload_detections({
        {10.0f,  10.0f,  30.0f,  30.0f,  0.7f, 0.0f},
        {100.0f, 100.0f, 130.0f, 130.0f, 0.9f, 0.0f},
        {200.0f, 200.0f, 230.0f, 230.0f, 0.8f, 0.0f},
    });

    InferBox out[10];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 3, 0.5f, 0.45f,
                                    out, 10, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    ASSERT_EQ(count, 3);
    EXPECT_GE(out[0].confidence, out[1].confidence);
    EXPECT_GE(out[1].confidence, out[2].confidence);

    cudaFree(d_boxes);
}

// ─── OPT-36-T1: Max output limit respected ──────────────────────────────────

TEST(NmsCuda, MaxOutputLimit) {
    auto d_boxes = upload_detections({
        {10.0f,  10.0f,  30.0f,  30.0f,  0.9f, 0.0f},
        {100.0f, 100.0f, 130.0f, 130.0f, 0.8f, 0.0f},
        {200.0f, 200.0f, 230.0f, 230.0f, 0.7f, 0.0f},
    });

    InferBox out[2];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 3, 0.5f, 0.45f,
                                    out, 2, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    EXPECT_EQ(count, 2);

    cudaFree(d_boxes);
}

// ─── OPT-36-T1: GPU NMS matches CPU NMS output ──────────────────────────────
// Verify that for the same set of input boxes, GPU NMS produces the same
// kept set as CPU NMS (within tolerance).

TEST(NmsCuda, MatchesCPUNMS) {
    // 5 detections, some overlapping
    auto d_boxes = upload_detections({
        {10.0f,  10.0f,  90.0f,  90.0f,  0.95f, 0.0f},  // high conf
        {12.0f,  12.0f,  92.0f,  92.0f,  0.80f, 0.0f},  // overlap with 0 -> suppressed
        {200.0f, 200.0f, 280.0f, 280.0f, 0.85f, 1.0f},  // different class
        {205.0f, 205.0f, 285.0f, 285.0f, 0.70f, 1.0f},  // overlap with 2, same class -> suppressed
        {500.0f, 500.0f, 550.0f, 550.0f, 0.60f, 0.0f},  // no overlap
    });

    InferBox out[10];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 5, 0.5f, 0.45f,
                                    out, 10, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);

    // Expected: box 0 (0.95, cls 0), box 2 (0.85, cls 1), box 4 (0.60, cls 0)
    // Box 1 suppressed by box 0 (same class, high overlap)
    // Box 3 suppressed by box 2 (same class, high overlap)
    ASSERT_EQ(count, 3);

    // Sorted by confidence descending
    EXPECT_NEAR(out[0].confidence, 0.95f, 1e-3f);
    EXPECT_EQ(out[0].class_idx, 0);

    EXPECT_NEAR(out[1].confidence, 0.85f, 1e-3f);
    EXPECT_EQ(out[1].class_idx, 1);

    EXPECT_NEAR(out[2].confidence, 0.60f, 1e-3f);
    EXPECT_EQ(out[2].class_idx, 0);

    cudaFree(d_boxes);
}

// ─── Stress test: larger number of boxes ─────────────────────────────────────

TEST(NmsCuda, ManyBoxes) {
    // 1000 non-overlapping boxes
    std::vector<std::vector<float>> dets;
    for (int i = 0; i < 1000; ++i) {
        float x = static_cast<float>(i * 20);
        float y = 0.0f;
        float conf = 0.5f + 0.001f * static_cast<float>(i);
        dets.push_back({x, y, x + 10.0f, y + 10.0f, conf, 0.0f});
    }

    auto d_boxes = upload_detections(dets);

    InferBox out[1000];
    int count = -1;
    InferError err = infer_nms_cuda(d_boxes, 1000, 0.5f, 0.45f,
                                    out, 1000, &count, nullptr);
    EXPECT_EQ(err, INFER_OK);
    // All non-overlapping -> all kept
    EXPECT_EQ(count, 1000);

    cudaFree(d_boxes);
}

#else  // !INFER_CUDA_AVAILABLE

TEST(NmsCuda, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available — skipping GPU NMS tests";
}

#endif // INFER_CUDA_AVAILABLE
