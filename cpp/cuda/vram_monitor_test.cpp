// cpp/cuda/vram_monitor_test.cpp
// Unit tests for VRAM monitor — only meaningful on a machine with a GPU.

#include "vram_monitor.hpp"
#include <gtest/gtest.h>

#ifdef INFER_CUDA_AVAILABLE

TEST(VRAMMonitor, TestVRAMFree) {
    size_t free_bytes = infergo::cuda_vram_free();
    EXPECT_GT(free_bytes, 0u) << "Expected free VRAM > 0 on a GPU machine";
}

TEST(VRAMMonitor, TestVRAMTotal) {
    size_t total = infergo::cuda_vram_total();
    size_t free_bytes = infergo::cuda_vram_free();
    EXPECT_GT(total, 0u) << "Expected total VRAM > 0 on a GPU machine";
    EXPECT_GE(total, free_bytes) << "Total VRAM must be >= free VRAM";
}

TEST(VRAMMonitor, TestVRAMUsedPct) {
    int pct = infergo::cuda_vram_used_pct();
    EXPECT_GE(pct, 0);
    EXPECT_LE(pct, 100);
}

TEST(VRAMMonitor, TestVRAMConsistency) {
    size_t total = infergo::cuda_vram_total();
    size_t free_bytes = infergo::cuda_vram_free();
    int pct = infergo::cuda_vram_used_pct();

    ASSERT_GT(total, 0u);

    size_t used = total - free_bytes;
    int expected_pct = static_cast<int>((used * 100) / total);

    // Allow 1% tolerance — VRAM usage can shift between calls.
    EXPECT_NEAR(pct, expected_pct, 1)
        << "used_pct should be consistent with (total - free) / total";
}

#else

TEST(VRAMMonitor, NoCUDA_ReturnsZero) {
    EXPECT_EQ(infergo::cuda_vram_free(), 0u);
    EXPECT_EQ(infergo::cuda_vram_total(), 0u);
    EXPECT_EQ(infergo::cuda_vram_used_pct(), 0);
}

#endif
