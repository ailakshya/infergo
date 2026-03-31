#include "kv_cache.hpp"

#include <algorithm>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

using infergo::KVCacheSlotManager;

// ─── Construction ─────────────────────────────────────────────────────────────

TEST(KVCacheSlotManager, ConstructValid) {
    EXPECT_NO_THROW(KVCacheSlotManager m(8));
}

TEST(KVCacheSlotManager, ConstructZeroThrows) {
    EXPECT_THROW(KVCacheSlotManager m(0), std::invalid_argument);
}

TEST(KVCacheSlotManager, ConstructNegativeThrows) {
    EXPECT_THROW(KVCacheSlotManager m(-1), std::invalid_argument);
}

// ─── Capacity / counts ────────────────────────────────────────────────────────

TEST(KVCacheSlotManager, CapacityMatchesConstructor) {
    KVCacheSlotManager m(16);
    EXPECT_EQ(m.Capacity(), 16);
}

TEST(KVCacheSlotManager, InitiallyAllFree) {
    KVCacheSlotManager m(8);
    EXPECT_EQ(m.FreeCount(), 8);
    EXPECT_EQ(m.UsedCount(), 0);
}

// ─── AllocSlot ────────────────────────────────────────────────────────────────

TEST(KVCacheSlotManager, AllocNSlotsAllSucceed) {
    const int N = 8;
    KVCacheSlotManager m(N);

    std::vector<int> allocated;
    for (int i = 0; i < N; ++i) {
        int slot = m.AllocSlot();
        ASSERT_GE(slot, 0) << "AllocSlot failed at i=" << i;
        ASSERT_LT(slot, N);
        allocated.push_back(slot);
    }

    // All returned IDs must be unique
    std::sort(allocated.begin(), allocated.end());
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(allocated[i], i) << "missing slot id " << i;
    }

    EXPECT_EQ(m.FreeCount(), 0);
    EXPECT_EQ(m.UsedCount(), N);
}

TEST(KVCacheSlotManager, AllocNPlusOneFailsGracefully) {
    const int N = 4;
    KVCacheSlotManager m(N);

    for (int i = 0; i < N; ++i) {
        ASSERT_GE(m.AllocSlot(), 0);
    }

    // N+1th alloc must return -1, not crash
    EXPECT_EQ(m.AllocSlot(), -1);
}

// ─── FreeSlot ─────────────────────────────────────────────────────────────────

TEST(KVCacheSlotManager, FreeReturnsSlotToPool) {
    KVCacheSlotManager m(4);
    int s0 = m.AllocSlot();
    int s1 = m.AllocSlot();
    ASSERT_GE(s0, 0);
    ASSERT_GE(s1, 0);

    m.FreeSlot(s0);
    EXPECT_EQ(m.FreeCount(), 3);

    // The freed slot must be reusable
    int s2 = m.AllocSlot();
    EXPECT_EQ(s2, s0);
}

TEST(KVCacheSlotManager, FreeInvalidIdIsNoop) {
    KVCacheSlotManager m(4);
    EXPECT_NO_THROW(m.FreeSlot(-1));
    EXPECT_NO_THROW(m.FreeSlot(4));
    EXPECT_NO_THROW(m.FreeSlot(99));
    EXPECT_EQ(m.FreeCount(), 4);  // pool unchanged
}

TEST(KVCacheSlotManager, FreeAlreadyFreeIdIsNoop) {
    KVCacheSlotManager m(4);
    int s = m.AllocSlot();
    m.FreeSlot(s);
    EXPECT_NO_THROW(m.FreeSlot(s));  // double-free must not crash
    EXPECT_EQ(m.FreeCount(), 4);
}

// ─── Full alloc-free cycle ────────────────────────────────────────────────────

TEST(KVCacheSlotManager, AllocFreeRepeatedly) {
    const int N = 4;
    KVCacheSlotManager m(N);

    for (int round = 0; round < 3; ++round) {
        std::vector<int> slots;
        for (int i = 0; i < N; ++i) {
            int s = m.AllocSlot();
            ASSERT_GE(s, 0) << "round=" << round << " i=" << i;
            slots.push_back(s);
        }
        EXPECT_EQ(m.AllocSlot(), -1);  // pool full
        for (int s : slots) m.FreeSlot(s);
        EXPECT_EQ(m.FreeCount(), N);   // fully restored
    }
}

// ─── Thread safety ────────────────────────────────────────────────────────────

TEST(KVCacheSlotManager, ConcurrentAllocFree) {
    const int N = 16;
    KVCacheSlotManager m(N);

    // 16 threads each alloc once, record their slot, then free it
    std::vector<std::thread> threads;
    std::vector<int> results(static_cast<size_t>(N), -2);

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&m, &results, i]() {
            results[static_cast<size_t>(i)] = m.AllocSlot();
        });
    }
    for (auto& t : threads) t.join();

    // All 16 allocations should have succeeded and returned distinct IDs
    std::vector<int> ids = results;
    std::sort(ids.begin(), ids.end());
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(ids[i], i) << "slot " << i << " missing from concurrent allocs";
    }

    for (int s : results) m.FreeSlot(s);
    EXPECT_EQ(m.FreeCount(), N);
}
