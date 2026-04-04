#include "kv_paged.hpp"

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>

namespace infergo {

// ── T1: AllocSlot/FreeSlot basic round-trip ───────────────────────────────────

TEST(KVPageAllocatorTest, T1_BasicAllocFree) {
    // page_size=16, n_slots=4, n_pages=64 (1024 tokens)
    KVPageAllocator alloc(16, 4, 64);

    EXPECT_EQ(alloc.FreeSlotCount(), 4);
    EXPECT_EQ(alloc.FreePages(), 64);
    EXPECT_EQ(alloc.UsedPages(), 0);

    int s0 = alloc.AllocSlot(32);  // needs 2 pages
    ASSERT_GE(s0, 0);
    EXPECT_EQ(alloc.FreeSlotCount(), 3);
    EXPECT_EQ(alloc.UsedPages(), 2);
    EXPECT_EQ(alloc.FreePages(), 62);

    int s1 = alloc.AllocSlot(48);  // needs 3 pages
    ASSERT_GE(s1, 0);
    EXPECT_NE(s1, s0);
    EXPECT_EQ(alloc.UsedPages(), 5);

    alloc.FreeSlot(s0);
    EXPECT_EQ(alloc.FreeSlotCount(), 3);
    EXPECT_EQ(alloc.UsedPages(), 3);
    EXPECT_EQ(alloc.FreePages(), 61);

    alloc.FreeSlot(s1);
    EXPECT_EQ(alloc.FreeSlotCount(), 4);
    EXPECT_EQ(alloc.UsedPages(), 0);
    EXPECT_EQ(alloc.FreePages(), 64);
}

// ── T2: Pages exhausted returns -1 ────────────────────────────────────────────

TEST(KVPageAllocatorTest, T2_PagesExhausted) {
    // 8 pages total, page_size=16, n_slots=16
    KVPageAllocator alloc(16, 16, 8);

    // Consume all 8 pages with one sequence (128 tokens = 8 pages)
    int s0 = alloc.AllocSlot(128);
    ASSERT_GE(s0, 0);
    EXPECT_EQ(alloc.UsedPages(), 8);
    EXPECT_EQ(alloc.FreePages(), 0);

    // Next alloc should fail — no pages left
    int s1 = alloc.AllocSlot(16);
    EXPECT_EQ(s1, -1);

    alloc.FreeSlot(s0);
}

// ── T3: Slot IDs exhausted returns -1 ─────────────────────────────────────────

TEST(KVPageAllocatorTest, T3_SlotsExhausted) {
    // n_slots=2 but many pages available
    KVPageAllocator alloc(16, 2, 1024);

    int s0 = alloc.AllocSlot(16);
    int s1 = alloc.AllocSlot(16);
    ASSERT_GE(s0, 0);
    ASSERT_GE(s1, 0);
    EXPECT_NE(s0, s1);

    // Third alloc must fail — no slot IDs left
    int s2 = alloc.AllocSlot(16);
    EXPECT_EQ(s2, -1);

    alloc.FreeSlot(s0);
    alloc.FreeSlot(s1);
}

// ── T4: FreeSlot releases pages for reuse ─────────────────────────────────────

TEST(KVPageAllocatorTest, T4_FreedPagesReused) {
    // 4 pages total, page_size=16, n_slots=4
    KVPageAllocator alloc(16, 4, 4);

    // Fill up: 2 sequences × 2 pages each = 4 pages
    int s0 = alloc.AllocSlot(32);  // 2 pages
    int s1 = alloc.AllocSlot(32);  // 2 pages
    ASSERT_GE(s0, 0);
    ASSERT_GE(s1, 0);
    EXPECT_EQ(alloc.FreePages(), 0);

    // No room for another sequence
    EXPECT_EQ(alloc.AllocSlot(16), -1);

    // Free one sequence — its 2 pages return to pool
    alloc.FreeSlot(s0);
    EXPECT_EQ(alloc.FreePages(), 2);

    // Now a new sequence can fit
    int s2 = alloc.AllocSlot(16);  // needs 1 page
    ASSERT_GE(s2, 0);
    EXPECT_EQ(alloc.UsedPages(), 3);

    alloc.FreeSlot(s1);
    alloc.FreeSlot(s2);
    EXPECT_EQ(alloc.UsedPages(), 0);
}

// ── T5: ExtendSlot grows allocation ───────────────────────────────────────────

TEST(KVPageAllocatorTest, T5_ExtendSlot) {
    KVPageAllocator alloc(16, 4, 64);

    int s0 = alloc.AllocSlot(16);  // 1 page
    ASSERT_GE(s0, 0);
    EXPECT_EQ(alloc.UsedPages(), 1);

    // Extend by 32 tokens = 2 pages
    bool ok = alloc.ExtendSlot(s0, 32);
    EXPECT_TRUE(ok);
    EXPECT_EQ(alloc.UsedPages(), 3);

    // Extend by 1 token = 1 page (ceiling)
    ok = alloc.ExtendSlot(s0, 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(alloc.UsedPages(), 4);

    // Extend by 0 tokens — no-op, still succeeds
    ok = alloc.ExtendSlot(s0, 0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(alloc.UsedPages(), 4);

    alloc.FreeSlot(s0);
    EXPECT_EQ(alloc.UsedPages(), 0);
}

// ── T6: ExtendSlot fails when pages exhausted ─────────────────────────────────

TEST(KVPageAllocatorTest, T6_ExtendSlotExhausted) {
    // 4 pages, page_size=16
    KVPageAllocator alloc(16, 4, 4);

    int s0 = alloc.AllocSlot(32);  // 2 pages
    int s1 = alloc.AllocSlot(32);  // 2 pages — pool now full
    ASSERT_GE(s0, 0);
    ASSERT_GE(s1, 0);
    EXPECT_EQ(alloc.FreePages(), 0);

    // Extending s0 should fail — no pages available
    bool ok = alloc.ExtendSlot(s0, 16);
    EXPECT_FALSE(ok);
    EXPECT_EQ(alloc.UsedPages(), 4);  // unchanged

    alloc.FreeSlot(s0);
    alloc.FreeSlot(s1);
}

// ── T7: Thread-safety with 8 concurrent allocators ────────────────────────────

TEST(KVPageAllocatorTest, T7_ThreadSafety) {
    // Enough pages and slots for 8 threads, each doing 10 alloc/free cycles.
    // page_size=16, n_slots=16, n_pages=128
    KVPageAllocator alloc(16, 16, 128);

    constexpr int kThreads = 8;
    constexpr int kCycles  = 10;

    std::atomic<int> failures{0};

    auto worker = [&]() {
        for (int i = 0; i < kCycles; ++i) {
            int slot = alloc.AllocSlot(16);  // 1 page each
            if (slot < 0) {
                ++failures;
                continue;
            }
            // Optionally extend
            if (!alloc.ExtendSlot(slot, 16)) {
                // Not a hard failure — pages may be tight at this moment.
                // Just free and move on.
            }
            alloc.FreeSlot(slot);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& th : threads) {
        th.join();
    }

    // All threads should complete without crashing.
    // With n_slots=16 and kThreads=8 there should be no slot exhaustion failures.
    EXPECT_EQ(failures.load(), 0);

    // After all threads finish, everything should be back to zero.
    EXPECT_EQ(alloc.UsedPages(), 0);
    EXPECT_EQ(alloc.FreeSlotCount(), 16);
}

// ── T8: FreePages/UsedPages accounting correct after alloc/free sequence ──────

TEST(KVPageAllocatorTest, T8_PageAccounting) {
    // page_size=16, n_slots=8, n_pages=32
    KVPageAllocator alloc(16, 8, 32);

    EXPECT_EQ(alloc.TotalPages(), 32);
    EXPECT_EQ(alloc.FreePages(), 32);
    EXPECT_EQ(alloc.UsedPages(), 0);

    // Alloc with initial_tokens=0 → minimum 1 page
    int s0 = alloc.AllocSlot(0);
    ASSERT_GE(s0, 0);
    EXPECT_EQ(alloc.UsedPages(), 1);
    EXPECT_EQ(alloc.FreePages(), 31);

    // Alloc 17 tokens → ceil(17/16) = 2 pages
    int s1 = alloc.AllocSlot(17);
    ASSERT_GE(s1, 0);
    EXPECT_EQ(alloc.UsedPages(), 3);
    EXPECT_EQ(alloc.FreePages(), 29);

    // Alloc 16 tokens → 1 page
    int s2 = alloc.AllocSlot(16);
    ASSERT_GE(s2, 0);
    EXPECT_EQ(alloc.UsedPages(), 4);

    // Free s1 (held 2 pages)
    alloc.FreeSlot(s1);
    EXPECT_EQ(alloc.UsedPages(), 2);
    EXPECT_EQ(alloc.FreePages(), 30);

    // Double-free of s1 is silently ignored
    alloc.FreeSlot(s1);
    EXPECT_EQ(alloc.UsedPages(), 2);

    // Free invalid slot — silently ignored
    alloc.FreeSlot(-1);
    alloc.FreeSlot(999);
    EXPECT_EQ(alloc.UsedPages(), 2);

    alloc.FreeSlot(s0);
    alloc.FreeSlot(s2);
    EXPECT_EQ(alloc.UsedPages(), 0);
    EXPECT_EQ(alloc.FreePages(), 32);
    EXPECT_EQ(alloc.FreeSlotCount(), 8);
}

} // namespace infergo
