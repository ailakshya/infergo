#pragma once
#include <cstddef>
#include <mutex>
#include <vector>

namespace infergo {

/// KVPageAllocator manages a fixed pool of KV cache pages.
///
/// Replaces KVCacheSlotManager. Instead of just tracking which slot IDs are in use,
/// it also tracks the total token budget consumed across all sequences.
/// This allows many short sequences to share the KV cache that fixed-slot
/// allocation would waste on pre-divided chunks.
///
/// Thread-safe.
class KVPageAllocator {
public:
    static constexpr int kDefaultPageSize = 16;

    // page_size: tokens per page (power of 2 recommended, e.g. 16)
    // n_slots:   max concurrent sequences (= llama.cpp n_seq_max; set to capacity/page_size or a fixed large value)
    // n_pages:   total KV cache pages = n_ctx / page_size
    KVPageAllocator(int page_size, int n_slots, int n_pages);

    // Allocate a slot for a new sequence that needs initial_tokens KV budget.
    // Reserves ceil(initial_tokens / page_size) pages and returns a slot_id in [0, n_slots).
    // Returns -1 if no slot is available OR if there are not enough free pages.
    int AllocSlot(int initial_tokens);

    // Extend an existing slot's KV page reservation by additional_tokens.
    // Allocates ceil(additional_tokens / page_size) more pages.
    // Returns true on success, false if KV budget is exhausted (caller should stop generation).
    bool ExtendSlot(int slot_id, int additional_tokens);

    // Release all pages held by a slot and return the slot to the free pool.
    // Silently ignores invalid or already-free slots.
    void FreeSlot(int slot_id);

    // ── Query ──────────────────────────────────────────────────────────────────
    int PageSize()    const noexcept { return page_size_; }
    int TotalPages()  const noexcept { return n_pages_; }
    int Capacity()    const noexcept { return n_slots_; }

    // Number of pages currently free (not reserved by any sequence).
    int FreePages()   const;

    // Number of pages currently in use.
    int UsedPages()   const;

    // Number of currently free slot IDs.
    int FreeSlotCount() const;

private:
    int page_size_;
    int n_slots_;
    int n_pages_;

    mutable std::mutex mu_;
    std::vector<bool> slot_in_use_;   // slot_in_use_[i] = true → slot i is allocated
    std::vector<int>  slot_pages_;    // slot_pages_[i] = pages currently held by slot i
    int pages_used_ = 0;              // total pages in use across all slots
};

} // namespace infergo
