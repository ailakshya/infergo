#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

namespace infergo {

/// Manages a fixed pool of KV cache sequence slots.
///
/// Each concurrent LLM sequence occupies one slot (a llama_seq_id).
/// Slot IDs are integers in [0, capacity). AllocSlot() returns the
/// next free ID; FreeSlot() returns it to the pool.
///
/// Thread-safe: all public methods are protected by an internal mutex.
class KVCacheSlotManager {
public:
    /// Create a pool with `capacity` slots.
    /// capacity must be > 0.
    explicit KVCacheSlotManager(int capacity);

    /// Allocate the next free slot.
    /// Returns a slot ID in [0, capacity) on success, or -1 if all slots are in use.
    int AllocSlot();

    /// Release a previously allocated slot back to the pool.
    /// Silently ignores invalid or already-free slot IDs.
    void FreeSlot(int slot_id);

    /// Returns the total number of slots (free + in-use).
    int Capacity() const noexcept { return capacity_; }

    /// Returns the number of currently free slots.
    int FreeCount() const;

    /// Returns the number of currently in-use slots.
    int UsedCount() const;

private:
    int capacity_;
    mutable std::mutex mu_;
    std::vector<bool> in_use_;  // in_use_[i] = true → slot i is allocated
};

} // namespace infergo
