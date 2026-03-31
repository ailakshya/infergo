#include "kv_cache.hpp"

#include <stdexcept>

namespace infergo {

KVCacheSlotManager::KVCacheSlotManager(int capacity)
    : capacity_(capacity), in_use_(static_cast<size_t>(capacity), false)
{
    if (capacity <= 0) {
        throw std::invalid_argument("KVCacheSlotManager: capacity must be > 0");
    }
}

int KVCacheSlotManager::AllocSlot() {
    std::lock_guard<std::mutex> lock(mu_);
    for (int i = 0; i < capacity_; ++i) {
        if (!in_use_[static_cast<size_t>(i)]) {
            in_use_[static_cast<size_t>(i)] = true;
            return i;
        }
    }
    return -1;  // pool exhausted
}

void KVCacheSlotManager::FreeSlot(int slot_id) {
    if (slot_id < 0 || slot_id >= capacity_) {
        return;  // ignore invalid IDs silently
    }
    std::lock_guard<std::mutex> lock(mu_);
    in_use_[static_cast<size_t>(slot_id)] = false;
}

int KVCacheSlotManager::FreeCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    int count = 0;
    for (int i = 0; i < capacity_; ++i) {
        if (!in_use_[static_cast<size_t>(i)]) ++count;
    }
    return count;
}

int KVCacheSlotManager::UsedCount() const {
    return capacity_ - FreeCount();
}

} // namespace infergo
