#include "kv_paged.hpp"

#include <stdexcept>

namespace infergo {

KVPageAllocator::KVPageAllocator(int page_size, int n_slots, int n_pages)
    : page_size_(page_size),
      n_slots_(n_slots),
      n_pages_(n_pages),
      slot_in_use_(static_cast<size_t>(n_slots), false),
      slot_pages_(static_cast<size_t>(n_slots), 0),
      pages_used_(0)
{
    if (page_size <= 0) {
        throw std::invalid_argument("KVPageAllocator: page_size must be > 0");
    }
    if (n_slots <= 0) {
        throw std::invalid_argument("KVPageAllocator: n_slots must be > 0");
    }
    if (n_pages <= 0) {
        throw std::invalid_argument("KVPageAllocator: n_pages must be > 0");
    }
}

int KVPageAllocator::AllocSlot(int initial_tokens) {
    // Minimum 1 page per sequence, even if initial_tokens == 0.
    int pages_needed = (initial_tokens <= 0)
        ? 1
        : (initial_tokens + page_size_ - 1) / page_size_;

    std::lock_guard<std::mutex> lock(mu_);

    // Check KV budget first.
    if (pages_used_ + pages_needed > n_pages_) {
        return -1;
    }

    // Find a free slot ID.
    for (int i = 0; i < n_slots_; ++i) {
        if (!slot_in_use_[static_cast<size_t>(i)]) {
            slot_in_use_[static_cast<size_t>(i)] = true;
            slot_pages_[static_cast<size_t>(i)]  = pages_needed;
            pages_used_ += pages_needed;
            return i;
        }
    }

    return -1;  // no free slot ID
}

bool KVPageAllocator::ExtendSlot(int slot_id, int additional_tokens) {
    if (slot_id < 0 || slot_id >= n_slots_) {
        return false;
    }
    if (additional_tokens <= 0) {
        return true;  // nothing to do
    }

    int pages_needed = (additional_tokens + page_size_ - 1) / page_size_;

    std::lock_guard<std::mutex> lock(mu_);

    if (!slot_in_use_[static_cast<size_t>(slot_id)]) {
        return false;  // slot not currently allocated
    }

    if (pages_used_ + pages_needed > n_pages_) {
        return false;  // KV budget exhausted
    }

    slot_pages_[static_cast<size_t>(slot_id)] += pages_needed;
    pages_used_ += pages_needed;
    return true;
}

void KVPageAllocator::FreeSlot(int slot_id) {
    if (slot_id < 0 || slot_id >= n_slots_) {
        return;  // ignore invalid IDs silently
    }

    std::lock_guard<std::mutex> lock(mu_);

    if (!slot_in_use_[static_cast<size_t>(slot_id)]) {
        return;  // already free — ignore silently
    }

    pages_used_ -= slot_pages_[static_cast<size_t>(slot_id)];
    slot_pages_[static_cast<size_t>(slot_id)]  = 0;
    slot_in_use_[static_cast<size_t>(slot_id)] = false;
}

int KVPageAllocator::FreePages() const {
    std::lock_guard<std::mutex> lock(mu_);
    return n_pages_ - pages_used_;
}

int KVPageAllocator::UsedPages() const {
    std::lock_guard<std::mutex> lock(mu_);
    return pages_used_;
}

int KVPageAllocator::FreeSlotCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    int count = 0;
    for (int i = 0; i < n_slots_; ++i) {
        if (!slot_in_use_[static_cast<size_t>(i)]) ++count;
    }
    return count;
}

} // namespace infergo
