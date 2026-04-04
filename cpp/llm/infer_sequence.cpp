#include "infer_sequence.hpp"

#include <stdexcept>
#include <utility>

namespace infergo {

// ─── Constructor (slot manager) ───────────────────────────────────────────────

InferSequence::InferSequence(KVCacheSlotManager&  slot_manager,
                             std::vector<int32_t> prompt_tokens,
                             int32_t              eos_token_id)
    : slot_manager_(&slot_manager),
      eos_token_id_(eos_token_id),
      tokens_(std::move(prompt_tokens)),
      prompt_len_(static_cast<int>(tokens_.size()))
{
    if (tokens_.empty()) {
        throw std::invalid_argument("InferSequence: prompt must not be empty");
    }
    slot_id_ = slot_manager_->AllocSlot();
    if (slot_id_ < 0) {
        throw std::runtime_error("InferSequence: no KV cache slot available");
    }
}

// ─── Constructor (paged allocator) ───────────────────────────────────────────

InferSequence::InferSequence(KVPageAllocator&     allocator,
                             std::vector<int32_t> prompt_tokens,
                             int32_t              eos_token_id)
    : slot_manager_(nullptr),
      page_alloc_(&allocator),
      eos_token_id_(eos_token_id),
      tokens_(std::move(prompt_tokens)),
      prompt_len_(static_cast<int>(tokens_.size()))
{
    if (tokens_.empty()) {
        throw std::invalid_argument("InferSequence: prompt must not be empty");
    }
    slot_id_ = allocator.AllocSlot(prompt_len_);
    if (slot_id_ < 0) {
        throw std::runtime_error("InferSequence: no KV page slot available");
    }
}

// ─── Destructor ───────────────────────────────────────────────────────────────

InferSequence::~InferSequence() {
    if (slot_id_ >= 0) {
        if (page_alloc_ != nullptr) {
            page_alloc_->FreeSlot(slot_id_);
        } else {
            slot_manager_->FreeSlot(slot_id_);
        }
        slot_id_ = -1;
    }
}

// ─── Move ─────────────────────────────────────────────────────────────────────

InferSequence::InferSequence(InferSequence&& other) noexcept
    : slot_manager_(other.slot_manager_),
      page_alloc_(other.page_alloc_),
      slot_id_(other.slot_id_),
      eos_token_id_(other.eos_token_id_),
      tokens_(std::move(other.tokens_)),
      prompt_len_(other.prompt_len_),
      kv_pos_(other.kv_pos_),
      done_(other.done_),
      first_decode_(other.first_decode_)
{
    other.slot_id_ = -1;  // prevent double-free
}

InferSequence& InferSequence::operator=(InferSequence&& other) noexcept {
    if (this != &other) {
        if (slot_id_ >= 0) {
            if (page_alloc_ != nullptr) {
                page_alloc_->FreeSlot(slot_id_);
            } else {
                slot_manager_->FreeSlot(slot_id_);
            }
        }
        slot_manager_  = other.slot_manager_;
        page_alloc_    = other.page_alloc_;
        slot_id_       = other.slot_id_;
        eos_token_id_  = other.eos_token_id_;
        tokens_        = std::move(other.tokens_);
        prompt_len_    = other.prompt_len_;
        kv_pos_        = other.kv_pos_;
        done_          = other.done_;
        first_decode_  = other.first_decode_;
        other.slot_id_ = -1;
    }
    return *this;
}

// ─── NextTokens ───────────────────────────────────────────────────────────────

std::vector<int32_t> InferSequence::NextTokens() const {
    if (first_decode_) {
        // Prefill: feed the entire prompt
        return tokens_;
    }
    // Decode: feed only the last token (just appended)
    return { tokens_.back() };
}

// ─── AppendToken ──────────────────────────────────────────────────────────────

void InferSequence::AppendToken(int32_t token_id) {
    if (done_) return;

    if (first_decode_) {
        // The prompt tokens have just been decoded; kv_pos_ advances by prompt length.
        kv_pos_      = prompt_len_;
        first_decode_ = false;
    } else {
        // The previous single token has been decoded.
        kv_pos_++;

        // If using the paged allocator and we've just crossed a page boundary,
        // extend the slot by one more page.
        if (page_alloc_ != nullptr && kv_pos_ % page_alloc_->PageSize() == 0) {
            page_alloc_->ExtendSlot(slot_id_, page_alloc_->PageSize());
        }
    }

    tokens_.push_back(token_id);

    if (token_id == eos_token_id_) {
        done_ = true;
    }
}

} // namespace infergo
