#pragma once

#include "kv_cache.hpp"
#include "kv_paged.hpp"
#include "sampler.hpp"

#include <cstdint>
#include <vector>

namespace infergo {

/// InferSequence represents one concurrent LLM request.
///
/// Lifecycle:
///   1. Construct with a slot_manager and the prompt token IDs.
///      Constructor calls slot_manager.AllocSlot(); throws if pool is full.
///   2. Call NextTokens() to get the SequenceInput for the engine's BatchDecode.
///   3. After BatchDecode, call AppendToken(sampled_id) to advance the sequence.
///   4. Check IsDone() after each step; when true the sequence has hit EOS/EOT.
///   5. Destructor calls slot_manager.FreeSlot() — always release the slot.
///
/// Not thread-safe — the caller must serialise access.
class InferSequence {
public:
    /// Allocate a KV slot and initialise the sequence with prompt tokens.
    /// Throws std::runtime_error if no slot is available.
    InferSequence(KVCacheSlotManager& slot_manager,
                  std::vector<int32_t> prompt_tokens,
                  int32_t eos_token_id);

    /// Allocate a paged KV slot and initialise the sequence with prompt tokens.
    /// Throws std::runtime_error if no page slot is available.
    InferSequence(KVPageAllocator& allocator,
                  std::vector<int32_t> prompt_tokens,
                  int32_t eos_token_id);

    ~InferSequence();

    // Not copyable — owns a KV slot
    InferSequence(const InferSequence&)            = delete;
    InferSequence& operator=(const InferSequence&) = delete;

    // Movable — transfers slot ownership
    InferSequence(InferSequence&& other) noexcept;
    InferSequence& operator=(InferSequence&& other) noexcept;

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// KV slot ID assigned by the slot manager (= llama_seq_id for this sequence).
    int SlotID() const noexcept { return slot_id_; }

    /// Current KV cache position (= number of tokens already decoded).
    int Position() const noexcept { return kv_pos_; }

    /// All token IDs processed so far (prompt + generated).
    const std::vector<int32_t>& Tokens() const noexcept { return tokens_; }

    /// The last token appended (prompt last token or most recently sampled).
    int32_t LastToken() const noexcept { return tokens_.back(); }

    /// True once the sequence has produced an end-of-generation token.
    bool IsDone() const noexcept { return done_; }

    // ── State machine ─────────────────────────────────────────────────────────

    /// Returns the token IDs that need to be decoded in the next BatchDecode call.
    /// Before the first decode this is the full prompt.
    /// After each AppendToken call it is exactly the one newly appended token.
    std::vector<int32_t> NextTokens() const;

    /// Append a sampled token to the history and advance the KV position.
    /// Sets IsDone()=true if the token matches eos_token_id.
    void AppendToken(int32_t token_id);

    /// Mark the sequence as done externally (e.g. max-length reached).
    void MarkDone() noexcept { done_ = true; }

    /// Number of tokens generated so far (excludes prompt).
    int GeneratedCount() const noexcept {
        return static_cast<int>(tokens_.size()) - prompt_len_;
    }

private:
    KVCacheSlotManager* slot_manager_ = nullptr;  // non-owning; null if using paged allocator
    KVPageAllocator*    page_alloc_   = nullptr;  // non-owning; null if using old slot manager
    int                 slot_id_     = -1;
    int32_t             eos_token_id_;
    std::vector<int32_t> tokens_;     // prompt + generated
    int                 prompt_len_  = 0;
    int                 kv_pos_      = 0;   // tokens already in KV cache
    bool                done_        = false;
    bool                first_decode_ = true;  // true before any AppendToken
};

} // namespace infergo
