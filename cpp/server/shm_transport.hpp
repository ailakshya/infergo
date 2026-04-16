// Shared Memory Transport — zero-copy, zero-syscall inference.
// Overhead: ~0.01ms (50 microseconds) per request.
//
// Memory layout:
//   [Header: 64 bytes]
//   [Request  slot 0: 64KB] [Request  slot 1: 64KB] ... [Request  slot N: 64KB]
//   [Response slot 0: 64KB] [Response slot 1: 64KB] ... [Response slot N: 64KB]
//
// Protocol:
//   1. Client writes request to slot, sets slot.state = READY
//   2. Server polls slots, finds READY, processes, writes response, sets state = DONE
//   3. Client polls slot.state, finds DONE, reads response, sets state = FREE
//
// No locks. No syscalls in the hot path. Atomic state transitions only.
#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <string>

namespace infergo {

// Slot states (atomic transitions, no locks)
enum class SlotState : uint32_t {
    FREE     = 0,  // Slot available for client to write
    READY    = 1,  // Client has written request, server should process
    BUSY     = 2,  // Server is processing
    DONE     = 3,  // Server has written response, client should read
};

// Each slot has a fixed-size buffer for request and response
static constexpr size_t SHM_SLOT_SIZE = 65536;  // 64KB per slot
static constexpr int    SHM_MAX_SLOTS = 16;      // max concurrent clients

struct SHMSlot {
    std::atomic<uint32_t> state;    // SlotState
    uint32_t request_len;           // bytes written by client
    uint32_t response_len;          // bytes written by server
    uint32_t client_id;             // which client owns this slot
    char     request[SHM_SLOT_SIZE];
    char     response[SHM_SLOT_SIZE];
    char     padding[16];           // cache line alignment
};

struct SHMHeader {
    uint32_t magic;         // 0x494E4647 ("INFG")
    uint32_t version;       // protocol version
    uint32_t n_slots;       // number of active slots
    uint32_t slot_size;     // size of each slot
    std::atomic<uint32_t> server_alive;  // 1 = server running
    char     reserved[44];  // pad to 64 bytes
};

struct SHMRegion {
    SHMHeader header;
    SHMSlot   slots[SHM_MAX_SLOTS];
};

// Server side: create shared memory, poll for requests
class SHMServer {
public:
    // Create shared memory region at /dev/shm/<name>
    bool Create(const std::string& name, int n_slots = 8);

    // Poll for ready requests. Returns slot index or -1 if none ready.
    int PollReady();

    // Get request data from a slot
    const char* GetRequest(int slot, int* out_len);

    // Write response to a slot and mark DONE
    void WriteResponse(int slot, const char* data, int len);

    // Cleanup
    void Destroy();

    bool Running() const;

private:
    SHMRegion* region_ = nullptr;
    int fd_ = -1;
    std::string name_;
    int n_slots_ = 0;
};

// Client side: connect to shared memory, send requests
class SHMClient {
public:
    // Connect to existing shared memory
    bool Connect(const std::string& name);

    // Acquire a free slot. Returns slot index or -1.
    int AcquireSlot();

    // Write request and signal server
    void SendRequest(int slot, const char* data, int len);

    // Wait for response (spins, then yields). Returns response data.
    const char* WaitResponse(int slot, int* out_len, int timeout_ms = 5000);

    // Release slot back to FREE
    void ReleaseSlot(int slot);

    // Disconnect
    void Disconnect();

private:
    SHMRegion* region_ = nullptr;
    int fd_ = -1;
    uint32_t client_id_ = 0;
};

} // namespace infergo
