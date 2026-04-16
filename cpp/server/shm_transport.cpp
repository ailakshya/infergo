// Shared Memory Transport — implementation.
#include "shm_transport.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>

namespace infergo {

// ─── Server ──────────────────────────────────────────────────────────────────

bool SHMServer::Create(const std::string& name, int n_slots) {
    if (n_slots <= 0 || n_slots > SHM_MAX_SLOTS) n_slots = 8;
    name_ = "/" + name;
    n_slots_ = n_slots;

    // Create shared memory
    fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ < 0) {
        fprintf(stderr, "[shm] shm_open(%s) failed: %m\n", name_.c_str());
        return false;
    }

    size_t size = sizeof(SHMRegion);
    if (ftruncate(fd_, static_cast<off_t>(size)) < 0) {
        fprintf(stderr, "[shm] ftruncate failed: %m\n");
        close(fd_);
        shm_unlink(name_.c_str());
        return false;
    }

    region_ = static_cast<SHMRegion*>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (region_ == MAP_FAILED) {
        fprintf(stderr, "[shm] mmap failed: %m\n");
        close(fd_);
        shm_unlink(name_.c_str());
        region_ = nullptr;
        return false;
    }

    // Initialize
    memset(region_, 0, size);
    region_->header.magic = 0x494E4647; // "INFG"
    region_->header.version = 1;
    region_->header.n_slots = static_cast<uint32_t>(n_slots);
    region_->header.slot_size = SHM_SLOT_SIZE;
    region_->header.server_alive.store(1, std::memory_order_release);

    for (int i = 0; i < n_slots; i++) {
        region_->slots[i].state.store(static_cast<uint32_t>(SlotState::FREE),
                                       std::memory_order_release);
    }

    fprintf(stderr, "[shm] created /dev/shm%s (%d slots, %zu bytes)\n",
            name_.c_str(), n_slots, size);
    return true;
}

int SHMServer::PollReady() {
    for (int i = 0; i < n_slots_; i++) {
        uint32_t expected = static_cast<uint32_t>(SlotState::READY);
        uint32_t desired = static_cast<uint32_t>(SlotState::BUSY);
        if (region_->slots[i].state.compare_exchange_strong(
                expected, desired, std::memory_order_acq_rel)) {
            return i;
        }
    }
    return -1;
}

const char* SHMServer::GetRequest(int slot, int* out_len) {
    if (slot < 0 || slot >= n_slots_) return nullptr;
    *out_len = static_cast<int>(region_->slots[slot].request_len);
    return region_->slots[slot].request;
}

void SHMServer::WriteResponse(int slot, const char* data, int len) {
    if (slot < 0 || slot >= n_slots_) return;
    auto& s = region_->slots[slot];
    int copy_len = len < static_cast<int>(SHM_SLOT_SIZE) ? len : static_cast<int>(SHM_SLOT_SIZE) - 1;
    memcpy(s.response, data, static_cast<size_t>(copy_len));
    s.response[copy_len] = '\0';
    s.response_len = static_cast<uint32_t>(copy_len);
    s.state.store(static_cast<uint32_t>(SlotState::DONE), std::memory_order_release);
}

void SHMServer::Destroy() {
    if (region_) {
        region_->header.server_alive.store(0, std::memory_order_release);
        munmap(region_, sizeof(SHMRegion));
        region_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    if (!name_.empty()) {
        shm_unlink(name_.c_str());
    }
}

bool SHMServer::Running() const {
    return region_ && region_->header.server_alive.load(std::memory_order_acquire);
}

// ─── Client ──────────────────────────────────────────────────────────────────

bool SHMClient::Connect(const std::string& name) {
    std::string shm_name = "/" + name;
    fd_ = shm_open(shm_name.c_str(), O_RDWR, 0666);
    if (fd_ < 0) {
        fprintf(stderr, "[shm-client] shm_open(%s) failed: %m\n", shm_name.c_str());
        return false;
    }

    region_ = static_cast<SHMRegion*>(
        mmap(nullptr, sizeof(SHMRegion), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (region_ == MAP_FAILED) {
        close(fd_);
        region_ = nullptr;
        return false;
    }

    // Verify magic
    if (region_->header.magic != 0x494E4647) {
        fprintf(stderr, "[shm-client] invalid magic: 0x%08X\n", region_->header.magic);
        Disconnect();
        return false;
    }

    // Check server alive
    if (!region_->header.server_alive.load(std::memory_order_acquire)) {
        fprintf(stderr, "[shm-client] server not alive\n");
        Disconnect();
        return false;
    }

    client_id_ = static_cast<uint32_t>(getpid());
    return true;
}

int SHMClient::AcquireSlot() {
    int n = static_cast<int>(region_->header.n_slots);
    for (int i = 0; i < n; i++) {
        uint32_t expected = static_cast<uint32_t>(SlotState::FREE);
        uint32_t desired = static_cast<uint32_t>(SlotState::FREE); // mark owned but not ready
        if (region_->slots[i].state.compare_exchange_strong(
                expected, desired, std::memory_order_acq_rel)) {
            region_->slots[i].client_id = client_id_;
            return i;
        }
    }
    return -1; // no free slots
}

void SHMClient::SendRequest(int slot, const char* data, int len) {
    auto& s = region_->slots[slot];
    int copy_len = len < static_cast<int>(SHM_SLOT_SIZE) ? len : static_cast<int>(SHM_SLOT_SIZE) - 1;
    memcpy(s.request, data, static_cast<size_t>(copy_len));
    s.request[copy_len] = '\0';
    s.request_len = static_cast<uint32_t>(copy_len);
    s.state.store(static_cast<uint32_t>(SlotState::READY), std::memory_order_release);
}

const char* SHMClient::WaitResponse(int slot, int* out_len, int timeout_ms) {
    auto& s = region_->slots[slot];
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    // Spin for 100us, then yield
    int spins = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        uint32_t state = s.state.load(std::memory_order_acquire);
        if (state == static_cast<uint32_t>(SlotState::DONE)) {
            *out_len = static_cast<int>(s.response_len);
            return s.response;
        }
        if (++spins < 1000) {
            // spin
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    *out_len = 0;
    return nullptr; // timeout
}

void SHMClient::ReleaseSlot(int slot) {
    region_->slots[slot].state.store(static_cast<uint32_t>(SlotState::FREE),
                                      std::memory_order_release);
}

void SHMClient::Disconnect() {
    if (region_) {
        munmap(region_, sizeof(SHMRegion));
        region_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

} // namespace infergo
