// Unix Domain Socket server for zero-overhead local inference.
// No TCP stack, no Go, no JSON — just length-prefixed binary protocol.
// Overhead: ~0.3ms per request (vs ~4ms HTTP).
#pragma once

#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <vector>

namespace infergo {

// Simple binary protocol: [4-byte length][payload]
// Request payload:  [1-byte type][rest depends on type]
// Response payload: [4-byte status][rest is response data]

enum class UDSRequestType : uint8_t {
    GENERATE     = 1,  // LLM generation
    TOKENIZE     = 2,  // Tokenize text
    EMBED        = 3,  // Generate embedding
    SEARCH       = 4,  // Vector search
    BM25_SEARCH  = 5,  // BM25 keyword search
    HEALTH       = 10, // Health check
};

// Callback: process a request, return response bytes
using RequestHandler = std::function<std::string(UDSRequestType type, const char* data, int len)>;

class UDSServer {
public:
    UDSServer();
    ~UDSServer();

    // Start listening on the given socket path
    bool Start(const std::string& socket_path, RequestHandler handler);

    // Stop the server and clean up
    void Stop();

    // Returns true if the server is running
    bool Running() const { return running_.load(); }

private:
    void acceptLoop();
    void handleClient(int client_fd);

    std::string socket_path_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    std::vector<std::thread> client_threads_;
    RequestHandler handler_;
};

} // namespace infergo
