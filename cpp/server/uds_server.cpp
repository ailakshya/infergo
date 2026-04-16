// Unix Domain Socket server — C++ implementation.
// Length-prefixed binary protocol. One thread per client.
#include "uds_server.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdio>

namespace infergo {

UDSServer::UDSServer() = default;

UDSServer::~UDSServer() {
    Stop();
}

bool UDSServer::Start(const std::string& socket_path, RequestHandler handler) {
    socket_path_ = socket_path;
    handler_ = handler;

    // Remove existing socket file
    unlink(socket_path.c_str());

    // Create socket
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        fprintf(stderr, "[uds] socket() failed: %s\n", strerror(errno));
        return false;
    }

    // Bind
    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "[uds] bind(%s) failed: %s\n", socket_path.c_str(), strerror(errno));
        close(server_fd_);
        return false;
    }

    // Listen
    if (listen(server_fd_, 32) < 0) {
        fprintf(stderr, "[uds] listen() failed: %s\n", strerror(errno));
        close(server_fd_);
        unlink(socket_path.c_str());
        return false;
    }

    running_ = true;
    accept_thread_ = std::thread(&UDSServer::acceptLoop, this);

    fprintf(stderr, "[uds] listening on %s\n", socket_path.c_str());
    return true;
}

void UDSServer::Stop() {
    if (!running_) return;
    running_ = false;

    // Close server socket to unblock accept()
    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }

    if (accept_thread_.joinable()) accept_thread_.join();
    for (auto& t : client_threads_) {
        if (t.joinable()) t.join();
    }
    client_threads_.clear();

    unlink(socket_path_.c_str());
}

void UDSServer::acceptLoop() {
    while (running_) {
        int client_fd = accept(server_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (running_) fprintf(stderr, "[uds] accept() failed: %s\n", strerror(errno));
            break;
        }
        client_threads_.emplace_back(&UDSServer::handleClient, this, client_fd);
    }
}

// Read exactly n bytes from fd
static bool read_exact(int fd, void* buf, size_t n) {
    size_t total = 0;
    while (total < n) {
        ssize_t r = read(fd, static_cast<char*>(buf) + total, n - total);
        if (r <= 0) return false;
        total += static_cast<size_t>(r);
    }
    return true;
}

// Write exactly n bytes to fd
static bool write_exact(int fd, const void* buf, size_t n) {
    size_t total = 0;
    while (total < n) {
        ssize_t w = write(fd, static_cast<const char*>(buf) + total, n - total);
        if (w <= 0) return false;
        total += static_cast<size_t>(w);
    }
    return true;
}

void UDSServer::handleClient(int client_fd) {
    // Protocol: [4-byte length][1-byte type][payload]
    // Response: [4-byte length][4-byte status][payload]
    while (running_) {
        // Read request length
        uint32_t req_len = 0;
        if (!read_exact(client_fd, &req_len, 4)) break;

        if (req_len == 0 || req_len > 16 * 1024 * 1024) break; // max 16MB

        // Read request body
        std::string req_data(req_len, '\0');
        if (!read_exact(client_fd, &req_data[0], req_len)) break;

        // Parse type
        auto type = static_cast<UDSRequestType>(static_cast<uint8_t>(req_data[0]));

        // Handle request
        std::string response;
        if (handler_) {
            response = handler_(type, req_data.data() + 1, static_cast<int>(req_len) - 1);
        }

        // Write response: [4-byte length][payload]
        uint32_t resp_len = static_cast<uint32_t>(response.size());
        if (!write_exact(client_fd, &resp_len, 4)) break;
        if (resp_len > 0) {
            if (!write_exact(client_fd, response.data(), resp_len)) break;
        }
    }
    close(client_fd);
}

} // namespace infergo
