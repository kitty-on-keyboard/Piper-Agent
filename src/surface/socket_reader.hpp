#pragma once
//
// Keep-warm daemon support: Unix domain socket listener and client forwarder.
// Allows lmp_sidecar --worker --serve to hold model weights resident in memory
// across multiple task slices, eliminating cold-load latency on Apple Silicon.
//
#include <optional>
#include <string>

namespace lmp::surface::worker {

struct DaemonConfig {
    std::string socket_path;
    std::string pid_path;
    double idle_timeout_s = 3600.0;
};

// Default socket path: ~/.piper/worker.sock (overridden by LMP_WORKER_SOCKET).
[[nodiscard]] std::string default_socket_path();

// Default pid path: ~/.piper/worker.pid.
[[nodiscard]] std::string default_pid_path();

// Probes whether a daemon is actively listening on socket_path.
// If socket exists but is unresponsive, unlinks stale socket & pid files.
[[nodiscard]] bool is_daemon_alive(const std::string& socket_path);

// Forward a task packet to the running daemon and wait for completion.
// Returns the exit code on success, or std::nullopt if daemon unreachable.
[[nodiscard]] std::optional<int> forward_to_daemon(
    const std::string& socket_path, const std::string& task_path, bool jsonl,
    bool auto_approve_irreversible = false, bool auto_approve_all = false);

class DaemonListener {
  public:
    explicit DaemonListener(DaemonConfig config);
    ~DaemonListener();
    DaemonListener(const DaemonListener&) = delete;
    DaemonListener& operator=(const DaemonListener&) = delete;

    // Bind and listen on socket_path. Writes pid_path. Returns false on error.
    [[nodiscard]] bool start();

    // Accept next client connection with idle timeout.
    // Returns client fd >= 0, or -1 on idle timeout or error.
    [[nodiscard]] int accept_client();

    void stop();

    [[nodiscard]] const DaemonConfig& config() const noexcept { return config_; }

  private:
    DaemonConfig config_;
    int listen_fd_ = -1;
    bool bound_ = false;
};

} // namespace lmp::surface::worker
