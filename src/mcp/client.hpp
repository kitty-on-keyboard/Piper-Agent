#pragma once
//
// MCP client.
//
// Correlation is keyed on Id, not uint64. Six of the seven cook-off clients used a
// uint64 map, which works exactly as long as the client is the only party minting ids;
// it silently maps every string id to 0. See src/mcp/message.hpp.
//
// Requests are concurrent by construction: send_async returns a future and the reader
// thread completes whichever reply arrives first. This is the only performance lever
// that matters over stdio -- a pipe round trip is microseconds and a real tool call is
// milliseconds, so throughput comes from having several calls outstanding, never from a
// faster byte pipe. See docs/MCP.md.
//
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "src/mcp/content.hpp"
#include "src/mcp/message.hpp"
#include "src/mcp/protocol.hpp"
#include "src/mcp/transport.hpp"

namespace lmp::mcp {

// A JSON-RPC error that came back from the server, or a local transport failure.
class McpError : public std::runtime_error {
public:
    McpError(int code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    [[nodiscard]] int code() const noexcept { return code_; }

private:
    int code_;
};

struct ServerInfo {
    std::string name;
    std::string version;
    std::string protocol_version;
    std::string instructions;
    nlohmann::json capabilities = nlohmann::json::object();

    [[nodiscard]] bool supports_tools() const { return capabilities.contains("tools"); }
    [[nodiscard]] bool supports_resources() const { return capabilities.contains("resources"); }
    [[nodiscard]] bool supports_prompts() const { return capabilities.contains("prompts"); }
    [[nodiscard]] bool supports_logging() const { return capabilities.contains("logging"); }
};

struct Root {
    std::string uri;  // must be a file:// URI
    std::string name;
};

class Client {
public:
    struct Info {
        std::string name = "lmp-mcp-client";
        std::string version = "0.1.0";
    };

    struct Options {
        std::chrono::milliseconds default_timeout{30000};
        // Sent in the initialize handshake. Declaring a capability we do not implement
        // invites the server to call something that will fail, so these default off and
        // are switched on by the corresponding setter.
        bool declare_roots = false;
        bool declare_sampling = false;
    };

    explicit Client(Info info);
    Client(Info info, Options options);
    ~Client();

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // --- connection --------------------------------------------------------
    // Takes ownership of the transport and starts its reader thread.
    void connect(std::unique_ptr<Transport> transport);

    // Convenience: spawn a server process and connect to it.
    void connect_stdio(Subprocess::Options options);

    // Handshake. Throws McpError if the server refuses or the version is unusable.
    ServerInfo initialize();

    // Idempotent. Fails every outstanding request with a meaningful McpError rather
    // than letting ~promise deliver a bare broken_promise -- entrant C7's one good idea.
    void close();

    [[nodiscard]] bool connected() const;
    [[nodiscard]] const ServerInfo& server() const noexcept { return server_info_; }

    // --- tools -------------------------------------------------------------
    // Follows nextCursor to completion, so the caller sees the whole list.
    [[nodiscard]] std::vector<Tool> list_tools();

    using ProgressFn = std::function<void(double progress, std::optional<double> total,
                                          std::string_view message)>;

    [[nodiscard]] ToolResult call_tool(std::string_view name, const nlohmann::json& arguments,
                                       ProgressFn on_progress = {},
                                       std::optional<std::chrono::milliseconds> timeout = {});

    // --- resources ---------------------------------------------------------
    [[nodiscard]] std::vector<Resource> list_resources();
    [[nodiscard]] std::vector<ResourceTemplate> list_resource_templates();
    [[nodiscard]] std::vector<ResourceContents> read_resource(std::string_view uri);
    void subscribe_resource(std::string_view uri);
    void unsubscribe_resource(std::string_view uri);

    // --- prompts -----------------------------------------------------------
    [[nodiscard]] std::vector<Prompt> list_prompts();
    [[nodiscard]] std::vector<PromptMessage> get_prompt(std::string_view name,
                                                        const nlohmann::json& arguments);

    // --- utilities ---------------------------------------------------------
    void ping();
    void set_log_level(LogLevel level);

    // --- server-initiated traffic ------------------------------------------
    void on_log(std::function<void(LogLevel, std::string_view logger, const nlohmann::json&)> fn);
    void on_tools_changed(std::function<void()> fn);
    void on_prompts_changed(std::function<void()> fn);
    void on_resources_changed(std::function<void()> fn);
    void on_resource_updated(std::function<void(std::string_view uri)> fn);
    void on_server_stderr(std::function<void(std::string_view)> fn);

    // Roots the server may enumerate. Enables the roots capability.
    void set_roots(std::vector<Root> roots);

    // --- raw ---------------------------------------------------------------
    // Escape hatch for methods this class does not model. Returns the `result` member,
    // or throws McpError carrying the server's code.
    nlohmann::json call(std::string_view method, const nlohmann::json& params,
                        std::optional<std::chrono::milliseconds> timeout = {});
    void notify(std::string_view method, const nlohmann::json& params);

private:
    struct Pending {
        std::promise<nlohmann::json> promise;
        ProgressFn on_progress;
    };

    void on_message(const nlohmann::json& raw);
    void handle_response(const Message& msg);
    void handle_server_request(const Message& msg);
    void handle_server_notification(const Message& msg);

    // Registers a pending entry and writes the request. The future resolves with the
    // `result` member, or breaks with an McpError.
    //
    // The id is returned rather than recomputed by the caller: it is minted inside, and
    // a caller that re-reads the counter to guess it races every other thread issuing a
    // request -- two concurrent calls would await each other's replies.
    struct Call {
        Id id;
        std::future<nlohmann::json> future;
    };
    Call send_async(std::string_view method, const nlohmann::json& params,
                    ProgressFn on_progress);

    nlohmann::json await(std::future<nlohmann::json> fut, const Id& id,
                         std::chrono::milliseconds timeout);

    [[nodiscard]] Id next_id();
    [[nodiscard]] nlohmann::json client_capabilities() const;

    // Walks nextCursor, concatenating `key` from each page.
    nlohmann::json list_paged(std::string_view method, const char* key);

    Info info_;
    Options options_;

    std::unique_ptr<Transport> transport_;
    ServerInfo server_info_;

    std::atomic<std::int64_t> next_id_{1};
    std::atomic<bool> closed_{false};
    std::atomic<bool> initialized_{false};

    mutable std::mutex pending_mutex_;
    std::unordered_map<Id, Pending, Id::Hash> pending_;

    std::mutex handler_mutex_;
    std::function<void(LogLevel, std::string_view, const nlohmann::json&)> on_log_;
    std::function<void()> on_tools_changed_;
    std::function<void()> on_prompts_changed_;
    std::function<void()> on_resources_changed_;
    std::function<void(std::string_view)> on_resource_updated_;
    std::function<void(std::string_view)> on_stderr_;

    std::vector<Root> roots_;
};

} // namespace lmp::mcp
