#pragma once
//
// MCP server.
//
// Two decisions here are direct consequences of the cook-off (docs/BAKEOFF_MCP.md).
//
// The lifecycle gate: nothing but `initialize` and `ping` is served until
// `notifications/initialized` has arrived. Six of the seven server entrants would answer
// `tools/list` from a peer that had never handshaken, and only S7 refused. It is two
// lines and it is in the spec.
//
// The concurrency model: requests run on a fixed pool that is drained and joined before
// run() returns; notifications are handled inline on the reader thread. Entrant S4
// spawned a detached thread per request and returned from main() with workers still
// writing, so it dropped replies nondeterministically -- 0, 1 or 2 responses to the same
// four-request input. Handling notifications inline is what keeps `notifications/
// cancelled` responsive while a slow tool occupies every worker; that is the whole
// reason cancellation is not just another queued item.
//
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "src/mcp/content.hpp"
#include "src/mcp/message.hpp"
#include "src/mcp/protocol.hpp"
#include "src/mcp/transport.hpp"

namespace lmp::mcp {

class Server;

// Handed to every handler. Carries the things a handler can only do by talking back to
// the peer mid-call: report progress, and notice it has been cancelled.
class RequestContext {
public:
    RequestContext(Server& server, Id id, nlohmann::json progress_token)
        : server_(server), id_(std::move(id)), progress_token_(std::move(progress_token)) {}

    // No-op unless the client supplied a progressToken in _meta, so a handler can call
    // it unconditionally.
    void report_progress(double progress, std::optional<double> total = std::nullopt,
                         std::string_view message = {});

    // Handlers doing long work should check this and return early. Returning a normal
    // result after cancellation is harmless -- the reply is suppressed.
    [[nodiscard]] bool cancelled() const;

    [[nodiscard]] const Id& id() const noexcept { return id_; }

private:
    Server& server_;
    Id id_;
    nlohmann::json progress_token_;
};

class Server {
public:
    struct Info {
        std::string name = "lmp-mcp";
        std::string version = "0.1.0";
        // Shown to the model as guidance on how to use this server. Optional.
        std::string instructions;
    };

    struct Options {
        // Requests in flight at once. Notifications never queue behind these.
        std::size_t worker_threads = 4;
        // tools/list and friends page at this size. Clients must follow nextCursor.
        std::size_t page_size = 100;
    };

    // Two overloads rather than a defaulted parameter: a default argument of `{}` would
    // need Options complete inside its own enclosing class definition.
    explicit Server(Info info);
    Server(Info info, Options options);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // --- registration (call before run) ------------------------------------
    using ToolFn = std::function<ToolResult(const nlohmann::json& args, RequestContext&)>;
    void add_tool(Tool tool, ToolFn fn);

    using ResourceReadFn =
        std::function<std::vector<ResourceContents>(const std::string& uri, RequestContext&)>;
    void add_resource(Resource resource, ResourceReadFn fn);
    void add_resource_template(ResourceTemplate tmpl, ResourceReadFn fn);

    using PromptFn =
        std::function<std::vector<PromptMessage>(const nlohmann::json& args, RequestContext&)>;
    void add_prompt(Prompt prompt, PromptFn fn);

    // Argument completion for prompt/resource-template parameters.
    using CompletionFn =
        std::function<std::vector<std::string>(std::string_view argument_name,
                                               std::string_view partial_value)>;
    void set_completion_handler(CompletionFn fn);

    // --- running -----------------------------------------------------------
    // Blocks until the transport closes. Joins every worker before returning, so no
    // handler can still be running when it does.
    //
    // This overload does NOT take ownership: the transport must outlive the call, which
    // is easy to get wrong when the server runs on its own thread and the transport was
    // a local at the call site. Prefer the owning overload below unless the transport is
    // genuinely owned elsewhere (StdioTransport on the main thread, typically).
    void run(Transport& transport);

    // Takes ownership for the duration, which removes the lifetime question entirely.
    void run(std::unique_ptr<Transport> transport);

    // Safe from any thread, including a signal-handler-adjacent context.
    void request_stop();

    // --- server-initiated traffic ------------------------------------------
    void log(LogLevel level, std::string_view logger, nlohmann::json data);
    void notify_tools_list_changed();
    void notify_prompts_list_changed();
    void notify_resources_list_changed();
    void notify_resource_updated(std::string_view uri);

    [[nodiscard]] const std::string& negotiated_version() const noexcept { return negotiated_version_; }

private:
    friend class RequestContext;

    struct ToolEntry {
        Tool tool;
        ToolFn fn;
    };
    struct ResourceEntry {
        Resource resource;
        ResourceReadFn fn;
    };
    struct TemplateEntry {
        ResourceTemplate tmpl;
        ResourceReadFn fn;
    };
    struct PromptEntry {
        Prompt prompt;
        PromptFn fn;
    };

    // --- dispatch ----------------------------------------------------------
    void on_message(const nlohmann::json& raw);
    void handle_request(const Message& msg);
    void handle_notification(const Message& msg);
    void run_request(Message msg);

    nlohmann::json dispatch(const Message& msg, RequestContext& ctx);

    nlohmann::json handle_initialize(const Message& msg);
    nlohmann::json handle_tools_list(const Message& msg);
    nlohmann::json handle_tools_call(const Message& msg, RequestContext& ctx);
    nlohmann::json handle_resources_list(const Message& msg);
    nlohmann::json handle_resource_templates_list(const Message& msg);
    nlohmann::json handle_resources_read(const Message& msg, RequestContext& ctx);
    nlohmann::json handle_prompts_list(const Message& msg);
    nlohmann::json handle_prompts_get(const Message& msg, RequestContext& ctx);
    nlohmann::json handle_completion(const Message& msg);
    nlohmann::json handle_logging_set_level(const Message& msg);

    void send(const nlohmann::json& j);
    void send_error(const Id& id, ErrorCode code, std::string_view message);

    [[nodiscard]] nlohmann::json capabilities() const;

    // Thrown by handlers to turn into a JSON-RPC error rather than a crash.
    struct Failure {
        ErrorCode code;
        std::string message;
    };
    [[noreturn]] static void fail(ErrorCode code, std::string message);

    // --- worker pool -------------------------------------------------------
    void start_workers();
    void stop_workers();
    void worker_loop();

    Info info_;
    Options options_;

    Transport* transport_ = nullptr;

    // Registries. Written during setup, read-only once run() starts, so no lock.
    std::vector<ToolEntry> tools_;
    std::unordered_map<std::string, std::size_t> tool_index_;
    std::vector<ResourceEntry> resources_;
    std::unordered_map<std::string, std::size_t> resource_index_;
    std::vector<TemplateEntry> templates_;
    std::vector<PromptEntry> prompts_;
    std::unordered_map<std::string, std::size_t> prompt_index_;
    CompletionFn completion_;

    // --- lifecycle ---------------------------------------------------------
    std::atomic<bool> initialize_received_{false};
    std::atomic<bool> initialized_{false};
    std::string negotiated_version_{std::string(kProtocolVersion)};
    std::atomic<int> log_level_{static_cast<int>(LogLevel::kInfo)};

    // --- in-flight bookkeeping ---------------------------------------------
    mutable std::mutex cancel_mutex_;
    std::unordered_set<Id, Id::Hash> cancelled_;

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<Message> queue_;
    std::vector<std::thread> workers_;
    bool draining_ = false;

    std::mutex done_mutex_;
    std::condition_variable done_cv_;
    bool closed_ = false;
};

} // namespace lmp::mcp
