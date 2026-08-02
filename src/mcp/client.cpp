#include "src/mcp/client.hpp"

#include <future>
#include <utility>

namespace lmp::mcp {

namespace {

const nlohmann::json& obj_or_empty(const nlohmann::json& j) {
    static const nlohmann::json kEmpty = nlohmann::json::object();
    return j.is_object() ? j : kEmpty;
}

std::string string_or(const nlohmann::json& j, const char* key, std::string fallback = {}) {
    const auto& o = obj_or_empty(j);
    if (o.contains(key) && o[key].is_string()) {
        return o[key].get<std::string>();
    }
    return fallback;
}

Tool tool_from_json(const nlohmann::json& j) {
    Tool t;
    t.name = string_or(j, "name");
    t.title = string_or(j, "title");
    t.description = string_or(j, "description");
    if (j.contains("inputSchema")) {
        t.input_schema = j["inputSchema"];
    }
    if (j.contains("outputSchema")) {
        t.output_schema = j["outputSchema"];
    }
    return t;
}

Resource resource_from_json(const nlohmann::json& j) {
    Resource r;
    r.uri = string_or(j, "uri");
    r.name = string_or(j, "name");
    r.title = string_or(j, "title");
    r.description = string_or(j, "description");
    r.mime_type = string_or(j, "mimeType");
    return r;
}

ResourceTemplate template_from_json(const nlohmann::json& j) {
    ResourceTemplate t;
    t.uri_template = string_or(j, "uriTemplate");
    t.name = string_or(j, "name");
    t.title = string_or(j, "title");
    t.description = string_or(j, "description");
    t.mime_type = string_or(j, "mimeType");
    return t;
}

Prompt prompt_from_json(const nlohmann::json& j) {
    Prompt p;
    p.name = string_or(j, "name");
    p.title = string_or(j, "title");
    p.description = string_or(j, "description");
    if (j.contains("arguments") && j["arguments"].is_array()) {
        for (const auto& a : j["arguments"]) {
            PromptArgument arg;
            arg.name = string_or(a, "name");
            arg.description = string_or(a, "description");
            arg.required = obj_or_empty(a).value("required", false);
            p.arguments.push_back(std::move(arg));
        }
    }
    return p;
}

} // namespace

Client::Client(Info info) : Client(std::move(info), Options{}) {}

Client::Client(Info info, Options options) : info_(std::move(info)), options_(options) {}

Client::~Client() {
    close();
}

Id Client::next_id() {
    return Id::number(next_id_.fetch_add(1, std::memory_order_relaxed));
}

void Client::connect(std::unique_ptr<Transport> transport) {
    transport_ = std::move(transport);
    closed_.store(false, std::memory_order_release);

    Transport::Handlers handlers;
    handlers.on_message = [this](const nlohmann::json& raw) { on_message(raw); };
    handlers.on_parse_error = [](std::string_view, std::string_view) {
        // A server that emits a non-JSON line on stdout is misbehaving, but one bad
        // line is not a reason to tear down a working session. The server's own stderr
        // is the place that will say why.
    };
    handlers.on_closed = [this] {
        // The server went away. Everyone waiting needs to hear about it now rather than
        // at their individual timeouts.
        std::unordered_map<Id, Pending, Id::Hash> taken;
        {
            const std::lock_guard<std::mutex> lock(pending_mutex_);
            taken.swap(pending_);
        }
        for (auto& [id, p] : taken) {
            p.promise.set_exception(std::make_exception_ptr(
                McpError(to_int(ErrorCode::kInternalError),
                         "MCP server closed the connection before answering request " + id.debug())));
        }
    };
    handlers.on_stderr = [this](std::string_view chunk) {
        std::function<void(std::string_view)> fn;
        {
            const std::lock_guard<std::mutex> lock(handler_mutex_);
            fn = on_stderr_;
        }
        if (fn) {
            fn(chunk);
        }
    };

    transport_->start(std::move(handlers));
}

void Client::connect_stdio(Subprocess::Options options) {
    connect(std::make_unique<SubprocessTransport>(std::move(options)));
}

bool Client::connected() const {
    return transport_ && transport_->is_open() && !closed_.load(std::memory_order_acquire);
}

void Client::close() {
    if (closed_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    if (transport_) {
        transport_->stop();
    }

    // Fail whatever is left rather than leaving callers on a promise whose only
    // outcome is std::future_errc::broken_promise, which says nothing about why.
    std::unordered_map<Id, Pending, Id::Hash> taken;
    {
        const std::lock_guard<std::mutex> lock(pending_mutex_);
        taken.swap(pending_);
    }
    for (auto& [id, p] : taken) {
        p.promise.set_exception(std::make_exception_ptr(
            McpError(to_int(ErrorCode::kInternalError),
                     "MCP client closed while request " + id.debug() + " was in flight")));
    }
    transport_.reset();
}

// ---------------------------------------------------------------------------

nlohmann::json Client::client_capabilities() const {
    nlohmann::json caps = nlohmann::json::object();
    if (options_.declare_roots || !roots_.empty()) {
        caps["roots"] = nlohmann::json{{"listChanged", true}};
    }
    if (options_.declare_sampling) {
        caps["sampling"] = nlohmann::json::object();
    }
    return caps;
}

ServerInfo Client::initialize() {
    nlohmann::json params{
        {"protocolVersion", std::string(kProtocolVersion)},
        {"capabilities", client_capabilities()},
        {"clientInfo", {{"name", info_.name}, {"version", info_.version}}},
    };

    const nlohmann::json result = call(method::kInitialize, params, options_.default_timeout);

    ServerInfo info;
    info.protocol_version = string_or(result, "protocolVersion");
    if (result.contains("capabilities")) {
        info.capabilities = result["capabilities"];
    }
    if (result.contains("serverInfo")) {
        info.name = string_or(result["serverInfo"], "name");
        info.version = string_or(result["serverInfo"], "version");
    }
    info.instructions = string_or(result, "instructions");

    // Version negotiation, the half every cook-off entrant skipped. The server answers
    // with the version it will actually speak; if we cannot speak it, saying so here is
    // far better than failing on an unexpected field ten calls later.
    if (!info.protocol_version.empty() && !is_supported_version(info.protocol_version)) {
        throw McpError(to_int(ErrorCode::kInvalidRequest),
                       "Server requires unsupported MCP protocol version '" +
                           info.protocol_version + "'");
    }

    server_info_ = info;
    initialized_.store(true, std::memory_order_release);

    // The handshake is only complete once this is sent, and a spec-conformant server
    // refuses everything else until it arrives.
    notify(notification::kInitialized, nullptr);
    return info;
}

// ---------------------------------------------------------------------------

Client::Call Client::send_async(std::string_view method_name, const nlohmann::json& params,
                                ProgressFn on_progress) {
    const Id id = next_id();

    nlohmann::json p = params;
    if (on_progress) {
        // The progress token is the request id. It only has to be unique per session,
        // and reusing the id means a progress notification needs no second lookup table.
        if (!p.is_object()) {
            p = nlohmann::json::object();
        }
        p["_meta"]["progressToken"] = id.to_json();
    }

    std::promise<nlohmann::json> promise;
    std::future<nlohmann::json> fut = promise.get_future();

    {
        const std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_.emplace(id, Pending{std::move(promise), std::move(on_progress)});
    }

    if (!transport_ || !transport_->send(make_request(id, method_name, p))) {
        // Retract the registration, or a failed write leaks an entry that nothing will
        // ever complete.
        Pending dead;
        {
            const std::lock_guard<std::mutex> lock(pending_mutex_);
            const auto it = pending_.find(id);
            if (it != pending_.end()) {
                dead = std::move(it->second);
                pending_.erase(it);
            }
        }
        dead.promise.set_exception(std::make_exception_ptr(
            McpError(to_int(ErrorCode::kInternalError),
                     "failed to write request to MCP server (connection closed)")));
    }

    return Call{id, std::move(fut)};
}

nlohmann::json Client::await(std::future<nlohmann::json> fut, const Id& id,
                             std::chrono::milliseconds timeout) {
    if (fut.wait_for(timeout) == std::future_status::timeout) {
        // Retire the entry and tell the server to stop working on it. Without the
        // notification the server keeps computing a result nobody will read.
        {
            const std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_.erase(id);
        }
        notify(notification::kCancelled,
               nlohmann::json{{"requestId", id.to_json()}, {"reason", "client timeout"}});
        throw McpError(to_int(ErrorCode::kRequestCancelled),
                       "MCP request " + id.debug() + " timed out after " +
                           std::to_string(timeout.count()) + " ms");
    }
    return fut.get();
}

nlohmann::json Client::call(std::string_view method_name, const nlohmann::json& params,
                            std::optional<std::chrono::milliseconds> timeout) {
    Call c = send_async(method_name, params, {});
    return await(std::move(c.future), c.id, timeout.value_or(options_.default_timeout));
}

void Client::notify(std::string_view method_name, const nlohmann::json& params) {
    if (transport_) {
        transport_->send(make_notification(method_name, params));
    }
}

// ---------------------------------------------------------------------------

void Client::on_message(const nlohmann::json& raw) {
    const Message msg = classify(raw);
    switch (msg.kind) {
    case MessageKind::kResponse:
        handle_response(msg);
        return;
    case MessageKind::kRequest:
        handle_server_request(msg);
        return;
    case MessageKind::kNotification:
        handle_server_notification(msg);
        return;
    case MessageKind::kInvalid:
        return;
    }
}

void Client::handle_response(const Message& msg) {
    Pending pending;
    bool found = false;
    {
        const std::lock_guard<std::mutex> lock(pending_mutex_);
        const auto it = pending_.find(msg.id);
        if (it != pending_.end()) {
            pending = std::move(it->second);
            pending_.erase(it);
            found = true;
        }
    }
    if (!found) {
        return; // late reply to something already timed out
    }

    if (msg.error.has_value()) {
        pending.promise.set_exception(
            std::make_exception_ptr(McpError(msg.error->code, msg.error->message)));
    } else {
        pending.promise.set_value(msg.result);
    }
}

void Client::handle_server_request(const Message& msg) {
    // A server may call back into the client. Answering `ping` keeps health checks
    // working; anything we did not declare a capability for gets a clean -32601 rather
    // than silence, because silence hangs the server until its own timeout.
    if (msg.method == method::kPing) {
        transport_->send(make_response(msg.id, nlohmann::json::object()));
        return;
    }
    if (msg.method == method::kRootsList) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : roots_) {
            nlohmann::json j{{"uri", r.uri}};
            if (!r.name.empty()) {
                j["name"] = r.name;
            }
            arr.push_back(std::move(j));
        }
        transport_->send(make_response(msg.id, nlohmann::json{{"roots", std::move(arr)}}));
        return;
    }
    transport_->send(make_error(msg.id, ErrorCode::kMethodNotFound,
                                "Client does not implement " + msg.method));
}

void Client::handle_server_notification(const Message& msg) {
    if (msg.method == notification::kProgress) {
        const auto& p = obj_or_empty(msg.params);
        if (!p.contains("progressToken")) {
            return;
        }
        const Id token = Id::from_json(p["progressToken"]);
        ProgressFn fn;
        {
            const std::lock_guard<std::mutex> lock(pending_mutex_);
            const auto it = pending_.find(token);
            if (it != pending_.end()) {
                fn = it->second.on_progress;
            }
        }
        if (fn) {
            const double progress = p.value("progress", 0.0);
            std::optional<double> total;
            if (p.contains("total") && p["total"].is_number()) {
                total = p["total"].get<double>();
            }
            fn(progress, total, string_or(msg.params, "message"));
        }
        return;
    }

    std::function<void()> simple;
    {
        const std::lock_guard<std::mutex> lock(handler_mutex_);
        if (msg.method == notification::kMessage) {
            if (on_log_) {
                LogLevel level = LogLevel::kInfo;
                // An unrecognised level is not worth dropping the message over; it
                // stays at the kInfo default.
                static_cast<void>(parse_log_level(string_or(msg.params, "level", "info"), level));
                const auto& p = obj_or_empty(msg.params);
                on_log_(level, string_or(msg.params, "logger"),
                        p.contains("data") ? p["data"] : nlohmann::json());
            }
            return;
        }
        if (msg.method == notification::kResourcesUpdated) {
            if (on_resource_updated_) {
                on_resource_updated_(string_or(msg.params, "uri"));
            }
            return;
        }
        if (msg.method == notification::kToolsListChanged) {
            simple = on_tools_changed_;
        } else if (msg.method == notification::kPromptsListChanged) {
            simple = on_prompts_changed_;
        } else if (msg.method == notification::kResourcesListChanged) {
            simple = on_resources_changed_;
        }
    }
    if (simple) {
        simple();
    }
}

// ---------------------------------------------------------------------------

nlohmann::json Client::list_paged(std::string_view method_name, const char* key) {
    nlohmann::json all = nlohmann::json::array();
    nlohmann::json params = nlohmann::json::object();

    for (;;) {
        const nlohmann::json page = call(method_name, params, options_.default_timeout);
        if (page.contains(key) && page[key].is_array()) {
            for (const auto& item : page[key]) {
                all.push_back(item);
            }
        }
        if (!page.contains("nextCursor") || !page["nextCursor"].is_string()) {
            break;
        }
        params["cursor"] = page["nextCursor"];
    }
    return all;
}

std::vector<Tool> Client::list_tools() {
    std::vector<Tool> out;
    for (const auto& j : list_paged(method::kToolsList, "tools")) {
        out.push_back(tool_from_json(j));
    }
    return out;
}

ToolResult Client::call_tool(std::string_view name, const nlohmann::json& arguments,
                             ProgressFn on_progress,
                             std::optional<std::chrono::milliseconds> timeout) {
    nlohmann::json params{{"name", std::string(name)}, {"arguments", arguments}};

    Call c = send_async(method::kToolsCall, params, std::move(on_progress));
    const nlohmann::json result =
        await(std::move(c.future), c.id, timeout.value_or(options_.default_timeout));

    ToolResult r;
    if (result.contains("content") && result["content"].is_array()) {
        r.content = result["content"];
    }
    r.is_error = obj_or_empty(result).value("isError", false);
    if (result.contains("structuredContent")) {
        r.structured = result["structuredContent"];
    }
    return r;
}

std::vector<Resource> Client::list_resources() {
    std::vector<Resource> out;
    for (const auto& j : list_paged(method::kResourcesList, "resources")) {
        out.push_back(resource_from_json(j));
    }
    return out;
}

std::vector<ResourceTemplate> Client::list_resource_templates() {
    std::vector<ResourceTemplate> out;
    for (const auto& j : list_paged(method::kResourcesTemplatesList, "resourceTemplates")) {
        out.push_back(template_from_json(j));
    }
    return out;
}

std::vector<ResourceContents> Client::read_resource(std::string_view uri) {
    const nlohmann::json result =
        call(method::kResourcesRead, nlohmann::json{{"uri", std::string(uri)}});

    std::vector<ResourceContents> out;
    if (result.contains("contents") && result["contents"].is_array()) {
        for (const auto& c : result["contents"]) {
            ResourceContents rc;
            rc.uri = string_or(c, "uri");
            rc.mime_type = string_or(c, "mimeType");
            const auto& o = obj_or_empty(c);
            if (o.contains("text") && o["text"].is_string()) {
                rc.text = o["text"].get<std::string>();
            }
            if (o.contains("blob") && o["blob"].is_string()) {
                rc.blob = o["blob"].get<std::string>();
            }
            out.push_back(std::move(rc));
        }
    }
    return out;
}

void Client::subscribe_resource(std::string_view uri) {
    call(method::kResourcesSubscribe, nlohmann::json{{"uri", std::string(uri)}});
}

void Client::unsubscribe_resource(std::string_view uri) {
    call(method::kResourcesUnsubscribe, nlohmann::json{{"uri", std::string(uri)}});
}

std::vector<Prompt> Client::list_prompts() {
    std::vector<Prompt> out;
    for (const auto& j : list_paged(method::kPromptsList, "prompts")) {
        out.push_back(prompt_from_json(j));
    }
    return out;
}

std::vector<PromptMessage> Client::get_prompt(std::string_view name,
                                              const nlohmann::json& arguments) {
    const nlohmann::json result =
        call(method::kPromptsGet,
             nlohmann::json{{"name", std::string(name)}, {"arguments", arguments}});

    std::vector<PromptMessage> out;
    if (result.contains("messages") && result["messages"].is_array()) {
        for (const auto& m : result["messages"]) {
            PromptMessage pm;
            pm.role = string_or(m, "role", "user");
            const auto& o = obj_or_empty(m);
            if (o.contains("content")) {
                pm.content = o["content"];
            }
            out.push_back(std::move(pm));
        }
    }
    return out;
}

void Client::ping() {
    call(method::kPing, nullptr);
}

void Client::set_log_level(LogLevel level) {
    call(method::kLoggingSetLevel,
         nlohmann::json{{"level", std::string(to_string(level))}});
}

// ---------------------------------------------------------------------------

void Client::on_log(
    std::function<void(LogLevel, std::string_view, const nlohmann::json&)> fn) {
    const std::lock_guard<std::mutex> lock(handler_mutex_);
    on_log_ = std::move(fn);
}

void Client::on_tools_changed(std::function<void()> fn) {
    const std::lock_guard<std::mutex> lock(handler_mutex_);
    on_tools_changed_ = std::move(fn);
}

void Client::on_prompts_changed(std::function<void()> fn) {
    const std::lock_guard<std::mutex> lock(handler_mutex_);
    on_prompts_changed_ = std::move(fn);
}

void Client::on_resources_changed(std::function<void()> fn) {
    const std::lock_guard<std::mutex> lock(handler_mutex_);
    on_resources_changed_ = std::move(fn);
}

void Client::on_resource_updated(std::function<void(std::string_view)> fn) {
    const std::lock_guard<std::mutex> lock(handler_mutex_);
    on_resource_updated_ = std::move(fn);
}

void Client::on_server_stderr(std::function<void(std::string_view)> fn) {
    const std::lock_guard<std::mutex> lock(handler_mutex_);
    on_stderr_ = std::move(fn);
}

void Client::set_roots(std::vector<Root> roots) {
    roots_ = std::move(roots);
    options_.declare_roots = true;
    if (initialized_.load(std::memory_order_acquire)) {
        notify(notification::kRootsListChanged, nullptr);
    }
}

} // namespace lmp::mcp
