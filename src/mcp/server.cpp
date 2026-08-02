#include "src/mcp/server.hpp"

#include <algorithm>
#include <charconv>
#include <utility>

namespace lmp::mcp {

namespace {

// Cursors are opaque to the client by contract, so an index is a legitimate encoding.
// It is parsed defensively anyway: a client that invents a cursor gets -32602, not a
// crash or a silently empty page.
bool parse_cursor(const nlohmann::json& params, std::size_t& out) {
    out = 0;
    if (!params.is_object() || !params.contains("cursor")) {
        return true;
    }
    const auto& c = params["cursor"];
    if (!c.is_string()) {
        return false;
    }
    const std::string s = c.get<std::string>();
    std::size_t value = 0;
    const char* begin = s.data();
    const char* end = begin + s.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end) {
        return false;
    }
    out = value;
    return true;
}

const nlohmann::json& params_or_empty(const nlohmann::json& params) {
    static const nlohmann::json kEmpty = nlohmann::json::object();
    return params.is_object() ? params : kEmpty;
}

std::string require_string(const nlohmann::json& params, const char* key) {
    const auto& p = params_or_empty(params);
    if (!p.contains(key) || !p[key].is_string()) {
        return {};
    }
    return p[key].get<std::string>();
}

// The progress token lives in params._meta.progressToken and may be a string or a
// number; it is echoed back verbatim, so it is carried as raw json.
nlohmann::json extract_progress_token(const nlohmann::json& params) {
    const auto& p = params_or_empty(params);
    if (!p.contains("_meta") || !p["_meta"].is_object()) {
        return nullptr;
    }
    const auto& meta = p["_meta"];
    if (!meta.contains("progressToken")) {
        return nullptr;
    }
    return meta["progressToken"];
}

} // namespace

// ---------------------------------------------------------------------------
// RequestContext
// ---------------------------------------------------------------------------

void RequestContext::report_progress(double progress, std::optional<double> total,
                                     std::string_view message) {
    if (progress_token_.is_null()) {
        return; // client did not ask for progress
    }
    nlohmann::json params{{"progressToken", progress_token_}, {"progress", progress}};
    if (total.has_value()) {
        params["total"] = *total;
    }
    if (!message.empty()) {
        params["message"] = std::string(message);
    }
    server_.send(make_notification(notification::kProgress, params));
}

bool RequestContext::cancelled() const {
    const std::lock_guard<std::mutex> lock(server_.cancel_mutex_);
    return server_.cancelled_.find(id_) != server_.cancelled_.end();
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

Server::Server(Info info) : Server(std::move(info), Options{}) {}

Server::Server(Info info, Options options) : info_(std::move(info)), options_(options) {
    if (options_.worker_threads == 0) {
        options_.worker_threads = 1;
    }
    if (options_.page_size == 0) {
        options_.page_size = 100;
    }
}

Server::~Server() {
    stop_workers();
}

void Server::fail(ErrorCode code, std::string message) {
    throw Failure{code, std::move(message)};
}

void Server::add_tool(Tool tool, ToolFn fn) {
    tool_index_[tool.name] = tools_.size();
    tools_.push_back(ToolEntry{std::move(tool), std::move(fn)});
}

void Server::add_resource(Resource resource, ResourceReadFn fn) {
    resource_index_[resource.uri] = resources_.size();
    resources_.push_back(ResourceEntry{std::move(resource), std::move(fn)});
}

void Server::add_resource_template(ResourceTemplate tmpl, ResourceReadFn fn) {
    templates_.push_back(TemplateEntry{std::move(tmpl), std::move(fn)});
}

void Server::add_prompt(Prompt prompt, PromptFn fn) {
    prompt_index_[prompt.name] = prompts_.size();
    prompts_.push_back(PromptEntry{std::move(prompt), std::move(fn)});
}

void Server::set_completion_handler(CompletionFn fn) {
    completion_ = std::move(fn);
}

nlohmann::json Server::capabilities() const {
    nlohmann::json caps = nlohmann::json::object();
    if (!tools_.empty()) {
        caps["tools"] = nlohmann::json{{"listChanged", true}};
    }
    if (!resources_.empty() || !templates_.empty()) {
        caps["resources"] = nlohmann::json{{"subscribe", true}, {"listChanged", true}};
    }
    if (!prompts_.empty()) {
        caps["prompts"] = nlohmann::json{{"listChanged", true}};
    }
    if (completion_) {
        caps["completions"] = nlohmann::json::object();
    }
    caps["logging"] = nlohmann::json::object();
    return caps;
}

void Server::send(const nlohmann::json& j) {
    if (transport_ != nullptr) {
        transport_->send(j);
    }
}

void Server::send_error(const Id& id, ErrorCode code, std::string_view message) {
    send(make_error(id, code, message));
}

// ---------------------------------------------------------------------------

void Server::run(std::unique_ptr<Transport> transport) {
    // Held here so it cannot outlive, or be outlived by, the session it carries.
    const std::unique_ptr<Transport> owned = std::move(transport);
    if (owned) {
        run(*owned);
    }
}

void Server::run(Transport& transport) {
    transport_ = &transport;
    closed_ = false;
    start_workers();

    Transport::Handlers handlers;
    handlers.on_message = [this](const nlohmann::json& raw) { on_message(raw); };
    handlers.on_parse_error = [this](std::string_view, std::string_view reason) {
        // A response to an unparseable request carries a null id -- there was no id to
        // echo. This is the one place a null id is correct rather than malformed.
        send(make_error(Id::none(), ErrorCode::kParseError, reason));
    };
    handlers.on_closed = [this] {
        const std::lock_guard<std::mutex> lock(done_mutex_);
        closed_ = true;
        done_cv_.notify_all();
    };

    transport.start(std::move(handlers));

    {
        std::unique_lock<std::mutex> lock(done_mutex_);
        done_cv_.wait(lock, [this] { return closed_; });
    }

    // Drain and join before returning. Entrant S4 returned from main() with detached
    // workers still writing to stdout and lost replies as a result; this is the fix, and
    // it is why the pool is joined rather than detached.
    stop_workers();
    transport_ = nullptr;
}

void Server::request_stop() {
    const std::lock_guard<std::mutex> lock(done_mutex_);
    closed_ = true;
    done_cv_.notify_all();
}

void Server::start_workers() {
    {
        const std::lock_guard<std::mutex> lock(queue_mutex_);
        draining_ = false;
    }
    workers_.reserve(options_.worker_threads);
    for (std::size_t i = 0; i < options_.worker_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

void Server::stop_workers() {
    {
        const std::lock_guard<std::mutex> lock(queue_mutex_);
        draining_ = true;
    }
    queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_.clear();
}

void Server::worker_loop() {
    for (;;) {
        Message msg;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return draining_ || !queue_.empty(); });
            // Finish what is queued even while draining: a request already accepted
            // deserves its reply.
            if (queue_.empty()) {
                if (draining_) {
                    return;
                }
                continue;
            }
            msg = std::move(queue_.front());
            queue_.pop_front();
        }
        run_request(std::move(msg));
    }
}

// ---------------------------------------------------------------------------

void Server::on_message(const nlohmann::json& raw) {
    const Message msg = classify(raw);

    switch (msg.kind) {
    case MessageKind::kRequest:
        handle_request(msg);
        return;
    case MessageKind::kNotification:
        handle_notification(msg);
        return;
    case MessageKind::kResponse:
        // A reply to something we asked the client (roots/list, sampling). Not wired
        // to a waiter yet; dropping it is correct until it is.
        return;
    case MessageKind::kInvalid:
        // Well-formed JSON, malformed JSON-RPC. Distinct from a parse error, and it
        // gets the distinct code the spec assigns it.
        send(make_error(msg.id, ErrorCode::kInvalidRequest, msg.invalid_reason));
        return;
    }
}

void Server::handle_notification(const Message& msg) {
    if (msg.method == notification::kInitialized) {
        initialized_.store(true, std::memory_order_release);
        return;
    }
    if (msg.method == notification::kCancelled) {
        const auto& p = params_or_empty(msg.params);
        if (p.contains("requestId")) {
            const Id target = Id::from_json(p["requestId"]);
            if (!target.is_none()) {
                const std::lock_guard<std::mutex> lock(cancel_mutex_);
                cancelled_.insert(target);
            }
        }
        return;
    }
    // Unknown notification: ignore in silence. Answering it is a protocol violation --
    // entrant S5 returned -32601 here, which is a reply to something that had no id.
}

void Server::handle_request(const Message& msg) {
    // The lifecycle gate. `initialize` and `ping` are the only two methods legal before
    // the handshake completes.
    const bool is_lifecycle =
        msg.method == method::kInitialize || msg.method == method::kPing;
    if (!is_lifecycle && !initialized_.load(std::memory_order_acquire)) {
        send_error(msg.id, ErrorCode::kNotInitialized,
                   "Server not initialized: send `initialize` and `notifications/initialized` first");
        return;
    }

    // initialize is answered inline. It is cheap, it must not race another initialize,
    // and the handshake should not queue behind a slow tool call.
    if (msg.method == method::kInitialize) {
        send(handle_initialize(msg));
        return;
    }

    const std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.push_back(msg);
    queue_cv_.notify_one();
}

void Server::run_request(Message msg) {
    RequestContext ctx(*this, msg.id, extract_progress_token(msg.params));

    nlohmann::json response;
    try {
        response = dispatch(msg, ctx);
    } catch (const Failure& f) {
        response = make_error(msg.id, f.code, f.message);
    } catch (const std::exception& e) {
        // A handler that throws must not take the server down with it.
        response = make_error(msg.id, ErrorCode::kInternalError, e.what());
    } catch (...) {
        response = make_error(msg.id, ErrorCode::kInternalError, "unknown exception in handler");
    }

    // A cancelled request gets no reply, per spec, and its id is retired here so the
    // cancelled set cannot grow without bound over a long session.
    bool was_cancelled = false;
    {
        const std::lock_guard<std::mutex> lock(cancel_mutex_);
        const auto it = cancelled_.find(msg.id);
        if (it != cancelled_.end()) {
            cancelled_.erase(it);
            was_cancelled = true;
        }
    }
    if (!was_cancelled) {
        send(response);
    }
}

nlohmann::json Server::dispatch(const Message& msg, RequestContext& ctx) {
    if (msg.method == method::kPing) {
        return make_response(msg.id, nlohmann::json::object());
    }
    if (msg.method == method::kToolsList) {
        return handle_tools_list(msg);
    }
    if (msg.method == method::kToolsCall) {
        return handle_tools_call(msg, ctx);
    }
    if (msg.method == method::kResourcesList) {
        return handle_resources_list(msg);
    }
    if (msg.method == method::kResourcesTemplatesList) {
        return handle_resource_templates_list(msg);
    }
    if (msg.method == method::kResourcesRead) {
        return handle_resources_read(msg, ctx);
    }
    if (msg.method == method::kResourcesSubscribe || msg.method == method::kResourcesUnsubscribe) {
        // Accepted so a client that subscribes does not see a hard failure; updates are
        // delivered by notify_resource_updated when the owner calls it.
        return make_response(msg.id, nlohmann::json::object());
    }
    if (msg.method == method::kPromptsList) {
        return handle_prompts_list(msg);
    }
    if (msg.method == method::kPromptsGet) {
        return handle_prompts_get(msg, ctx);
    }
    if (msg.method == method::kCompletionComplete) {
        return handle_completion(msg);
    }
    if (msg.method == method::kLoggingSetLevel) {
        return handle_logging_set_level(msg);
    }
    return make_error(msg.id, ErrorCode::kMethodNotFound, "Method not found: " + msg.method);
}

// ---------------------------------------------------------------------------

nlohmann::json Server::handle_initialize(const Message& msg) {
    initialize_received_.store(true, std::memory_order_release);

    // Version negotiation, which not one cook-off entrant attempted. The rule: if we
    // support what they asked for, agree to it; otherwise answer with our latest and
    // let the client decide whether it can live with that.
    const std::string requested = require_string(msg.params, "protocolVersion");
    negotiated_version_ = is_supported_version(requested) ? requested
                                                          : std::string(kProtocolVersion);

    nlohmann::json result{
        {"protocolVersion", negotiated_version_},
        {"capabilities", capabilities()},
        {"serverInfo", {{"name", info_.name}, {"version", info_.version}}},
    };
    if (!info_.instructions.empty()) {
        result["instructions"] = info_.instructions;
    }
    return make_response(msg.id, result);
}

nlohmann::json Server::handle_tools_list(const Message& msg) {
    std::size_t cursor = 0;
    if (!parse_cursor(msg.params, cursor)) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "invalid cursor");
    }
    if (cursor > tools_.size()) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "cursor out of range");
    }

    nlohmann::json list = nlohmann::json::array();
    const std::size_t end = std::min(cursor + options_.page_size, tools_.size());
    for (std::size_t i = cursor; i < end; ++i) {
        list.push_back(tools_[i].tool.to_json());
    }
    nlohmann::json result{{"tools", std::move(list)}};
    if (end < tools_.size()) {
        result["nextCursor"] = std::to_string(end);
    }
    return make_response(msg.id, result);
}

nlohmann::json Server::handle_tools_call(const Message& msg, RequestContext& ctx) {
    const std::string name = require_string(msg.params, "name");
    if (name.empty()) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "params.name is required");
    }
    const auto it = tool_index_.find(name);
    if (it == tool_index_.end()) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "Unknown tool: " + name);
    }

    const auto& p = params_or_empty(msg.params);
    const nlohmann::json args =
        p.contains("arguments") ? p["arguments"] : nlohmann::json::object();

    // A tool that throws yields isError:true rather than a JSON-RPC error: the call was
    // made and it failed, which is information the model should see and act on.
    ToolResult result;
    try {
        result = tools_[it->second].fn(args, ctx);
    } catch (const Failure& f) {
        throw;
    } catch (const std::exception& e) {
        result = ToolResult::failure(e.what());
    }
    return make_response(msg.id, result.to_json());
}

nlohmann::json Server::handle_resources_list(const Message& msg) {
    std::size_t cursor = 0;
    if (!parse_cursor(msg.params, cursor)) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "invalid cursor");
    }
    if (cursor > resources_.size()) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "cursor out of range");
    }
    nlohmann::json list = nlohmann::json::array();
    const std::size_t end = std::min(cursor + options_.page_size, resources_.size());
    for (std::size_t i = cursor; i < end; ++i) {
        list.push_back(resources_[i].resource.to_json());
    }
    nlohmann::json result{{"resources", std::move(list)}};
    if (end < resources_.size()) {
        result["nextCursor"] = std::to_string(end);
    }
    return make_response(msg.id, result);
}

nlohmann::json Server::handle_resource_templates_list(const Message& msg) {
    nlohmann::json list = nlohmann::json::array();
    for (const auto& t : templates_) {
        list.push_back(t.tmpl.to_json());
    }
    return make_response(msg.id, nlohmann::json{{"resourceTemplates", std::move(list)}});
}

nlohmann::json Server::handle_resources_read(const Message& msg, RequestContext& ctx) {
    const std::string uri = require_string(msg.params, "uri");
    if (uri.empty()) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "params.uri is required");
    }

    const auto it = resource_index_.find(uri);
    ResourceReadFn fn;
    if (it != resource_index_.end()) {
        fn = resources_[it->second].fn;
    } else if (!templates_.empty()) {
        // No URI-template matching yet: the first template handler is offered the URI
        // and may decline by returning empty.
        fn = templates_.front().fn;
    }
    if (!fn) {
        return make_error(msg.id, ErrorCode::kResourceNotFound, "Resource not found: " + uri);
    }

    const std::vector<ResourceContents> contents = fn(uri, ctx);
    if (contents.empty()) {
        return make_error(msg.id, ErrorCode::kResourceNotFound, "Resource not found: " + uri);
    }
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& c : contents) {
        arr.push_back(c.to_json());
    }
    return make_response(msg.id, nlohmann::json{{"contents", std::move(arr)}});
}

nlohmann::json Server::handle_prompts_list(const Message& msg) {
    std::size_t cursor = 0;
    if (!parse_cursor(msg.params, cursor)) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "invalid cursor");
    }
    if (cursor > prompts_.size()) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "cursor out of range");
    }
    nlohmann::json list = nlohmann::json::array();
    const std::size_t end = std::min(cursor + options_.page_size, prompts_.size());
    for (std::size_t i = cursor; i < end; ++i) {
        list.push_back(prompts_[i].prompt.to_json());
    }
    nlohmann::json result{{"prompts", std::move(list)}};
    if (end < prompts_.size()) {
        result["nextCursor"] = std::to_string(end);
    }
    return make_response(msg.id, result);
}

nlohmann::json Server::handle_prompts_get(const Message& msg, RequestContext& ctx) {
    const std::string name = require_string(msg.params, "name");
    if (name.empty()) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "params.name is required");
    }
    const auto it = prompt_index_.find(name);
    if (it == prompt_index_.end()) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "Unknown prompt: " + name);
    }

    const auto& p = params_or_empty(msg.params);
    const nlohmann::json args =
        p.contains("arguments") ? p["arguments"] : nlohmann::json::object();

    const std::vector<PromptMessage> messages = prompts_[it->second].fn(args, ctx);
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& m : messages) {
        arr.push_back(m.to_json());
    }
    nlohmann::json result{{"messages", std::move(arr)}};
    if (!prompts_[it->second].prompt.description.empty()) {
        result["description"] = prompts_[it->second].prompt.description;
    }
    return make_response(msg.id, result);
}

nlohmann::json Server::handle_completion(const Message& msg) {
    if (!completion_) {
        return make_error(msg.id, ErrorCode::kMethodNotFound, "Completion is not supported");
    }
    const auto& p = params_or_empty(msg.params);
    std::string arg_name;
    std::string partial;
    if (p.contains("argument") && p["argument"].is_object()) {
        const auto& a = p["argument"];
        if (a.contains("name") && a["name"].is_string()) {
            arg_name = a["name"].get<std::string>();
        }
        if (a.contains("value") && a["value"].is_string()) {
            partial = a["value"].get<std::string>();
        }
    }

    std::vector<std::string> values = completion_(arg_name, partial);
    const bool truncated = values.size() > 100;
    if (truncated) {
        values.resize(100); // spec caps a completion page at 100
    }
    return make_response(msg.id, nlohmann::json{{"completion",
                                                 {{"values", values},
                                                  {"total", values.size()},
                                                  {"hasMore", truncated}}}});
}

nlohmann::json Server::handle_logging_set_level(const Message& msg) {
    const std::string level_name = require_string(msg.params, "level");
    LogLevel level = LogLevel::kInfo;
    if (!parse_log_level(level_name, level)) {
        return make_error(msg.id, ErrorCode::kInvalidParams, "Unknown log level: " + level_name);
    }
    log_level_.store(static_cast<int>(level), std::memory_order_release);
    return make_response(msg.id, nlohmann::json::object());
}

// ---------------------------------------------------------------------------

void Server::log(LogLevel level, std::string_view logger, nlohmann::json data) {
    if (static_cast<int>(level) < log_level_.load(std::memory_order_acquire)) {
        return;
    }
    nlohmann::json params{{"level", std::string(to_string(level))}, {"data", std::move(data)}};
    if (!logger.empty()) {
        params["logger"] = std::string(logger);
    }
    send(make_notification(notification::kMessage, params));
}

void Server::notify_tools_list_changed() {
    send(make_notification(notification::kToolsListChanged, nullptr));
}

void Server::notify_prompts_list_changed() {
    send(make_notification(notification::kPromptsListChanged, nullptr));
}

void Server::notify_resources_list_changed() {
    send(make_notification(notification::kResourcesListChanged, nullptr));
}

void Server::notify_resource_updated(std::string_view uri) {
    send(make_notification(notification::kResourcesUpdated,
                           nlohmann::json{{"uri", std::string(uri)}}));
}

} // namespace lmp::mcp
