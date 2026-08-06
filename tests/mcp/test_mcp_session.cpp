// A full client/server session over the in-process transport.
//
// Both halves are ours, so this exercises the protocol logic without a subprocess: the
// handshake, every feature, progress, cancellation, and -- the check that matters most
// for correctness under load -- concurrent requests all landing on the right future.

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "src/mcp/client.hpp"
#include "src/mcp/inproc_transport.hpp"
#include "src/mcp/server.hpp"

#include "tests/check.hpp"

using namespace lmp::mcp;

namespace {

// Owns a running server and a connected client, and shuts both down in the right order.
class Session {
public:
    Session() {
        Server::Info info;
        info.name = "test-server";
        info.version = "9.9.9";
        info.instructions = "test instructions";
        server_ = std::make_unique<Server>(info);
        install(*server_);

        InProcessTransport::Pair pair = InProcessTransport::make_pair();
        // The server half must be OWNED for the life of the session. Holding only a
        // raw pointer here and letting `pair` die at the end of this constructor is a
        // use-after-free that segfaults on the first message.
        server_side_ = std::move(pair.server);
        thread_ = std::thread([this] { server_->run(*server_side_); });

        client_ = std::make_unique<Client>(Client::Info{"test-client", "1.0"});
        client_->connect(std::move(pair.client));
    }

    ~Session() {
        client_->close();
        server_->request_stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    Client& client() { return *client_; }
    Server& server() { return *server_; }

    static std::atomic<int>& cancel_observations() {
        static std::atomic<int> n{0};
        return n;
    }

private:
    static void install(Server& s) {
        Tool echo;
        echo.name = "echo";
        echo.description = "echo";
        echo.input_schema = {{"type", "object"},
                             {"properties", {{"text", {{"type", "string"}}}}}};
        s.add_tool(std::move(echo), [](const nlohmann::json& args, RequestContext&) {
            return ToolResult::text(args.value("text", ""));
        });

        // Returns its own argument so a concurrency test can prove each reply reached
        // the caller that asked for it.
        Tool ident;
        ident.name = "identity";
        ident.description = "returns the n it was given";
        ident.input_schema = {{"type", "object"}, {"properties", {{"n", {{"type", "integer"}}}}}};
        s.add_tool(std::move(ident), [](const nlohmann::json& args, RequestContext&) {
            return ToolResult::text(std::to_string(args.value("n", -1)));
        });

        Tool boom;
        boom.name = "boom";
        boom.description = "always fails";
        s.add_tool(std::move(boom), [](const nlohmann::json&, RequestContext&) {
            return ToolResult::failure("deliberate failure");
        });

        Tool thrower;
        thrower.name = "thrower";
        thrower.description = "throws a C++ exception";
        s.add_tool(std::move(thrower), [](const nlohmann::json&, RequestContext&) -> ToolResult {
            throw std::runtime_error("handler exploded");
        });

        Tool slow;
        slow.name = "slow";
        slow.description = "reports progress, honours cancellation";
        slow.input_schema = {{"type", "object"}, {"properties", {{"steps", {{"type", "integer"}}}}}};
        s.add_tool(std::move(slow), [](const nlohmann::json& args, RequestContext& ctx) {
            const int steps = args.value("steps", 4);
            for (int i = 0; i < steps; ++i) {
                if (ctx.cancelled()) {
                    cancel_observations().fetch_add(1);
                    return ToolResult::failure("cancelled");
                }
                ctx.report_progress(static_cast<double>(i + 1), static_cast<double>(steps), "tick");
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            return ToolResult::text("done");
        });

        Resource r;
        r.uri = "test://doc";
        r.name = "doc";
        r.mime_type = "text/plain";
        s.add_resource(std::move(r), [](const std::string& uri, RequestContext&) {
            return std::vector<ResourceContents>{ResourceContents::from_text(uri, "body text")};
        });

        Prompt p;
        p.name = "greet";
        p.description = "a greeting";
        p.arguments.push_back(PromptArgument{"who", "who to greet", true});
        s.add_prompt(std::move(p), [](const nlohmann::json& args, RequestContext&) {
            return std::vector<PromptMessage>{
                PromptMessage::user_text("hello " + args.value("who", "world"))};
        });
    }

    std::unique_ptr<Server> server_;
    std::unique_ptr<Client> client_;
    std::unique_ptr<InProcessTransport> server_side_;
    std::thread thread_;
};

std::string text_of(const ToolResult& r) {
    if (r.content.is_array() && !r.content.empty() && r.content[0].contains("text")) {
        return r.content[0]["text"].get<std::string>();
    }
    return {};
}

} // namespace

TEST(handshake_negotiates_and_reports_server_identity) {
    Session s;
    const ServerInfo info = s.client().initialize();
    CHECK_EQ(info.name, std::string("test-server"));
    CHECK_EQ(info.version, std::string("9.9.9"));
    CHECK_EQ(info.protocol_version, std::string(kProtocolVersion));
    CHECK_EQ(info.instructions, std::string("test instructions"));
    CHECK(info.supports_tools());
    CHECK(info.supports_resources());
    CHECK(info.supports_prompts());
}

TEST(requests_before_initialize_are_refused) {
    // The lifecycle gate only S7 of seven entrants implemented.
    Session s;
    bool threw = false;
    try {
        static_cast<void>(s.client().list_tools());
    } catch (const McpError& e) {
        threw = true;
        CHECK_EQ(e.code(), to_int(ErrorCode::kNotInitialized));
    }
    CHECK(threw);
}

TEST(tools_round_trip) {
    Session s;
    static_cast<void>(s.client().initialize());

    const std::vector<Tool> tools = s.client().list_tools();
    CHECK_EQ(tools.size(), std::size_t(5));

    const ToolResult r = s.client().call_tool("echo", nlohmann::json{{"text", "hi there"}});
    CHECK(!r.is_error);
    CHECK_EQ(text_of(r), std::string("hi there"));
}

TEST(tool_failure_is_a_result_not_a_protocol_error) {
    Session s;
    static_cast<void>(s.client().initialize());
    const ToolResult r = s.client().call_tool("boom", nlohmann::json::object());
    CHECK(r.is_error);
    CHECK_EQ(text_of(r), std::string("deliberate failure"));
}

TEST(a_throwing_handler_becomes_a_tool_failure_not_a_crash) {
    Session s;
    static_cast<void>(s.client().initialize());
    const ToolResult r = s.client().call_tool("thrower", nlohmann::json::object());
    CHECK(r.is_error);
    CHECK_EQ(text_of(r), std::string("handler exploded"));

    // And the session survives it.
    const ToolResult after = s.client().call_tool("echo", nlohmann::json{{"text", "still alive"}});
    CHECK_EQ(text_of(after), std::string("still alive"));
}

TEST(unknown_tool_is_rejected) {
    Session s;
    static_cast<void>(s.client().initialize());
    bool threw = false;
    try {
        static_cast<void>(s.client().call_tool("no_such_tool", nlohmann::json::object()));
    } catch (const McpError& e) {
        threw = true;
        CHECK_EQ(e.code(), to_int(ErrorCode::kInvalidParams));
    }
    CHECK(threw);
}

TEST(resources_and_prompts_round_trip) {
    Session s;
    static_cast<void>(s.client().initialize());

    const std::vector<Resource> res = s.client().list_resources();
    REQUIRE(res.size() == 1);
    CHECK_EQ(res[0].uri, std::string("test://doc"));

    const std::vector<ResourceContents> body = s.client().read_resource("test://doc");
    REQUIRE(body.size() == 1);
    CHECK(body[0].text.has_value());
    CHECK_EQ(*body[0].text, std::string("body text"));

    const std::vector<Prompt> prompts = s.client().list_prompts();
    REQUIRE(prompts.size() == 1);
    CHECK_EQ(prompts[0].name, std::string("greet"));

    const std::vector<PromptMessage> msgs =
        s.client().get_prompt("greet", nlohmann::json{{"who", "sean"}});
    REQUIRE(msgs.size() == 1);
    CHECK_EQ(msgs[0].content.value("text", ""), std::string("hello sean"));
}

TEST(progress_notifications_reach_the_caller) {
    Session s;
    static_cast<void>(s.client().initialize());

    std::atomic<int> ticks{0};
    const ToolResult r = s.client().call_tool(
        "slow", nlohmann::json{{"steps", 4}},
        [&](double, std::optional<double>, std::string_view) { ticks.fetch_add(1); });

    CHECK(!r.is_error);
    CHECK_EQ(ticks.load(), 4);
}

TEST(a_timeout_cancels_the_request_server_side) {
    Session s;
    static_cast<void>(s.client().initialize());

    const int before = Session::cancel_observations().load();
    bool timed_out = false;
    try {
        // 12 steps at 25ms is ~300ms; give it 80ms.
        static_cast<void>(s.client().call_tool("slow", nlohmann::json{{"steps", 12}}, {},
                                               std::chrono::milliseconds(80)));
    } catch (const McpError& e) {
        timed_out = true;
        CHECK_EQ(e.code(), to_int(ErrorCode::kRequestCancelled));
    }
    CHECK(timed_out);

    // The server must actually observe the cancellation, not merely have its reply
    // discarded -- otherwise a timed-out call keeps burning a worker to completion.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(Session::cancel_observations().load() > before);
}

TEST(a_cancel_fn_cancels_the_request_server_side) {
    Session s;
    static_cast<void>(s.client().initialize());

    const int before = Session::cancel_observations().load();
    std::atomic<bool> flag{false};
    std::thread flipper([&flag]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        flag.store(true);
    });
    bool cancelled = false;
    try {
        static_cast<void>(s.client().call_tool(
            "slow", nlohmann::json{{"steps", 40}}, {}, std::chrono::milliseconds(5000),
            [&flag]() { return flag.load(); }));
    } catch (const McpError& e) {
        cancelled = true;
        CHECK_EQ(e.code(), to_int(ErrorCode::kRequestCancelled));
        CHECK(std::string(e.what()).find("cancelled") != std::string::npos);
    }
    flipper.join();
    CHECK(cancelled);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(Session::cancel_observations().load() > before);
}

TEST(concurrent_requests_each_get_their_own_reply) {
    // The check that would have caught a uint64 correlation map, and the one that
    // catches any future mix-up between in-flight requests. Every caller asks for a
    // distinct n and must get that same n back.
    Session s;
    static_cast<void>(s.client().initialize());

    constexpr int kThreads = 8;
    constexpr int kPerThread = 40;
    std::atomic<int> mismatches{0};
    std::atomic<int> completed{0};

    auto issue_batch = [&](int thread_index) {
        for (int i = 0; i < kPerThread; ++i) {
            const int n = thread_index * kPerThread + i;
            const ToolResult r = s.client().call_tool("identity", nlohmann::json{{"n", n}});
            mismatches.fetch_add(text_of(r) == std::to_string(n) ? 0 : 1);
            completed.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(issue_batch, t);
    }
    for (auto& th : threads) {
        th.join();
    }

    CHECK_EQ(completed.load(), kThreads * kPerThread);
    CHECK_EQ(mismatches.load(), 0);
}

TEST(pagination_is_followed_to_the_end) {
    // A server with more tools than one page, driven through the client's cursor loop.
    Server::Info info;
    info.name = "paged";
    Server::Options opts;
    opts.page_size = 3;
    Server server(info, opts);
    for (int i = 0; i < 10; ++i) {
        Tool t;
        t.name = "tool_" + std::to_string(i);
        server.add_tool(std::move(t), [](const nlohmann::json&, RequestContext&) {
            return ToolResult::text("ok");
        });
    }

    InProcessTransport::Pair pair = InProcessTransport::make_pair();
    Transport* server_side = pair.server.get();
    std::thread th([&] { server.run(*server_side); });

    Client client(Client::Info{"c", "1"});
    client.connect(std::move(pair.client));
    static_cast<void>(client.initialize());

    const std::vector<Tool> tools = client.list_tools();
    CHECK_EQ(tools.size(), std::size_t(10));
    CHECK_EQ(tools.front().name, std::string("tool_0"));
    CHECK_EQ(tools.back().name, std::string("tool_9"));

    client.close();
    server.request_stop();
    th.join();
}

TEST(closing_the_client_fails_waiters_with_a_real_reason) {
    // Entrant C7's one good idea: a caller stranded by a teardown should learn why,
    // not receive a bare std::future_errc::broken_promise.
    Session s;
    static_cast<void>(s.client().initialize());

    std::atomic<bool> got_mcp_error{false};
    std::thread waiter([&] {
        try {
            static_cast<void>(s.client().call_tool("slow", nlohmann::json{{"steps", 40}}));
        } catch (const McpError&) {
            got_mcp_error.store(true);
        } catch (...) {
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    s.client().close();
    waiter.join();
    CHECK(got_mcp_error.load());
}

TEST(the_session_checks_can_actually_fail) {
    EXPECT_FAILING_CHECKS(2, {
        Session s;
        const ServerInfo info = s.client().initialize();
        CHECK_EQ(info.name, std::string("not-the-server-name"));
        CHECK(!info.supports_tools());
    });
}
