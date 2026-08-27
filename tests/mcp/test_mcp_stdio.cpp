// The stdio path, end to end, against the real demo server binary in a real subprocess.
//
// The in-process tests cover the protocol; this one covers everything the in-process
// transport skips and that the cook-off entrants actually broke: spawning, pipe
// framing, stderr backpressure, EOF handling, and reaping the child.

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "src/mcp/client.hpp"

#include "tests/check.hpp"

using namespace lmp::mcp;

namespace {

// Set by CMake to $<TARGET_FILE:mcp_demo_server>.
const char* demo_server_path() { return LMP_MCP_DEMO_SERVER; }

std::string text_of(const ToolResult& r) {
    if (r.content.is_array() && !r.content.empty() && r.content[0].contains("text")) {
        return r.content[0]["text"].get<std::string>();
    }
    return {};
}

Subprocess::Options demo_options(StderrMode mode = StderrMode::kDiscard) {
    Subprocess::Options o;
    o.program = demo_server_path();
    o.stderr_mode = mode;
    return o;
}

} // namespace

TEST(stdio_session_against_a_real_subprocess) {
    Client client(Client::Info{"stdio-test", "1.0"});
    client.connect_stdio(demo_options());

    const ServerInfo info = client.initialize();
    CHECK_EQ(info.name, std::string("lmp-mcp-demo"));
    CHECK_EQ(info.protocol_version, std::string(kProtocolVersion));

    const std::vector<Tool> tools = client.list_tools();
    CHECK_EQ(tools.size(), std::size_t(5)); // + screenshot, the image round-trip
    bool saw_echo_ro = false;
    bool saw_add_unannotated = false;
    for (const Tool& t : tools) {
        if (t.name == "echo") {
            REQUIRE(t.annotations.has_value());
            CHECK(t.annotations->at("readOnlyHint").get<bool>());
            saw_echo_ro = true;
        }
        if (t.name == "add") {
            CHECK(!t.annotations.has_value());
            saw_add_unannotated = true;
        }
    }
    CHECK(saw_echo_ro);
    CHECK(saw_add_unannotated);

    const ToolResult r = client.call_tool("echo", nlohmann::json{{"text", "over a pipe"}});
    CHECK(!r.is_error);
    CHECK_EQ(text_of(r), std::string("over a pipe"));

    client.close();
}

TEST(structured_content_survives_the_pipe) {
    Client client(Client::Info{"stdio-test", "1.0"});
    client.connect_stdio(demo_options());
    static_cast<void>(client.initialize());

    const ToolResult r = client.call_tool("add", nlohmann::json{{"a", 20}, {"b", 22}});
    REQUIRE(r.structured.has_value());
    CHECK_EQ(r.structured->value("sum", 0.0), 42.0);
    client.close();
}

TEST(a_large_payload_spans_many_reads_intact) {
    // 512 KB through a 64 KB pipe buffer: the message is reassembled across many
    // read() boundaries, which is exactly what the framer exists for.
    Client client(Client::Info{"stdio-test", "1.0"});
    client.connect_stdio(demo_options());
    static_cast<void>(client.initialize());

    const std::string big(512 * 1024, 'z');
    const ToolResult r = client.call_tool("echo", nlohmann::json{{"text", big}});
    CHECK(!r.is_error);
    CHECK_EQ(text_of(r).size(), big.size());
    CHECK(text_of(r) == big);
    client.close();
}

TEST(a_payload_full_of_newlines_does_not_desync_the_stream) {
    // If encoding ever let an interior newline through, this splits one message into
    // thousands and the session never recovers.
    Client client(Client::Info{"stdio-test", "1.0"});
    client.connect_stdio(demo_options());
    static_cast<void>(client.initialize());

    std::string newlines;
    for (int i = 0; i < 2000; ++i) {
        newlines += "line " + std::to_string(i) + "\n";
    }
    const ToolResult r = client.call_tool("echo", nlohmann::json{{"text", newlines}});
    CHECK(!r.is_error);
    CHECK(text_of(r) == newlines);

    // And the next call still works, which is the real proof it did not desync.
    const ToolResult after = client.call_tool("echo", nlohmann::json{{"text", "after"}});
    CHECK_EQ(text_of(after), std::string("after"));
    client.close();
}

TEST(concurrent_calls_over_one_pipe_are_correlated_correctly) {
    Client client(Client::Info{"stdio-test", "1.0"});
    client.connect_stdio(demo_options());
    static_cast<void>(client.initialize());

    constexpr int kThreads = 6;
    constexpr int kPerThread = 25;
    std::atomic<int> mismatches{0};

    auto issue_batch = [&](int thread_index) {
        for (int i = 0; i < kPerThread; ++i) {
            const std::string want = "t" + std::to_string(thread_index) + "-" + std::to_string(i);
            const ToolResult r = client.call_tool("echo", nlohmann::json{{"text", want}});
            mismatches.fetch_add(text_of(r) == want ? 0 : 1);
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
    CHECK_EQ(mismatches.load(), 0);
    client.close();
}

TEST(captured_stderr_is_drained_and_delivered) {
    // Entrant C6 piped stderr and never read it, and deadlocked against any server that
    // logs. The demo server writes a banner to stderr at startup; capturing it must
    // both work and not wedge the session.
    Client client(Client::Info{"stdio-test", "1.0"});

    std::atomic<int> bytes{0};
    client.on_server_stderr([&](std::string_view chunk) {
        bytes.fetch_add(static_cast<int>(chunk.size()));
    });
    client.connect_stdio(demo_options(StderrMode::kCapture));
    static_cast<void>(client.initialize());

    const ToolResult r = client.call_tool("echo", nlohmann::json{{"text", "ok"}});
    CHECK_EQ(text_of(r), std::string("ok"));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(bytes.load() > 0);
    client.close();
}

TEST(cancellation_reaches_a_real_subprocess) {
    Client client(Client::Info{"stdio-test", "1.0"});
    client.connect_stdio(demo_options());
    static_cast<void>(client.initialize());

    bool timed_out = false;
    try {
        // 20 steps at 100ms each; allow 150ms.
        static_cast<void>(client.call_tool("slow_count", nlohmann::json{{"n", 20}}, {},
                                           std::chrono::milliseconds(150)));
    } catch (const McpError& e) {
        timed_out = true;
        CHECK_EQ(e.code(), to_int(ErrorCode::kRequestCancelled));
    }
    CHECK(timed_out);

    // The session must still be usable after a cancelled call.
    const ToolResult after = client.call_tool("echo", nlohmann::json{{"text", "recovered"}});
    CHECK_EQ(text_of(after), std::string("recovered"));
    client.close();
}

TEST(spawning_a_nonexistent_server_fails_loudly) {
    Client client(Client::Info{"stdio-test", "1.0"});
    Subprocess::Options o;
    o.program = "/nonexistent/definitely-not-a-server";
    bool threw = false;
    try {
        client.connect_stdio(std::move(o));
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(many_sequential_sessions_leave_no_processes_behind) {
    // Spawn and tear down repeatedly. A leaked child or an unreaped zombie shows up
    // here as a hang or an fd exhaustion rather than as a quiet leak in production.
    for (int i = 0; i < 20; ++i) {
        Client client(Client::Info{"stdio-test", "1.0"});
        client.connect_stdio(demo_options());
        static_cast<void>(client.initialize());
        const ToolResult r = client.call_tool("echo", nlohmann::json{{"text", "x"}});
        CHECK_EQ(text_of(r), std::string("x"));
        client.close();
    }
}

TEST(the_stdio_checks_can_actually_fail) {
    EXPECT_FAILING_CHECKS(1, {
        Client client(Client::Info{"stdio-test", "1.0"});
        client.connect_stdio(demo_options());
        const ServerInfo info = client.initialize();
        CHECK_EQ(info.name, std::string("wrong-name"));
        client.close();
    });
}

// --- where the child starts ------------------------------------------------------
//
// AN MCP SERVER IS A DIFFERENT PROCESS, and a relative path means whatever ITS working
// directory says it means. The sidecar's own is whatever the editor launched it with --
// on macOS, a GUI-launched process gets `/` -- so a server asked for the relative path the
// prompt tells the model to use resolved it against the root of the filesystem.
//
// Measured on r-18ced29746aa7728-2ea858f4: fifteen consecutive godoer calls came back
// "'/game' is not a Godot project (no project.godot found)". Verified against godoer
// directly the same day: the identical command run from `/` fails and run from the
// workspace root returns real diagnostics.

TEST(a_child_starts_in_the_working_directory_it_was_given) {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base != nullptr ? base : "/tmp") + "/lmp_cwd_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    REQUIRE(made != nullptr);
    const std::string dir(made);

    Subprocess::Options o;
    o.program = "/bin/pwd";
    o.working_dir = dir;
    o.stderr_mode = StderrMode::kDiscard;
    Subprocess proc(std::move(o));

    std::string out;
    char chunk[512];
    for (;;) {
        const ssize_t n = ::read(proc.stdout_fd(), chunk, sizeof(chunk));
        if (n <= 0) {
            break;
        }
        out.append(chunk, static_cast<std::size_t>(n));
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    // mkdtemp hands back a path under TMPDIR, which on macOS is itself a symlink
    // (/var -> /private/var), and pwd resolves it. Comparing the tail is what makes this
    // assert "it started where we said" without asserting the platform's symlink policy.
    CHECK(out.size() >= dir.size() - 5);
    CHECK(out.find(dir.substr(dir.rfind('/'))) != std::string::npos);
}

TEST(an_unset_working_directory_still_inherits_ours) {
    // The default must not change: every existing caller passes nothing and expects the
    // parent's directory, and a test that only proves the new path works would not notice
    // the old one breaking.
    Subprocess::Options o;
    o.program = "/bin/pwd";
    o.stderr_mode = StderrMode::kDiscard;
    Subprocess proc(std::move(o));

    std::string out;
    char chunk[512];
    for (;;) {
        const ssize_t n = ::read(proc.stdout_fd(), chunk, sizeof(chunk));
        if (n <= 0) {
            break;
        }
        out.append(chunk, static_cast<std::size_t>(n));
    }
    CHECK(!out.empty());
}

TEST(a_working_directory_that_does_not_exist_fails_the_spawn) {
    // REFUSED, not ignored. A server that silently starts in the wrong place answers every
    // call with a path error, which reads as a broken tool rather than a bad config -- the
    // exact failure this option exists to prevent.
    Subprocess::Options o;
    o.program = "/bin/pwd";
    o.working_dir = "/nonexistent/definitely-not-a-directory";
    o.stderr_mode = StderrMode::kDiscard;
    bool threw = false;
    try {
        Subprocess proc(std::move(o));
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}
