#include "src/surface/worker.hpp"
#include "src/surface/socket_reader.hpp"

#include <arpa/inet.h>
#include <filesystem>
#include <fstream>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "tests/check.hpp"

using namespace lmp::surface::worker;

TEST(strip_think_leak_removes_tags) {
    const std::string in = "<think>internal reasoning\nmore reasoning</think>All tasks finished.";
    CHECK_EQ(strip_think_leak(in), "All tasks finished.");

    const std::string unclosed = "<think>unclosed thought";
    CHECK_EQ(strip_think_leak(unclosed), "");

    const std::string clean = "Straight to the point.";
    CHECK_EQ(strip_think_leak(clean), "Straight to the point.");
}

TEST(strip_think_leak_removes_leading_preambles) {
    const std::string with_preamble = "Let me verify the file.\nHere is the summary of what changed.";
    CHECK_EQ(strip_think_leak(with_preamble), "Here is the summary of what changed.");

    const std::string will_now = "I will now proceed with testing.\nCompleted with 0 errors.";
    CHECK_EQ(strip_think_leak(will_now), "Completed with 0 errors.");
}

TEST(strip_think_leak_caps_at_limit) {
    std::string long_str(3000, 'x');
    CHECK_EQ(strip_think_leak(long_str).size(), std::size_t{2000});
}

TEST(load_packet_validates_json_and_fields) {
    std::string error;
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_packet";
    std::filesystem::create_directories(tmp_dir);

    // 1. Invalid JSON
    {
        std::ofstream f((tmp_dir / "task.json").string());
        f << "{not json}";
    }
    CHECK(!load_packet((tmp_dir / "task.json").string(), error).has_value());
    CHECK(error.find("valid JSON") != std::string::npos);

    // 2. Missing id
    {
        nlohmann::json j = {
            {"cwd", tmp_dir.string()},
            {"prompt", "do something"}
        };
        std::ofstream f((tmp_dir / "task.json").string());
        f << j.dump();
    }
    CHECK(!load_packet((tmp_dir / "task.json").string(), error).has_value());
    CHECK(error.find("id") != std::string::npos);

    // 3. Valid with sibling prompt.md
    {
        nlohmann::json j = {
            {"id", "slice-01"},
            {"cwd", tmp_dir.string()},
            {"model_dir", tmp_dir.string()}
        };
        std::ofstream f((tmp_dir / "task.json").string());
        f << j.dump();

        std::ofstream pf((tmp_dir / "prompt.md").string());
        pf << "Do the task from prompt.md";
    }
    auto packet = load_packet((tmp_dir / "task.json").string(), error);
    CHECK(packet.has_value());
    if (packet) {
        CHECK_EQ(packet->id, "slice-01");
        CHECK_EQ(packet->prompt, "Do the task from prompt.md");
        CHECK_EQ(packet->mode, "agent");
        CHECK_EQ(packet->timeout_s, 900.0);
        CHECK(packet->auto_approve_exec);
    }

    // 4. Invalid mode
    {
        nlohmann::json j = {
            {"id", "slice-01"},
            {"cwd", tmp_dir.string()},
            {"model_dir", tmp_dir.string()},
            {"prompt", "hello"},
            {"mode", "invalid_mode"}
        };
        std::ofstream f((tmp_dir / "task.json").string());
        f << j.dump();
    }
    CHECK(!load_packet((tmp_dir / "task.json").string(), error).has_value());
    CHECK(error.find("mode") != std::string::npos);

    std::filesystem::remove_all(tmp_dir);
}

TEST(build_start_message_formats_proper_jsonrpc) {
    TaskPacket packet;
    packet.id = "test-task";
    packet.cwd = "/tmp";
    packet.prompt = "Test prompt";
    packet.model_dir = "/tmp/model";
    packet.mode = "plan";
    packet.timeout_s = 300.0;
    packet.check_command = "npm test";

    std::string msg_str = build_start_message(packet, "42");
    auto j = nlohmann::json::parse(msg_str, nullptr, false);
    CHECK(!j.is_discarded());
    CHECK_EQ(j["id"].get<std::string>(), "42");
    CHECK_EQ(j["method"].get<std::string>(), "lmp/start");
    CHECK_EQ(j["params"]["mission"].get<std::string>(), "Test prompt");
    CHECK_EQ(j["params"]["settings"]["mode"].get<std::string>(), "plan");
    CHECK_EQ(j["params"]["settings"]["wall_clock_seconds"].get<int>(), 300);
    CHECK_EQ(j["params"]["settings"]["verify_contract"].get<std::string>(), "npm test");
    CHECK_EQ(j["params"]["settings"]["auto_approve_writes"].get<bool>(), true);
    CHECK_EQ(j["params"]["settings"]["auto_approve_irreversible"].get<bool>(), false);
}

TEST(collect_files_touched_parses_event_log) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_log";
    std::filesystem::create_directories(tmp_dir);
    std::string log_file = (tmp_dir / "events.jsonl").string();

    {
        std::ofstream f(log_file);
        f << "{\"kind\":\"turn\"}\n";
        f << "{\"kind\":\"write\",\"path\":\"" << (tmp_dir / "file1.txt").string() << "\",\"changed\":\"10\"}\n";
        f << "{\"kind\":\"write\",\"path\":\"" << (tmp_dir / "file1.txt").string() << "\",\"changed\":\"5\"}\n"; // duplicate
        f << "{\"kind\":\"write\",\"path\":\"" << (tmp_dir / "file2.txt").string() << "\",\"changed\":\"0\"}\n"; // unchanged
        f << "{\"kind\":\"write\",\"path\":\"" << (tmp_dir / "sub/file3.txt").string() << "\",\"changed\":25}\n";
    }

    auto touched = collect_files_touched(log_file, tmp_dir.string());
    CHECK_EQ(touched.size(), std::size_t{2});
    if (touched.size() == 2) {
        CHECK_EQ(touched[0], "file1.txt");
        CHECK_EQ(touched[1], "sub/file3.txt");
    }

    std::filesystem::remove_all(tmp_dir);
}

TEST(write_result_writes_atomic_json) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_result";
    std::filesystem::create_directories(tmp_dir);
    std::string res_file = (tmp_dir / "result.json").string();

    RunResult r;
    r.task_id = "t-1";
    r.status = "ok";
    r.message = "Clean message";
    r.cwd = tmp_dir.string();
    r.model_dir = "/models/qwen";
    r.wall_seconds = 12.5;
    r.turns = 3;
    r.generated_tokens = 450;
    r.files_touched = {"a.py", "b.py"};
    r.diff_stat = {15, 2, 2};
    r.test = {true, 0, "pytest", "tests passed"};

    write_result(res_file, r);

    std::ifstream f(res_file);
    CHECK(f.is_open());
    auto j = nlohmann::json::parse(f, nullptr, false);
    CHECK(!j.is_discarded());
    CHECK_EQ(j["task_id"].get<std::string>(), "t-1");
    CHECK_EQ(j["status"].get<std::string>(), "ok");
    CHECK_EQ(j["message"].get<std::string>(), "Clean message");
    CHECK_EQ(j["wall_seconds"].get<double>(), 12.5);
    CHECK_EQ(j["turns"].get<int>(), 3);
    CHECK_EQ(j["generated_tokens"].get<int>(), 450);
    CHECK_EQ(j["files_touched"].size(), std::size_t{2});
    CHECK_EQ(j["diff_stat"]["insertions"].get<int>(), 15);
    CHECK_EQ(j["test"]["ran"].get<bool>(), true);
    CHECK_EQ(j["test"]["exit_code"].get<int>(), 0);

    std::filesystem::remove_all(tmp_dir);
}

TEST(find_last_ask_user_finds_event_and_populates_fields) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_ask_user";
    std::filesystem::create_directories(tmp_dir);
    std::string log_file = (tmp_dir / "events.ndjson").string();

    {
        std::ofstream f(log_file);
        f << "{\"seq\":1,\"kind\":\"run_begin\"}\n";
        f << "{\"seq\":10,\"kind\":\"ask_user\",\"question\":\"First question?\",\"options\":\"1,2\"}\n";
        f << "{\"seq\":11,\"kind\":\"turn\"}\n";
        f << "{\"seq\":35,\"kind\":\"ask_user\",\"question\":\"Which scene format?\",\"options\":\"Option A\\nOption B\"}\n";
        f << "{\"seq\":36,\"kind\":\"turn\"}\n";
        f << "{\"seq\":37,\"kind\":\"run_end\",\"termination_reason\":\"awaiting_user\"}\n";
    }

    auto ask = find_last_ask_user(log_file, "r-12345");
    REQUIRE(ask.has_value());
    CHECK_EQ(ask->question, "Which scene format?");
    CHECK_EQ(ask->options, "Option A\nOption B");
    CHECK_EQ(ask->run_id, "r-12345");
    CHECK_EQ(ask->seq, uint64_t{35});

    std::filesystem::remove_all(tmp_dir);
}

TEST(write_awaiting_user_writes_valid_json) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_awaiting";
    std::filesystem::create_directories(tmp_dir);
    std::string awaiting_file = (tmp_dir / "awaiting_user.json").string();

    AwaitingUserInfo info;
    info.question = "Should we proceed?";
    info.options = "yes,no";
    info.run_id = "r-test";
    info.seq = 42;

    write_awaiting_user(awaiting_file, info);

    std::ifstream f(awaiting_file);
    CHECK(f.is_open());
    auto j = nlohmann::json::parse(f, nullptr, false);
    CHECK(!j.is_discarded());
    CHECK_EQ(j["question"].get<std::string>(), "Should we proceed?");
    CHECK_EQ(j["options"].get<std::string>(), "yes,no");
    CHECK_EQ(j["run_id"].get<std::string>(), "r-test");
    CHECK_EQ(j["seq"].get<uint64_t>(), uint64_t{42});

    std::filesystem::remove_all(tmp_dir);
}

TEST(read_and_consume_answer_reads_formats_and_deletes_files) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_consume";
    std::filesystem::create_directories(tmp_dir);
    std::string ans_file = (tmp_dir / "answer.json").string();
    std::string await_file = (tmp_dir / "awaiting_user.json").string();

    // 1. Missing answer.json returns nullopt
    CHECK(!read_and_consume_answer(ans_file, await_file).has_value());

    // 2. Object with "text"
    {
        std::ofstream af(await_file);
        af << "{\"question\":\"q\"}";
        std::ofstream f(ans_file);
        f << "{\"text\":\"Proceed with option A\"}";
    }
    auto ans1 = read_and_consume_answer(ans_file, await_file);
    REQUIRE(ans1.has_value());
    CHECK_EQ(*ans1, "Proceed with option A");
    CHECK(!std::filesystem::exists(ans_file));
    CHECK(!std::filesystem::exists(await_file));

    // 3. Object with "answer"
    {
        std::ofstream af(await_file);
        af << "{\"question\":\"q\"}";
        std::ofstream f(ans_file);
        f << "{\"answer\":\"Proceed with option B\"}";
    }
    auto ans2 = read_and_consume_answer(ans_file, await_file);
    REQUIRE(ans2.has_value());
    CHECK_EQ(*ans2, "Proceed with option B");
    CHECK(!std::filesystem::exists(ans_file));
    CHECK(!std::filesystem::exists(await_file));

    // 4. Raw JSON string
    {
        std::ofstream af(await_file);
        af << "{\"question\":\"q\"}";
        std::ofstream f(ans_file);
        f << "\"Direct JSON string answer\"";
    }
    auto ans3 = read_and_consume_answer(ans_file, await_file);
    REQUIRE(ans3.has_value());
    CHECK_EQ(*ans3, "Direct JSON string answer");
    CHECK(!std::filesystem::exists(ans_file));

    // 5. Raw plain text
    {
        std::ofstream f(ans_file);
        f << "Plain text reply from orchestrator\n";
    }
    auto ans4 = read_and_consume_answer(ans_file, await_file);
    REQUIRE(ans4.has_value());
    CHECK_EQ(*ans4, "Plain text reply from orchestrator");
    CHECK(!std::filesystem::exists(ans_file));

    std::filesystem::remove_all(tmp_dir);
}

TEST(find_last_ask_user_handles_missing_or_empty_log) {
    CHECK(!find_last_ask_user("", "r-1").has_value());
    CHECK(!find_last_ask_user("/no/such/path/events.ndjson", "r-1").has_value());

    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_no_ask";
    std::filesystem::create_directories(tmp_dir);
    std::string log_file = (tmp_dir / "events.ndjson").string();
    {
        std::ofstream f(log_file);
        f << "{\"seq\":1,\"kind\":\"run_begin\"}\n";
        f << "{\"seq\":2,\"kind\":\"turn\"}\n";
    }
    CHECK(!find_last_ask_user(log_file, "r-1").has_value());
    std::filesystem::remove_all(tmp_dir);
}

TEST(load_packet_validates_trust_mcp_against_mcp_json) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_trust_mcp";
    std::filesystem::create_directories(tmp_dir);
    std::string task_file = (tmp_dir / "task.json").string();
    std::string mcp_file = (tmp_dir / ".mcp.json").string();
    std::string error;

    // 1. trust_mcp not array
    {
        nlohmann::json j = {
            {"id", "t-1"}, {"cwd", tmp_dir.string()}, {"model_dir", tmp_dir.string()},
            {"prompt", "p"}, {"trust_mcp", "godoer"}
        };
        std::ofstream f(task_file);
        f << j.dump();
    }
    CHECK(!load_packet(task_file, error).has_value());
    CHECK(error.find("array") != std::string::npos);

    // 2. trust_mcp contains non-string
    {
        nlohmann::json j = {
            {"id", "t-1"}, {"cwd", tmp_dir.string()}, {"model_dir", tmp_dir.string()},
            {"prompt", "p"}, {"trust_mcp", {123}}
        };
        std::ofstream f(task_file);
        f << j.dump();
    }
    CHECK(!load_packet(task_file, error).has_value());
    CHECK(error.find("elements must be strings") != std::string::npos);

    // 3. trust_mcp names server but .mcp.json missing
    {
        nlohmann::json j = {
            {"id", "t-1"}, {"cwd", tmp_dir.string()}, {"model_dir", tmp_dir.string()},
            {"prompt", "p"}, {"trust_mcp", {"godoer"}}
        };
        std::ofstream f(task_file);
        f << j.dump();
        std::filesystem::remove(mcp_file);
    }
    CHECK(!load_packet(task_file, error).has_value());
    CHECK(error.find(".mcp.json was not found") != std::string::npos);

    // 4. trust_mcp names server absent from .mcp.json
    {
        nlohmann::json mcp_j = {
            {"mcpServers", {
                {"other_server", {{"command", "/bin/echo"}}}
            }}
        };
        std::ofstream mf(mcp_file);
        mf << mcp_j.dump();
    }
    CHECK(!load_packet(task_file, error).has_value());
    CHECK(error.find("absent from") != std::string::npos);

    // 5. trust_mcp names server with empty command
    {
        nlohmann::json mcp_j = {
            {"mcpServers", {
                {"godoer", {{"command", ""}}}
            }}
        };
        std::ofstream mf(mcp_file);
        mf << mcp_j.dump();
    }
    CHECK(!load_packet(task_file, error).has_value());
    CHECK(error.find("missing or empty 'command'") != std::string::npos);

    // 6. Valid trust_mcp matching server in .mcp.json
    {
        nlohmann::json mcp_j = {
            {"mcpServers", {
                {"godoer", {
                    {"command", "/opt/godoer"},
                    {"args", {"--headless"}},
                    {"env", {{"GODOT_BIN", "/bin/godot"}}}
                }}
            }}
        };
        std::ofstream mf(mcp_file);
        mf << mcp_j.dump();
    }
    auto packet = load_packet(task_file, error);
    REQUIRE(packet.has_value());
    CHECK_EQ(packet->trust_mcp.size(), std::size_t{1});
    CHECK_EQ(packet->trust_mcp[0], "godoer");

    std::filesystem::remove_all(tmp_dir);
}

TEST(build_start_message_includes_trusted_mcp_servers) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_start_msg";
    std::filesystem::create_directories(tmp_dir);
    std::string mcp_file = (tmp_dir / ".mcp.json").string();

    {
        nlohmann::json mcp_j = {
            {"mcpServers", {
                {"godoer", {
                    {"command", "/usr/bin/godoer"},
                    {"args", {"serve", "--port", "8000"}},
                    {"env", {{"VAR1", "VAL1"}, {"VAR2", "VAL2"}}}
                }},
                {"untrusted_srv", {
                    {"command", "/usr/bin/untrusted"}
                }}
            }}
        };
        std::ofstream mf(mcp_file);
        mf << mcp_j.dump();
    }

    TaskPacket packet;
    packet.id = "task-trust";
    packet.cwd = tmp_dir.string();
    packet.prompt = "test prompt";
    packet.model_dir = tmp_dir.string();
    packet.trust_mcp = {"godoer"};

    std::string msg_str = build_start_message(packet, "1");
    auto j = nlohmann::json::parse(msg_str, nullptr, false);
    CHECK(!j.is_discarded());
    const auto& settings = j["params"]["settings"];
    CHECK(settings.contains("mcp_servers"));
    const auto& servers = settings["mcp_servers"];
    CHECK_EQ(servers.size(), std::size_t{1});
    if (servers.size() == 1) {
        CHECK_EQ(servers[0]["name"].get<std::string>(), "godoer");
        CHECK_EQ(servers[0]["command"].get<std::string>(), "/usr/bin/godoer");
        CHECK(servers[0]["trusted"].get<bool>());
        CHECK_EQ(servers[0]["args"].size(), std::size_t{3});
        CHECK_EQ(servers[0]["args"][0].get<std::string>(), "serve");
        CHECK_EQ(servers[0]["env"].size(), std::size_t{2});
        bool has_var1 = false;
        for (const auto& e : servers[0]["env"]) {
            if (e.get<std::string>() == "VAR1=VAL1") has_var1 = true;
        }
        CHECK(has_var1);
    }

    // Packet with empty trust_mcp has no mcp_servers
    packet.trust_mcp.clear();
    std::string msg_no_trust = build_start_message(packet, "2");
    auto j_no_trust = nlohmann::json::parse(msg_no_trust, nullptr, false);
    CHECK(!j_no_trust["params"]["settings"].contains("mcp_servers"));

    std::filesystem::remove_all(tmp_dir);
}

TEST(load_packet_parses_orch_webhook) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_orch_webhook";
    std::filesystem::create_directories(tmp_dir);
    std::string task_file = (tmp_dir / "task.json").string();
    std::string err;

    // 1. Packet with orch_webhook
    {
        nlohmann::json j = {
            {"id", "webhook-task"},
            {"cwd", tmp_dir.string()},
            {"prompt", "hello"},
            {"model_dir", tmp_dir.string()},
            {"orch_webhook", "https://api.example.com/wake"}
        };
        std::ofstream f(task_file);
        f << j.dump();
    }
    auto p1 = load_packet(task_file, err);
    CHECK(p1.has_value());
    if (p1) {
        CHECK_EQ(p1->orch_webhook, "https://api.example.com/wake");
    }

    // 2. Packet without orch_webhook, but env set
    ::setenv("LMP_ORCH_WEBHOOK", "https://env.example.com/wake", 1);
    {
        nlohmann::json j = {
            {"id", "webhook-task-env"},
            {"cwd", tmp_dir.string()},
            {"prompt", "hello"},
            {"model_dir", tmp_dir.string()}
        };
        std::ofstream f(task_file);
        f << j.dump();
    }
    auto p2 = load_packet(task_file, err);
    CHECK(p2.has_value());
    if (p2) {
        CHECK_EQ(p2->orch_webhook, "https://env.example.com/wake");
    }

    // 3. Unset env
    ::unsetenv("LMP_ORCH_WEBHOOK");
    auto p3 = load_packet(task_file, err);
    CHECK(p3.has_value());
    if (p3) {
        CHECK_EQ(p3->orch_webhook, "");
    }

    std::filesystem::remove_all(tmp_dir);
}

TEST(post_orch_webhook_network_and_schema) {
    int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(server_fd >= 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(0);

    int opt = 1;
    ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    CHECK_EQ(::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    CHECK_EQ(::listen(server_fd, 1), 0);

    socklen_t len = sizeof(addr);
    CHECK_EQ(::getsockname(server_fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    int port = ntohs(addr.sin_port);

    std::vector<std::string> received_bodies;
    std::thread server_thread([server_fd, &received_bodies]() {
        for (int i = 0; i < 4; ++i) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
            if (client_fd >= 0) {
                char buf[4096];
                std::string req;
                ssize_t n = ::read(client_fd, buf, sizeof(buf));
                if (n > 0) {
                    req.assign(buf, static_cast<size_t>(n));
                    auto body_pos = req.find("\r\n\r\n");
                    if (body_pos != std::string::npos) {
                        received_bodies.push_back(req.substr(body_pos + 4));
                    }
                }
                std::string resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
                (void)::write(client_fd, resp.data(), resp.size());
                ::close(client_fd);
            }
        }
        ::close(server_fd);
    });

    std::string url = "http://127.0.0.1:" + std::to_string(port) + "/webhook";

    // 1. ask
    WebhookPayload ask_p;
    ask_p.kind = "ask";
    ask_p.task_id = "test-task-123";
    ask_p.run_id = "run-abc";
    ask_p.cwd = "/tmp/test";
    ask_p.result_path = "/tmp/test/result.json";
    ask_p.seq = 42;
    ask_p.status = "";
    ask_p.question = "Shall I proceed?";
    CHECK(post_orch_webhook(url, ask_p, 5.0));

    // 2. done
    WebhookPayload done_p;
    done_p.kind = "done";
    done_p.task_id = "test-task-123";
    done_p.run_id = "run-abc";
    done_p.cwd = "/tmp/test";
    done_p.result_path = "/tmp/test/result.json";
    done_p.seq = 0;
    done_p.status = "ok";
    done_p.question = "";
    CHECK(post_orch_webhook(url, done_p, 5.0));

    // 3. stalled
    WebhookPayload stalled_p;
    stalled_p.kind = "stalled";
    stalled_p.task_id = "test-task-123";
    stalled_p.run_id = "run-abc";
    stalled_p.cwd = "/tmp/test";
    stalled_p.result_path = "/tmp/test/result.json";
    stalled_p.seq = 0;
    stalled_p.status = "error";
    stalled_p.question = "";
    CHECK(post_orch_webhook(url, stalled_p, 5.0));

    // 4. died
    WebhookPayload died_p;
    died_p.kind = "died";
    died_p.task_id = "test-task-123";
    died_p.run_id = "";
    died_p.cwd = "/tmp/test";
    died_p.result_path = "/tmp/test/result.json";
    died_p.seq = 0;
    died_p.status = "";
    died_p.question = "";
    CHECK(post_orch_webhook(url, died_p, 5.0));

    server_thread.join();

    CHECK_EQ(received_bodies.size(), std::size_t{4});
    if (received_bodies.size() == 4) {
        auto j0 = nlohmann::json::parse(received_bodies[0], nullptr, false);
        CHECK_EQ(j0["kind"].get<std::string>(), "ask");
        CHECK_EQ(j0["seq"].get<uint64_t>(), uint64_t{42});
        CHECK_EQ(j0["question"].get<std::string>(), "Shall I proceed?");
        CHECK_EQ(j0["status"].get<std::string>(), "");

        auto j1 = nlohmann::json::parse(received_bodies[1], nullptr, false);
        CHECK_EQ(j1["kind"].get<std::string>(), "done");
        CHECK_EQ(j1["status"].get<std::string>(), "ok");
        CHECK_EQ(j1["seq"].get<uint64_t>(), uint64_t{0});
        CHECK_EQ(j1["question"].get<std::string>(), "");

        auto j2 = nlohmann::json::parse(received_bodies[2], nullptr, false);
        CHECK_EQ(j2["kind"].get<std::string>(), "stalled");
        CHECK_EQ(j2["status"].get<std::string>(), "error");
        CHECK_EQ(j2["seq"].get<uint64_t>(), uint64_t{0});
        CHECK_EQ(j2["question"].get<std::string>(), "");

        auto j3 = nlohmann::json::parse(received_bodies[3], nullptr, false);
        CHECK_EQ(j3["kind"].get<std::string>(), "died");
        CHECK_EQ(j3["run_id"].get<std::string>(), "");
        CHECK_EQ(j3["status"].get<std::string>(), "");
        CHECK_EQ(j3["seq"].get<uint64_t>(), uint64_t{0});
        CHECK_EQ(j3["question"].get<std::string>(), "");
    }

    bool dead_ok = post_orch_webhook("http://127.0.0.1:1/dead", ask_p, 1.0);
    CHECK(!dead_ok);
}

TEST(parse_approval_answer_handles_various_formats) {
    // True / Allow cases
    CHECK(parse_approval_answer("allow"));
    CHECK(parse_approval_answer("Allow"));
    CHECK(parse_approval_answer("ALLOWED"));
    CHECK(parse_approval_answer("approve"));
    CHECK(parse_approval_answer("Approved"));
    CHECK(parse_approval_answer("yes"));
    CHECK(parse_approval_answer("YES"));
    CHECK(parse_approval_answer("y"));
    CHECK(parse_approval_answer("true"));
    CHECK(parse_approval_answer("1"));
    CHECK(parse_approval_answer("\"allow\""));
    CHECK(parse_approval_answer("{\"text\":\"allow\"}"));
    CHECK(parse_approval_answer("{\"answer\":\"allow\"}"));
    CHECK(parse_approval_answer("{\"allow\":true}"));
    CHECK(parse_approval_answer("{\"approved\":true}"));
    CHECK(parse_approval_answer("{\"answer\":true}"));
    CHECK(parse_approval_answer("{\"text\":\"approved\"}"));

    // False / Deny cases
    CHECK(!parse_approval_answer("deny"));
    CHECK(!parse_approval_answer("Deny"));
    CHECK(!parse_approval_answer("DENIED"));
    CHECK(!parse_approval_answer("reject"));
    CHECK(!parse_approval_answer("Rejected"));
    CHECK(!parse_approval_answer("no"));
    CHECK(!parse_approval_answer("NO"));
    CHECK(!parse_approval_answer("n"));
    CHECK(!parse_approval_answer("false"));
    CHECK(!parse_approval_answer("0"));
    CHECK(!parse_approval_answer("{\"text\":\"deny\"}"));
    CHECK(!parse_approval_answer("{\"answer\":\"denied\"}"));
    CHECK(!parse_approval_answer("{\"allow\":false}"));
    CHECK(!parse_approval_answer("{\"approved\":false}"));
    CHECK(!parse_approval_answer("{\"answer\":false}"));

    // Fallback / unrecognized
    CHECK(!parse_approval_answer(""));
    CHECK(!parse_approval_answer("   "));
    CHECK(!parse_approval_answer("something random"));
    CHECK(!parse_approval_answer("{not json"));
}

TEST(format_irreversible_question_contains_tool_and_command) {
    std::string q = format_irreversible_question("bash", "cat > /tmp/floor_spec.json << 'EOF'");
    CHECK(q.find("bash") != std::string::npos);
    CHECK(q.find("cat > /tmp/floor_spec.json << 'EOF'") != std::string::npos);
    CHECK(q.find("irreversible") != std::string::npos);
    CHECK(q.find("allow or deny") != std::string::npos);
}

TEST(handle_irreversible_ask_allow_path) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_ask_allow";
    std::filesystem::create_directories(tmp_dir);
    std::string await_file = (tmp_dir / "awaiting_user.json").string();
    std::string ans_file = (tmp_dir / "answer.json").string();

    // Pre-populate answer.json so handle_irreversible_ask consumes it immediately
    {
        std::ofstream f(ans_file);
        f << "{\"text\":\"allow\"}\n";
    }

    IrreversibleAskParams params;
    params.tool = "bash";
    params.command_or_preview = "cat > /tmp/test.txt";
    params.run_id = "r-test-allow";
    params.task_id = "task-allow-1";
    params.cwd = tmp_dir.string();
    params.result_path = (tmp_dir / "result.json").string();
    params.awaiting_path = await_file;
    params.answer_path = ans_file;
    params.seq = 7;
    params.timeout_s = 5.0;
    params.wall_start = std::chrono::steady_clock::now();

    IrreversibleAskResult res = handle_irreversible_ask(params);
    CHECK(res == IrreversibleAskResult::Allowed);

    // Both files must have been consumed
    CHECK(!std::filesystem::exists(ans_file));
    CHECK(!std::filesystem::exists(await_file));

    std::filesystem::remove_all(tmp_dir);
}

TEST(handle_irreversible_ask_deny_path) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_ask_deny";
    std::filesystem::create_directories(tmp_dir);
    std::string await_file = (tmp_dir / "awaiting_user.json").string();
    std::string ans_file = (tmp_dir / "answer.json").string();

    {
        std::ofstream f(ans_file);
        f << "{\"answer\":\"deny\"}\n";
    }

    IrreversibleAskParams params;
    params.tool = "bash";
    params.command_or_preview = "rm -rf /tmp/test";
    params.run_id = "r-test-deny";
    params.task_id = "task-deny-1";
    params.cwd = tmp_dir.string();
    params.result_path = (tmp_dir / "result.json").string();
    params.awaiting_path = await_file;
    params.answer_path = ans_file;
    params.seq = 10;
    params.timeout_s = 5.0;
    params.wall_start = std::chrono::steady_clock::now();

    IrreversibleAskResult res = handle_irreversible_ask(params);
    CHECK(res == IrreversibleAskResult::Denied);

    CHECK(!std::filesystem::exists(ans_file));
    CHECK(!std::filesystem::exists(await_file));

    std::filesystem::remove_all(tmp_dir);
}

TEST(handle_irreversible_ask_timeout_path) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_ask_timeout";
    std::filesystem::create_directories(tmp_dir);
    std::string await_file = (tmp_dir / "awaiting_user.json").string();
    std::string ans_file = (tmp_dir / "answer.json").string();

    IrreversibleAskParams params;
    params.tool = "bash";
    params.command_or_preview = "cat > /tmp/spec.json";
    params.run_id = "r-test-timeout";
    params.task_id = "task-timeout-1";
    params.cwd = tmp_dir.string();
    params.result_path = (tmp_dir / "result.json").string();
    params.awaiting_path = await_file;
    params.answer_path = ans_file;
    params.seq = 12;
    params.timeout_s = 0.05; // 50ms timeout
    params.wall_start = std::chrono::steady_clock::now();

    IrreversibleAskResult res = handle_irreversible_ask(params);
    CHECK(res == IrreversibleAskResult::Timeout);

    // On timeout, awaiting_user.json was written and NOT deleted
    CHECK(std::filesystem::exists(await_file));
    std::ifstream f(await_file);
    auto j = nlohmann::json::parse(f, nullptr, false);
    CHECK(!j.is_discarded());
    CHECK_EQ(j["run_id"].get<std::string>(), "r-test-timeout");
    CHECK_EQ(j["seq"].get<uint64_t>(), uint64_t{12});

    std::filesystem::remove_all(tmp_dir);
}

TEST(handle_irreversible_ask_cancelled_path) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_worker_ask_cancel";
    std::filesystem::create_directories(tmp_dir);
    std::string await_file = (tmp_dir / "awaiting_user.json").string();
    std::string ans_file = (tmp_dir / "answer.json").string();

    IrreversibleAskParams params;
    params.tool = "bash";
    params.command_or_preview = "cat > /tmp/spec.json";
    params.run_id = "r-test-cancel";
    params.task_id = "task-cancel-1";
    params.cwd = tmp_dir.string();
    params.result_path = (tmp_dir / "result.json").string();
    params.awaiting_path = await_file;
    params.answer_path = ans_file;
    params.seq = 15;
    params.timeout_s = 10.0;
    params.wall_start = std::chrono::steady_clock::now();
    params.is_cancelled = []() { return true; };

    IrreversibleAskResult res = handle_irreversible_ask(params);
    CHECK(res == IrreversibleAskResult::Cancelled);

    std::filesystem::remove_all(tmp_dir);
}

TEST(repeated_denied_command_caching_behavior) {
    std::unordered_set<std::string> denied_commands;
    std::string cmd = "cat > /tmp/floor_spec.json << 'EOF'";

    // First call: not denied yet
    CHECK_EQ(denied_commands.count(cmd), std::size_t{0});

    // Simulate denial recorded by worker approver
    denied_commands.insert(cmd);

    // Second call with same command: denied immediately
    CHECK_EQ(denied_commands.count(cmd), std::size_t{1});

    // Different command: not denied
    CHECK_EQ(denied_commands.count("cat > /tmp/other.json"), std::size_t{0});
}

TEST(detached_launch_detection_and_gating) {
    // 1. cli_detach flag forces true
    CHECK(is_detached_launch(true));

    // 2. LMP_DAEMONIZE=1 forces true even if cli_detach is false
    ::setenv("LMP_DAEMONIZE", "1", 1);
    CHECK(is_detached_launch(false));

    // 3. LMP_DAEMONIZE=0 forces false even if stdin/stdout are redirected
    ::setenv("LMP_DAEMONIZE", "0", 1);
    CHECK(!is_detached_launch(false));

    ::unsetenv("LMP_DAEMONIZE");
}

TEST(init_project_creates_files_and_preserves_godoer) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_piper_init";
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);

    // Pre-populate Godoer briefs and game directory
    const std::string agents_content = "# Agent Godoer Instructions\nDo not overwrite.";
    const std::string gemini_content = "system instruction for gemini";
    const std::string claude_content = "claude instructions";
    const std::string mcp_content = "{\"mcpServers\":{\"db\":{\"command\":\"sqlite\"}}}";

    {
        std::ofstream f((tmp_dir / "AGENTS.md").string());
        f << agents_content;
    }
    {
        std::ofstream f((tmp_dir / "GEMINI.md").string());
        f << gemini_content;
    }
    {
        std::ofstream f((tmp_dir / "CLAUDE.md").string());
        f << claude_content;
    }
    {
        std::ofstream f((tmp_dir / ".mcp.json").string());
        f << mcp_content;
    }
    std::filesystem::create_directories(tmp_dir / "game");
    {
        std::ofstream f((tmp_dir / "game" / "main.cpp").string());
        f << "int main() { return 0; }";
    }

    // Call init_project
    int code = init_project(tmp_dir.string());
    CHECK_EQ(code, kExitOk);

    // Verify PIPER.md exists and contains the orchestrator guide and wake standard
    std::filesystem::path piper_md = tmp_dir / "PIPER.md";
    CHECK(std::filesystem::exists(piper_md));
    std::ifstream pf(piper_md.string());
    std::string piper_text((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
    CHECK(piper_text.find("Piper worker wake standard") != std::string::npos);
    CHECK(piper_text.find("Cloud directs, local writes") != std::string::npos);
    CHECK(piper_text.find("task.json") != std::string::npos);
    CHECK(piper_text.find("result.json") != std::string::npos);

    // Verify .cursor/rules/piper-parent.mdc exists and has frontmatter and orchestrator instructions
    std::filesystem::path cursor_rule = tmp_dir / ".cursor" / "rules" / "piper-parent.mdc";
    CHECK(std::filesystem::exists(cursor_rule));
    std::ifstream cf(cursor_rule.string());
    std::string cursor_text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    CHECK(cursor_text.find("alwaysApply: true") != std::string::npos);
    CHECK(cursor_text.find("piper: you are the parent.") != std::string::npos);
    CHECK(cursor_text.find("Cloud directs, local writes") != std::string::npos);
    CHECK(cursor_text.find("task.json") != std::string::npos);
    CHECK(cursor_text.find("piper run --task") != std::string::npos);

    // Verify Godoer briefs remain completely intact
    auto read_file = [](const std::filesystem::path& p) -> std::string {
        std::ifstream f(p.string());
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };
    CHECK_EQ(read_file(tmp_dir / "AGENTS.md"), agents_content);
    CHECK_EQ(read_file(tmp_dir / "GEMINI.md"), gemini_content);
    CHECK_EQ(read_file(tmp_dir / "CLAUDE.md"), claude_content);
    CHECK_EQ(read_file(tmp_dir / ".mcp.json"), mcp_content);
    CHECK_EQ(read_file(tmp_dir / "game" / "main.cpp"), "int main() { return 0; }");

    // Idempotency: run init_project again
    int code2 = init_project(tmp_dir.string());
    CHECK_EQ(code2, kExitOk);
    CHECK_EQ(read_file(piper_md), piper_text);
    CHECK_EQ(read_file(cursor_rule), cursor_text);
    CHECK_EQ(read_file(tmp_dir / "AGENTS.md"), agents_content);

    std::filesystem::remove_all(tmp_dir);
}

TEST(load_packet_parses_auto_approve_irreversible) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_load_auto_irr";
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);

    std::string err;

    // 1. Default when omitted: auto_approve_writes=true, auto_approve_irreversible=false
    {
        nlohmann::json j = {
            {"id", "test-default"},
            {"cwd", tmp_dir.string()},
            {"prompt", "hello"},
            {"model_dir", tmp_dir.string()}
        };
        std::ofstream f((tmp_dir / "task.json").string());
        f << j.dump();
    }
    auto p1 = load_packet((tmp_dir / "task.json").string(), err);
    CHECK(p1.has_value());
    CHECK(p1->auto_approve_writes);
    CHECK(!p1->auto_approve_irreversible);

    // 2. Explicit true
    {
        nlohmann::json j = {
            {"id", "test-irr-true"},
            {"cwd", tmp_dir.string()},
            {"prompt", "hello"},
            {"model_dir", tmp_dir.string()},
            {"auto_approve_irreversible", true}
        };
        std::ofstream f((tmp_dir / "task.json").string());
        f << j.dump();
    }
    auto p2 = load_packet((tmp_dir / "task.json").string(), err);
    CHECK(p2.has_value());
    CHECK(p2->auto_approve_irreversible);

    // 3. String "true" and "1"
    {
        nlohmann::json j = {
            {"id", "test-irr-str"},
            {"cwd", tmp_dir.string()},
            {"prompt", "hello"},
            {"model_dir", tmp_dir.string()},
            {"auto_approve_irreversible", "true"}
        };
        std::ofstream f((tmp_dir / "task.json").string());
        f << j.dump();
    }
    auto p3 = load_packet((tmp_dir / "task.json").string(), err);
    CHECK(p3.has_value());
    CHECK(p3->auto_approve_irreversible);

    // 4. Explicit false
    {
        nlohmann::json j = {
            {"id", "test-irr-false"},
            {"cwd", tmp_dir.string()},
            {"prompt", "hello"},
            {"model_dir", tmp_dir.string()},
            {"auto_approve_irreversible", false}
        };
        std::ofstream f((tmp_dir / "task.json").string());
        f << j.dump();
    }
    auto p4 = load_packet((tmp_dir / "task.json").string(), err);
    CHECK(p4.has_value());
    CHECK(!p4->auto_approve_irreversible);

    std::filesystem::remove_all(tmp_dir);
}

TEST(forward_to_daemon_exits_on_result_without_hanging) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_fwd_daemon";
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);

    std::string sock_path = (tmp_dir / "test.sock").string();
    int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    CHECK(listen_fd >= 0);

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
    CHECK_EQ(::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)), 0);
    CHECK_EQ(::listen(listen_fd, 1), 0);

    std::thread server_thread([listen_fd]() {
        int client = ::accept(listen_fd, nullptr, nullptr);
        if (client < 0) return;

        // Read request
        char buf[1024];
        ssize_t n = ::read(client, buf, sizeof(buf));
        (void)n;

        // Send intermediate jsonl line then final result with exit_code: 0
        std::string line1 = "{\"kind\":\"status\",\"step\":1}\n";
        (void)::write(client, line1.data(), line1.size());

        std::string line2 = "{\"status\":\"ok\",\"exit_code\":0,\"result_path\":\"/tmp/r.json\"}\n";
        (void)::write(client, line2.data(), line2.size());

        // Intentionally keep client socket open for 500ms to verify client does NOT block waiting for EOF!
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        ::shutdown(client, SHUT_RDWR);
        ::close(client);
    });

    auto start_t = std::chrono::steady_clock::now();
    auto res = forward_to_daemon(sock_path, "/tmp/task.json", false);
    auto end_t = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();

    CHECK(res.has_value());
    CHECK_EQ(*res, 0);
    // Client should have returned immediately upon reading exit_code, well before the 500ms delay!
    CHECK(elapsed_ms < 350.0);

    server_thread.join();
    ::close(listen_fd);
    ::unlink(sock_path.c_str());
    std::filesystem::remove_all(tmp_dir);
}

TEST(forward_to_daemon_handles_clean_eof_and_exit_code) {
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "test_fwd_eof";
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);

    std::string sock_path = (tmp_dir / "test_eof.sock").string();
    int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    CHECK(listen_fd >= 0);

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
    CHECK_EQ(::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)), 0);
    CHECK_EQ(::listen(listen_fd, 1), 0);

    std::thread server_thread([listen_fd]() {
        int client = ::accept(listen_fd, nullptr, nullptr);
        if (client < 0) return;

        char buf[1024];
        (void)::read(client, buf, sizeof(buf));

        std::string line = "{\"status\":\"error\",\"exit_code\":2,\"error\":\"failed\"}\n";
        (void)::write(client, line.data(), line.size());
        ::shutdown(client, SHUT_RDWR);
        ::close(client);
    });

    auto res = forward_to_daemon(sock_path, "/tmp/task.json", false);
    CHECK(res.has_value());
    CHECK_EQ(*res, 2);

    server_thread.join();
    ::close(listen_fd);
    ::unlink(sock_path.c_str());
    std::filesystem::remove_all(tmp_dir);
}
