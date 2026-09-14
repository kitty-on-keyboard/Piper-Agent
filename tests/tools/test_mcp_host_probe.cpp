#include <string>
#include <vector>

#include "src/tools/mcp_host.hpp"
#include "src/tools/mcp_host_probe.hpp"

#include "tests/check.hpp"

namespace {

bool run_parse_args(const std::vector<std::string>& args,
                    lmp::tools::McpServerConfig& cfg,
                    std::string& call_tool,
                    std::string& call_args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    return lmp::tools::parse_mcp_host_probe_args(
        static_cast<int>(argv.size()), argv.data(), cfg, call_tool, call_args);
}

} // namespace

TEST(parse_args_basic_command) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    const std::vector<std::string> args = {"mcp_host_probe", "--", "node", "server.js", "arg1"};
    const bool ok = run_parse_args(args, cfg, call_tool, call_args);

    CHECK(ok);
    CHECK(!cfg.trusted);
    CHECK_EQ(cfg.command, std::string("node"));
    REQUIRE(cfg.args.size() == 2);
    CHECK_EQ(cfg.args[0], std::string("server.js"));
    CHECK_EQ(cfg.args[1], std::string("arg1"));
    CHECK(call_tool.empty());
    CHECK_EQ(call_args, std::string("{}"));
}

TEST(parse_args_trusted_flag) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    const std::vector<std::string> args = {"mcp_host_probe", "--trusted", "--", "node", "server.js"};
    const bool ok = run_parse_args(args, cfg, call_tool, call_args);

    CHECK(ok);
    CHECK(cfg.trusted);
    CHECK_EQ(cfg.command, std::string("node"));
    REQUIRE(cfg.args.size() == 1);
    CHECK_EQ(cfg.args[0], std::string("server.js"));
}

TEST(parse_args_call_and_args_flags) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    const std::vector<std::string> args = {
        "mcp_host_probe", "--call", "my_tool", "--args", "{\"key\":\"val\"}", "--", "node", "server.js"};
    const bool ok = run_parse_args(args, cfg, call_tool, call_args);

    CHECK(ok);
    CHECK(!cfg.trusted);
    CHECK_EQ(call_tool, std::string("my_tool"));
    CHECK_EQ(call_args, std::string("{\"key\":\"val\"}"));
    CHECK_EQ(cfg.command, std::string("node"));
}

TEST(parse_args_combined_flags) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    const std::vector<std::string> args = {
        "mcp_host_probe", "--trusted", "--call", "tool1", "--args", "{\"a\":1}", "--", "python", "script.py", "foo", "bar"};
    const bool ok = run_parse_args(args, cfg, call_tool, call_args);

    CHECK(ok);
    CHECK(cfg.trusted);
    CHECK_EQ(call_tool, std::string("tool1"));
    CHECK_EQ(call_args, std::string("{\"a\":1}"));
    CHECK_EQ(cfg.command, std::string("python"));
    REQUIRE(cfg.args.size() == 3);
    CHECK_EQ(cfg.args[0], std::string("script.py"));
    CHECK_EQ(cfg.args[1], std::string("foo"));
    CHECK_EQ(cfg.args[2], std::string("bar"));
}

TEST(parse_args_missing_double_dash) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    const std::vector<std::string> args = {"mcp_host_probe", "node", "server.js"};
    const bool ok = run_parse_args(args, cfg, call_tool, call_args);

    CHECK(!ok);
}

TEST(parse_args_missing_command_after_double_dash) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    const std::vector<std::string> args = {"mcp_host_probe", "--trusted", "--"};
    const bool ok = run_parse_args(args, cfg, call_tool, call_args);

    CHECK(!ok);
}

TEST(parse_args_missing_flag_value) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    {
        const std::vector<std::string> args = {"mcp_host_probe", "--call"};
        CHECK(!run_parse_args(args, cfg, call_tool, call_args));
    }
    {
        const std::vector<std::string> args = {"mcp_host_probe", "--args"};
        CHECK(!run_parse_args(args, cfg, call_tool, call_args));
    }
}

TEST(parse_args_unknown_flag) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    const std::vector<std::string> args = {"mcp_host_probe", "--unknown", "--", "node", "server.js"};
    const bool ok = run_parse_args(args, cfg, call_tool, call_args);

    CHECK(!ok);
}

TEST(parse_args_empty_argv) {
    lmp::tools::McpServerConfig cfg;
    std::string call_tool;
    std::string call_args = "{}";

    const std::vector<std::string> args = {"mcp_host_probe"};
    const bool ok = run_parse_args(args, cfg, call_tool, call_args);

    CHECK(!ok);
}
