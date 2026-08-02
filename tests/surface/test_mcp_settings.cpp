// Reading the MCP server list off the wire (M1).
//
// This is a settings parser, which is normally dull. It is here because ONE of its fields
// decides whether a tool that runs outside the sandbox may be called without an approval
// card, and S7.5 says no security input gets a permissive default. So `trusted` is tested
// against every shape a settings file can produce that is not a literal boolean true.

#include <string>

#include "src/surface/mcp_settings.hpp"

#include "tests/check.hpp"

using lmp::surface::parse_mcp_servers;
using lmp::tools::McpServerConfig;

namespace {

std::string start_message(const std::string& servers_json) {
    return R"({"jsonrpc":"2.0","id":"1","method":"lmp/start","params":{)"
           R"("mission":"m","model_dir":"/m","workspace_root":"/w",)"
           R"("mcp_servers":)" +
           servers_json + "}}";
}

} // namespace

TEST(a_server_list_is_read_off_the_start_message) {
    std::string sig;
    const auto servers = parse_mcp_servers(
        start_message(R"([{"name":"fs","command":"node","args":["srv.js","/tmp"],)"
                      R"("env":["A=1","B=2"],"trusted":true}])"),
        sig);
    REQUIRE(servers.size() == 1);
    CHECK_EQ(servers[0].name, std::string("fs"));
    CHECK_EQ(servers[0].command, std::string("node"));
    REQUIRE(servers[0].args.size() == 2);
    CHECK_EQ(servers[0].args[1], std::string("/tmp"));
    REQUIRE(servers[0].env.size() == 2);
    CHECK_EQ(servers[0].env[0], std::string("A=1"));
    CHECK(servers[0].trusted);
    CHECK(!sig.empty());
}

TEST(trusted_is_a_literal_true_and_nothing_else) {
    // Every one of these is a plausible thing to find in a hand-edited settings file, and
    // every one of them means "I did not say yes". A coercing parser would turn the first
    // three into permission to run an unsandboxed tool with no card.
    const char* not_yes[] = {R"("true")", "1", "null", R"("yes")", "0", "false"};
    for (const char* value : not_yes) {
        std::string sig;
        const auto servers = parse_mcp_servers(
            start_message(std::string(R"([{"name":"s","command":"c","trusted":)") + value +
                          "}]"),
            sig);
        REQUIRE(servers.size() == 1);
        if (servers[0].trusted) {
            lmp::test::record_failure(__FILE__, __LINE__,
                                      std::string("trusted:") + value + " was read as yes");
        }
        ++lmp::test::reg().checks;
    }

    // ...and absent is no.
    std::string sig;
    const auto absent =
        parse_mcp_servers(start_message(R"([{"name":"s","command":"c"}])"), sig);
    REQUIRE(absent.size() == 1);
    CHECK(!absent[0].trusted);
}

TEST(a_message_with_no_servers_is_normal_and_not_an_error) {
    // The overwhelmingly common case. None of these may be a failure mode.
    std::string sig;
    CHECK(parse_mcp_servers(start_message("[]"), sig).empty());
    CHECK(parse_mcp_servers(R"({"params":{"mission":"m"}})", sig).empty());
    CHECK(parse_mcp_servers("{}", sig).empty());
    CHECK(parse_mcp_servers("", sig).empty());
    CHECK(parse_mcp_servers("not json at all {{{", sig).empty());
    // A wrong-typed field is also not a crash.
    CHECK(parse_mcp_servers(start_message(R"("a string")"), sig).empty());
}

TEST(malformed_entries_are_skipped_rather_than_taken_apart) {
    std::string sig;
    const auto servers = parse_mcp_servers(
        start_message(R"([7,"nope",{"name":"ok","command":"c"},{"args":"not-an-array"}])"),
        sig);
    // The two non-objects are skipped; the two objects survive, the second with empty
    // name/command, which McpHost rejects with a reported reason rather than here.
    REQUIRE(servers.size() == 2);
    CHECK_EQ(servers[0].name, std::string("ok"));
    CHECK(servers[1].name.empty());
    CHECK(servers[1].args.empty());
}

TEST(the_signature_changes_when_the_configured_set_changes) {
    // The signature is what tells the sidecar to rebuild the registry: the Registry has no
    // unregister, so missing a change here would leave remote tools alive under settings
    // that no longer authorise them.
    std::string a;
    std::string b;
    std::string c;
    (void)parse_mcp_servers(start_message(R"([{"name":"s","command":"c"}])"), a);
    (void)parse_mcp_servers(start_message(R"([{"name":"s","command":"c"}])"), b);
    (void)parse_mcp_servers(start_message(R"([{"name":"s","command":"c","trusted":true}])"),
                            c);
    CHECK_EQ(a, b);
    CHECK(a != c);

    // No servers must not look like "unchanged" against a list that had some.
    std::string empty_sig;
    (void)parse_mcp_servers(start_message("[]"), empty_sig);
    CHECK(empty_sig != a);
}

// The case that proves these checks can fail: everything above asserts on a parse that
// succeeded, so a parser that returned a fixed one-element list would pass most of them.
TEST(the_parser_can_return_more_than_one_and_can_return_none) {
    std::string sig;
    const auto two = parse_mcp_servers(
        start_message(R"([{"name":"a","command":"x"},{"name":"b","command":"y"}])"), sig);
    REQUIRE(two.size() == 2);
    CHECK_EQ(two[0].name, std::string("a"));
    CHECK_EQ(two[1].name, std::string("b"));
    CHECK(parse_mcp_servers(start_message("[]"), sig).empty());
}
