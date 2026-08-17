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

// The FLAT shape: mcp_servers directly under params. Headless drivers send this, and it
// is what this file used to test exclusively -- which is precisely why the parser's real
// bug went unseen for months. Kept, because those drivers are real; no longer alone.
std::string start_message(const std::string& servers_json) {
    return R"({"jsonrpc":"2.0","id":"1","method":"lmp/start","params":{)"
           R"("mission":"m","model_dir":"/m","workspace_root":"/w",)"
           R"("mcp_servers":)" +
           servers_json + "}}";
}

// THE SHAPE THE EXTENSION ACTUALLY SENDS. lmp/start carries StartParams -- {mission,
// settings, image_paths} -- so every RunSettings field, mcp_servers included, is nested one
// level deeper than the helper above pretends. See the note in parse_mcp_servers.
std::string real_start_message(const std::string& servers_json) {
    return R"({"jsonrpc":"2.0","id":"1","method":"lmp/start","params":{)"
           R"("mission":"m","image_paths":[],"settings":{)"
           R"("model_dir":"/m","workspace_root":"/w","mode":"agent",)"
           R"("mcp_servers":)" +
           servers_json + "}}}";
}

} // namespace

// NO MCP SERVER HAD EVER CONNECTED.
//
// parse_mcp_servers read params.mcp_servers; the extension sends params.settings.
// mcp_servers. The two never met, so `mcp_servers` was silently empty on every run ever
// started from the editor -- MEASURED: a 21 MB event log spanning months contains zero
// `mcp_server` events, and a run whose mission explicitly said "use the godoer mcp server"
// reported 26 tools, the built-in count, and fell back to driving the CLI through the
// shell.
//
// It is the ONLY reader in the sidecar that walks real JSON. Every other setting goes
// through surface::string_field, a flat substring search that finds a nested key by
// accident and so cannot be wrong about nesting. Being the one that has to know the shape
// is what made this the one that could be wrong about it -- and the test agreed with the
// parser instead of with the protocol, so both were wrong together and the gate was green.
TEST(the_server_list_is_read_from_the_shape_the_extension_sends) {
    std::string sig;
    const auto servers = parse_mcp_servers(
        real_start_message(R"([{"name":"godoer","command":"/opt/venv/bin/godoer",)"
                           R"("args":["mcp"],"trusted":true}])"),
        sig);
    REQUIRE(servers.size() == 1);
    CHECK_EQ(servers[0].name, std::string("godoer"));
    CHECK_EQ(servers[0].command, std::string("/opt/venv/bin/godoer"));
    REQUIRE(servers[0].args.size() == 1);
    CHECK_EQ(servers[0].args[0], std::string("mcp"));
    CHECK(servers[0].trusted);
    CHECK(!sig.empty());

    // `trusted` must survive the nesting too: it is the field that decides whether a tool
    // runs outside Seatbelt with no card, and a parser that found the server but lost the
    // flag would be the worse failure of the two.
    std::string sig2;
    const auto untrusted = parse_mcp_servers(
        real_start_message(R"([{"name":"s","command":"c","trusted":false}])"), sig2);
    REQUIRE(untrusted.size() == 1);
    CHECK(!untrusted[0].trusted);

    // An absent list is still an empty list, not a parse failure.
    std::string sig3;
    CHECK(parse_mcp_servers(
              R"({"method":"lmp/start","params":{"mission":"m","settings":{"mode":"agent"}}})",
              sig3)
              .empty());
}

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

// --- repeated string fields ----------------------------------------------------
//
// `image_paths` on lmp/start and lmp/message. The string_field extractors are a substring
// search for a scalar and cannot walk an array, so this is the parser those requests use.
TEST(a_repeated_string_field_is_parsed_from_params_or_the_root) {
    const std::string nested =
        R"({"method":"lmp/message","params":{"text":"look","image_paths":["a.png","b/c.jpg"]}})";
    const std::vector<std::string> got = lmp::surface::parse_string_array(nested, "image_paths");
    REQUIRE(got.size() == std::size_t{2});
    CHECK_EQ(got[0], std::string("a.png"));
    CHECK_EQ(got[1], std::string("b/c.jpg"));

    // Same message shape without the params wrapper, which is how several of these
    // requests arrive.
    const std::string flat = R"({"image_paths":["only.png"]})";
    CHECK_EQ(lmp::surface::parse_string_array(flat, "image_paths").size(), std::size_t{1});
}

TEST(a_missing_or_malformed_repeated_field_is_empty_rather_than_wrong) {
    CHECK(lmp::surface::parse_string_array(R"({"params":{}})", "image_paths").empty());
    CHECK(lmp::surface::parse_string_array("not json at all", "image_paths").empty());
    // A scalar where an array belongs is not silently promoted to a one-element list.
    CHECK(lmp::surface::parse_string_array(R"({"image_paths":"a.png"})", "image_paths").empty());
    // Non-strings and empties are DROPPED: an empty path would reach the prompt builder
    // as an unreadable image and cost the turn a note about a file nobody attached.
    const std::string mixed = R"({"image_paths":["ok.png","",7,null,"two.png"]})";
    const std::vector<std::string> got = lmp::surface::parse_string_array(mixed, "image_paths");
    REQUIRE(got.size() == std::size_t{2});
    CHECK_EQ(got[0], std::string("ok.png"));
    CHECK_EQ(got[1], std::string("two.png"));
}
