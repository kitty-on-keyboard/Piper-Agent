// Line framing and the JSON-RPC message model.
//
// These are the two layers where the cook-off entrants failed silently rather than
// loudly (docs/BAKEOFF_MCP.md): three clients wrote LSP `Content-Length:` preambles
// into a newline-delimited stream, and six keyed their correlation map on uint64 so
// every string id collapsed to 0. Both bugs pass a happy-path test. The checks here
// are the ones that would have caught them.

#include <string>
#include <vector>

#include "src/mcp/framing.hpp"
#include "src/mcp/message.hpp"

#include "tests/check.hpp"

using namespace lmp::mcp;

namespace {

std::vector<std::string> collect(LineFramer& framer, std::string_view bytes) {
    std::vector<std::string> out;
    framer.feed(bytes, [&](std::string_view line) { out.emplace_back(line); });
    return out;
}

} // namespace

TEST(framer_splits_on_newlines) {
    LineFramer f;
    const auto lines = collect(f, "{\"a\":1}\n{\"b\":2}\n");
    REQUIRE(lines.size() == 2);
    CHECK_EQ(lines[0], std::string("{\"a\":1}"));
    CHECK_EQ(lines[1], std::string("{\"b\":2}"));
    CHECK_EQ(f.buffered(), std::size_t(0));
}

TEST(framer_holds_partial_line_across_feeds) {
    // The case a getline() loop gets wrong: a read boundary lands mid-message. This is
    // the normal case on a pipe, not an edge case.
    LineFramer f;
    auto lines = collect(f, "{\"jsonrpc\":\"2.0\",\"id\"");
    CHECK_EQ(lines.size(), std::size_t(0));
    CHECK(f.buffered() > 0);

    lines = collect(f, ":1,\"method\":\"ping\"}\n");
    REQUIRE(lines.size() == 1);
    CHECK_EQ(lines[0], std::string("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}"));
    CHECK_EQ(f.buffered(), std::size_t(0));
}

TEST(framer_splits_one_byte_at_a_time) {
    // The pathological version of the above: every byte its own read.
    const std::string msg = "{\"id\":7,\"method\":\"ping\"}";
    LineFramer f;
    std::vector<std::string> lines;
    const std::string stream = msg + "\n";
    for (const char c : stream) {
        f.feed(std::string_view(&c, 1), [&](std::string_view l) { lines.emplace_back(l); });
    }
    REQUIRE(lines.size() == 1);
    CHECK_EQ(lines[0], msg);
}

TEST(framer_tolerates_crlf_and_blank_lines) {
    LineFramer f;
    const auto lines = collect(f, "{\"a\":1}\r\n\r\n\n{\"b\":2}\r\n");
    REQUIRE(lines.size() == 2);
    CHECK_EQ(lines[0], std::string("{\"a\":1}"));
    CHECK_EQ(lines[1], std::string("{\"b\":2}"));
}

TEST(framer_drops_oversized_message_instead_of_growing) {
    LineFramer f;
    bool overflowed = false;
    const std::string big(1024 * 1024, 'x');
    for (int i = 0; i < 40 && !overflowed; ++i) {
        f.feed(big, [](std::string_view) {}, [&](std::size_t) { overflowed = true; });
    }
    CHECK(overflowed);
    CHECK_EQ(f.buffered(), std::size_t(0));
}

TEST(encode_line_emits_exactly_one_trailing_newline) {
    const std::string s = encode_line(nlohmann::json{{"a", 1}});
    CHECK_EQ(s.back(), '\n');
    CHECK_EQ(s.find('\n'), s.size() - 1);
}

TEST(encode_line_escapes_interior_newlines) {
    // The framing's whole invariant: a newline inside a string must not reach the wire
    // as a newline, or it splits one message into two and desyncs the peer forever.
    const std::string s = encode_line(nlohmann::json{{"text", "line one\nline two"}});
    CHECK_EQ(s.find('\n'), s.size() - 1);
    CHECK(s.find("\\n") != std::string::npos);
}

// --- message model ---------------------------------------------------------

TEST(string_ids_survive_a_round_trip) {
    // The uint64-map bug, stated as a check.
    const Id id = Id::from_json(nlohmann::json("abc-1"));
    CHECK(id.is_string());
    CHECK_EQ(id.as_string(), std::string("abc-1"));
    CHECK_EQ(id.to_json().dump(), std::string("\"abc-1\""));
}

TEST(string_and_number_ids_do_not_collide) {
    const Id as_num = Id::number(1);
    const Id as_str = Id::string("1");
    CHECK(!(as_num == as_str));

    // And they must land in different buckets of the correlation map.
    const Id::Hash hash;
    CHECK(hash(as_num) != hash(as_str));
}

TEST(absent_id_is_distinct_from_null_id) {
    // A message with no id is a notification and must never be answered. `"id": null`
    // is a malformed request and must be. Collapsing them makes a server answer its
    // peer's notifications, which is what entrant S5 did.
    const Message notif = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","method":"x"})"));
    CHECK(notif.is_notification());
    CHECK(notif.id.is_none());

    const Message null_id =
        classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":null,"method":"x"})"));
    CHECK(null_id.is_invalid());
}

TEST(classify_recognises_each_message_shape) {
    CHECK(classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1,"method":"ping"})")).is_request());
    CHECK(classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","method":"notifications/x"})"))
              .is_notification());
    CHECK(classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1,"result":{}})")).is_response());
    CHECK(classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"x"}})"))
              .is_response());

    CHECK(classify(nlohmann::json::parse(R"([1,2,3])")).is_invalid());
    CHECK(classify(nlohmann::json::parse(R"({"jsonrpc":"1.0","id":1,"method":"ping"})")).is_invalid());
    CHECK(classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1})")).is_invalid());
    CHECK(classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1,"result":{},"error":{}})"))
              .is_invalid());
}

TEST(response_error_is_parsed_out) {
    const Message m = classify(nlohmann::json::parse(
        R"({"jsonrpc":"2.0","id":9,"error":{"code":-32601,"message":"nope","data":{"x":1}}})"));
    REQUIRE(m.is_response());
    REQUIRE(m.error.has_value());
    CHECK_EQ(m.error->code, -32601);
    CHECK_EQ(m.error->message, std::string("nope"));
    CHECK(m.error->data.has_value());
}

TEST(notifications_carry_no_id_on_the_wire) {
    const nlohmann::json n = make_notification("notifications/initialized", nullptr);
    CHECK(!n.contains("id"));
    CHECK(!n.contains("params"));
    CHECK_EQ(n["jsonrpc"].get<std::string>(), std::string("2.0"));
}

TEST(the_framing_checks_can_actually_fail) {
    // The repo's rule: a green check counts only once it has been shown capable of
    // going red. Without this, every check above is asserting against a framework
    // nobody has falsified.
    EXPECT_FAILING_CHECKS(2, {
        LineFramer f;
        const auto lines = collect(f, "only-one-line\n");
        CHECK_EQ(lines.size(), std::size_t(2)); // it is 1
        CHECK(classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1,"method":"ping"})"))
                  .is_notification()); // it is a request
    });
}
