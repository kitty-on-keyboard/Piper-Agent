// The framing contract (S4.2, S4.3). Both v1 transport bugs are pinned here as tests
// that would have caught them.

#include <string>
#include <vector>

#include "src/model/backend.hpp"
#include "src/platform/spsc_channel.hpp"
#include "src/surface/transport.hpp"

#include "tests/check.hpp"

using namespace lmp;
using surface::StdinReader;

namespace {

std::vector<std::string> drain(platform::SpscChannel<std::string>& ch) {
    std::vector<std::string> out;
    std::string m;
    while (ch.try_pop(m)) {
        out.push_back(m);
    }
    return out;
}

} // namespace

TEST(a_message_split_across_reads_arrives_whole) {
    // v1 read 127 bytes at a time into a fixed ring and reassembled in the CONSUMER, so
    // a control check in the reader saw an arbitrary byte window. Here the producer
    // frames, so a chunk boundary anywhere -- including mid-method-name -- is invisible.
    platform::SpscChannel<std::string> ch(64);
    model::CancelToken cancel;
    StdinReader reader(ch, cancel);

    const std::string msg = R"({"jsonrpc":"2.0","id":"1","method":"lmp/start"})";
    for (std::size_t split = 1; split < msg.size(); split += 7) {
        reader.feed_for_test(msg.substr(0, split));
        CHECK(drain(ch).empty()); // nothing surfaces until the newline
        reader.feed_for_test(msg.substr(split));
        reader.feed_for_test("\n");
        const std::vector<std::string> got = drain(ch);
        REQUIRE(got.size() == 1);
        CHECK_EQ(got[0], msg);
    }
}

TEST(discussing_cancellation_does_not_cancel_anything) {
    // v1 did chunk.find("agent/cancel") on raw transport bytes, so a chat message that
    // mentioned cancelling could cancel a running mission. THIS is that test.
    platform::SpscChannel<std::string> ch(64);
    model::CancelToken cancel;
    StdinReader reader(ch, cancel);

    reader.feed_for_test(
        R"({"jsonrpc":"2.0","id":"1","method":"lmp/start","params":)"
        R"({"mission":"explain how lmp/cancel works and when to send it"}})"
        "\n");
    CHECK(!cancel.cancelled());
    CHECK_EQ(reader.cancels_seen(), std::size_t{0});
    CHECK_EQ(drain(ch).size(), std::size_t{1}); // still delivered, just not obeyed
}

TEST(a_real_cancel_is_seen_by_the_reader_not_the_consumer) {
    // Deliverable while the model is generating (S4.3): the READER sets the token, so
    // it does not wait behind a busy consumer.
    platform::SpscChannel<std::string> ch(64);
    model::CancelToken cancel;
    StdinReader reader(ch, cancel);
    reader.feed_for_test(R"({"jsonrpc":"2.0","id":"9","method":"lmp/cancel"})"
                         "\n");
    CHECK(cancel.cancelled());
    CHECK_EQ(reader.cancels_seen(), std::size_t{1});
}

TEST(several_messages_in_one_read_all_arrive_in_order) {
    platform::SpscChannel<std::string> ch(64);
    model::CancelToken cancel;
    StdinReader reader(ch, cancel);
    reader.feed_for_test(R"({"method":"a"})"
                         "\n"
                         R"({"method":"b"})"
                         "\n"
                         R"({"method":"c"})"
                         "\n");
    const std::vector<std::string> got = drain(ch);
    REQUIRE(got.size() == 3);
    CHECK_EQ(surface::method_of(got[0]), std::string("a"));
    CHECK_EQ(surface::method_of(got[1]), std::string("b"));
    CHECK_EQ(surface::method_of(got[2]), std::string("c"));
}

TEST(method_extraction_is_field_parsing_not_substring_matching) {
    CHECK_EQ(surface::method_of(R"({"method":"lmp/start"})"), std::string("lmp/start"));
    CHECK_EQ(surface::method_of(R"({"method": "lmp/start"})"), std::string("lmp/start"));
    // The literal appears only as DATA -- there is no method field at all.
    CHECK_EQ(surface::method_of(R"({"params":{"text":"\"method\":\"lmp/cancel\""}})"),
             std::string());
    CHECK_EQ(surface::method_of("{}"), std::string());
}

TEST(a_partial_trailing_message_is_held_not_delivered) {
    platform::SpscChannel<std::string> ch(64);
    model::CancelToken cancel;
    StdinReader reader(ch, cancel);
    reader.feed_for_test(R"({"method":"complete"})"
                         "\n"
                         R"({"method":"incompl)");
    const std::vector<std::string> got = drain(ch);
    CHECK_EQ(got.size(), std::size_t{1});
    CHECK_EQ(surface::method_of(got[0]), std::string("complete"));
    // The remainder is still buffered; it must not be delivered as a truncated message.
    reader.feed_for_test("ete\"}\n");
    const std::vector<std::string> rest = drain(ch);
    REQUIRE(rest.size() == 1);
    CHECK_EQ(surface::method_of(rest[0]), std::string("incomplete"));
}

TEST(an_approval_boolean_is_true_only_for_a_literal_true) {
    // Deny-by-default lives in this function. The approval reply is the one place a
    // parse failure could turn into an execution, so anything that is not an
    // unambiguous JSON `true` must read as "not approved" (S7.2).
    CHECK(surface::bool_field(R"({"approved":true})", "approved"));
    CHECK(surface::bool_field(R"({"approved": true})", "approved"));

    CHECK(!surface::bool_field(R"({"approved":false})", "approved"));
    CHECK(!surface::bool_field(R"({})", "approved"));
    // A quoted "true" is a string, not the boolean the schema declares.
    CHECK(!surface::bool_field(R"({"approved":"true"})", "approved"));
    CHECK(!surface::bool_field(R"({"approved":1})", "approved"));
    // A key that merely STARTS with the one we want must not answer for it.
    CHECK(!surface::bool_field(R"({"approved_by":"sean"})", "approved"));
}

TEST(an_approval_reply_is_matched_by_request_id_not_by_arrival_order) {
    // The stale-card path in the sidecar's ApprovalBridge turns on these two fields
    // being readable from one framed message. request_id must not be shadowed by the
    // envelope's own id, which appears FIRST in the message.
    const std::string reply =
        R"({"jsonrpc":"2.0","id":"7","method":"lmp/approve",)"
        R"("params":{"request_id":"3/2","approved":true}})";
    CHECK_EQ(surface::method_of(reply), std::string("lmp/approve"));
    CHECK_EQ(surface::string_field(reply, "id"), std::string("7"));
    CHECK_EQ(surface::string_field(reply, "request_id"), std::string("3/2"));
    CHECK(surface::bool_field(reply, "approved"));
}

TEST(a_numeric_setting_falls_back_to_the_value_the_caller_already_holds) {
    // The sampling block carries Qwen3's recommended operating point (S5.9). A missing
    // field must therefore keep the caller's pinned default, NOT collapse to zero --
    // temperature 0 is greedy decoding, which Qwen3 specifically warns against, and it
    // is exactly what a fallback of 0.0 would silently produce.
    const std::string settings =
        R"({"sampling":{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,)"
        R"("repetition_penalty":1.05,"seed":0}})";
    CHECK_EQ(surface::double_field(settings, "temperature", 9.0), 0.6);
    CHECK_EQ(surface::double_field(settings, "top_p", 9.0), 0.95);
    CHECK_EQ(surface::double_field(settings, "top_k", 9.0), 20.0);
    CHECK_EQ(surface::double_field(settings, "min_p", 9.0), 0.0);
    CHECK_EQ(surface::double_field(settings, "repetition_penalty", 9.0), 1.05);

    // Absent, and so kept.
    CHECK_EQ(surface::double_field(settings, "top_a", 0.6), 0.6);
    CHECK_EQ(surface::double_field("{}", "temperature", 0.6), 0.6);
    // A quoted number is a string. Coercing it would make a settings typo look chosen.
    CHECK_EQ(surface::double_field(R"({"temperature":"0.1"})", "temperature", 0.6), 0.6);
}
