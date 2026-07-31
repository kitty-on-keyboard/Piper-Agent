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
