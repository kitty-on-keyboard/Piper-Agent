#include "src/platform/event_log.hpp"

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tests/check.hpp"

using namespace lmp::platform;

namespace {

std::string escaped(std::string_view in) {
    std::string out;
    (void)append_json_string(out, in);
    return out;
}

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_evlog_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

std::vector<std::string> read_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

} // namespace

// --- pure core -------------------------------------------------------------

TEST(json_escaping_covers_the_control_range) {
    CHECK_EQ(escaped("plain"), std::string("\"plain\""));
    CHECK_EQ(escaped("a\"b"), std::string("\"a\\\"b\""));
    CHECK_EQ(escaped("a\\b"), std::string("\"a\\\\b\""));
    CHECK_EQ(escaped("a\nb"), std::string("\"a\\nb\""));
    CHECK_EQ(escaped("a\tb"), std::string("\"a\\tb\""));
    CHECK_EQ(escaped(std::string("a\x01" "b")), std::string("\"a\\u0001b\""));
    CHECK_EQ(escaped(std::string("\x7f")), std::string("\"\x7f\""));
}

TEST(utf8_validator_rejects_the_three_classic_forgeries) {
    CHECK(is_valid_utf8("hello"));
    CHECK(is_valid_utf8("\xF0\x9F\x98\x80"));           // U+1F600, well formed
    CHECK(is_valid_utf8(""));

    CHECK(!is_valid_utf8(std::string("\xC0\xAF")));      // overlong '/'
    CHECK(!is_valid_utf8(std::string("\xED\xA0\x80")));  // U+D800, a surrogate half
    CHECK(!is_valid_utf8(std::string("\xF0\x9F\x98")));  // truncated 4-byte sequence
    CHECK(!is_valid_utf8(std::string("\xF5\x80\x80\x80")));  // beyond U+10FFFF
    CHECK(!is_valid_utf8(std::string("\x80")));          // lone continuation byte
}

TEST(a_codepoint_split_across_two_tokens_survives_as_bytes) {
    // This is the S5.3 scenario in miniature: 944 of Qwen's tokens are byte fragments
    // that are not valid standalone UTF-8, and an emoji routinely spans two of them.
    // v1's answer was a 231-line external decoder bolted on after the fact. The log's
    // job here is narrower and absolute: whatever bytes it was handed come back out.
    const std::string half1 = "\xF0\x9F"; // first half of U+1F600
    const std::string half2 = "\x98\x80"; // second half

    CHECK(!is_valid_utf8(half1));
    CHECK(!is_valid_utf8(half2));

    Event ev;
    ev.kind = "token";
    ev.fields = {{"piece", half1}};
    const std::string line = serialize_event(ev);

    // The display form is lossy and says so, U+FFFD.
    CHECK(line.find("\xEF\xBF\xBD") != std::string::npos);
    // The exact bytes are carried alongside it.
    CHECK(line.find("\"piece__b64\":") != std::string::npos);

    const std::string b64 = base64_encode(half1);
    CHECK(line.find(b64) != std::string::npos);

    std::string round;
    CHECK(base64_decode(b64, round));
    CHECK_EQ(round, half1);

    // And the two halves concatenate back to the original codepoint.
    std::string r1;
    std::string r2;
    CHECK(base64_decode(base64_encode(half1), r1));
    CHECK(base64_decode(base64_encode(half2), r2));
    CHECK(is_valid_utf8(r1 + r2));
    CHECK_EQ(r1 + r2, std::string("\xF0\x9F\x98\x80"));
}

TEST(valid_utf8_pays_no_base64_tax) {
    Event ev;
    ev.kind = "token";
    ev.fields = {{"piece", "\xF0\x9F\x98\x80"}};
    const std::string line = serialize_event(ev);
    CHECK(line.find("__b64") == std::string::npos);
    CHECK(line.find("\xF0\x9F\x98\x80") != std::string::npos);
}

TEST(base64_round_trips_every_byte_value) {
    std::string all;
    for (int i = 0; i < 256; ++i) {
        all.push_back(static_cast<char>(i));
    }
    for (std::size_t len = 0; len <= all.size(); ++len) {
        const std::string in = all.substr(0, len);
        std::string out;
        REQUIRE(base64_decode(base64_encode(in), out));
        if (out != in) {
            CHECK_EQ(out.size(), in.size());
            CHECK(false);
            return;
        }
    }
    CHECK(true);
}

TEST(base64_decode_rejects_garbage) {
    std::string out;
    CHECK(!base64_decode("abc", out));      // length not a multiple of 4
    CHECK(!base64_decode("ab*d", out));     // outside the alphabet
    CHECK(!base64_decode("a=cd", out));     // data after padding
}

TEST(serialize_event_is_deterministic_and_ordered) {
    Event a;
    a.seq = 7;
    a.wall_ns = 123;
    a.mono_us = 45;
    a.kind = "tool_result";
    a.fields = {{"tool", "read_file"}, {"status", "Ok"}};

    const std::string line = serialize_event(a);
    CHECK_EQ(line,
             std::string("{\"seq\":7,\"t_wall_ns\":123,\"t_mono_us\":45,"
                         "\"kind\":\"tool_result\",\"tool\":\"read_file\","
                         "\"status\":\"Ok\"}\n"));
    CHECK_EQ(serialize_event(a), line);
    CHECK_EQ(line.back(), '\n');
}

TEST(rotated_path_inserts_before_the_extension) {
    CHECK_EQ(rotated_path("events.jsonl", 1), std::string("events.1.jsonl"));
    CHECK_EQ(rotated_path("/a/b/events.jsonl", 3), std::string("/a/b/events.3.jsonl"));
    CHECK_EQ(rotated_path("events", 2), std::string("events.2"));
    // A dot in a directory name is not an extension.
    CHECK_EQ(rotated_path("/a.b/events", 1), std::string("/a.b/events.1"));
}

// --- adapter ---------------------------------------------------------------

TEST(writer_assigns_seq_and_timestamps_as_outputs) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    const std::string path = dir + "/events.jsonl";

    ManualClock clock;
    EventLogWriter w;
    const OpenResult r = w.open({path, 1 << 20, 3});
    REQUIRE(r.ok);

    Event ev;
    ev.seq = 999;       // a caller cannot forge ordering...
    ev.wall_ns = -1;    // ...or timestamps
    ev.kind = "first";
    clock.advance(std::chrono::seconds{5});
    w.append(ev, clock);
    CHECK_EQ(ev.seq, std::uint64_t{1});
    CHECK_EQ(ev.wall_ns, static_cast<std::int64_t>(5'000'000'000));

    Event ev2;
    ev2.kind = "second";
    w.append(ev2, clock);
    CHECK_EQ(ev2.seq, std::uint64_t{2});
    CHECK_EQ(w.events_written(), std::uint64_t{2});

    w.close();
    const std::vector<std::string> lines = read_lines(path);
    CHECK_EQ(lines.size(), std::size_t{2});
}

TEST(writer_refuses_a_second_writer_on_the_same_path) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    const std::string path = dir + "/events.jsonl";

    EventLogWriter a;
    EventLogWriter b;
    CHECK(a.open({path, 1 << 20, 2}).ok);

    const OpenResult second = b.open({path, 1 << 20, 2});
    // "One writer" is enforced, not documented. Two writers interleave partial lines and
    // the corrupted replay input reads as a model bug.
    CHECK(!second.ok);
    CHECK(second.error.find("already holds") != std::string::npos);

    a.close();
    CHECK(b.open({path, 1 << 20, 2}).ok); // released on close
}

TEST(writer_rotates_and_stays_bounded) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    const std::string path = dir + "/events.jsonl";

    ManualClock clock;
    EventLogWriter w;
    REQUIRE(w.open({path, 200, 3}).ok); // 3 files total: live + .1 + .2

    for (int i = 0; i < 200; ++i) {
        Event ev;
        ev.kind = "e";
        ev.fields = {{"i", std::to_string(i)}};
        w.append(ev, clock);
        clock.advance(std::chrono::milliseconds{1});
    }
    w.close();

    CHECK(w.rotations() > 0);
    CHECK(w.events_written() == std::uint64_t{200});

    // Bounded: the live file plus exactly two archives, and nothing beyond max_files.
    std::ifstream live(path);
    std::ifstream a1(rotated_path(path, 1));
    std::ifstream a2(rotated_path(path, 2));
    std::ifstream a3(rotated_path(path, 3));
    CHECK(live.good());
    CHECK(a1.good());
    CHECK(a2.good());
    CHECK(!a3.good());
}

TEST(appending_to_a_closed_writer_is_a_no_op_not_a_crash) {
    ManualClock clock;
    EventLogWriter w;
    Event ev;
    ev.kind = "orphan";
    w.append(ev, clock);
    CHECK_EQ(w.events_written(), std::uint64_t{0});
    CHECK(!w.is_open());
}
