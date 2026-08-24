// The transcript that goes on the wire when a run is resumed.
//
// Everything here is about a bound holding. The context store is unbounded by design and
// one real observation in the log is 708,670 bytes, so the failure this guards is not a
// wrong picture -- it is a reply large enough to stall the extension host, or a string cut
// through the middle of a codepoint, which loses the ENTIRE transcript to a parse error at
// the other end rather than one observation.

#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/surface/transcript.hpp"
#include "tests/check.hpp"

using lmp::context::ContextStore;
using lmp::context::TurnRecord;
using namespace lmp::surface;

namespace {

TurnRecord tool_turn(const std::string& name, const std::string& observation) {
    TurnRecord t;
    t.tool_name = name;
    t.tool_args_summary = "path=x";
    t.observation = observation;
    return t;
}

} // namespace

TEST(a_transcript_carries_the_conversation_in_order) {
    ContextStore ctx("fix the build");
    TurnRecord a;
    a.user_text = "please start";
    ctx.add_turn(std::move(a));
    TurnRecord b;
    b.assistant_text = "looking now";
    b.tool_name = "read_file";
    b.tool_args_summary = "path=main.cpp";
    b.observation = "int main() {}";
    ctx.add_turn(std::move(b));

    int omitted = -1;
    const std::vector<lmp::protocol::TranscriptEntry> t = transcript::build(ctx, omitted);
    CHECK_EQ(omitted, 0);
    REQUIRE(t.size() == 3);
    CHECK_EQ(t[0].role, std::string{"user"});
    CHECK_EQ(t[0].text, std::string{"please start"});
    CHECK_EQ(t[1].role, std::string{"assistant"});
    CHECK_EQ(t[1].text, std::string{"looking now"});
    // The call and its result are ONE entry, because that is how the pane draws a live
    // tool turn: one row whose summary is the call and whose body is what came back.
    CHECK_EQ(t[2].role, std::string{"tool"});
    CHECK_EQ(t[2].tool, std::string{"read_file"});
    CHECK_EQ(t[2].args, std::string{"path=main.cpp"});
    CHECK_EQ(t[2].text, std::string{"int main() {}"});
    for (const auto& e : t) {
        CHECK(!e.truncated);
    }
}

TEST(an_observation_larger_than_the_entry_cap_is_cut_and_says_so) {
    ContextStore ctx("read a big file");
    ctx.add_turn(tool_turn("read_file", std::string(transcript::kMaxEntryChars * 3, 'x')));

    int omitted = -1;
    const std::vector<lmp::protocol::TranscriptEntry> t = transcript::build(ctx, omitted);
    REQUIRE(t.size() == 1);
    CHECK(t[0].text.size() <= transcript::kMaxEntryChars);
    // A silently shortened file is indistinguishable from a short file, which is the
    // whole reason this flag exists.
    CHECK(t[0].truncated);
}

TEST(a_cut_never_lands_inside_a_utf8_codepoint) {
    // THE FAILURE THIS PREVENTS IS TOTAL, not partial: half a codepoint in a JSON string
    // is a parse error at the other end, so cutting one observation badly loses every
    // other entry with it. The padding is sized so the cap falls mid-character.
    for (std::size_t pad = 0; pad < 4; ++pad) {
        std::string body(transcript::kMaxEntryChars - pad, 'a');
        body += "\xE2\x9C\x93"; // U+2713, three bytes
        body += std::string(64, 'b');
        ContextStore ctx("m");
        ctx.add_turn(tool_turn("read_file", body));
        int omitted = -1;
        const std::vector<lmp::protocol::TranscriptEntry> t = transcript::build(ctx, omitted);
        REQUIRE(t.size() == 1);
        CHECK(t[0].truncated);
        // The test is that the result is VALID UTF-8, not that it ends in ASCII: a string
        // ending with a complete three-byte character legitimately ends on a continuation
        // byte. Asserting the narrower thing was wrong about the code, not about the risk.
        const std::string& out = t[0].text;
        REQUIRE(!out.empty());
        std::size_t i = 0;
        bool valid = true;
        while (i < out.size()) {
            const auto c = static_cast<unsigned char>(out[i]);
            std::size_t len = 0;
            if ((c & 0x80U) == 0) {
                len = 1;
            } else if ((c & 0xE0U) == 0xC0U) {
                len = 2;
            } else if ((c & 0xF0U) == 0xE0U) {
                len = 3;
            } else if ((c & 0xF8U) == 0xF0U) {
                len = 4;
            } else {
                valid = false; // a continuation byte where a leader belongs
                break;
            }
            if (i + len > out.size()) {
                valid = false; // a sequence the cut left unfinished -- the real failure
                break;
            }
            for (std::size_t k = 1; k < len; ++k) {
                if ((static_cast<unsigned char>(out[i + k]) & 0xC0U) != 0x80U) {
                    valid = false;
                }
            }
            i += len;
        }
        CHECK(valid);
    }
}

TEST(the_total_cap_keeps_the_TAIL_and_reports_what_it_dropped) {
    // A conversation is read from where it left off, and the turns that explain why a run
    // died are the last ones -- so the oldest are the ones to lose.
    ContextStore ctx("a long run");
    const std::size_t big = transcript::kMaxEntryChars;
    const int n = static_cast<int>((transcript::kMaxTotalChars / big) + 20);
    for (int i = 0; i < n; ++i) {
        ctx.add_turn(tool_turn("read_file", std::string(big, static_cast<char>('a' + (i % 26)))));
    }

    int omitted = -1;
    const std::vector<lmp::protocol::TranscriptEntry> t = transcript::build(ctx, omitted);
    CHECK(omitted > 0);
    CHECK(static_cast<int>(t.size()) + omitted == n);

    std::size_t total = 0;
    for (const auto& e : t) {
        total += e.text.size();
    }
    CHECK(total <= transcript::kMaxTotalChars);
    // The LAST turn is present -- dropping from the wrong end would show the reader the
    // beginning of a run whose ending is what they came back for.
    REQUIRE(!t.empty());
    CHECK_EQ(t.back().text.front(), static_cast<char>('a' + ((n - 1) % 26)));
}

TEST(an_empty_context_yields_an_empty_transcript_rather_than_a_blank_entry) {
    ContextStore ctx("nothing happened yet");
    int omitted = -1;
    const std::vector<lmp::protocol::TranscriptEntry> t = transcript::build(ctx, omitted);
    CHECK(t.empty());
    CHECK_EQ(omitted, 0);
}

// A tool observation is a FILE, and files contain the characters JSON cares about.
//
// The transcript is the first thing in this protocol to put arbitrary file bytes on the
// wire in bulk. If one observation's quotes or backslashes escape wrongly the whole reply
// fails to parse and the resume looks like a sidecar crash, so the escaping is asserted
// here rather than assumed from the generator.
TEST(an_observation_full_of_json_metacharacters_serializes_and_parses) {
    lmp::protocol::TranscriptEntry e;
    e.role = "tool";
    e.tool = "read_file";
    e.args = "path=\"weird\".json";
    e.text = "line one\n\"quoted\"\tand a backslash \\ and a brace } and \x01 control";
    e.is_error = false;
    e.truncated = true;

    std::string out;
    lmp::protocol::append_value(out, e);

    // Every quote inside a value must be escaped: an unescaped one closes the string early
    // and everything after it is parsed as structure.
    CHECK(out.find("\\\"quoted\\\"") != std::string::npos);
    // A real newline may never appear raw inside a JSON string.
    CHECK(out.find('\n') == std::string::npos);
    // And the shape is still one object with the fields we put in it.
    CHECK(out.front() == '{');
    CHECK(out.back() == '}');
    CHECK(out.find("\"role\":\"tool\"") != std::string::npos);
    CHECK(out.find("\"truncated\":true") != std::string::npos);
}
