#include "markdown_stream.hpp"
#include <gtest/gtest.h>

using namespace md;

namespace md {
inline bool operator==(const Event& a, const Event& b) {
    return a.kind == b.kind && a.text == b.text && a.info == b.info && a.level == b.level;
}

inline std::ostream& operator<<(std::ostream& os, const Event& e) {
    os << "Event(kind=" << static_cast<int>(e.kind) << ", text='" << e.text << "', info='" << e.info << "', level=" << e.level << ")";
    return os;
}
}

std::vector<Event> ProcessAll(const std::string& input) {
    MarkdownStream stream;
    auto events = stream.feed(input);
    auto finish_events = stream.finish();
    events.insert(events.end(), finish_events.begin(), finish_events.end());
    return events;
}

std::vector<Event> ProcessSplit(const std::string& input, size_t split_pos) {
    MarkdownStream stream;
    auto events = stream.feed(input.substr(0, split_pos));
    auto events2 = stream.feed(input.substr(split_pos));
    auto finish_events = stream.finish();
    events.insert(events.end(), events2.begin(), events2.end());
    events.insert(events.end(), finish_events.begin(), finish_events.end());

    // Merge consecutive text events to allow comparison
    std::vector<Event> merged;
    for (const auto& e : events) {
        if (!merged.empty() && merged.back().kind == e.kind && (e.kind == EventKind::Text || e.kind == EventKind::CodeBlockText)) {
            merged.back().text += e.text;
        } else {
            merged.push_back(e);
        }
    }
    return merged;
}

void CheckSplitInvariance(const std::string& input) {
    auto full_events = ProcessAll(input);

    // Also merge full_events to compare fairly against merged split events
    std::vector<Event> full_merged;
    for (const auto& e : full_events) {
        if (!full_merged.empty() && full_merged.back().kind == e.kind && (e.kind == EventKind::Text || e.kind == EventKind::CodeBlockText)) {
            full_merged.back().text += e.text;
        } else {
            full_merged.push_back(e);
        }
    }

    for (size_t i = 0; i <= input.size(); ++i) {
        auto split_events = ProcessSplit(input, i);
        ASSERT_EQ(full_merged.size(), split_events.size()) << "Mismatch at split " << i << " for input:\n" << input;
        for (size_t j = 0; j < full_merged.size(); ++j) {
            EXPECT_EQ(full_merged[j], split_events[j]) << "Mismatch at event " << j << ", split " << i << " for input:\n" << input;
        }
    }
}

TEST(MarkdownStreamTest, SplitInvarianceExhaustive) {
    const std::vector<std::string> snippets = {
        "Just some text.",
        "`inline`",
        "```\ncode\n```",
        "```cpp\nint main() {}\n```\n",
        "````\nfour ticks\n````",
        "# H1",
        "## H2",
        "### H3\nText",
        "* List item",
        "  * Nested",
        "    * Double nested",
        "1. Ordered",
        "   1. Nested ordered",
        "2. Any digit",
        "Mixed:\n* List\n```\ncode\n```\n# Heading",
        "```unclosed\ncode here\n",
        "`unclosed inline",
        "#Unspaced",
        "  # Spaced heading",
        "* List item\n* List item 2",
        "```\n`backtick inside`\n```",
        "Text\n\nParagraph\n"
    };

    for (const auto& snippet : snippets) {
        CheckSplitInvariance(snippet);
    }
}

TEST(MarkdownStreamTest, AdversarialFences) {
    MarkdownStream stream;
    auto ev = stream.feed("```cp");
    EXPECT_TRUE(stream.pending()); // Holding back language tag
    auto ev2 = stream.feed("p\n");
    ASSERT_EQ(ev2.size(), 1);
    EXPECT_EQ(ev2[0].kind, EventKind::CodeBlockOpen);
    EXPECT_EQ(ev2[0].info, "cpp");

    // Backticks inside code block
    auto ev3 = stream.feed("``");
    EXPECT_TRUE(stream.pending()); // Might be closing fence
    auto ev4 = stream.feed("not fence");
    EXPECT_EQ(ev4.size(), 1);
    EXPECT_EQ(ev4[0].kind, EventKind::CodeBlockText);
    EXPECT_EQ(ev4[0].text, "``not fence");
}

TEST(MarkdownStreamTest, NoSwallowingOnFinish) {
    MarkdownStream stream;
    auto ev = stream.feed("```cpp\nint x = 0;\n"); // unclosed code block
    auto fin = stream.finish();

    ASSERT_EQ(ev.size(), 2);
    EXPECT_EQ(ev[0].kind, EventKind::CodeBlockOpen);
    EXPECT_EQ(ev[1].kind, EventKind::CodeBlockText);
    EXPECT_EQ(ev[1].text, "int x = 0;\n");

    ASSERT_EQ(fin.size(), 1);
    EXPECT_EQ(fin[0].kind, EventKind::CodeBlockClose);
}

TEST(MarkdownStreamTest, HoldbackBound) {
    MarkdownStream stream;
    const size_t MAX_HOLDBACK = 130;

    // Send a very long line of spaces to ensure it doesn't hang holding back spaces forever.
    size_t consecutive_empty = 0;
    for (int i = 0; i < 200; ++i) {
        auto evs = stream.feed(" ");
        if (evs.empty()) consecutive_empty++;
        else consecutive_empty = 0;
        EXPECT_LE(consecutive_empty, MAX_HOLDBACK);
    }

    stream.reset();

    // Send a long language tag
    (void)stream.feed("```");
    consecutive_empty = 0;
    for (int i = 0; i < 200; ++i) {
        auto evs = stream.feed("a");
        if (evs.empty()) consecutive_empty++;
        else consecutive_empty = 0;
        EXPECT_LE(consecutive_empty, MAX_HOLDBACK);
    }
}

// Deliberately broken stream to prove tests can fail
class BrokenMarkdownStream : public MarkdownStream {
public:
    std::vector<Event> feed_broken(std::string_view chunk) {
        if (chunk.size() == 1) return {};
        return feed(chunk);
    }
};

std::vector<Event> ProcessAllBroken(const std::string& input) {
    BrokenMarkdownStream stream;
    auto events = stream.feed_broken(input);
    auto finish_events = stream.finish();
    events.insert(events.end(), finish_events.begin(), finish_events.end());
    return events;
}

std::vector<Event> ProcessSplitBroken(const std::string& input, size_t split_pos) {
    BrokenMarkdownStream stream;
    auto events = stream.feed_broken(input.substr(0, split_pos));
    auto events2 = stream.feed_broken(input.substr(split_pos));
    auto finish_events = stream.finish();
    events.insert(events.end(), events2.begin(), events2.end());
    events.insert(events.end(), finish_events.begin(), finish_events.end());
    return events;
}

TEST(MarkdownStreamTest, Falsification) {
    std::string input = "Hello world";
    auto full = ProcessAllBroken(input);
    auto split = ProcessSplitBroken(input, 5);

    bool same = full.size() == split.size();
    if (same) {
        for (size_t i = 0; i < full.size(); ++i) {
            if (!(full[i] == split[i])) {
                same = false;
                break;
            }
        }
    }
    EXPECT_FALSE(same);
}
