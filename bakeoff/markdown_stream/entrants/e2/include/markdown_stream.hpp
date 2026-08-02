#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>

namespace md {

enum class EventKind : std::uint8_t {
    Text,           // plain prose to append
    CodeBlockOpen,  // `info` carries the language tag, possibly empty
    CodeBlockText,  // raw code to append, never interpreted
    CodeBlockClose,
    InlineCodeOpen,
    InlineCodeClose,
    HeadingOpen,    // `level` is 1..6
    HeadingClose,
    ListItemOpen,   // `level` is the nesting depth, starting at 0
    ListItemClose,
    ParagraphBreak,
};

struct Event {
    EventKind kind = EventKind::Text;
    std::string text;   // Text / CodeBlockText payload
    std::string info;   // language tag for CodeBlockOpen
    int level = 0;      // heading level or list depth

    bool operator==(const Event& other) const {
        return kind == other.kind && text == other.text && info == other.info && level == other.level;
    }
};

class MarkdownStream {
  public:
    [[nodiscard]] std::vector<Event> feed(std::string_view chunk);
    [[nodiscard]] std::vector<Event> finish();
    void reset();
    [[nodiscard]] bool pending() const noexcept;

  private:
    std::string buffer_;
    bool in_fenced_code_ = false;
    std::string fence_marker_;

    bool in_inline_code_ = false;
    int inline_fence_length_ = 0;

    bool in_heading_ = false;

    std::vector<int> list_stack_;
    bool at_line_start_ = true;

    enum class ParseRes {
        Matched,
        NeedMore,
        NoMatch
    };

    struct ParseResult {
        ParseRes res;
        size_t consumed;
    };

    std::vector<Event> process_buffer(bool is_finish);
    ParseResult try_parse_fenced_code(std::vector<Event>& events, bool is_finish);

    void close_inline_and_heading(std::vector<Event>& events);
    void close_all_lists(std::vector<Event>& events);
};

} // namespace md
