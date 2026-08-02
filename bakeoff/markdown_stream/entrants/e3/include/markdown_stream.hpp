#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

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

    // Equality operator for tests
    bool operator==(const Event& other) const {
        return kind == other.kind && text == other.text && info == other.info && level == other.level;
    }
};

class MarkdownStream {
  public:
    // Feed the next chunk. Returns only events that are now CERTAIN. A chunk ending in
    // "``" emits nothing for those bytes: they may become a fence or may be literal.
    [[nodiscard]] std::vector<Event> feed(std::string_view chunk);

    // End of stream. Flushes held-back bytes as text and closes anything still open, so an
    // unterminated code block does not swallow the output.
    [[nodiscard]] std::vector<Event> finish();

    void reset();

    // True when bytes are being withheld pending disambiguation. The UI can show a caret.
    [[nodiscard]] bool pending() const noexcept;

  private:
    std::string holdback_;

    // State machine status
    bool in_fenced_code_ = false;
    std::string fence_marker_; // The exact backticks used to open
    std::string current_language_tag_;
    bool collecting_language_tag_ = false;

    bool in_inline_code_ = false;
    std::string inline_code_marker_; // The exact backticks used to open

    int current_heading_level_ = 0; // 0 if not in a heading

    // Ordered or unordered lists can be nested. We just need to track the levels of open lists.
    std::vector<int> open_list_depths_;

    bool at_line_start_ = true;

    void process_holdback(std::vector<Event>& events);
    void close_all_lists(std::vector<Event>& events);
    int get_list_depth_and_close_others(int indent, std::vector<Event>& events);
};

} // namespace md
