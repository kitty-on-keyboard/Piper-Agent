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
    std::string m_pending;

    enum class State {
        Normal,
        InHeading,
        CodeFenceInfo,
        InCodeBlock,
        InInlineCode
    };

    State m_state = State::Normal;
    bool m_at_line_start = true;
    int m_consecutive_newlines = 0;

    // For lists (track nesting via indentation spaces)
    std::vector<int> m_list_indentations;

    // For code blocks
    int m_fence_length = 0;
    char m_fence_char = '\0';

    // For inline code
    int m_inline_fence_length = 0;

    void consume(std::vector<Event>& out, bool eof);
    size_t try_consume(std::string_view pending, std::vector<Event>& out, bool eof);
    void emit_text(std::vector<Event>& out, std::string_view text);
    void emit_code_text(std::vector<Event>& out, std::string_view text);
    void handle_list_item(int spaces, std::vector<Event>& out);
    size_t consume_as_text(std::string_view pending, std::vector<Event>& out);
};

} // namespace md
