#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace md {

enum class EventKind : std::uint8_t {
    Text,
    CodeBlockOpen,
    CodeBlockText,
    CodeBlockClose,
    InlineCodeOpen,
    InlineCodeClose,
    HeadingOpen,
    HeadingClose,
    ListItemOpen,
    ListItemClose,
    ParagraphBreak,
};

struct Event {
    EventKind kind = EventKind::Text;
    std::string text;
    std::string info;
    int level = 0;

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
    std::vector<Event> m_events;
    std::string m_holdback;

    enum class State {
        Normal,
        FencedCodeInfo,
        FencedCodeContent,
        InlineCode,
        Heading
    };

    State m_state = State::Normal;
    bool m_at_line_start = true;
    std::vector<int> m_list_indents;

    int m_fenced_ticks = 0;
    std::string m_fenced_info;

    int m_inline_ticks = 0;

    int m_heading_level = 0;

    void process(bool is_finish);

    void emit_text(std::string_view text);
    void close_lists(int up_to_indent);
    void close_all_lists();

    // Limits
    static constexpr size_t MAX_HOLDBACK = 64;
};

} // namespace md
