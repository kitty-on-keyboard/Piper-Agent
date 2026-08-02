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
};

class MarkdownStream {
  public:
    [[nodiscard]] std::vector<Event> feed(std::string_view chunk);
    [[nodiscard]] std::vector<Event> finish();
    void reset();
    [[nodiscard]] bool pending() const noexcept;

  private:
    void emit(Event e);
    void emitText(const std::string& s);
    void emitCodeBlockText(const std::string& s);
    void openListItems(int target_level);
    void closeListItems(int target);
    void process_pending(bool flush);

    std::string pending_;
    std::vector<Event> events_;

    bool in_code_block_ = false;
    int opening_fence_length_ = 0;
    bool in_code_block_info_ = false;
    std::string code_block_info_;
    bool in_inline_code_ = false;
    bool in_heading_ = false;
    int current_list_depth_ = -1;
    bool at_line_start_ = true;
};

} // namespace md
