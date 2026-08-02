// Falsifier base: a straightforward incremental MarkdownStream, written against Brief E's
// header before any entrant was read.
//
// It exists for two reasons. First, a scoreboard on which nothing can pass is as useless as
// one on which nothing can fail -- this is the proof the columns are satisfiable. Second,
// each falsifier is this file plus ONE planted defect, so the delta against the base row is
// unambiguous rather than a guess about which of several changes fired which column.
//
// It is not an entry and it is not tuned. Where the brief leaves a choice open it takes the
// obvious one.
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
    [[nodiscard]] bool pending() const noexcept { return !hold_.empty(); }

  private:
    enum class Mode : std::uint8_t { LineStart, Body, FenceLineStart, FenceBody };

    void pump(bool at_end);
    void emit(EventKind k, std::string text = {}, std::string info = {}, int level = 0);

    std::string hold_;
    std::vector<Event> out_;
    std::size_t fence_len_ = 0;  // backticks in the OPENING fence; the close must match it
    Mode mode_ = Mode::LineStart;
    bool inline_open_ = false;
    bool heading_open_ = false;
    bool list_open_ = false;
    bool fence_open_ = false;
};

} // namespace md
