
#include "markdown_stream.hpp"
#include <string_view>
#include <iostream>

namespace md {

enum class Match { Yes, No, NeedMore };

struct MatchRes {
    Match status;
    size_t length;
    std::string tag;
};

static MatchRes match_paragraph_break(std::string_view s) {
    if (s.size() > 0 && s[0] != '\n') return {Match::No, 0, ""};
    size_t count = 0;
    while (count < s.size() && s[count] == '\n') count++;
    if (count >= 2) return {Match::Yes, count, ""};
    if (count == 1 && s.size() == 1) return {Match::NeedMore, 0, ""};
    return {Match::No, 0, ""};
}

static MatchRes match_heading(std::string_view s) {
    size_t hashes = 0;
    while (hashes < s.size() && s[hashes] == '#') hashes++;
    if (hashes == 0) return {Match::No, 0, ""};
    if (hashes > 6) return {Match::No, 0, ""};
    if (hashes == s.size()) return {Match::NeedMore, 0, ""};
    if (s[hashes] == ' ') return {Match::Yes, hashes + 1, ""};
    return {Match::No, 0, ""};
}

static MatchRes match_list_item(std::string_view s) {
    size_t i = 0;
    while (i < s.size() && s[i] == ' ') i++;
    if (i == s.size()) {
        if (i <= 10) return {Match::NeedMore, 0, ""};
        else return {Match::No, 0, ""};
    }

    if (s[i] == '-' || s[i] == '*') {
        if (i + 1 == s.size()) return {Match::NeedMore, 0, ""};
        if (s[i+1] == ' ') return {Match::Yes, i + 2, ""};
    }

    size_t digits = 0;
    while (i + digits < s.size() && s[i + digits] >= '0' && s[i + digits] <= '9') digits++;
    if (i + digits == s.size()) {
        if (digits > 0 && digits <= 9) return {Match::NeedMore, 0, ""};
    } else if (digits > 0) {
        size_t pos = i + digits;
        if (pos == s.size()) return {Match::NeedMore, 0, ""};
        if (s[pos] == '.') {
            if (pos + 1 == s.size()) return {Match::NeedMore, 0, ""};
            if (s[pos+1] == ' ') return {Match::Yes, pos + 2, ""};
        }
    }
    return {Match::No, 0, ""};
}

static MatchRes match_fence_open(std::string_view s) {
    size_t i = 0;
    while (i < s.size() && s[i] == '`') i++;
    if (i == s.size()) {
        if (i < 10) return {Match::NeedMore, 0, ""};
    }
    if (i >= 3) {
        // Collect language tag until newline or bounded limit
        size_t j = i;
        while (j < s.size() && s[j] != '\n') {
            if (j - i > 64) break; // bounded
            j++;
        }
        if (j == s.size()) {
            if (s.size() - i <= 64) return {Match::NeedMore, 0, ""};
        }

        std::string tag = std::string(s.substr(i, j - i));
        // Strip trailing spaces from tag if needed, but not strictly required.
        // The prompt just says 'info' carries the language tag.
        size_t consume = (j < s.size() && s[j] == '\n') ? j + 1 : j;
        return {Match::Yes, consume, tag};
    }
    return {Match::No, 0, ""};
}

static MatchRes match_fence_close(std::string_view s, const std::string& fence_marker) {
    size_t i = 0;
    while (i < s.size() && s[i] == ' ') i++;
    if (i == s.size()) {
        if (i <= 3) return {Match::NeedMore, 0, ""}; // up to 3 leading spaces allowed in markdown, but let's be strict or lenient?
        // Strict: no leading spaces for simplicity unless specified. Wait, standard markdown allows up to 3 spaces. Let's just say 0 for simplicity.
    }

    // We only care about EXACT match or longer? Fences close with N or more backticks.
    size_t ticks = 0;
    while (i + ticks < s.size() && s[i + ticks] == '`') ticks++;

    if (i + ticks == s.size()) {
        if (ticks <= fence_marker.size() + 5) return {Match::NeedMore, 0, ""};
    }

    if (ticks >= fence_marker.size()) {
        size_t j = i + ticks;
        while (j < s.size() && s[j] == ' ') j++;
        if (j == s.size()) {
            if (j - (i + ticks) <= 20) return {Match::NeedMore, 0, ""};
        }
        if (j == s.size() || s[j] == '\n') {
            size_t consume = (j < s.size() && s[j] == '\n') ? j + 1 : j;
            return {Match::Yes, consume, ""};
        }
    }
    return {Match::No, 0, ""};
}

static MatchRes match_inline_code_marker(std::string_view s) {
    size_t i = 0;
    while (i < s.size() && s[i] == '`') i++;
    if (i == s.size()) {
        if (i < 8) return {Match::NeedMore, 0, ""};
    }
    if (i > 0) return {Match::Yes, i, ""};
    return {Match::No, 0, ""};
}

std::vector<Event> MarkdownStream::feed(std::string_view chunk) {
    std::vector<Event> events;
    holdback_.append(chunk);
    process_holdback(events);
    return events;
}

void MarkdownStream::close_all_lists(std::vector<Event>& events) {
    while (!open_list_depths_.empty()) {
        Event ev; ev.kind = EventKind::ListItemClose; events.push_back(ev);
        open_list_depths_.pop_back();
    }
}

int MarkdownStream::get_list_depth_and_close_others(int depth, std::vector<Event>& events) {
    while (!open_list_depths_.empty() && open_list_depths_.back() > depth) {
        Event ev; ev.kind = EventKind::ListItemClose; events.push_back(ev);
        open_list_depths_.pop_back();
    }
    if (!open_list_depths_.empty() && open_list_depths_.back() == depth) {
        Event ev; ev.kind = EventKind::ListItemClose; events.push_back(ev);
        open_list_depths_.pop_back();
    }
    return depth;
}

void MarkdownStream::process_holdback(std::vector<Event>& events) {
    size_t pos = 0;
    size_t text_start = 0;

    auto emit_text = [&]() {
        if (pos > text_start) {
            std::string t = holdback_.substr(text_start, pos - text_start);
            EventKind expected = in_fenced_code_ ? EventKind::CodeBlockText : EventKind::Text;
            if (!events.empty() && events.back().kind == expected) {
                events.back().text += t;
            } else {
                Event ev; ev.kind = expected; ev.text = t; events.push_back(ev);
            }
        }
        text_start = pos;
    };

    while (pos < holdback_.size()) {
        std::string_view s(holdback_.data() + pos, holdback_.size() - pos);

        if (in_fenced_code_) {
            if (at_line_start_) {
                auto close = match_fence_close(s, fence_marker_);
                if (close.status == Match::NeedMore) break;
                if (close.status == Match::Yes) {
                    emit_text();
                    Event ev; ev.kind = EventKind::CodeBlockClose; events.push_back(ev);
                    in_fenced_code_ = false;
                    pos += close.length;
                    text_start = pos;
                    at_line_start_ = true;
                    continue;
                }
            }

            size_t next_nl = s.find('\n');
            if (next_nl == std::string_view::npos) {
                pos += s.size();
                at_line_start_ = false;
            } else {
                pos += next_nl + 1;
                at_line_start_ = true;
            }
            continue;
        }

        if (in_inline_code_) {
            size_t i = 0;
            while (i < s.size() && s[i] == '`') i++;
            if (i == s.size()) {
                if (i <= inline_code_marker_.size()) break; // need more
            }
            if (i == inline_code_marker_.size()) {
                emit_text();
                Event ev; ev.kind = EventKind::InlineCodeClose; events.push_back(ev);
                in_inline_code_ = false;
                pos += i;
                text_start = pos;
                continue;
            } else if (i > 0) {
                pos += i;
                continue;
            }
            pos++;
            continue;
        }

        auto pb = match_paragraph_break(s);
        if (pb.status == Match::NeedMore) break;
        if (pb.status == Match::Yes) {
            emit_text();
            if (current_heading_level_ > 0) {
                Event ev; ev.kind = EventKind::HeadingClose; events.push_back(ev);
                current_heading_level_ = 0;
            }
            close_all_lists(events);
            Event ev; ev.kind = EventKind::ParagraphBreak; events.push_back(ev);
            pos += pb.length;
            text_start = pos;
            at_line_start_ = true;
            continue;
        }

        if (at_line_start_) {
            auto f = match_fence_open(s);
            if (f.status == Match::NeedMore) break;
            if (f.status == Match::Yes) {
                emit_text();
                if (current_heading_level_ > 0) {
                    Event ev; ev.kind = EventKind::HeadingClose; events.push_back(ev);
                    current_heading_level_ = 0;
                }
                close_all_lists(events);

                in_fenced_code_ = true;
                // find the exact backticks used
                size_t ticks = 0;
                while (ticks < s.size() && s[ticks] == '`') ticks++;
                fence_marker_ = std::string(s.substr(0, ticks));

                Event ev; ev.kind = EventKind::CodeBlockOpen; ev.info = f.tag; events.push_back(ev);

                pos += f.length;
                text_start = pos;
                at_line_start_ = true; // The newline was consumed, so we are at line start
                continue;
            }

            auto h = match_heading(s);
            if (h.status == Match::NeedMore) break;
            if (h.status == Match::Yes) {
                emit_text();
                close_all_lists(events);

                current_heading_level_ = h.length - 1;
                Event ev; ev.kind = EventKind::HeadingOpen; ev.level = current_heading_level_;
                events.push_back(ev);

                pos += h.length;
                text_start = pos;
                at_line_start_ = false;
                continue;
            }

            auto li = match_list_item(s);
            if (li.status == Match::NeedMore) break;
            if (li.status == Match::Yes) {
                emit_text();
                if (current_heading_level_ > 0) {
                    Event ev; ev.kind = EventKind::HeadingClose; events.push_back(ev);
                    current_heading_level_ = 0;
                }

                size_t spaces = 0;
                while (s[spaces] == ' ') spaces++;
                int depth = spaces / 2; // Assuming 2 spaces per level

                get_list_depth_and_close_others(depth, events);
                if (open_list_depths_.empty() || open_list_depths_.back() < depth) {
                    open_list_depths_.push_back(depth);
                    Event ev; ev.kind = EventKind::ListItemOpen; ev.level = depth;
                    events.push_back(ev);
                }

                pos += li.length;
                text_start = pos;
                at_line_start_ = false;
                continue;
            }
        }

        auto ic = match_inline_code_marker(s);
        if (ic.status == Match::NeedMore) break;
        if (ic.status == Match::Yes) {
            emit_text();
            Event ev; ev.kind = EventKind::InlineCodeOpen; events.push_back(ev);
            in_inline_code_ = true;
            inline_code_marker_ = std::string(s.substr(0, ic.length));
            pos += ic.length;
            text_start = pos;
            at_line_start_ = false;
            continue;
        }

        if (s[0] == '\n') {
            if (s.size() == 1) break; // Might become \n\n

            emit_text();
            if (current_heading_level_ > 0) {
                Event ev; ev.kind = EventKind::HeadingClose; events.push_back(ev);
                current_heading_level_ = 0;
            }

            // Output the single newline as text
            Event ev; ev.kind = EventKind::Text; ev.text = "\n";
            if (!events.empty() && events.back().kind == EventKind::Text) {
                events.back().text += "\n";
            } else {
                events.push_back(ev);
            }

            pos += 1;
            at_line_start_ = true;
            text_start = pos;
            continue;
        } else {
            pos += 1;
            at_line_start_ = false;
        }
    }

    emit_text();
    holdback_ = holdback_.substr(text_start);
}

std::vector<Event> MarkdownStream::finish() {
    std::vector<Event> events;

    if (!holdback_.empty()) {
        if (in_fenced_code_) {
            Event ev; ev.kind = EventKind::CodeBlockText; ev.text = holdback_; events.push_back(ev);
        } else {
            Event ev; ev.kind = EventKind::Text; ev.text = holdback_; events.push_back(ev);
        }
        holdback_.clear();
    }

    if (in_fenced_code_) {
        Event ev; ev.kind = EventKind::CodeBlockClose; events.push_back(ev);
        in_fenced_code_ = false;
    }

    if (in_inline_code_) {
        Event ev; ev.kind = EventKind::InlineCodeClose; events.push_back(ev);
        in_inline_code_ = false;
    }

    if (current_heading_level_ > 0) {
        Event ev; ev.kind = EventKind::HeadingClose; events.push_back(ev);
        current_heading_level_ = 0;
    }

    close_all_lists(events);

    return events;
}

void MarkdownStream::reset() {
    holdback_.clear();
    in_fenced_code_ = false;
    fence_marker_.clear();
    current_language_tag_.clear();
    collecting_language_tag_ = false;
    in_inline_code_ = false;
    inline_code_marker_.clear();
    current_heading_level_ = 0;
    open_list_depths_.clear();
    at_line_start_ = true;
}

bool MarkdownStream::pending() const noexcept {
    return !holdback_.empty();
}

} // namespace md
