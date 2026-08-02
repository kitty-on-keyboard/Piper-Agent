#include "markdown_stream.hpp"

namespace md {

std::vector<Event> MarkdownStream::feed(std::string_view chunk) {
    buffer_.append(chunk);
    return process_buffer(false);
}

std::vector<Event> MarkdownStream::finish() {
    std::vector<Event> events = process_buffer(true);

    if (in_fenced_code_) {
        events.push_back({EventKind::CodeBlockClose});
        in_fenced_code_ = false;
    }

    close_inline_and_heading(events);
    close_all_lists(events);

    if (!buffer_.empty()) {
        events.push_back({EventKind::Text, buffer_});
        buffer_.clear();
    }

    reset();
    return events;
}

void MarkdownStream::reset() {
    buffer_.clear();
    in_fenced_code_ = false;
    fence_marker_.clear();
    in_inline_code_ = false;
    inline_fence_length_ = 0;
    in_heading_ = false;
    list_stack_.clear();
    at_line_start_ = true;
}

bool MarkdownStream::pending() const noexcept {
    return !buffer_.empty();
}

void MarkdownStream::close_inline_and_heading(std::vector<Event>& events) {
    if (in_inline_code_) {
        events.push_back({EventKind::InlineCodeClose});
        in_inline_code_ = false;
    }
    if (in_heading_) {
        events.push_back({EventKind::HeadingClose});
        in_heading_ = false;
    }
}

void MarkdownStream::close_all_lists(std::vector<Event>& events) {
    while (!list_stack_.empty()) {
        events.push_back({EventKind::ListItemClose, "", "", list_stack_.back()});
        list_stack_.pop_back();
    }
}

MarkdownStream::ParseResult MarkdownStream::try_parse_fenced_code(std::vector<Event>& events, bool is_finish) {
    size_t processed = 0;

    while (processed < buffer_.size()) {
        size_t newline = buffer_.find('\n', processed);
        if (newline == std::string::npos) {
            // No newline yet
            if (!is_finish) {
                // If there's a potential fence character, hold back
                size_t potential_fence = buffer_.find(fence_marker_[0], processed);
                if (potential_fence != std::string::npos) {
                    if (potential_fence > processed) {
                        events.push_back({EventKind::CodeBlockText, buffer_.substr(processed, potential_fence - processed)});
                        processed = potential_fence;
                    }

                    return {ParseRes::NeedMore, processed};
                } else {
                    events.push_back({EventKind::CodeBlockText, buffer_.substr(processed)});
                    processed = buffer_.size();
                    return {ParseRes::NeedMore, processed};
                }
            } else {
                events.push_back({EventKind::CodeBlockText, buffer_.substr(processed)});
                processed = buffer_.size();
                return {ParseRes::NeedMore, processed}; // End of buffer
            }
        } else {
            // Newline found
            std::string line = buffer_.substr(processed, newline - processed);

            // Allow up to 3 spaces indentation for closing fence
            size_t i = 0;
            while (i < line.size() && line[i] == ' ' && i < 3) i++;

            if (i < line.size() && line.substr(i, fence_marker_.size()) == fence_marker_) {
                // Check if remaining characters are only spaces
                size_t j = i + fence_marker_.size();
                while (j < line.size() && (line[j] == ' ' || line[j] == '`' || line[j] == '~')) {
                     // actually, closing fence can have more characters of same type
                    if (line[j] != ' ' && line[j] != fence_marker_[0]) break;
                    j++;
                }

                size_t first_non_space = j;
                while (first_non_space < line.size() && line[first_non_space] == ' ') first_non_space++;

                if (first_non_space == line.size()) {
                    events.push_back({EventKind::CodeBlockClose});
                    in_fenced_code_ = false;
                    processed = newline + 1;
                    at_line_start_ = true;
                    return {ParseRes::Matched, processed};
                }
            }

            // Not a closing fence
            events.push_back({EventKind::CodeBlockText, buffer_.substr(processed, newline - processed + 1)});
            processed = newline + 1;
        }
    }

    return {ParseRes::NeedMore, processed};
}

std::vector<Event> MarkdownStream::process_buffer(bool is_finish) {
    std::vector<Event> events;
    size_t processed = 0;

    while (processed < buffer_.size()) {
        if (in_fenced_code_) {
            std::string temp_buffer = buffer_.substr(processed);
            buffer_ = temp_buffer;
            processed = 0;

            ParseResult pr = try_parse_fenced_code(events, is_finish);
            processed = pr.consumed;

            if (in_fenced_code_) {
                // Still in fenced code, wait for more
                break;
            } else {
                continue;
            }
        }

        // Not in fenced code
        if (at_line_start_) {
            // Skip up to 3 spaces
            size_t spaces = 0;
            while (processed + spaces < buffer_.size() && buffer_[processed + spaces] == ' ' && spaces < 3) {
                spaces++;
            }

            // Paragraph break (blank line)
            // A blank line has only spaces and a newline
            size_t pcount = spaces;
            if (processed + pcount < buffer_.size() && buffer_[processed + pcount] == '\n') {
                close_inline_and_heading(events);
                close_all_lists(events); // End lists on paragraph break
                events.push_back({EventKind::ParagraphBreak});
                processed += pcount + 1;
                at_line_start_ = true;
                continue;
            }
            if (processed + pcount == buffer_.size() && is_finish) {
                close_inline_and_heading(events);
                close_all_lists(events);
                events.push_back({EventKind::ParagraphBreak});
                processed += pcount;
                at_line_start_ = true;
                continue;
            }
            if (processed + pcount == buffer_.size() && !is_finish) {
                // need to wait to see if it's a newline
                break;
            }

            if (processed + spaces + 2 < buffer_.size()) { // Needs at least ``` or ~~~
                char first_char = buffer_[processed + spaces];
                if (first_char == '`' || first_char == '~') {
                    size_t count = 1;
                    while (processed + spaces + count < buffer_.size() && buffer_[processed + spaces + count] == first_char) {
                        count++;
                    }
                    if (count >= 3) {
                        // Check for newline
                        size_t newline = buffer_.find('\n', processed + spaces + count);
                        if (newline == std::string::npos) {
                            if (!is_finish) {
                                // Hold back until we get newline for the info string
                                break;
                            } else {
                                newline = buffer_.size();
                            }
                        }

                        fence_marker_ = std::string(count, first_char);

                        std::string info = buffer_.substr(processed + spaces + count, newline - (processed + spaces + count));

                        // Extract info string (trimming)
                        size_t start = info.find_first_not_of(" \t");
                        if (start != std::string::npos) {
                            info = info.substr(start);
                            size_t end = info.find_last_not_of(" \t\r");
                            if (end != std::string::npos) {
                                info = info.substr(0, end + 1);
                            }
                        } else {
                            info = "";
                        }

                        // Fences can't have backticks in info string
                        if (first_char == '`' && info.find('`') != std::string::npos) {
                            // Not a fence
                        } else {
                            close_inline_and_heading(events);
                            close_all_lists(events);
                            events.push_back({EventKind::CodeBlockOpen, "", info});
                            in_fenced_code_ = true;
                            processed = newline < buffer_.size() ? newline + 1 : newline;
                            at_line_start_ = true;
                            continue;
                        }
                    }
                }
            } else if (!is_finish && buffer_.size() - processed <= 5) {
                // Hold back potentially starting markers
                bool possible_fence = false;
                for (size_t i = processed; i < buffer_.size(); ++i) {
                    if (buffer_[i] == '`' || buffer_[i] == '~' || buffer_[i] == '#' || buffer_[i] == '-' || buffer_[i] == '*' || buffer_[i] == '+') possible_fence = true;
                }
                if (possible_fence) break;
            }

            // Check for ATX headings
            if (processed + spaces < buffer_.size() && buffer_[processed + spaces] == '#') {
                size_t hcount = 0;
                while (processed + spaces + hcount < buffer_.size() && buffer_[processed + spaces + hcount] == '#') {
                    hcount++;
                }
                if (hcount >= 1 && hcount <= 6) {
                    if (processed + spaces + hcount == buffer_.size() && !is_finish) {
                        break;
                    }
                    if (processed + spaces + hcount == buffer_.size() || buffer_[processed + spaces + hcount] == ' ' || buffer_[processed + spaces + hcount] == '\n') {
                        close_inline_and_heading(events);
                        close_all_lists(events);
                        events.push_back({EventKind::HeadingOpen, "", "", static_cast<int>(hcount)});
                        in_heading_ = true;
                        processed += spaces + hcount;
                        if (processed < buffer_.size() && buffer_[processed] == ' ') processed++;
                        at_line_start_ = false;
                        continue;
                    }
                }
            }

            // Unordered Lists
            if (processed + spaces < buffer_.size()) {
                char c = buffer_[processed + spaces];
                if (c == '-' || c == '*' || c == '+') {
                    if (processed + spaces + 1 == buffer_.size() && !is_finish) {
                        break; // Wait for space
                    }
                    if (processed + spaces + 1 < buffer_.size() && (buffer_[processed + spaces + 1] == ' ' || buffer_[processed + spaces + 1] == '\n')) {
                        int depth = list_stack_.size();
                        if (list_stack_.empty() || spaces > static_cast<size_t>(list_stack_.back())) {
                            events.push_back({EventKind::ListItemOpen, "", "", depth});
                            list_stack_.push_back(spaces);
                        } else if (!list_stack_.empty() && spaces == static_cast<size_t>(list_stack_.back())) {
                            // Close the previous item at this depth and open a new one
                            events.push_back({EventKind::ListItemClose, "", "", list_stack_.back()});
                            events.push_back({EventKind::ListItemOpen, "", "", depth - 1});
                        }
                        processed += spaces + 1;
                        if (processed < buffer_.size() && buffer_[processed] == ' ') processed++;
                        at_line_start_ = false;
                        continue;
                    }
                }
            }

            // Ordered Lists (very simplified, digits then . or ))
            if (processed + spaces < buffer_.size() && isdigit(buffer_[processed + spaces])) {
                size_t dcount = 0;
                while (processed + spaces + dcount < buffer_.size() && isdigit(buffer_[processed + spaces + dcount])) {
                    dcount++;
                }
                if (dcount <= 9) {
                    if (processed + spaces + dcount == buffer_.size() && !is_finish) {
                        break;
                    }
                    if (processed + spaces + dcount < buffer_.size()) {
                        char m = buffer_[processed + spaces + dcount];
                        if (m == '.' || m == ')') {
                            if (processed + spaces + dcount + 1 == buffer_.size() && !is_finish) {
                                break;
                            }
                            if (processed + spaces + dcount + 1 < buffer_.size() && (buffer_[processed + spaces + dcount + 1] == ' ' || buffer_[processed + spaces + dcount + 1] == '\n')) {
                                int depth = list_stack_.size();
                                if (list_stack_.empty() || spaces > static_cast<size_t>(list_stack_.back())) {
                                    events.push_back({EventKind::ListItemOpen, "", "", depth});
                                    list_stack_.push_back(spaces);
                                } else if (!list_stack_.empty() && spaces == static_cast<size_t>(list_stack_.back())) {
                                    events.push_back({EventKind::ListItemClose, "", "", list_stack_.back()});
                                    events.push_back({EventKind::ListItemOpen, "", "", depth - 1});
                                }
                                processed += spaces + dcount + 1;
                                if (processed < buffer_.size() && buffer_[processed] == ' ') processed++;
                                at_line_start_ = false;
                                continue;
                            }
                        }
                    }
                }
            }

        }

        // Inline parsing
        // Find next interesting character or newline
        size_t next_char = buffer_.find_first_of("\n`", processed);

        if (next_char == std::string::npos) {
            if (!is_finish) {
                // Maybe the buffer ends with a backtick
                size_t backtick = buffer_.find('`', processed);
                if (backtick != std::string::npos) {
                    if (backtick > processed) {
                        events.push_back({EventKind::Text, buffer_.substr(processed, backtick - processed)});
                        processed = backtick;
                    }
                    break; // Wait for more
                }
            }

            if (processed < buffer_.size()) {
                events.push_back({EventKind::Text, buffer_.substr(processed)});
                processed = buffer_.size();
            }
            break;
        } else {
            if (next_char > processed) {
                events.push_back({EventKind::Text, buffer_.substr(processed, next_char - processed)});
                processed = next_char;
            }

            if (buffer_[processed] == '\n') {
                if (in_heading_) {
                    events.push_back({EventKind::HeadingClose});
                    in_heading_ = false;
                }
                events.push_back({EventKind::Text, "\n"});
                processed++;
                at_line_start_ = true;
            } else if (buffer_[processed] == '`') {
                size_t count = 0;
                while (processed + count < buffer_.size() && buffer_[processed + count] == '`') {
                    count++;
                }

                if (processed + count == buffer_.size() && !is_finish) {
                    break; // Wait for more backticks
                }

                if (in_inline_code_) {
                    if (count == inline_fence_length_) {
                        events.push_back({EventKind::InlineCodeClose});
                        in_inline_code_ = false;
                        inline_fence_length_ = 0;
                        processed += count;
                    } else {
                        events.push_back({EventKind::Text, std::string(count, '`')});
                        processed += count;
                    }
                } else {
                    events.push_back({EventKind::InlineCodeOpen});
                    in_inline_code_ = true;
                    inline_fence_length_ = count;
                    processed += count;
                }
                at_line_start_ = false;
            }
        }
    }

    buffer_ = buffer_.substr(processed);

    return events;
}

} // namespace md
