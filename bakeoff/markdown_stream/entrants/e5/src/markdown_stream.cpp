#include "../include/markdown_stream.hpp"

namespace md {

std::vector<Event> MarkdownStream::feed(std::string_view chunk) {
    m_pending.append(chunk);
    std::vector<Event> out;
    consume(out, false);
    return out;
}

std::vector<Event> MarkdownStream::finish() {
    std::vector<Event> out;
    consume(out, true);

    if (m_state == State::InHeading) out.push_back({EventKind::HeadingClose});
    if (m_state == State::CodeFenceInfo) {
        out.push_back({EventKind::CodeBlockOpen, "", m_pending});
        m_pending.clear();
        out.push_back({EventKind::CodeBlockClose});
    } else if (m_state == State::InCodeBlock) {
        if (!m_pending.empty()) emit_code_text(out, m_pending);
        m_pending.clear();
        out.push_back({EventKind::CodeBlockClose});
    } else if (m_state == State::InInlineCode) {
        if (!m_pending.empty()) emit_text(out, m_pending);
        m_pending.clear();
        out.push_back({EventKind::InlineCodeClose});
    } else {
        if (!m_pending.empty()) emit_text(out, m_pending);
        m_pending.clear();
    }

    while (!m_list_indentations.empty()) {
        m_list_indentations.pop_back();
        out.push_back({EventKind::ListItemClose});
    }

    reset();
    return out;
}

void MarkdownStream::reset() {
    m_pending.clear();
    m_state = State::Normal;
    m_at_line_start = true;
    m_consecutive_newlines = 0;
    m_list_indentations.clear();
}

bool MarkdownStream::pending() const noexcept {
    return !m_pending.empty();
}

void MarkdownStream::emit_text(std::vector<Event>& out, std::string_view text) {
    if (text.empty()) return;
    if (!out.empty() && out.back().kind == EventKind::Text) {
        out.back().text += text;
    } else {
        out.push_back({EventKind::Text, std::string(text)});
    }
}

void MarkdownStream::emit_code_text(std::vector<Event>& out, std::string_view text) {
    if (text.empty()) return;
    if (!out.empty() && out.back().kind == EventKind::CodeBlockText) {
        out.back().text += text;
    } else {
        out.push_back({EventKind::CodeBlockText, std::string(text)});
    }
}

size_t MarkdownStream::consume_as_text(std::string_view pending, std::vector<Event>& out) {
    emit_text(out, pending);
    m_at_line_start = (!pending.empty() && pending.back() == '\n');
    return pending.size();
}

void MarkdownStream::handle_list_item(int spaces, std::vector<Event>& out) {
    while (!m_list_indentations.empty() && spaces <= m_list_indentations.back()) {
        m_list_indentations.pop_back();
        out.push_back({EventKind::ListItemClose});
    }
    int level = m_list_indentations.size();
    m_list_indentations.push_back(spaces);
    out.push_back({EventKind::ListItemOpen, "", "", level});
}

void MarkdownStream::consume(std::vector<Event>& out, bool eof) {
    while (!m_pending.empty()) {
        size_t consumed = try_consume(m_pending, out, eof);
        if (consumed == 0) {
            if (!eof && m_pending.size() > 1024) { // holdback bound ~1KB
                if (m_state == State::InCodeBlock) emit_code_text(out, m_pending.substr(0, 1));
                else emit_text(out, m_pending.substr(0, 1));
                m_at_line_start = (m_pending[0] == '\n');
                consumed = 1;
            } else {
                break; // wait for more bytes
            }
        }
        m_pending.erase(0, consumed);
    }
}

size_t MarkdownStream::try_consume(std::string_view pending, std::vector<Event>& out, bool eof) {
    if (m_state == State::CodeFenceInfo) {
        size_t nl = pending.find('\n');
        if (nl != std::string_view::npos) {
            std::string info(pending.substr(0, nl));
            out.push_back({EventKind::CodeBlockOpen, "", info});
            m_state = State::InCodeBlock;
            m_at_line_start = true;
            return nl + 1;
        } else {
            if (eof) {
                out.push_back({EventKind::CodeBlockOpen, "", std::string(pending)});
                m_state = State::InCodeBlock;
                m_at_line_start = false;
                return pending.size();
            }
            return 0; // wait
        }
    }

    if (m_state == State::InCodeBlock) {
        if (m_at_line_start) {
            size_t i = 0;
            int spaces = 0;
            while (i < pending.size() && pending[i] == ' ' && spaces < 3) { spaces++; i++; }
            if (i == pending.size()) return eof ? consume_as_text(pending, out) : 0; // if eof consume as text? actually in codeblock it should just be text of code block, but handled fine by consume_as_text doing emit_text, WAIT, no, need emit_code_text
            // actually if we are in code block, eof just means we append whatever spaces as code block text
            if (i == pending.size()) {
                if (eof) { emit_code_text(out, pending); return pending.size(); }
                return 0;
            }

            if (pending[i] == m_fence_char) {
                size_t fence_start = i;
                int count = 0;
                while (i < pending.size() && pending[i] == m_fence_char) { count++; i++; }
                if (i == pending.size()) {
                    if (eof) {
                        if (count >= m_fence_length) {
                            out.push_back({EventKind::CodeBlockClose});
                            m_state = State::Normal;
                            m_at_line_start = false;
                            return pending.size();
                        } else {
                            emit_code_text(out, pending);
                            return pending.size();
                        }
                    }
                    return 0;
                }

                if (count >= m_fence_length) {
                    size_t j = i;
                    while (j < pending.size() && (pending[j] == ' ' || pending[j] == '\t')) j++;
                    if (j == pending.size()) {
                        if (eof) {
                            out.push_back({EventKind::CodeBlockClose});
                            m_state = State::Normal;
                            m_at_line_start = false;
                            return pending.size();
                        }
                        return 0;
                    }
                    if (pending[j] == '\n' || pending[j] == '\r') {
                        size_t to_consume = j;
                        if (pending[j] == '\r' && j + 1 < pending.size() && pending[j+1] == '\n') to_consume++;
                        if (to_consume < pending.size() && pending[to_consume] == '\n') to_consume++;
                        out.push_back({EventKind::CodeBlockClose});
                        m_state = State::Normal;
                        m_at_line_start = true;
                        return to_consume;
                    }
                }
            }
            m_at_line_start = false;
            emit_code_text(out, pending.substr(0, 1));
            return 1;
        } else {
            size_t nl = pending.find('\n');
            if (nl != std::string_view::npos) {
                emit_code_text(out, pending.substr(0, nl + 1));
                m_at_line_start = true;
                return nl + 1;
            } else {
                emit_code_text(out, pending);
                return pending.size();
            }
        }
    }

    if (m_state == State::InInlineCode) {
        size_t bt = pending.find('`');
        if (bt == std::string_view::npos) {
            emit_text(out, pending);
            return pending.size();
        }
        if (bt > 0) {
            emit_text(out, pending.substr(0, bt));
            return bt;
        }

        size_t count = 0;
        while (count < pending.size() && pending[count] == '`') count++;
        if (count == pending.size()) {
            if (eof) {
                if (count == m_inline_fence_length) {
                    out.push_back({EventKind::InlineCodeClose});
                    m_state = State::Normal;
                } else {
                    emit_text(out, pending);
                }
                return count;
            }
            return 0; // wait
        }

        if (count == m_inline_fence_length) {
            out.push_back({EventKind::InlineCodeClose});
            m_state = State::Normal;
            return count;
        } else {
            emit_text(out, pending.substr(0, count));
            return count;
        }
    }

    if (m_state == State::Normal || m_state == State::InHeading) {
        if (m_at_line_start) {
            size_t i = 0;
            while (i < pending.size() && pending[i] == ' ' && i < 3) i++;
            if (i == pending.size()) return eof ? (i > 0 ? consume_as_text(pending, out) : 0) : 0;

            char c = pending[i];

            if (c == '#') {
                size_t hc = 0;
                while (i + hc < pending.size() && pending[i + hc] == '#') hc++;
                if (i + hc == pending.size()) return eof ? consume_as_text(pending, out) : 0;
                if (hc > 0 && hc <= 6 && pending[i + hc] == ' ') {
                    if (m_state == State::InHeading) out.push_back({EventKind::HeadingClose});
                    m_state = State::InHeading;
                    m_at_line_start = false;
                    out.push_back({EventKind::HeadingOpen, "", "", (int)hc});
                    return i + hc + 1;
                }
            }

            if (c == '`' || c == '~') {
                size_t count = 0;
                while (i + count < pending.size() && pending[i + count] == c) count++;
                if (i + count == pending.size()) return eof ? consume_as_text(pending, out) : 0;
                if (count >= 3) {
                    m_fence_char = c;
                    m_fence_length = count;
                    m_state = State::CodeFenceInfo;
                    m_at_line_start = false;
                    if (m_state == State::InHeading) out.push_back({EventKind::HeadingClose});
                    return i + count;
                }
            }

            if (c == '*' || c == '-' || c == '+') {
                if (i + 1 == pending.size()) return eof ? consume_as_text(pending, out) : 0;
                if (pending[i+1] == ' ') {
                    handle_list_item(i, out);
                    m_at_line_start = false;
                    return i + 2;
                }
            }

            if (c >= '0' && c <= '9') {
                size_t d = i;
                while (d < pending.size() && pending[d] >= '0' && pending[d] <= '9') d++;
                if (d == pending.size()) return eof ? consume_as_text(pending, out) : 0;
                if (pending[d] == '.') {
                    if (d + 1 == pending.size()) return eof ? consume_as_text(pending, out) : 0;
                    if (pending[d+1] == ' ') {
                        handle_list_item(i, out);
                        m_at_line_start = false;
                        return d + 2;
                    }
                }
            }

            if (c == '\n') {
                if (m_state == State::InHeading) {
                    out.push_back({EventKind::HeadingClose});
                    m_state = State::Normal;
                } else if (m_consecutive_newlines == 1) {
                    out.push_back({EventKind::ParagraphBreak});
                    while (!m_list_indentations.empty()) {
                        m_list_indentations.pop_back();
                        out.push_back({EventKind::ListItemClose});
                    }
                }
                m_consecutive_newlines++;
                m_at_line_start = true;
                return i + 1;
            }

            m_at_line_start = false;
            m_consecutive_newlines = 0;
        }

        size_t next_special = pending.find_first_of("\n`");
        if (next_special == std::string_view::npos) {
            emit_text(out, pending);
            return pending.size();
        }

        if (next_special > 0) {
            emit_text(out, pending.substr(0, next_special));
            return next_special;
        }

        if (pending[0] == '\n') {
            if (m_state == State::InHeading) {
                out.push_back({EventKind::HeadingClose});
                m_state = State::Normal;
            } else {
                emit_text(out, "\n");
            }
            m_at_line_start = true;
            m_consecutive_newlines = 1;
            return 1;
        }

        if (pending[0] == '`') {
            size_t count = 0;
            while (count < pending.size() && pending[count] == '`') count++;
            if (count == pending.size()) return eof ? consume_as_text(pending, out) : 0;

            m_state = State::InInlineCode;
            m_inline_fence_length = count;
            out.push_back({EventKind::InlineCodeOpen});
            return count;
        }
    }

    return 0;
}

} // namespace md
