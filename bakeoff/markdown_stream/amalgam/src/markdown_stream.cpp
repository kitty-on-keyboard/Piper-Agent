#include "markdown_stream.hpp"

namespace md {

void MarkdownStream::emit_text(std::string_view text) {
    if (text.empty()) return;
    if (!m_events.empty() && m_events.back().kind == EventKind::Text) {
        m_events.back().text.append(text);
    } else {
        m_events.push_back({EventKind::Text, std::string(text), "", 0});
    }
}

void MarkdownStream::close_lists(int target_indent) {
    while (!m_list_indents.empty() && m_list_indents.back() >= target_indent) {
        m_events.push_back({EventKind::ListItemClose, "", "", (int)m_list_indents.size() - 1});
        m_list_indents.pop_back();
    }
}

void MarkdownStream::close_all_lists() {
    close_lists(0);
}

void MarkdownStream::reset() {
    m_holdback.clear();
    m_state = State::Normal;
    m_at_line_start = true;
    m_list_indents.clear();
    m_fenced_ticks = 0;
    m_fenced_info.clear();
    m_inline_ticks = 0;
    m_heading_level = 0;
    m_events.clear();
}

bool MarkdownStream::pending() const noexcept {
    return !m_holdback.empty();
}

void MarkdownStream::process(bool is_finish) {
    while (!m_holdback.empty()) {
        if (m_state == State::Normal) {
            if (m_at_line_start) {
                size_t spaces = 0;
                while (spaces < m_holdback.size() && m_holdback[spaces] == ' ' && spaces < 32) spaces++;

                if (spaces == m_holdback.size()) {
                    if (is_finish) {
                        emit_text(m_holdback);
                        m_holdback.clear();
                    } else if (m_holdback.size() >= MAX_HOLDBACK) {
                        emit_text(m_holdback.substr(0, 1));
                        m_holdback.erase(0, 1);
                    }
                    return;
                }

                char c = m_holdback[spaces];

                if (c == '\n') {
                    close_all_lists();
                    m_events.push_back({EventKind::ParagraphBreak, "", "", 0});
                    m_holdback.erase(0, spaces + 1);
                    m_at_line_start = true;
                    continue;
                }

                if (c == '#') {
                    int hashes = 0;
                    size_t i = spaces;
                    while (i < m_holdback.size() && m_holdback[i] == '#' && hashes < 6) { hashes++; i++; }

                    if (i == m_holdback.size()) {
                        if (is_finish) {
                            emit_text(m_holdback);
                            m_holdback.clear();
                        } else if (m_holdback.size() >= MAX_HOLDBACK) {
                            emit_text(m_holdback.substr(0, 1));
                            m_holdback.erase(0, 1);
                            m_at_line_start = false;
                        }
                        return;
                    }

                    if (m_holdback[i] == ' ') {
                        close_all_lists();
                        m_heading_level = hashes;
                        m_events.push_back({EventKind::HeadingOpen, "", "", m_heading_level});
                        m_state = State::Heading;
                        m_holdback.erase(0, i + 1);
                        m_at_line_start = false;
                        continue;
                    } else if (m_holdback[i] == '\n') {
                        close_all_lists();
                        m_events.push_back({EventKind::HeadingOpen, "", "", hashes});
                        m_events.push_back({EventKind::HeadingClose, "", "", hashes});
                        m_holdback.erase(0, i + 1);
                        m_at_line_start = true;
                        continue;
                    }
                }

                if (c == '`') {
                    int ticks = 0;
                    size_t i = spaces;
                    while (i < m_holdback.size() && m_holdback[i] == '`') { ticks++; i++; }
                    if (i == m_holdback.size()) {
                        if (is_finish) {
                            emit_text(m_holdback);
                            m_holdback.clear();
                        } else if (m_holdback.size() >= MAX_HOLDBACK) {
                            emit_text(m_holdback.substr(0, 1));
                            m_holdback.erase(0, 1);
                            m_at_line_start = false;
                        }
                        return;
                    }
                    if (ticks >= 3) {
                        close_all_lists();
                        m_fenced_ticks = ticks;
                        m_fenced_info.clear();
                        // wait, do not push CodeBlockOpen yet, wait for \n to get the info!
                        // This fixes the split-invariance problem with CodeBlockOpen.
                        m_state = State::FencedCodeInfo;
                        m_holdback.erase(0, i);
                        m_at_line_start = false;
                        continue;
                    }
                }

                if (c == '-' || c == '*') {
                    if (spaces + 1 == m_holdback.size()) {
                        if (is_finish) {
                            emit_text(m_holdback);
                            m_holdback.clear();
                        }
                        return;
                    }
                    char next_c = m_holdback[spaces + 1];
                    if (next_c == ' ' || next_c == '\n') {
                        int indent = spaces + 2;
                        close_lists(indent);
                        int level = m_list_indents.size();
                        m_events.push_back({EventKind::ListItemOpen, "", "", level});
                        m_list_indents.push_back(indent);
                        m_holdback.erase(0, spaces + (next_c == ' ' ? 2 : 1));
                        m_at_line_start = false;
                        if (next_c == '\n') m_at_line_start = true;
                        continue;
                    }
                }

                if (c >= '0' && c <= '9') {
                    size_t i = spaces;
                    while (i < m_holdback.size() && m_holdback[i] >= '0' && m_holdback[i] <= '9') i++;
                    if (i == m_holdback.size()) {
                        if (i - spaces < 9 && !is_finish) {
                            if (m_holdback.size() >= MAX_HOLDBACK) {
                                emit_text(m_holdback.substr(0, 1));
                                m_holdback.erase(0, 1);
                                m_at_line_start = false;
                            } else {
                                return;
                            }
                        } else {
                            emit_text(m_holdback);
                            m_holdback.clear();
                        }
                        return;
                    } else if (m_holdback[i] == '.') {
                        if (i + 1 == m_holdback.size()) {
                            if (is_finish) {
                                emit_text(m_holdback);
                                m_holdback.clear();
                            }
                            return;
                        }
                        char next_c = m_holdback[i + 1];
                        if (next_c == ' ' || next_c == '\n') {
                            int indent = i + 2;
                            close_lists(indent);
                            int level = m_list_indents.size();
                            m_events.push_back({EventKind::ListItemOpen, "", "", level});
                            m_list_indents.push_back(indent);
                            m_holdback.erase(0, i + (next_c == ' ' ? 2 : 1));
                            m_at_line_start = false;
                            if (next_c == '\n') m_at_line_start = true;
                            continue;
                        }
                    }
                }
            } // at_line_start

            size_t i = 0;
            while (i < m_holdback.size() && m_holdback[i] != '`' && m_holdback[i] != '\n') {
                i++;
            }

            if (i > 0) {
                emit_text(m_holdback.substr(0, i));
                m_holdback.erase(0, i);
                m_at_line_start = false;
                continue;
            }

            if (m_holdback.empty()) return;

            if (m_holdback[0] == '\n') {
                emit_text("\n");
                m_holdback.erase(0, 1);
                m_at_line_start = true;
                continue;
            }

            if (m_holdback[0] == '`') {
                int ticks = 0;
                size_t j = 0;
                while (j < m_holdback.size() && m_holdback[j] == '`') { ticks++; j++; }

                if (j == m_holdback.size()) {
                    if (is_finish) {
                        emit_text(m_holdback);
                        m_holdback.clear();
                    } else if (m_holdback.size() >= MAX_HOLDBACK) {
                        emit_text("`");
                        m_holdback.erase(0, 1);
                    }
                    return;
                }

                m_inline_ticks = ticks;
                m_events.push_back({EventKind::InlineCodeOpen, "", "", 0});
                m_state = State::InlineCode;
                m_holdback.erase(0, j);
                m_at_line_start = false;
                continue;
            }
        } else if (m_state == State::Heading) {
            size_t nl = m_holdback.find('\n');
            if (nl == std::string::npos) {
                if (is_finish) {
                    emit_text(m_holdback);
                    m_holdback.clear();
                } else if (m_holdback.size() >= MAX_HOLDBACK) {
                    emit_text(m_holdback.substr(0, 1));
                    m_holdback.erase(0, 1);
                }
                return;
            } else {
                emit_text(m_holdback.substr(0, nl));
                m_events.push_back({EventKind::HeadingClose, "", "", m_heading_level});
                m_holdback.erase(0, nl + 1);
                m_state = State::Normal;
                m_at_line_start = true;
                continue;
            }
        } else if (m_state == State::FencedCodeInfo) {
            size_t nl = m_holdback.find('\n');
            if (nl == std::string::npos) {
                if (is_finish) {
                    m_fenced_info += m_holdback;
                    m_events.push_back({EventKind::CodeBlockOpen, "", m_fenced_info, 0});
                    m_holdback.clear();
                    m_state = State::FencedCodeContent;
                    m_at_line_start = true;
                } else if (m_fenced_info.size() + m_holdback.size() >= MAX_HOLDBACK) {
                    m_fenced_info += m_holdback.substr(0, 1);
                    m_holdback.erase(0, 1);
                }
                return;
            } else {
                m_fenced_info += m_holdback.substr(0, nl);
                m_events.push_back({EventKind::CodeBlockOpen, "", m_fenced_info, 0});
                m_holdback.erase(0, nl + 1);
                m_state = State::FencedCodeContent;
                m_at_line_start = true;
                continue;
            }
        } else if (m_state == State::FencedCodeContent) {
            if (m_at_line_start) {
                size_t spaces = 0;
                while (spaces < m_holdback.size() && m_holdback[spaces] == ' ' && spaces < 32) spaces++;

                if (spaces == m_holdback.size()) {
                    if (is_finish) {
                        if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                            m_events.back().text += m_holdback;
                        } else {
                            m_events.push_back({EventKind::CodeBlockText, m_holdback, "", 0});
                        }
                        m_holdback.clear();
                    } else if (m_holdback.size() >= MAX_HOLDBACK) {
                        if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                            m_events.back().text += m_holdback.substr(0, 1);
                        } else {
                            m_events.push_back({EventKind::CodeBlockText, m_holdback.substr(0, 1), "", 0});
                        }
                        m_holdback.erase(0, 1);
                        m_at_line_start = false;
                    }
                    return;
                }

                if (m_holdback[spaces] == '`') {
                    int ticks = 0;
                    size_t i = spaces;
                    while (i < m_holdback.size() && m_holdback[i] == '`') { ticks++; i++; }

                    if (i == m_holdback.size()) {
                        if (is_finish) {
                            if (ticks >= m_fenced_ticks) {
                                m_events.push_back({EventKind::CodeBlockClose, "", "", 0});
                                m_holdback.clear();
                                m_state = State::Normal;
                                m_at_line_start = true;
                            } else {
                                if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                                    m_events.back().text += m_holdback;
                                } else {
                                    m_events.push_back({EventKind::CodeBlockText, m_holdback, "", 0});
                                }
                                m_holdback.clear();
                            }
                        } else if (m_holdback.size() >= MAX_HOLDBACK) {
                            if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                                m_events.back().text += m_holdback.substr(0, 1);
                            } else {
                                m_events.push_back({EventKind::CodeBlockText, m_holdback.substr(0, 1), "", 0});
                            }
                            m_holdback.erase(0, 1);
                            m_at_line_start = false;
                        }
                        return;
                    }

                    if (ticks >= m_fenced_ticks) {
                        size_t j = i;
                        while (j < m_holdback.size() && (m_holdback[j] == ' ' || m_holdback[j] == '\t')) j++;
                        if (j == m_holdback.size()) {
                            if (is_finish) {
                                m_events.push_back({EventKind::CodeBlockClose, "", "", 0});
                                m_holdback.clear();
                                m_state = State::Normal;
                                m_at_line_start = true;
                            } else if (m_holdback.size() >= MAX_HOLDBACK) {
                                if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                                    m_events.back().text += m_holdback.substr(0, 1);
                                } else {
                                    m_events.push_back({EventKind::CodeBlockText, m_holdback.substr(0, 1), "", 0});
                                }
                                m_holdback.erase(0, 1);
                                m_at_line_start = false;
                            }
                            return;
                        }
                        if (m_holdback[j] == '\n') {
                            m_events.push_back({EventKind::CodeBlockClose, "", "", 0});
                            m_holdback.erase(0, j + 1);
                            m_state = State::Normal;
                            m_at_line_start = true;
                            continue;
                        }
                    }
                }

                if (m_holdback[0] == '\n') {
                    if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                        m_events.back().text += "\n";
                    } else {
                        m_events.push_back({EventKind::CodeBlockText, "\n", "", 0});
                    }
                    m_holdback.erase(0, 1);
                    m_at_line_start = true;
                } else {
                    if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                        m_events.back().text += m_holdback.substr(0, 1);
                    } else {
                        m_events.push_back({EventKind::CodeBlockText, m_holdback.substr(0, 1), "", 0});
                    }
                    m_holdback.erase(0, 1);
                    m_at_line_start = false;
                }
                continue;
            } else {
                size_t nl = m_holdback.find('\n');
                if (nl == std::string::npos) {
                    if (is_finish) {
                        if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                            m_events.back().text += m_holdback;
                        } else {
                            m_events.push_back({EventKind::CodeBlockText, m_holdback, "", 0});
                        }
                        m_holdback.clear();
                    } else if (m_holdback.size() >= MAX_HOLDBACK) {
                        if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                            m_events.back().text += m_holdback.substr(0, 1);
                        } else {
                            m_events.push_back({EventKind::CodeBlockText, m_holdback.substr(0, 1), "", 0});
                        }
                        m_holdback.erase(0, 1);
                    }
                    return;
                } else {
                    if (!m_events.empty() && m_events.back().kind == EventKind::CodeBlockText) {
                        m_events.back().text += m_holdback.substr(0, nl + 1);
                    } else {
                        m_events.push_back({EventKind::CodeBlockText, m_holdback.substr(0, nl + 1), "", 0});
                    }
                    m_holdback.erase(0, nl + 1);
                    m_at_line_start = true;
                    continue;
                }
            }
        } else if (m_state == State::InlineCode) {
            int ticks = 0;
            size_t j = 0;
            while (j < m_holdback.size() && m_holdback[j] == '`') { ticks++; j++; }

            if (j == m_holdback.size()) {
                if (is_finish) {
                    emit_text(m_holdback);
                    m_holdback.clear();
                } else if (m_holdback.size() >= MAX_HOLDBACK) {
                    emit_text("`");
                    m_holdback.erase(0, 1);
                }
                return;
            }

            if (ticks == m_inline_ticks) {
                m_events.push_back({EventKind::InlineCodeClose, "", "", 0});
                m_state = State::Normal;
                m_holdback.erase(0, ticks);
                continue;
            } else if (ticks > 0) {
                emit_text(m_holdback.substr(0, ticks));
                m_holdback.erase(0, ticks);
                continue;
            }

            size_t i = 0;
            while (i < m_holdback.size() && m_holdback[i] != '`') {
                i++;
            }

            emit_text(m_holdback.substr(0, i));
            m_holdback.erase(0, i);
            continue;
        }
    }
}

std::vector<Event> MarkdownStream::feed(std::string_view chunk) {
    m_holdback.append(chunk);
    m_events.clear();
    process(false);
    return m_events;
}

std::vector<Event> MarkdownStream::finish() {
    m_events.clear();
    process(true);
    if (m_state == State::Heading) {
        m_events.push_back({EventKind::HeadingClose, "", "", m_heading_level});
    } else if (m_state == State::FencedCodeInfo || m_state == State::FencedCodeContent) {
        m_events.push_back({EventKind::CodeBlockClose, "", "", 0});
    } else if (m_state == State::InlineCode) {
        m_events.push_back({EventKind::InlineCodeClose, "", "", 0});
    }
    close_all_lists();

    // Do not clear m_events on reset since we are returning it, wait, reset clears holdback etc.
    // Copy events before reset.
    auto ret = m_events;
    reset();
    return ret;
}

} // namespace md
