#include "markdown_stream.hpp"

namespace md {

void MarkdownStream::emit(Event e) {
    if (!events_.empty()) {
        auto& last = events_.back();
        if (last.kind == e.kind && (e.kind == EventKind::Text || e.kind == EventKind::CodeBlockText)) {
            last.text += e.text;
            return;
        }
    }
    events_.push_back(std::move(e));
}

void MarkdownStream::emitText(const std::string& s) {
    if (s.empty()) return;
    emit({EventKind::Text, s, "", 0});
}

void MarkdownStream::emitCodeBlockText(const std::string& s) {
    if (s.empty()) return;
    emit({EventKind::CodeBlockText, s, "", 0});
}

void MarkdownStream::openListItems(int target_level) {
    while (current_list_depth_ < target_level) {
        current_list_depth_++;
        emit({EventKind::ListItemOpen, "", "", current_list_depth_});
    }
    while (current_list_depth_ > target_level) {
        emit({EventKind::ListItemClose, "", "", current_list_depth_});
        current_list_depth_--;
    }
}

void MarkdownStream::closeListItems(int target) {
    while (current_list_depth_ > target) {
        emit({EventKind::ListItemClose, "", "", current_list_depth_});
        current_list_depth_--;
    }
}

void MarkdownStream::process_pending(bool flush) {
    while (!pending_.empty()) {
        bool consumed_something = false;

        if (in_code_block_) {
            if (in_code_block_info_) {
                size_t nl = pending_.find('\n');
                if (nl != std::string::npos) {
                    std::string info = code_block_info_ + pending_.substr(0, nl);
                    if (!info.empty() && info.back() == '\r') info.pop_back();
                    emit({EventKind::CodeBlockOpen, "", info, 0});
                    in_code_block_info_ = false;
                    pending_.erase(0, nl + 1);
                    at_line_start_ = true;
                    consumed_something = true;
                } else if (flush || code_block_info_.size() + pending_.size() > 128) {
                    std::string info = code_block_info_ + pending_;
                    emit({EventKind::CodeBlockOpen, "", info, 0});
                    in_code_block_info_ = false;
                    pending_.clear();
                    at_line_start_ = false;
                    consumed_something = true;
                }
            } else {
                if (at_line_start_) {
                    size_t ticks = pending_.find_first_not_of('`');
                    if (ticks == std::string::npos) {
                        if (flush) {
                            emitCodeBlockText(pending_);
                            pending_.clear();
                            at_line_start_ = false;
                        } else {
                            if (pending_.size() > 64) {
                                emitCodeBlockText(std::string(1, pending_[0]));
                                pending_.erase(0, 1);
                                consumed_something = true;
                                continue;
                            } else {
                                break;
                            }
                        }
                    } else if (static_cast<int>(ticks) >= opening_fence_length_) {
                        emit({EventKind::CodeBlockClose, "", "", 0});
                        in_code_block_ = false;
                        pending_.erase(0, ticks);
                        at_line_start_ = false;
                        consumed_something = true;
                    } else {
                        emitCodeBlockText(std::string(1, pending_[0]));
                        if (pending_[0] == '\n') at_line_start_ = true;
                        else at_line_start_ = false;
                        pending_.erase(0, 1);
                        consumed_something = true;
                    }
                } else {
                    size_t nl = pending_.find('\n');
                    if (nl != std::string::npos) {
                        emitCodeBlockText(pending_.substr(0, nl + 1));
                        pending_.erase(0, nl + 1);
                        at_line_start_ = true;
                        consumed_something = true;
                    } else {
                        emitCodeBlockText(pending_);
                        pending_.clear();
                        consumed_something = true;
                    }
                }
            }
        } else {
            if (at_line_start_) {
                size_t first_non_space = pending_.find_first_not_of(' ');
                if (first_non_space == std::string::npos) {
                    if (flush) {
                        emitText(pending_);
                        pending_.clear();
                    } else if (pending_.size() > 64) {
                        emitText(std::string(1, pending_[0]));
                        pending_.erase(0, 1);
                        consumed_something = true;
                        continue;
                    } else {
                        break;
                    }
                } else {
                    int spaces = first_non_space;
                    std::string after_spaces = pending_.substr(spaces);

                    if (after_spaces[0] == '#') {
                        size_t hashes = after_spaces.find_first_not_of('#');
                        if (hashes == std::string::npos) {
                            if (flush) {
                                emitText(pending_);
                                pending_.clear();
                            } else if (after_spaces.size() > 6) {
                                emitText(pending_.substr(0, spaces + 1));
                                pending_.erase(0, spaces + 1);
                                at_line_start_ = false;
                                consumed_something = true;
                                continue;
                            } else {
                                if (pending_.size() > 64) {
                                    emitText(std::string(1, pending_[0]));
                                    pending_.erase(0, 1);
                                    consumed_something = true;
                                    continue;
                                } else {
                                    break;
                                }
                            }
                        } else {
                            if (after_spaces[hashes] == ' ' && hashes <= 6) {
                                closeListItems(-1);
                                emit({EventKind::HeadingOpen, "", "", static_cast<int>(hashes)});
                                in_heading_ = true;
                                pending_.erase(0, spaces + hashes + 1);
                                at_line_start_ = false;
                                consumed_something = true;
                                continue;
                            } else {
                                emitText(pending_.substr(0, spaces + 1));
                                pending_.erase(0, spaces + 1);
                                at_line_start_ = false;
                                consumed_something = true;
                                continue;
                            }
                        }
                    }

                    if (after_spaces[0] == '`') {
                        size_t ticks = after_spaces.find_first_not_of('`');
                        if (ticks == std::string::npos) {
                            if (flush) {
                                emitText(pending_);
                                pending_.clear();
                                at_line_start_ = false;
                            } else {
                                if (pending_.size() > 64) {
                                    emitText(std::string(1, pending_[0]));
                                    pending_.erase(0, 1);
                                    consumed_something = true;
                                    continue;
                                } else {
                                    break;
                                }
                            }
                        } else if (ticks >= 3) {
                            closeListItems(-1);
                            if (in_heading_) {
                                emit({EventKind::HeadingClose, "", "", 0});
                                in_heading_ = false;
                            }
                            in_code_block_ = true;
                            opening_fence_length_ = ticks;
                            in_code_block_info_ = true;
                            code_block_info_.clear();
                            pending_.erase(0, spaces + ticks);
                            at_line_start_ = false;
                            consumed_something = true;
                            continue;
                        } else {
                            emitText(pending_.substr(0, spaces + 1));
                            pending_.erase(0, spaces + 1);
                            at_line_start_ = false;
                            consumed_something = true;
                            continue;
                        }
                    }

                    if (after_spaces[0] == '*' || after_spaces[0] == '-' || after_spaces[0] == '+') {
                        if (after_spaces.size() == 1) {
                            if (flush) {
                                emitText(pending_);
                                pending_.clear();
                                at_line_start_ = false;
                            } else {
                                if (pending_.size() > 64) {
                                    emitText(std::string(1, pending_[0]));
                                    pending_.erase(0, 1);
                                    consumed_something = true;
                                    continue;
                                } else {
                                    break;
                                }
                            }
                        } else if (after_spaces[1] == ' ') {
                            int level = spaces / 2;
                            openListItems(level);
                            pending_.erase(0, spaces + 2);
                            at_line_start_ = false;
                            consumed_something = true;
                            continue;
                        }
                    }

                    if (isdigit(after_spaces[0])) {
                        size_t dot_pos = after_spaces.find('.');
                        if (dot_pos != std::string::npos) {
                            bool all_digits = true;
                            for (size_t i = 0; i < dot_pos; ++i) {
                                if (!isdigit(after_spaces[i])) {
                                    all_digits = false; break;
                                }
                            }
                            if (all_digits && dot_pos > 0) {
                                if (after_spaces.size() == dot_pos + 1) {
                                    if (flush) {
                                        emitText(pending_);
                                        pending_.clear();
                                        at_line_start_ = false;
                                    } else {
                                        if (pending_.size() > 64) {
                                            emitText(std::string(1, pending_[0]));
                                            pending_.erase(0, 1);
                                            consumed_something = true;
                                            continue;
                                        } else {
                                            break;
                                        }
                                    }
                                } else if (after_spaces[dot_pos + 1] == ' ') {
                                    int level = spaces / 2;
                                    openListItems(level);
                                    pending_.erase(0, spaces + dot_pos + 2);
                                    at_line_start_ = false;
                                    consumed_something = true;
                                    continue;
                                }
                            }
                        } else {
                            bool all_digits = true;
                            for (char c : after_spaces) {
                                if (!isdigit(c)) { all_digits = false; break; }
                            }
                            if (all_digits) {
                                if (flush) {
                                    emitText(pending_);
                                    pending_.clear();
                                    at_line_start_ = false;
                                } else {
                                    if (pending_.size() > 64) {
                                        emitText(std::string(1, pending_[0]));
                                        pending_.erase(0, 1);
                                        consumed_something = true;
                                        continue;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if (after_spaces[0] == '\n') {
                        closeListItems(-1);
                        if (in_heading_) {
                            emit({EventKind::HeadingClose, "", "", 0});
                            in_heading_ = false;
                        }
                        emit({EventKind::ParagraphBreak, "", "", 0});
                        pending_.erase(0, spaces + 1);
                        at_line_start_ = true;
                        consumed_something = true;
                        continue;
                    }

                    at_line_start_ = false;
                }
            }

            if (!at_line_start_ && !pending_.empty()) {
                char c = pending_[0];
                if (c == '`') {
                    if (in_inline_code_) {
                        emit({EventKind::InlineCodeClose, "", "", 0});
                        in_inline_code_ = false;
                    } else {
                        emit({EventKind::InlineCodeOpen, "", "", 0});
                        in_inline_code_ = true;
                    }
                    pending_.erase(0, 1);
                    consumed_something = true;
                } else if (c == '\n') {
                    if (in_heading_) {
                        emit({EventKind::HeadingClose, "", "", 0});
                        in_heading_ = false;
                    }
                    emitText("\n");
                    pending_.erase(0, 1);
                    at_line_start_ = true;
                    consumed_something = true;
                } else {
                    size_t next_special = pending_.find_first_of("`\n");
                    if (next_special == std::string::npos) {
                        emitText(pending_);
                        pending_.clear();
                        consumed_something = true;
                    } else {
                        emitText(pending_.substr(0, next_special));
                        pending_.erase(0, next_special);
                        consumed_something = true;
                    }
                }
            }
        }

        if (!consumed_something) {
            if (pending_.size() > 64) {
                emitText(std::string(1, pending_[0]));
                if (pending_[0] == '\n') at_line_start_ = true;
                else if (pending_[0] != ' ') at_line_start_ = false;
                pending_.erase(0, 1);
            } else {
                break;
            }
        }
    }
}

std::vector<Event> MarkdownStream::feed(std::string_view chunk) {
    for (char c : chunk) {
        pending_ += c;
        process_pending(false);
    }
    std::vector<Event> result = std::move(events_);
    events_.clear();
    return result;
}

std::vector<Event> MarkdownStream::finish() {
    process_pending(true);

    if (in_code_block_) {
        emit({EventKind::CodeBlockClose, "", "", 0});
        in_code_block_ = false;
    }
    if (in_inline_code_) {
        emit({EventKind::InlineCodeClose, "", "", 0});
        in_inline_code_ = false;
    }
    if (in_heading_) {
        emit({EventKind::HeadingClose, "", "", 0});
        in_heading_ = false;
    }
    closeListItems(-1);

    std::vector<Event> result = std::move(events_);
    events_.clear();
    return result;
}

void MarkdownStream::reset() {
    pending_.clear();
    events_.clear();
    in_code_block_ = false;
    opening_fence_length_ = 0;
    in_code_block_info_ = false;
    code_block_info_.clear();
    in_inline_code_ = false;
    in_heading_ = false;
    current_list_depth_ = -1;
    at_line_start_ = true;
}

bool MarkdownStream::pending() const noexcept {
    return !pending_.empty();
}

} // namespace md
