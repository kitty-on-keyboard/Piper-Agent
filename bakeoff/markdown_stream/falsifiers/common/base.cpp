// Shared body for the falsifier trees. Each falsifier defines exactly one MS_DEFECT_* macro
// before including this file; with none defined it is the clean base.
//
//   MS_DEFECT_HOLDBACK  drop the holdback cap, so an unresolved marker buffers forever
//   MS_DEFECT_SWALLOW   finish() does not close a still-open fence
//   MS_DEFECT_SPLIT     resolve '#' at a chunk boundary instead of waiting for the next byte

#include "base.hpp"

#include <algorithm>

namespace md {

namespace {

// The stated holdback bound. Nothing is withheld past this many bytes; past it, whatever is
// held is flushed as text, because a marker that has not resolved in 64 bytes is not a
// marker.
constexpr std::size_t kMaxHold = 64;

bool is_hspace(char c) { return c == ' ' || c == '\t'; }
bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

std::string trim(std::string_view s) {
    std::size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) ++b;
    while (e > b && is_space(s[e - 1])) --e;
    return std::string(s.substr(b, e - b));
}

} // namespace

void MarkdownStream::emit(EventKind k, std::string text, std::string info, int level) {
    out_.push_back(Event{k, std::move(text), std::move(info), level});
}

std::vector<Event> MarkdownStream::feed(std::string_view chunk) {
    out_.clear();
    hold_.append(chunk);
    pump(false);
    return std::move(out_);
}

std::vector<Event> MarkdownStream::finish() {
    out_.clear();
    pump(true);
    if (inline_open_) { emit(EventKind::InlineCodeClose); inline_open_ = false; }
    if (heading_open_) { emit(EventKind::HeadingClose); heading_open_ = false; }
    if (list_open_) { emit(EventKind::ListItemClose); list_open_ = false; }
#ifndef MS_DEFECT_SWALLOW
    if (fence_open_) { emit(EventKind::CodeBlockClose); fence_open_ = false; }
#endif
    return std::move(out_);
}

void MarkdownStream::reset() {
    hold_.clear();
    out_.clear();
    mode_ = Mode::LineStart;
    fence_len_ = 0;
    inline_open_ = heading_open_ = list_open_ = fence_open_ = false;
}

void MarkdownStream::pump(bool at_end) {
    for (;;) {
        if (hold_.empty()) return;

#ifdef MS_DEFECT_HOLDBACK
        const bool capped = false;
#else
        const bool capped = hold_.size() >= kMaxHold;
#endif

        switch (mode_) {
            case Mode::LineStart: {
                if (hold_[0] == '\n') {
                    emit(EventKind::ParagraphBreak);
                    hold_.erase(0, 1);
                    continue;
                }
                std::size_t i = 0;
                while (i < hold_.size() && is_hspace(hold_[i])) ++i;
                if (i == hold_.size()) {
                    if (!at_end && !capped) return;
                    emit(EventKind::Text, hold_);
                    hold_.clear();
                    mode_ = Mode::Body;
                    continue;
                }

                const char c = hold_[i];

                if (c == '#') {
                    std::size_t n = 0;
                    while (i + n < hold_.size() && hold_[i + n] == '#') ++n;
                    if (i + n == hold_.size() && n <= 6) {
#ifdef MS_DEFECT_SPLIT
                        // Planted: resolve at the chunk boundary instead of waiting to see
                        // whether the next byte is the space that makes this a heading.
                        mode_ = Mode::Body;
                        continue;
#else
                        if (!at_end && !capped) return;
#endif
                    }
                    const bool ok = n >= 1 && n <= 6 &&
                                    (i + n >= hold_.size() || hold_[i + n] == ' ' ||
                                     hold_[i + n] == '\n');
                    if (ok) {
                        emit(EventKind::HeadingOpen, {}, {}, static_cast<int>(n));
                        heading_open_ = true;
                        std::size_t adv = i + n;
                        if (adv < hold_.size() && hold_[adv] == ' ') ++adv;
                        hold_.erase(0, adv);
                    }
                    mode_ = Mode::Body;
                    continue;
                }

                if (c == '`') {
                    std::size_t n = 0;
                    while (i + n < hold_.size() && hold_[i + n] == '`') ++n;
                    if (n < 3) {
                        if (i + n == hold_.size() && !at_end && !capped) return;
                        mode_ = Mode::Body;
                        continue;
                    }
                    const std::size_t nl = hold_.find('\n', i + n);
                    if (nl == std::string::npos) {
                        if (!at_end && !capped) return;
                        if (capped && !at_end) { // info tag longer than the bound: not a fence
                            mode_ = Mode::Body;
                            continue;
                        }
                        if (at_end && i + n == hold_.size()) {
                            // No newline ever arrived, so the info tag was never known to be
                            // complete. These are held-back bytes: flush them as text rather
                            // than opening a code block nobody asked for.
                            emit(EventKind::Text, hold_);
                            hold_.clear();
                            return;
                        }
                        emit(EventKind::CodeBlockOpen, {}, trim(std::string_view(hold_).substr(i + n)));
                        fence_open_ = true;
                        fence_len_ = n;
                        hold_.clear();
                        mode_ = Mode::FenceLineStart;
                        continue;
                    }
                    emit(EventKind::CodeBlockOpen, {},
                         trim(std::string_view(hold_).substr(i + n, nl - i - n)));
                    fence_open_ = true;
                    fence_len_ = n;
                    hold_.erase(0, nl + 1);
                    mode_ = Mode::FenceLineStart;
                    continue;
                }

                if (c == '-' || c == '*' || c == '+') {
                    if (i + 1 == hold_.size() && !at_end && !capped) return;
                    if (i + 1 < hold_.size() && hold_[i + 1] == ' ') {
                        emit(EventKind::ListItemOpen, {}, {}, static_cast<int>(i / 2));
                        list_open_ = true;
                        hold_.erase(0, i + 2);
                    }
                    mode_ = Mode::Body;
                    continue;
                }

                if (c >= '0' && c <= '9') {
                    std::size_t d = 0;
                    while (i + d < hold_.size() && hold_[i + d] >= '0' && hold_[i + d] <= '9') ++d;
                    if (d <= 12 && i + d + 1 >= hold_.size() && !at_end && !capped) return;
                    if (d <= 12 && i + d + 1 < hold_.size() && hold_[i + d] == '.' &&
                        hold_[i + d + 1] == ' ') {
                        emit(EventKind::ListItemOpen, {}, {}, static_cast<int>(i / 2));
                        list_open_ = true;
                        hold_.erase(0, i + d + 2);
                    }
                    mode_ = Mode::Body;
                    continue;
                }

                mode_ = Mode::Body;
                continue;
            }

            case Mode::Body: {
                std::size_t p = 0;
                while (p < hold_.size() && hold_[p] != '\n' && hold_[p] != '`') ++p;
                if (p > 0) emit(EventKind::Text, hold_.substr(0, p));
                if (p == hold_.size()) {
                    hold_.clear();
                    return;
                }
                if (hold_[p] == '\n') {
                    if (inline_open_) { emit(EventKind::InlineCodeClose); inline_open_ = false; }
                    if (heading_open_) { emit(EventKind::HeadingClose); heading_open_ = false; }
                    if (list_open_) { emit(EventKind::ListItemClose); list_open_ = false; }
                    hold_.erase(0, p + 1);
                    mode_ = Mode::LineStart;
                    continue;
                }
                // A backtick run that reaches the end of the buffer is withheld: it may yet
                // become a longer run, and a partial marker must never be emitted as text.
                std::size_t n = 0;
                while (p + n < hold_.size() && hold_[p + n] == '`') ++n;
                if (p + n == hold_.size() && !at_end && !capped) {
                    hold_.erase(0, p);
                    return;
                }
                if (p + n == hold_.size() && at_end && !inline_open_) {
                    emit(EventKind::Text, hold_.substr(p, n));
                    hold_.clear();
                    return;
                }
                emit(inline_open_ ? EventKind::InlineCodeClose : EventKind::InlineCodeOpen);
                inline_open_ = !inline_open_;
                hold_.erase(0, p + 1);
                continue;
            }

            case Mode::FenceLineStart: {
                const std::size_t nl = hold_.find('\n');
                if (nl == std::string::npos) {
                    if (!at_end && !capped) return;
                    emit(EventKind::CodeBlockText, hold_);
                    hold_.clear();
                    mode_ = Mode::FenceBody;
                    continue;
                }
                const std::string line = trim(std::string_view(hold_).substr(0, nl));
                std::size_t m = 0;
                while (m < line.size() && line[m] == '`') ++m;
                if (m >= fence_len_ && trim(std::string_view(line).substr(m)).empty()) {
                    emit(EventKind::CodeBlockClose);
                    fence_open_ = false;
                    hold_.erase(0, nl + 1);
                    mode_ = Mode::LineStart;
                    continue;
                }
                emit(EventKind::CodeBlockText, hold_.substr(0, nl + 1));
                hold_.erase(0, nl + 1);
                continue;
            }

            case Mode::FenceBody: {
                const std::size_t nl = hold_.find('\n');
                if (nl == std::string::npos) {
                    emit(EventKind::CodeBlockText, hold_);
                    hold_.clear();
                    return;
                }
                emit(EventKind::CodeBlockText, hold_.substr(0, nl + 1));
                hold_.erase(0, nl + 1);
                mode_ = Mode::FenceLineStart;
                continue;
            }
        }
    }
}

} // namespace md
