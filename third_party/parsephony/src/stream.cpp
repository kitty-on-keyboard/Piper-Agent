#include "parsephony/stream.hpp"

#include <cstring>

namespace parsephony {

namespace {
inline bool is_ws(unsigned char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
inline bool is_digit(unsigned char c) noexcept { return c >= '0' && c <= '9'; }
inline bool is_hex(unsigned char c) noexcept {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
inline int hex_val(unsigned char c) noexcept {
    if (c <= '9') return c - '0';
    if (c <= 'F') return c - 'A' + 10;
    return c - 'a' + 10;
}
inline void encode_utf8(uint32_t cp, std::string& out) {
    if (cp <= 0x7F) out.push_back(char(cp));
    else if (cp <= 0x7FF) {
        out.push_back(char(0xC0 | (cp >> 6)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(char(0xE0 | (cp >> 12)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(char(0xF0 | (cp >> 18)));
        out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    }
}
} // namespace

StreamParser::StreamParser(Handler* h, Options o) : h_(h), opts_(o) {}

void StreamParser::reset() {
    st_ = St::Value;
    sub_ = Sub::None;
    stack_.clear();
    done_ = false;
    key_pending_ = false;
    buf_.clear();
    literal_ = {};
    literal_idx_ = 0;
    uni_ = 0;
    uni_digits_ = 0;
    high_surrogate_ = 0;
    expect_low_escape_ = false;
}

Error StreamParser::feed(std::string_view bytes) {
    for (unsigned char c : bytes) {
        Error e = push_byte(c);
        if (e != Error::Ok) return e;
    }
    return Error::Ok;
}

void StreamParser::emit_string_or_key() {
    if (!h_) return;
    if (key_pending_) h_->on_key(buf_);
    else h_->on_string(buf_);
}

Error StreamParser::flush_number() {
    // The number sub-machine has already validated the grammar byte by byte;
    // all that is left is to confirm it did not stop mid-token (e.g. "1e").
    if (buf_.empty()) return Error::BadNumber;
    unsigned char last = static_cast<unsigned char>(buf_.back());
    if (!is_digit(last)) return Error::BadNumber;
    if (h_) h_->on_number(buf_);
    sub_ = Sub::None;
    buf_.clear();
    return end_value();
}

Error StreamParser::close_container(bool is_object) {
    if (stack_.empty()) return Error::UnexpectedChar;
    if (static_cast<bool>(stack_.back()) != is_object) return Error::UnexpectedChar;
    stack_.pop_back();
    if (h_) { if (is_object) h_->on_object_end(); else h_->on_array_end(); }
    return end_value();
}

// A complete value has just been produced; decide what may follow it.
Error StreamParser::end_value() {
    if (stack_.empty()) {
        st_ = St::Done;
        done_ = true;
        return Error::Ok;
    }
    st_ = stack_.back() ? St::ObjCommaOrEnd : St::ArrCommaOrEnd;
    return Error::Ok;
}

Error StreamParser::begin_value(unsigned char c) {
    switch (c) {
        case '{':
            if (stack_.size() >= opts_.max_depth) return Error::DepthExceeded;
            stack_.push_back(1);
            if (h_) h_->on_object_begin();
            st_ = St::ObjKeyOrEnd;
            return Error::Ok;
        case '[':
            if (stack_.size() >= opts_.max_depth) return Error::DepthExceeded;
            stack_.push_back(0);
            if (h_) h_->on_array_begin();
            st_ = St::ArrValueOrEnd;
            return Error::Ok;
        case '"':
            sub_ = Sub::String;
            key_pending_ = false;
            buf_.clear();
            return Error::Ok;
        case 't': literal_ = "true";  literal_idx_ = 1; sub_ = Sub::Literal; return Error::Ok;
        case 'f': literal_ = "false"; literal_idx_ = 1; sub_ = Sub::Literal; return Error::Ok;
        case 'n': literal_ = "null";  literal_idx_ = 1; sub_ = Sub::Literal; return Error::Ok;
        default:
            if (c == '-' || is_digit(c)) {
                sub_ = Sub::Number;
                buf_.clear();
                buf_.push_back(char(c));
                return Error::Ok;
            }
            return Error::UnexpectedChar;
    }
}

Error StreamParser::push_byte(unsigned char c) {
    // ---- sub-machines that can straddle chunk boundaries -------------------
    switch (sub_) {
        case Sub::String:
            // Mid surrogate pair: the only legal continuation is the backslash
            // that opens the low surrogate's \u escape.
            if (expect_low_escape_ && c != '\\') return Error::BadUnicode;
            if (c == '"') {
                sub_ = Sub::None;
                emit_string_or_key();
                if (key_pending_) { st_ = St::ObjColon; key_pending_ = false; return Error::Ok; }
                return end_value();
            }
            if (c == '\\') { sub_ = Sub::StringEscape; return Error::Ok; }
            if (c < 0x20) return Error::ControlChar;
            buf_.push_back(char(c));
            return Error::Ok;

        case Sub::StringEscape:
            if (expect_low_escape_) {
                // Mid surrogate pair: only "\u" is legal here.
                if (c != 'u') return Error::BadUnicode;
                sub_ = Sub::StringUnicode;
                uni_ = 0; uni_digits_ = 0;
                return Error::Ok;
            }
            switch (c) {
                case '"':  buf_.push_back('"');  break;
                case '\\': buf_.push_back('\\'); break;
                case '/':  buf_.push_back('/');  break;
                case 'b':  buf_.push_back('\b'); break;
                case 'f':  buf_.push_back('\f'); break;
                case 'n':  buf_.push_back('\n'); break;
                case 'r':  buf_.push_back('\r'); break;
                case 't':  buf_.push_back('\t'); break;
                case 'u':  sub_ = Sub::StringUnicode; uni_ = 0; uni_digits_ = 0; return Error::Ok;
                default:   return Error::BadEscape;
            }
            sub_ = Sub::String;
            return Error::Ok;

        case Sub::StringUnicode: {
            if (!is_hex(c)) return Error::BadUnicode;
            uni_ = (uni_ << 4) | uint32_t(hex_val(c));
            if (++uni_digits_ < 4) return Error::Ok;

            if (high_surrogate_) {
                if (uni_ < 0xDC00 || uni_ > 0xDFFF) return Error::BadUnicode;
                uint32_t cp = 0x10000 + ((high_surrogate_ - 0xD800) << 10) + (uni_ - 0xDC00);
                encode_utf8(cp, buf_);
                high_surrogate_ = 0;
                expect_low_escape_ = false;
            } else if (uni_ >= 0xD800 && uni_ <= 0xDBFF) {
                // High surrogate. Return to the string body, but latched so that
                // only "\uDC00..\uDFFF" can follow — the backslash arrives as an
                // ordinary byte, possibly in the next chunk.
                high_surrogate_ = uni_;
                expect_low_escape_ = true;
                sub_ = Sub::String;
                return Error::Ok;
            } else if (uni_ >= 0xDC00 && uni_ <= 0xDFFF) {
                return Error::BadUnicode;    // lone low surrogate
            } else {
                encode_utf8(uni_, buf_);
            }
            sub_ = Sub::String;
            return Error::Ok;
        }

        case Sub::Number: {
            // Accept anything that can continue a number; validate the shape as
            // we go, and let any other byte terminate it.
            char last = buf_.back();
            bool ok = false;
            if (is_digit(c)) {
                // No leading zeros: "0" may not be followed by a digit.
                if (buf_ == "0" || buf_ == "-0") return Error::BadNumber;
                ok = true;
            } else if (c == '.') {
                ok = is_digit(static_cast<unsigned char>(last)) &&
                     buf_.find('.') == std::string::npos &&
                     buf_.find_first_of("eE") == std::string::npos;
            } else if (c == 'e' || c == 'E') {
                ok = is_digit(static_cast<unsigned char>(last)) &&
                     buf_.find_first_of("eE") == std::string::npos;
            } else if (c == '+' || c == '-') {
                ok = (last == 'e' || last == 'E');
            }
            if (ok) { buf_.push_back(char(c)); return Error::Ok; }

            // Terminator: finish the number, then reprocess this byte.
            Error e = flush_number();
            if (e != Error::Ok) return e;
            break;  // fall through to the structural machine below
        }

        case Sub::Literal:
            if (literal_idx_ >= literal_.size() || c != static_cast<unsigned char>(literal_[literal_idx_]))
                return Error::UnexpectedChar;
            if (++literal_idx_ == literal_.size()) {
                if (h_) {
                    if (literal_ == "true") h_->on_bool(true);
                    else if (literal_ == "false") h_->on_bool(false);
                    else h_->on_null();
                }
                sub_ = Sub::None;
                return end_value();
            }
            return Error::Ok;

        case Sub::None:
            break;
    }

    // ---- structural machine -------------------------------------------------
    if (is_ws(c)) return Error::Ok;

    switch (st_) {
        case St::Value:
            return begin_value(c);

        case St::ObjKeyOrEnd:
            if (c == '}') return close_container(true);
            if (c == '"') { sub_ = Sub::String; key_pending_ = true; buf_.clear(); return Error::Ok; }
            return Error::UnexpectedChar;

        case St::ObjKey:
            if (c == '"') { sub_ = Sub::String; key_pending_ = true; buf_.clear(); return Error::Ok; }
            return Error::UnexpectedChar;

        case St::ObjColon:
            if (c == ':') { st_ = St::Value; return Error::Ok; }
            return Error::UnexpectedChar;

        case St::ObjCommaOrEnd:
            if (c == ',') { st_ = St::ObjKey; return Error::Ok; }
            if (c == '}') return close_container(true);
            return Error::UnexpectedChar;

        case St::ArrValueOrEnd:
            if (c == ']') return close_container(false);
            return begin_value(c);

        case St::ArrCommaOrEnd:
            if (c == ',') { st_ = St::Value; return Error::Ok; }
            if (c == ']') return close_container(false);
            return Error::UnexpectedChar;

        case St::Done:
            return Error::TrailingContent;
    }
    return Error::UnexpectedChar;
}

Error StreamParser::finish() {
    if (sub_ == Sub::Number) {
        Error e = flush_number();
        if (e != Error::Ok) return e;
    }
    if (sub_ != Sub::None) return Error::UnexpectedEnd;
    if (!stack_.empty()) return Error::UnexpectedEnd;
    if (!done_) return Error::Empty;
    return Error::Ok;
}

// ---------------------------------------------------------------------------
// Constraint side
// ---------------------------------------------------------------------------

ByteSet StreamParser::allowed_bytes() const {
    ByteSet s;

    auto add_value_starts = [&] {
        s.add('{'); s.add('['); s.add('"');
        s.add('-'); s.add_range('0', '9');
        s.add('t'); s.add('f'); s.add('n');
    };
    auto add_ws = [&] { s.add(' '); s.add('\t'); s.add('\n'); s.add('\r'); };

    switch (sub_) {
        case Sub::String:
            if (expect_low_escape_) { s.add('\\'); return s; }  // must continue the pair
            s.add_range(0x20, 0x7F);
            s.non_ascii = true;
            return s;

        case Sub::StringEscape:
            if (expect_low_escape_) { s.add('u'); return s; }
            for (char e : {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'})
                s.add(static_cast<unsigned char>(e));
            return s;

        case Sub::StringUnicode:
            s.add_range('0', '9'); s.add_range('a', 'f'); s.add_range('A', 'F');
            return s;

        case Sub::Number: {
            char last = buf_.empty() ? '\0' : buf_.back();
            bool has_dot = buf_.find('.') != std::string::npos;
            bool has_exp = buf_.find_first_of("eE") != std::string::npos;
            if (!(buf_ == "0" || buf_ == "-0")) s.add_range('0', '9');
            if (is_digit(static_cast<unsigned char>(last))) {
                if (!has_dot && !has_exp) s.add('.');
                if (!has_exp) { s.add('e'); s.add('E'); }
            }
            if (last == 'e' || last == 'E') { s.add('+'); s.add('-'); }
            // A number may also simply end here, if it is well-formed so far.
            if (is_digit(static_cast<unsigned char>(last))) {
                add_ws();
                if (!stack_.empty()) {
                    s.add(',');
                    s.add(stack_.back() ? '}' : ']');
                }
            }
            return s;
        }

        case Sub::Literal:
            if (literal_idx_ < literal_.size())
                s.add(static_cast<unsigned char>(literal_[literal_idx_]));
            return s;

        case Sub::None:
            break;
    }

    switch (st_) {
        case St::Value:          add_ws(); add_value_starts(); break;
        case St::ObjKeyOrEnd:    add_ws(); s.add('"'); s.add('}'); break;
        case St::ObjKey:         add_ws(); s.add('"'); break;
        case St::ObjColon:       add_ws(); s.add(':'); break;
        case St::ObjCommaOrEnd:  add_ws(); s.add(','); s.add('}'); break;
        case St::ArrValueOrEnd:  add_ws(); add_value_starts(); s.add(']'); break;
        case St::ArrCommaOrEnd:  add_ws(); s.add(','); s.add(']'); break;
        case St::Done:           add_ws(); break;
    }
    return s;
}

uint64_t StreamParser::state_signature() const noexcept {
    // Everything that can affect which continuations are legal.
    uint64_t sig = uint64_t(st_);
    sig = sig * 31 + uint64_t(sub_);
    sig = sig * 31 + (stack_.empty() ? 2u : uint64_t(stack_.back()));
    sig = sig * 31 + (stack_.empty() ? 0u : 1u);
    sig = sig * 31 + uint64_t(expect_low_escape_);
    sig = sig * 31 + uint64_t(key_pending_);
    sig = sig * 31 + uint64_t(uni_digits_);
    sig = sig * 31 + uint64_t(literal_idx_);
    // Depth matters only in that a token may close more containers than are
    // open; fold in the exact depth up to a small cap.
    sig = sig * 31 + uint64_t(stack_.size() > 8 ? 8 : stack_.size());
    // The number sub-machine's legal set depends on the token so far.
    if (sub_ == Sub::Number) {
        bool has_dot = buf_.find('.') != std::string::npos;
        bool has_exp = buf_.find_first_of("eE") != std::string::npos;
        char last = buf_.empty() ? '\0' : buf_.back();
        sig = sig * 31 + uint64_t(has_dot) * 2 + uint64_t(has_exp);
        sig = sig * 31 + uint64_t(static_cast<unsigned char>(last));
        sig = sig * 31 + uint64_t(buf_ == "0" || buf_ == "-0");
    }
    if (sub_ == Sub::Literal) {
        sig = sig * 31 + uint64_t(literal_.empty() ? 0 : literal_[0]);
    }
    return sig;
}

} // namespace parsephony
