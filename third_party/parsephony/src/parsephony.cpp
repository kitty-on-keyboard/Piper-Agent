#include "parsephony/parsephony.hpp"
#include "parsephony/swar.hpp"

#include <xlocale.h>

#include <cerrno>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace parsephony {

const char* error_message(Error e) noexcept {
    switch (e) {
        case Error::Ok:              return "ok";
        case Error::Empty:           return "empty input";
        case Error::UnexpectedChar:  return "unexpected character";
        case Error::UnexpectedEnd:   return "unexpected end of input";
        case Error::TrailingContent: return "trailing content after document";
        case Error::DepthExceeded:   return "maximum nesting depth exceeded";
        case Error::BadNumber:       return "malformed number";
        case Error::BadEscape:       return "invalid escape sequence";
        case Error::BadUnicode:      return "invalid \\u escape or unpaired surrogate";
        case Error::BadUtf8:         return "invalid UTF-8 in source";
        case Error::ControlChar:     return "unescaped control character in string";
    }
    return "unknown error";
}

namespace {

inline bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

// Length of the UTF-8 sequence starting at p, or 0 if it is not valid.
// Rejects overlong forms, surrogates, and anything past U+10FFFF, matching what
// nlohmann accepts so the differential oracle stays meaningful.
inline int utf8_sequence_length(const unsigned char* p, size_t avail) noexcept {
    unsigned char c = p[0];
    if (c < 0x80) return 1;

    auto cont = [](unsigned char b) { return (b & 0xC0) == 0x80; };

    if (c >= 0xC2 && c <= 0xDF) {
        if (avail < 2 || !cont(p[1])) return 0;
        return 2;
    }
    if (c == 0xE0) {
        if (avail < 3 || p[1] < 0xA0 || p[1] > 0xBF || !cont(p[2])) return 0;
        return 3;
    }
    if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) {
        if (avail < 3 || !cont(p[1]) || !cont(p[2])) return 0;
        return 3;
    }
    if (c == 0xED) {  // exclude U+D800..U+DFFF
        if (avail < 3 || p[1] < 0x80 || p[1] > 0x9F || !cont(p[2])) return 0;
        return 3;
    }
    if (c == 0xF0) {
        if (avail < 4 || p[1] < 0x90 || p[1] > 0xBF || !cont(p[2]) || !cont(p[3])) return 0;
        return 4;
    }
    if (c >= 0xF1 && c <= 0xF3) {
        if (avail < 4 || !cont(p[1]) || !cont(p[2]) || !cont(p[3])) return 0;
        return 4;
    }
    if (c == 0xF4) {  // U+100000..U+10FFFF
        if (avail < 4 || p[1] < 0x80 || p[1] > 0x8F || !cont(p[2]) || !cont(p[3])) return 0;
        return 4;
    }
    return 0;  // 0xC0, 0xC1, 0xF5..0xFF are never valid lead bytes
}

inline bool hex4(const char* p, uint32_t& out) noexcept {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9')      v |= uint32_t(c - '0');
        else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
        else return false;
    }
    out = v;
    return true;
}

inline void encode_utf8(uint32_t cp, std::string& out) {
    if (cp <= 0x7F) {
        out.push_back(char(cp));
    } else if (cp <= 0x7FF) {
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

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

Error Parser::parse(std::string_view json, Document& doc) {
    doc.clear();
    doc.src_ = json;
    err_off_ = 0;

    const char* const base = json.data();
    const char* p   = base;
    const char* const end = base + json.size();

    // Every node begins at a distinct source byte — a container at its opening
    // bracket, a scalar at its first character — so the node count can never
    // exceed the byte count. Sizing to that bound up front lets the whole parse
    // write through a raw cursor with no per-node capacity check.
    //
    // The tempting tighter bound of size/2 (assuming a container pays for both
    // brackets) is wrong: "[[[[[..." is one node per byte and would overrun
    // before the depth limit ever fires. The tape stays allocated between
    // parses, so this is a one-time cost per payload size.
    const size_t max_nodes = json.size() + 4;
    if (doc.tape_.size() < max_nodes) doc.tape_.resize(max_nodes);
    Node* const tape = doc.tape_.data();
    uint32_t n = 0;
    stack_.clear();

    auto fail = [&](Error e, const char* at) {
        err_off_ = size_t(at - base);
        return e;
    };

    auto skip_ws = [&]() {
        p += swar::skip_whitespace(p, size_t(end - p));
    };

    // Scans a string body. On entry p points just past the opening quote; on
    // return it points just past the closing quote. Zero-copy unless the string
    // actually contains a backslash escape, in which case it is unescaped once
    // into doc.decoded_ and the node points there instead.
    auto scan_string = [&](uint32_t& out_off, uint32_t& out_len,
                           uint8_t& out_flags) -> Error {
        const char* body = p;
        bool escaped = false;
        size_t dec_start = 0;

        for (;;) {
            size_t adv = swar::scan_string_body(p, size_t(end - p));
            if (escaped && adv) doc.decoded_.append(p, adv);
            p += adv;

            if (p == end) return fail(Error::UnexpectedEnd, p);
            unsigned char c = static_cast<unsigned char>(*p);

            if (c == '"') {
                if (escaped) {
                    out_off = uint32_t(dec_start);
                    out_len = uint32_t(doc.decoded_.size() - dec_start);
                    out_flags |= Node::kEscaped;
                } else {
                    out_off = uint32_t(body - base);
                    out_len = uint32_t(p - body);
                }
                ++p;
                return Error::Ok;
            }

            if (c < 0x20) return fail(Error::ControlChar, p);

            if (c == '\\') {
                if (!escaped) {
                    escaped = true;
                    dec_start = doc.decoded_.size();
                    doc.decoded_.append(body, size_t(p - body));
                }
                ++p;
                if (p == end) return fail(Error::UnexpectedEnd, p);
                char e = *p;
                switch (e) {
                    case '"':  doc.decoded_.push_back('"');  ++p; break;
                    case '\\': doc.decoded_.push_back('\\'); ++p; break;
                    case '/':  doc.decoded_.push_back('/');  ++p; break;
                    case 'b':  doc.decoded_.push_back('\b'); ++p; break;
                    case 'f':  doc.decoded_.push_back('\f'); ++p; break;
                    case 'n':  doc.decoded_.push_back('\n'); ++p; break;
                    case 'r':  doc.decoded_.push_back('\r'); ++p; break;
                    case 't':  doc.decoded_.push_back('\t'); ++p; break;
                    case 'u': {
                        ++p;
                        if (size_t(end - p) < 4) return fail(Error::UnexpectedEnd, p);
                        uint32_t cp;
                        if (!hex4(p, cp)) return fail(Error::BadUnicode, p);
                        p += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            // High surrogate — a low surrogate must follow.
                            if (size_t(end - p) < 6 || p[0] != '\\' || p[1] != 'u')
                                return fail(Error::BadUnicode, p);
                            uint32_t lo;
                            if (!hex4(p + 2, lo)) return fail(Error::BadUnicode, p);
                            if (lo < 0xDC00 || lo > 0xDFFF) return fail(Error::BadUnicode, p);
                            p += 6;
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            return fail(Error::BadUnicode, p);  // lone low surrogate
                        }
                        encode_utf8(cp, doc.decoded_);
                        break;
                    }
                    default: return fail(Error::BadEscape, p);
                }
                continue;
            }

            // c >= 0x80 — multi-byte UTF-8.
            int n = opts_.validate_utf8
                  ? utf8_sequence_length(reinterpret_cast<const unsigned char*>(p),
                                         size_t(end - p))
                  : 1;
            if (n == 0) return fail(Error::BadUtf8, p);
            if (escaped) doc.decoded_.append(p, size_t(n));
            p += n;
        }
    };

    skip_ws();
    if (p == end) return fail(Error::Empty, p);

    Error err = Error::Ok;

parse_value:
    if (p == end) return fail(Error::UnexpectedEnd, p);
    switch (*p) {
        case '{': {
            if (stack_.size() >= opts_.max_depth) return fail(Error::DepthExceeded, p);
            uint32_t idx = n;
            tape[n++] = Node{0, 0, Type::Object, 0, 0};
            stack_.push_back(Frame{idx, 0, true});
            ++p;
            skip_ws();
            if (p == end) return fail(Error::UnexpectedEnd, p);
            if (*p == '}') { ++p; goto close_container; }
            goto parse_key;
        }
        case '[': {
            if (stack_.size() >= opts_.max_depth) return fail(Error::DepthExceeded, p);
            uint32_t idx = n;
            tape[n++] = Node{0, 0, Type::Array, 0, 0};
            stack_.push_back(Frame{idx, 0, false});
            ++p;
            skip_ws();
            if (p == end) return fail(Error::UnexpectedEnd, p);
            if (*p == ']') { ++p; goto close_container; }
            goto parse_value;
        }
        case '"': {
            ++p;
            uint32_t off = 0, len = 0; uint8_t flags = 0;
            if ((err = scan_string(off, len, flags)) != Error::Ok) return err;
            tape[n++] = Node{off, len, Type::String, flags, 0};
            goto after_value;
        }
        case 't': {
            if (size_t(end - p) < 4 || std::memcmp(p, "true", 4) != 0)
                return fail(Error::UnexpectedChar, p);
            tape[n++] = Node{uint32_t(p - base), 4, Type::True, 0, 0};
            p += 4;
            goto after_value;
        }
        case 'f': {
            if (size_t(end - p) < 5 || std::memcmp(p, "false", 5) != 0)
                return fail(Error::UnexpectedChar, p);
            tape[n++] = Node{uint32_t(p - base), 5, Type::False, 0, 0};
            p += 5;
            goto after_value;
        }
        case 'n': {
            if (size_t(end - p) < 4 || std::memcmp(p, "null", 4) != 0)
                return fail(Error::UnexpectedChar, p);
            tape[n++] = Node{uint32_t(p - base), 4, Type::Null, 0, 0};
            p += 4;
            goto after_value;
        }
        default: {
            // Strict JSON number grammar: no leading '+', no leading zeros, no
            // bare '.5' or '5.', exponent must carry at least one digit.
            const char* start = p;
            uint8_t flags = 0;
            if (*p == '-') ++p;
            if (p == end) return fail(Error::BadNumber, p);
            if (*p == '0') {
                ++p;
            } else if (is_digit(*p)) {
                while (p < end && is_digit(*p)) ++p;
            } else {
                return fail(Error::UnexpectedChar, p);
            }
            if (p < end && *p == '.') {
                flags |= Node::kFloat;
                ++p;
                if (p == end || !is_digit(*p)) return fail(Error::BadNumber, p);
                while (p < end && is_digit(*p)) ++p;
            }
            if (p < end && (*p == 'e' || *p == 'E')) {
                flags |= Node::kFloat;
                ++p;
                if (p < end && (*p == '+' || *p == '-')) ++p;
                if (p == end || !is_digit(*p)) return fail(Error::BadNumber, p);
                while (p < end && is_digit(*p)) ++p;
            }
            tape[n++] = Node{uint32_t(start - base), uint32_t(p - start), Type::Number, flags, 0};
            goto after_value;
        }
    }

parse_key: {
        skip_ws();
        if (p == end) return fail(Error::UnexpectedEnd, p);
        if (*p != '"') return fail(Error::UnexpectedChar, p);
        ++p;
        uint32_t off = 0, len = 0; uint8_t flags = 0;
        if ((err = scan_string(off, len, flags)) != Error::Ok) return err;
        tape[n++] = Node{off, len, Type::Key, flags, 0};
        skip_ws();
        if (p == end) return fail(Error::UnexpectedEnd, p);
        if (*p != ':') return fail(Error::UnexpectedChar, p);
        ++p;
        skip_ws();
        goto parse_value;
    }

close_container: {
        Frame f = stack_.back();
        stack_.pop_back();
        Node& cn = tape[f.node];
        cn.off = n;                 // skip pointer: one past this subtree
        cn.len = f.count;
        goto after_value;
    }

after_value:
    if (stack_.empty()) {
        skip_ws();
        if (p != end) return fail(Error::TrailingContent, p);
        doc.tape_count_ = n;
        return Error::Ok;
    }
    {
        Frame& f = stack_.back();
        ++f.count;
        skip_ws();
        if (p == end) return fail(Error::UnexpectedEnd, p);
        if (f.is_object) {
            if (*p == ',') { ++p; goto parse_key; }
            if (*p == '}') { ++p; goto close_container; }
            return fail(Error::UnexpectedChar, p);
        } else {
            if (*p == ',') { ++p; skip_ws(); goto parse_value; }
            if (*p == ']') { ++p; goto close_container; }
            return fail(Error::UnexpectedChar, p);
        }
    }
}

// ---------------------------------------------------------------------------
// Value accessors
// ---------------------------------------------------------------------------

namespace {

inline uint32_t next_sibling(const Document* doc, uint32_t idx) noexcept {
    const Node& n = doc->node(idx);
    if (n.type == Type::Array || n.type == Type::Object) return n.off;  // skip pointer
    return idx + 1;
}

} // namespace

Type Value::type() const noexcept { return doc_->node(idx_).type; }

std::string_view Value::get_string() const noexcept {
    if (!doc_) return {};
    const Node& n = doc_->node(idx_);
    if (n.type != Type::String && n.type != Type::Key) return {};
    // Escaped strings were unescaped once at parse time into the decode buffer;
    // everything else is a view straight into the source.
    const char* base = n.escaped() ? doc_->decoded_.data() : doc_->src_.data();
    return std::string_view(base + n.off, n.len);
}

std::string_view Value::raw() const noexcept {
    if (!doc_) return {};
    const Node& n = doc_->node(idx_);
    if (n.type == Type::Array || n.type == Type::Object) return {};
    if (n.escaped()) return get_string();
    return std::string_view(doc_->source().data() + n.off, n.len);
}

size_t Value::size() const noexcept {
    if (!doc_) return 0;
    const Node& n = doc_->node(idx_);
    if (n.type != Type::Array && n.type != Type::Object) return 0;
    return n.len;
}

// PORTABILITY, patched in LM_Pipe_2 (2026-08-02). Vendored otherwise unmodified.
//
// This was std::from_chars(const char*, const char*, double&). The INTEGER overloads are
// everywhere; the floating-point ones are not -- libc++ only implemented them in LLVM 20,
// and the libc++ shipped with Xcode 16.4 `= delete`s the overload that a double argument
// resolves to. So this built on a machine with a newer Xcode and failed on the macos-15
// CI runner, which is exactly the class of break that hides until somebody else builds it.
//
// strtod_l with an explicit C locale rather than plain strtod: strtod's decimal separator
// follows LC_NUMERIC, so under a comma locale "1.5" would parse as 1. A JSON number is
// defined with a '.' regardless of who is running the process, and a parser that disagrees
// depending on the host's locale is a worse bug than the one being fixed.
//
// The saturate-on-overflow behaviour is preserved exactly: strtod sets ERANGE and returns
// +/-HUGE_VAL, which is what the from_chars branch below returned by hand. The bytes are a
// number token the PDA already validated, so the wider syntax strtod accepts (leading '+',
// "inf", hex) is unreachable here, and the NUL-terminated copy bounds it to n.len anyway.
double Value::get_double() const noexcept {
    if (!doc_) return 0.0;
    const Node& n = doc_->node(idx_);
    if (n.type != Type::Number) return 0.0;
    const char* s = doc_->source().data() + n.off;

    // Long enough for any double's shortest round-trip form many times over; a token
    // longer than this cannot carry more precision, and truncating its tail changes only
    // digits the format cannot represent.
    char buf[512];
    const size_t len = n.len < sizeof(buf) - 1 ? n.len : sizeof(buf) - 1;
    std::memcpy(buf, s, len);
    buf[len] = '\0';

    static locale_t c_locale = ::newlocale(LC_NUMERIC_MASK, "C", nullptr);
    const int saved = errno;
    errno = 0;
    const double v = c_locale != nullptr ? ::strtod_l(buf, nullptr, c_locale)
                                         : std::strtod(buf, nullptr);
    const bool overflowed = errno == ERANGE;
    errno = saved;
    if (overflowed) {
        // Matches nlohmann: overflow saturates to infinity rather than failing.
        return (*s == '-') ? -HUGE_VAL : HUGE_VAL;
    }
    return v;
}

std::optional<int64_t> Value::get_int() const noexcept {
    if (!doc_) return std::nullopt;
    const Node& n = doc_->node(idx_);
    if (n.type != Type::Number || n.is_float()) return std::nullopt;
    const char* s = doc_->source().data() + n.off;
    int64_t v = 0;
    auto r = std::from_chars(s, s + n.len, v);
    if (r.ec != std::errc() || r.ptr != s + n.len) return std::nullopt;
    return v;
}

Value::iterator& Value::iterator::operator++() noexcept {
    idx_ = next_sibling(doc_, idx_);
    return *this;
}

Value::iterator Value::begin() const noexcept {
    if (!doc_) return iterator(doc_, 0);
    const Node& n = doc_->node(idx_);
    if (n.type != Type::Array) return iterator(doc_, idx_);
    return iterator(doc_, idx_ + 1);
}

Value::iterator Value::end() const noexcept {
    if (!doc_) return iterator(doc_, 0);
    const Node& n = doc_->node(idx_);
    if (n.type != Type::Array) return iterator(doc_, idx_);
    return iterator(doc_, n.off);
}

std::pair<std::string_view, Value> Value::member_iterator::operator*() const noexcept {
    return { Value(doc_, idx_).get_string(), Value(doc_, idx_ + 1) };
}

Value::member_iterator& Value::member_iterator::operator++() noexcept {
    idx_ = next_sibling(doc_, idx_ + 1);  // step past key, then past its value
    return *this;
}

Value::members_range Value::members() const noexcept {
    if (!doc_) return {nullptr, 0, 0};
    const Node& n = doc_->node(idx_);
    if (n.type != Type::Object) return {doc_, idx_, idx_};
    return {doc_, idx_ + 1, n.off};
}

Value Value::operator[](size_t i) const noexcept {
    if (!doc_) return {};
    const Node& n = doc_->node(idx_);
    if (n.type != Type::Array || i >= n.len) return {};
    uint32_t cur = idx_ + 1;
    for (size_t k = 0; k < i; ++k) cur = next_sibling(doc_, cur);
    return Value(doc_, cur);
}

Value Value::operator[](std::string_view key) const noexcept {
    if (!doc_) return {};
    const Node& n = doc_->node(idx_);
    if (n.type != Type::Object) return {};
    // Duplicate keys resolve to the last occurrence, matching nlohmann,
    // JavaScript and Python. That rules out an early exit on first match, but
    // tool-call objects are a handful of members, so the scan is short.
    uint32_t cur = idx_ + 1;
    uint32_t found = 0;
    bool have = false;
    for (uint32_t k = 0; k < n.len; ++k) {
        Value kv(doc_, cur);
        if (kv.get_string() == key) { found = cur + 1; have = true; }
        cur = next_sibling(doc_, cur + 1);
    }
    return have ? Value(doc_, found) : Value();
}

Value Value::at(std::string_view path) const noexcept {
    Value cur = *this;
    size_t pos = 0;
    while (cur.valid() && pos <= path.size()) {
        size_t dot = path.find('.', pos);
        std::string_view seg = path.substr(pos, dot == std::string_view::npos
                                                ? std::string_view::npos : dot - pos);
        if (seg.empty()) break;

        if (cur.is_array()) {
            size_t i = 0;
            bool numeric = !seg.empty();
            for (char c : seg) {
                if (c < '0' || c > '9') { numeric = false; break; }
                i = i * 10 + size_t(c - '0');
            }
            cur = numeric ? cur[i] : Value();
        } else {
            cur = cur[seg];
        }

        if (dot == std::string_view::npos) break;
        pos = dot + 1;
    }
    return cur;
}

} // namespace parsephony
