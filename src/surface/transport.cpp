#include "src/surface/transport.hpp"

#include <unistd.h>

#include <cstdlib>
#include <vector>

namespace lmp::surface {
namespace {

// UTF-8 encode one code point. \uXXXX is the one JSON escape that is not a single byte,
// and dropping it would put the letter u and four hex digits into the value.
void append_utf8(std::string& out, unsigned int cp) {
    if (cp < 0x80U) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (cp >> 6)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else if (cp < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (cp >> 12)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (cp >> 18)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 12) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
}

// Four hex digits at `at`, or -1. Does not advance the caller's cursor.
[[nodiscard]] int hex4(std::string_view s, std::size_t at) {
    if (at + 4 > s.size()) {
        return -1;
    }
    int value = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const char c = s[at + i];
        int digit = 0;
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            digit = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            digit = c - 'A' + 10;
        } else {
            return -1;
        }
        value = value * 16 + digit;
    }
    return value;
}

// Finds "key" as a top-level-ish JSON string field and returns its value.
//
// THE ESCAPES ARE DECODED. `\n` IS A NEWLINE, NOT THE LETTER n.
//
// This used to skip the backslash and push the character after it verbatim, which is
// correct for exactly two escapes -- \" and \\, where the escaped byte IS the intended
// byte -- and silently wrong for every other one. \n became `n`, \t became `t`, \r became
// `r`, and \uXXXX became `uXXXX`.
//
// MEASURED, in the shipped log (events.jsonl, run_id 5): a 5,000-character plan handed
// from plan mode to agent mode arrived as ONE RUN-ON LINE, every newline replaced by a
// letter n -- `"Plan\n\n## Overview"` reaching the model as `Plannn## Overview`. Every
// heading, bullet and phase boundary in the plan was destroyed on the way in, and the
// run that had to follow it flailed and ended having written 118 bytes. This is upstream
// of the prompt, the context store and the loop: they were all faithfully carrying
// mangled text.
//
// The comment this replaces said "minimal on purpose: the reader thread needs the method
// and an id, nothing more". That stopped being true when start_mission began reading
// `mission`, `model_dir` and `workspace_root` through here, and steering messages after
// it -- an id and a method have no escapes worth decoding, which is why nothing noticed.
// The scope grew and the parser did not, which is the whole defect.
std::string extract_string(std::string_view message, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    std::size_t at = message.find(needle);
    if (at == std::string_view::npos) {
        return {};
    }
    at = message.find(':', at + needle.size());
    if (at == std::string_view::npos) {
        return {};
    }
    while (at < message.size() && (message[at] == ':' || message[at] == ' ')) {
        ++at;
    }
    if (at >= message.size() || message[at] != '"') {
        return {};
    }
    ++at;
    std::string out;
    while (at < message.size() && message[at] != '"') {
        if (message[at] != '\\') {
            out.push_back(message[at]);
            ++at;
            continue;
        }
        if (at + 1 >= message.size()) {
            break; // trailing backslash at end of input; nothing to escape
        }
        const char esc = message[at + 1];
        at += 2;
        switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                const int hi = hex4(message, at);
                if (hi < 0) {
                    // Malformed: keep the bytes rather than invent a character.
                    out.append("\\u");
                    break;
                }
                at += 4;
                unsigned int cp = static_cast<unsigned int>(hi);
                // A surrogate PAIR is one code point in two escapes; a lone surrogate is
                // not encodable, and U+FFFD says so rather than emitting invalid UTF-8.
                if (cp >= 0xD800U && cp <= 0xDBFFU) {
                    if (at + 1 < message.size() && message[at] == '\\' &&
                        message[at + 1] == 'u') {
                        const int lo = hex4(message, at + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            at += 6;
                            cp = 0x10000U + ((cp - 0xD800U) << 10) +
                                 (static_cast<unsigned int>(lo) - 0xDC00U);
                        } else {
                            cp = 0xFFFDU;
                        }
                    } else {
                        cp = 0xFFFDU;
                    }
                } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
                    cp = 0xFFFDU; // orphaned low surrogate
                }
                append_utf8(out, cp);
                break;
            }
            default:
                // Not a JSON escape at all. Keep both bytes: inventing one byte from two
                // is what the old code did, and it is how `\n` became `n`.
                out.push_back('\\');
                out.push_back(esc);
                break;
        }
    }
    return out;
}

// Finds "key" and returns the raw token that follows the colon, unquoted and
// untrimmed at the far end -- enough to recognise a JSON literal.
std::string_view extract_literal(std::string_view message, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    std::size_t at = message.find(needle);
    if (at == std::string_view::npos) {
        return {};
    }
    at = message.find(':', at + needle.size());
    if (at == std::string_view::npos) {
        return {};
    }
    ++at;
    while (at < message.size() && message[at] == ' ') {
        ++at;
    }
    return message.substr(at);
}

} // namespace

std::string method_of(std::string_view message) {
    return extract_string(message, "method");
}

std::string string_field(std::string_view message, std::string_view key) {
    return extract_string(message, key);
}

bool bool_field(std::string_view message, std::string_view key) {
    return extract_literal(message, key).substr(0, 4) == "true";
}

bool has_field(std::string_view message, std::string_view key) {
    return !extract_literal(message, key).empty();
}

double double_field(std::string_view message, std::string_view key, double fallback) {
    const std::string_view literal = extract_literal(message, key);
    if (literal.empty()) {
        return fallback;
    }
    // strtod needs a terminator, and the view is the whole message tail.
    const std::string text(literal.substr(0, 64));
    const char* begin = text.c_str();
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    return end == begin ? fallback : value;
}

StdinReader::~StdinReader() { join(); }

void StdinReader::deliver(std::string message) {
    // The control-plane check runs on a COMPLETE message, and it parses the method
    // field. It is not a substring search over transport bytes: a chat message
    // containing the text "lmp/cancel" must not cancel anything.
    if (method_of(message) == "lmp/cancel") {
        cancel_.cancel();
        cancels_.fetch_add(1, std::memory_order_release);
    }
    // Backpressure: if the consumer is behind, spin rather than drop. Dropping a
    // request here would be indistinguishable from the extension never sending it.
    while (!out_.try_push(std::move(message))) {
        std::this_thread::yield();
    }
}

void StdinReader::drain_accumulator() {
    std::size_t start = 0;
    while (true) {
        const std::size_t nl = accumulator_.find('\n', start);
        if (nl == std::string::npos) {
            break;
        }
        std::string message = accumulator_.substr(start, nl - start);
        start = nl + 1;
        if (!message.empty()) {
            deliver(std::move(message));
        }
    }
    accumulator_.erase(0, start);
}

void StdinReader::feed_for_test(std::string_view bytes) {
    accumulator_.append(bytes);
    drain_accumulator();
}

namespace {

// -1 on EOF or a real error, otherwise the byte count. EINTR retries here so the
// caller's loop stays flat.
[[nodiscard]] ssize_t read_block(int fd, char* buf, std::size_t cap) {
    while (true) {
        const ssize_t n = ::read(fd, buf, cap);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        return n <= 0 ? -1 : n;
    }
}

} // namespace

void StdinReader::start(int fd) {
    thread_ = std::thread([this, fd] {
        std::vector<char> block(kReadBlockBytes);
        while (true) {
            const ssize_t n = read_block(fd, block.data(), block.size());
            if (n < 0) {
                break; // EOF (the parent is gone) or an unrecoverable error
            }
            accumulator_.append(block.data(), static_cast<std::size_t>(n));
            drain_accumulator();
        }
        out_.close();
    });
}

void StdinReader::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

} // namespace lmp::surface
