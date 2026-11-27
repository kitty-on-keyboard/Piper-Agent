#include "src/surface/transport.hpp"

#include <nlohmann/json.hpp>
#include <unistd.h>

#include <cctype>
#include <cstdlib>
#include <vector>

namespace lmp::surface {
namespace {

constexpr std::size_t kMaxAccumulatorBytes = 64U << 20;

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

std::string decode_json_string_content(std::string_view message, std::size_t& at) {
    // Assumes message[at] is the opening '"' and advances `at` past the closing '"'.
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
            break;
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
                    out.append("\\u");
                    break;
                }
                at += 4;
                unsigned int cp = static_cast<unsigned int>(hi);
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
                    cp = 0xFFFDU;
                }
                append_utf8(out, cp);
                break;
            }
            default:
                out.push_back('\\');
                out.push_back(esc);
                break;
        }
    }
    if (at < message.size() && message[at] == '"') {
        ++at;
    }
    return out;
}

void skip_ws(std::string_view s, std::size_t& at) {
    while (at < s.size() && std::isspace(static_cast<unsigned char>(s[at])) != 0) {
        ++at;
    }
}

void skip_json_value(std::string_view s, std::size_t& at) {
    skip_ws(s, at);
    if (at >= s.size()) return;
    if (s[at] == '"') {
        ++at;
        while (at < s.size()) {
            if (s[at] == '\\' && at + 1 < s.size()) {
                at += 2;
                continue;
            }
            if (s[at] == '"') {
                ++at;
                return;
            }
            ++at;
        }
        return;
    }
    if (s[at] == '{' || s[at] == '[') {
        const char open = s[at];
        const char close = (open == '{') ? '}' : ']';
        int depth = 1;
        ++at;
        while (at < s.size() && depth > 0) {
            if (s[at] == '"') {
                skip_json_value(s, at);
                continue;
            }
            if (s[at] == open) ++depth;
            else if (s[at] == close) --depth;
            ++at;
        }
        return;
    }
    while (at < s.size() && s[at] != ',' && s[at] != '}' && s[at] != ']' &&
           std::isspace(static_cast<unsigned char>(s[at])) == 0) {
        ++at;
    }
}

// Structural JSON object field lookup. Only descends into object-typed container keys
// (`params`, `settings`, `sampling`), never matching inside string values.
const nlohmann::json* find_in_ast(const nlohmann::json& j, std::string_view key, int depth = 0) {
    if (!j.is_object() || depth > 3) {
        return nullptr;
    }
    const auto it = j.find(key);
    if (it != j.end()) {
        return &(*it);
    }
    for (const char* child : {"params", "settings", "sampling"}) {
        const auto cit = j.find(child);
        if (cit != j.end() && cit->is_object()) {
            if (const nlohmann::json* hit = find_in_ast(*cit, key, depth + 1)) {
                return hit;
            }
        }
    }
    return nullptr;
}

// Structural fallback when nlohmann::json::parse rejects legacy malformed escapes
// (e.g. \uZZZZ or lone surrogates pinned by unit tests). Still parses actual object
// structure so string values cannot spoof keys.
bool find_raw_value_span(std::string_view s, std::size_t obj_at, std::string_view key,
                         int depth, std::size_t& val_start, bool top_only) {
    skip_ws(s, obj_at);
    if (obj_at >= s.size() || s[obj_at] != '{' || depth > 3) {
        return false;
    }
    std::size_t at = obj_at + 1;
    std::vector<std::size_t> nested_containers;
    while (at < s.size()) {
        skip_ws(s, at);
        if (at >= s.size() || s[at] == '}') break;
        if (s[at] == ',') {
            ++at;
            continue;
        }
        if (s[at] != '"') break;
        const std::string field_name = decode_json_string_content(s, at);
        skip_ws(s, at);
        if (at >= s.size() || s[at] != ':') break;
        ++at;
        skip_ws(s, at);
        if (field_name == key) {
            val_start = at;
            return true;
        }
        if (!top_only && at < s.size() && s[at] == '{' &&
            (field_name == "params" || field_name == "settings" || field_name == "sampling")) {
            nested_containers.push_back(at);
        }
        skip_json_value(s, at);
    }
    for (std::size_t sub : nested_containers) {
        if (find_raw_value_span(s, sub, key, depth + 1, val_start, false)) {
            return true;
        }
    }
    return false;
}

} // namespace

std::string method_of(std::string_view message) {
    const auto j = nlohmann::json::parse(message, nullptr, false);
    if (!j.is_discarded()) {
        if (!j.is_object()) return {};
        const auto it = j.find("method");
        return (it != j.end() && it->is_string()) ? it->get<std::string>() : std::string{};
    }
    std::size_t val_at = 0;
    if (!find_raw_value_span(message, 0, "method", 0, val_at, true)) {
        return {};
    }
    return decode_json_string_content(message, val_at);
}

std::string string_field(std::string_view message, std::string_view key) {
    const auto j = nlohmann::json::parse(message, nullptr, false);
    if (!j.is_discarded()) {
        const nlohmann::json* v = find_in_ast(j, key);
        return (v != nullptr && v->is_string()) ? v->get<std::string>() : std::string{};
    }
    std::size_t val_at = 0;
    if (!find_raw_value_span(message, 0, key, 0, val_at, false)) {
        return {};
    }
    return decode_json_string_content(message, val_at);
}

bool bool_field(std::string_view message, std::string_view key) {
    const auto j = nlohmann::json::parse(message, nullptr, false);
    if (!j.is_discarded()) {
        const nlohmann::json* v = find_in_ast(j, key);
        return v != nullptr && v->is_boolean() && v->get<bool>();
    }
    std::size_t val_at = 0;
    if (!find_raw_value_span(message, 0, key, 0, val_at, false)) {
        return false;
    }
    return message.substr(val_at, 4) == "true";
}

bool has_field(std::string_view message, std::string_view key) {
    const auto j = nlohmann::json::parse(message, nullptr, false);
    if (!j.is_discarded()) {
        return find_in_ast(j, key) != nullptr;
    }
    std::size_t val_at = 0;
    return find_raw_value_span(message, 0, key, 0, val_at, false) && val_at < message.size();
}

double double_field(std::string_view message, std::string_view key, double fallback) {
    const auto j = nlohmann::json::parse(message, nullptr, false);
    if (!j.is_discarded()) {
        const nlohmann::json* v = find_in_ast(j, key);
        return (v != nullptr && v->is_number()) ? v->get<double>() : fallback;
    }
    std::size_t val_at = 0;
    if (!find_raw_value_span(message, 0, key, 0, val_at, false) || val_at >= message.size()) {
        return fallback;
    }
    const std::string text(message.substr(val_at, 64));
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
    if (accumulator_.size() + bytes.size() > kMaxAccumulatorBytes) {
        accumulator_.clear();
        return;
    }
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
        bool discarding_overlong = false;
        while (true) {
            const ssize_t n = read_block(fd, block.data(), block.size());
            if (n < 0) {
                break; // EOF (the parent is gone) or an unrecoverable error
            }
            const std::string_view chunk(block.data(), static_cast<std::size_t>(n));
            if (discarding_overlong) {
                const std::size_t nl = chunk.find('\n');
                if (nl == std::string_view::npos) continue;
                discarding_overlong = false;
                accumulator_.append(chunk.substr(nl + 1));
            } else if (accumulator_.size() + chunk.size() > kMaxAccumulatorBytes) {
                accumulator_.clear();
                const std::size_t nl = chunk.find('\n');
                if (nl == std::string_view::npos) {
                    discarding_overlong = true;
                } else {
                    accumulator_.append(chunk.substr(nl + 1));
                }
            } else {
                accumulator_.append(chunk);
            }
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
