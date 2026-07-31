#include "src/surface/transport.hpp"

#include <unistd.h>

#include <cstdlib>
#include <vector>

namespace lmp::surface {
namespace {

// Finds "key" as a top-level-ish JSON string field and returns its value. Minimal on
// purpose: the reader thread needs the method and an id, nothing more, and a full
// parser on the hot control path is a dependency the cancel guarantee does not need.
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
        if (message[at] == '\\' && at + 1 < message.size()) {
            ++at;
        }
        out.push_back(message[at]);
        ++at;
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
