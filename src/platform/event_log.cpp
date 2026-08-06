#include "src/platform/event_log.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace lmp::platform {
namespace {

// --- UTF-8 -----------------------------------------------------------------

// Length of the sequence a lead byte introduces, or 0 if it cannot lead one.
[[nodiscard]] int lead_len(unsigned char b) noexcept {
    if (b < 0x80U) {
        return 1;
    }
    if ((b & 0xE0U) == 0xC0U) {
        return 2;
    }
    if ((b & 0xF0U) == 0xE0U) {
        return 3;
    }
    if ((b & 0xF8U) == 0xF0U) {
        return 4;
    }
    return 0; // continuation byte or 0xF8..0xFF -- never a lead
}

[[nodiscard]] std::uint32_t lead_bits(unsigned char b, int len) noexcept {
    switch (len) {
        case 2:
            return b & 0x1FU;
        case 3:
            return b & 0x0FU;
        case 4:
            return b & 0x07U;
        default:
            return b;
    }
}

// Rejects overlong encodings, surrogate halves and anything above U+10FFFF. A decoder
// that accepts overlongs lets the same codepoint have two spellings, which is how a
// byte-faithful trace stops being byte-faithful.
[[nodiscard]] bool codepoint_ok(std::uint32_t cp, int len) noexcept {
    if (cp > 0x10FFFFU) {
        return false;
    }
    if (cp >= 0xD800U && cp <= 0xDFFFU) {
        return false;
    }
    const std::uint32_t minimum[5] = {0U, 0U, 0x80U, 0x800U, 0x10000U};
    return cp >= minimum[len];
}

// Length of the well-formed sequence at `p`, or 0 if there is not one.
[[nodiscard]] int decode_len(const unsigned char* p, std::size_t avail) noexcept {
    const int len = lead_len(p[0]);
    if (len == 0 || static_cast<std::size_t>(len) > avail) {
        return 0;
    }
    if (len == 1) {
        return 1;
    }
    std::uint32_t cp = lead_bits(p[0], len);
    for (int i = 1; i < len; ++i) {
        if ((p[static_cast<std::size_t>(i)] & 0xC0U) != 0x80U) {
            return 0;
        }
        cp = (cp << 6U) | (p[static_cast<std::size_t>(i)] & 0x3FU);
    }
    return codepoint_ok(cp, len) ? len : 0;
}

void append_ascii_escaped(std::string& out, unsigned char c) {
    switch (c) {
        case '"':
            out.append("\\\"");
            return;
        case '\\':
            out.append("\\\\");
            return;
        case '\b':
            out.append("\\b");
            return;
        case '\f':
            out.append("\\f");
            return;
        case '\n':
            out.append("\\n");
            return;
        case '\r':
            out.append("\\r");
            return;
        case '\t':
            out.append("\\t");
            return;
        default:
            break;
    }
    if (c < 0x20U) {
        char buf[7];
        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
        out.append(buf);
        return;
    }
    out.push_back(static_cast<char>(c));
}

// --- base64 ----------------------------------------------------------------

constexpr std::string_view kB64 =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

[[nodiscard]] int b64_value(char c) noexcept {
    const std::size_t pos = kB64.find(c);
    return pos == std::string_view::npos ? -1 : static_cast<int>(pos);
}

// --- one-writer registry ---------------------------------------------------
//
// S3 permits exactly two pieces of mutable global state: the cancel token and the event
// log writer. This is that exception, and it is deliberately the smallest form of it --
// a set of paths, not a singleton log. Two writers appending to one file interleave
// partial lines, and the resulting corruption reads as a model bug.

struct Registry {
    std::mutex mu;
    std::vector<std::string> paths;
};

[[nodiscard]] Registry& registry() {
    static Registry r;
    return r;
}

[[nodiscard]] bool registry_claim(const std::string& path) {
    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mu);
    if (std::find(r.paths.begin(), r.paths.end(), path) != r.paths.end()) {
        return false;
    }
    r.paths.push_back(path);
    return true;
}

void registry_release(const std::string& path) {
    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mu);
    const auto it = std::find(r.paths.begin(), r.paths.end(), path);
    if (it != r.paths.end()) {
        r.paths.erase(it);
    }
}

[[nodiscard]] bool write_all(int fd, const char* data, std::size_t len) {
    std::size_t done = 0;
    while (done < len) {
        const ssize_t n = ::write(fd, data + done, len - done);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

} // namespace

// --- pure core -------------------------------------------------------------

bool is_valid_utf8(std::string_view in) noexcept {
    const auto* p = reinterpret_cast<const unsigned char*>(in.data());
    std::size_t i = 0;
    while (i < in.size()) {
        const int len = decode_len(p + i, in.size() - i);
        if (len == 0) {
            return false;
        }
        i += static_cast<std::size_t>(len);
    }
    return true;
}

bool append_json_string(std::string& out, std::string_view in) {
    bool valid = true;
    out.push_back('"');
    const auto* p = reinterpret_cast<const unsigned char*>(in.data());
    std::size_t i = 0;
    while (i < in.size()) {
        if (p[i] < 0x80U) {
            append_ascii_escaped(out, p[i]);
            ++i;
            continue;
        }
        const int len = decode_len(p + i, in.size() - i);
        if (len == 0) {
            out.append("\xEF\xBF\xBD"); // U+FFFD; exact bytes go to the __b64 sibling
            valid = false;
            ++i;
            continue;
        }
        out.append(in.substr(i, static_cast<std::size_t>(len)));
        i += static_cast<std::size_t>(len);
    }
    out.push_back('"');
    return valid;
}

std::string base64_encode(std::string_view in) {
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 2 < in.size()) {
        const std::uint32_t v = (static_cast<std::uint32_t>(static_cast<unsigned char>(in[i])) << 16U) |
                                (static_cast<std::uint32_t>(static_cast<unsigned char>(in[i + 1])) << 8U) |
                                static_cast<std::uint32_t>(static_cast<unsigned char>(in[i + 2]));
        out.push_back(kB64[(v >> 18U) & 0x3FU]);
        out.push_back(kB64[(v >> 12U) & 0x3FU]);
        out.push_back(kB64[(v >> 6U) & 0x3FU]);
        out.push_back(kB64[v & 0x3FU]);
        i += 3;
    }
    const std::size_t rem = in.size() - i;
    if (rem == 0) {
        return out;
    }
    std::uint32_t v = static_cast<std::uint32_t>(static_cast<unsigned char>(in[i])) << 16U;
    if (rem == 2) {
        v |= static_cast<std::uint32_t>(static_cast<unsigned char>(in[i + 1])) << 8U;
    }
    out.push_back(kB64[(v >> 18U) & 0x3FU]);
    out.push_back(kB64[(v >> 12U) & 0x3FU]);
    out.push_back(rem == 2 ? kB64[(v >> 6U) & 0x3FU] : '=');
    out.push_back('=');
    return out;
}

bool base64_decode(std::string_view in, std::string& out) {
    out.clear();
    if (in.size() % 4 != 0) {
        return false;
    }
    for (std::size_t i = 0; i < in.size(); i += 4) {
        std::uint32_t v = 0;
        int pad = 0;
        for (std::size_t k = 0; k < 4; ++k) {
            const char c = in[i + k];
            const int d = (c == '=') ? (++pad, 0) : b64_value(c);
            if (d < 0 || (pad > 0 && c != '=')) {
                return false;
            }
            v = (v << 6U) | static_cast<std::uint32_t>(d);
        }
        out.push_back(static_cast<char>((v >> 16U) & 0xFFU));
        if (pad < 2) {
            out.push_back(static_cast<char>((v >> 8U) & 0xFFU));
        }
        if (pad < 1) {
            out.push_back(static_cast<char>(v & 0xFFU));
        }
    }
    return true;
}

namespace {

// Emits "key":"value", plus "key__b64":"..." when the value was not valid UTF-8.
void append_field(std::string& out, std::string_view key, std::string_view value) {
    out.push_back(',');
    (void)append_json_string(out, key);
    out.push_back(':');
    const bool valid = append_json_string(out, value);
    if (valid) {
        return;
    }
    out.push_back(',');
    (void)append_json_string(out, std::string(key) + "__b64");
    out.push_back(':');
    (void)append_json_string(out, base64_encode(value));
}

} // namespace

std::string default_event_log_path(const char* pinned, const char* home) {
    // An explicit path wins outright, so a harness can pin one run's trace to a known
    // file without caring where this function would otherwise have put it.
    if (pinned != nullptr && *pinned != '\0') {
        return pinned;
    }
    if (home == nullptr || *home == '\0') {
        // No HOME is a broken environment rather than a supported one. Returning the old
        // relative path keeps the process alive where it used to be alive, instead of
        // trading a bad default for no sidecar at all.
        return "lmp_events.jsonl";
    }
    // ~/Library/Logs is where a macOS user looks for an application's diagnostics and
    // where they can delete them without breaking anything. Resume reads the same path
    // back via read_event_log; it remains diagnostics-shaped (append-only, rotatable).
    return std::string(home) + "/Library/Logs/LM_Pipe/events.jsonl";
}

std::string serialize_event(const Event& ev) {
    std::string out;
    out.reserve(128);
    out.append("{\"seq\":");
    out.append(std::to_string(ev.seq));
    out.append(",\"t_wall_ns\":");
    out.append(std::to_string(ev.wall_ns));
    out.append(",\"t_mono_us\":");
    out.append(std::to_string(ev.mono_us));
    out.append(",\"kind\":");
    (void)append_json_string(out, ev.kind);
    for (const EventField& f : ev.fields) {
        append_field(out, f.key, f.value);
    }
    out.append("}\n");
    return out;
}

std::string rotated_path(std::string_view path, std::size_t index) {
    const std::size_t slash = path.find_last_of('/');
    const std::size_t dot = path.find_last_of('.');
    const bool has_ext = dot != std::string_view::npos &&
                         (slash == std::string_view::npos || dot > slash);
    const std::string suffix = "." + std::to_string(index);
    if (!has_ext) {
        return std::string(path) + suffix;
    }
    return std::string(path.substr(0, dot)) + suffix + std::string(path.substr(dot));
}

// --- adapter ---------------------------------------------------------------

EventLogWriter::~EventLogWriter() { close(); }

OpenResult EventLogWriter::open(const EventLogOptions& opts) {
    if (is_open()) {
        return {false, "already open"};
    }
    if (opts.max_files < 1) {
        return {false, "max_files must be >= 1"};
    }
    if (opts.max_bytes_per_file == 0) {
        return {false, "max_bytes_per_file must be > 0"};
    }
    if (!registry_claim(opts.path)) {
        return {false, "another EventLogWriter in this process already holds " + opts.path};
    }
    const int fd = ::open(opts.path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        registry_release(opts.path);
        return {false, std::string("open failed: ") + std::strerror(errno)};
    }
    struct stat st {};
    cur_bytes_ = (::fstat(fd, &st) == 0) ? static_cast<std::size_t>(st.st_size) : 0;
    fd_ = fd;
    path_ = opts.path;
    max_bytes_ = opts.max_bytes_per_file;
    max_files_ = opts.max_files;
    return {true, {}};
}

void EventLogWriter::append(Event& ev, const Clock& clock) {
    if (!is_open()) {
        return;
    }
    ev.seq = next_seq_++;
    ev.wall_ns = to_ns(clock.wall());
    ev.mono_us = to_us(clock.mono());
    const std::string line = serialize_event(ev);
    if (cur_bytes_ + line.size() > max_bytes_ && cur_bytes_ > 0) {
        rotate();
    }
    if (write_all(fd_, line.data(), line.size())) {
        cur_bytes_ += line.size();
    }
}

void EventLogWriter::rotate() {
    ::close(fd_);
    fd_ = -1;
    if (max_files_ > 1) {
        ::unlink(rotated_path(path_, max_files_ - 1).c_str());
        for (std::size_t i = max_files_ - 1; i >= 2; --i) {
            ::rename(rotated_path(path_, i - 1).c_str(), rotated_path(path_, i).c_str());
        }
        ::rename(path_.c_str(), rotated_path(path_, 1).c_str());
    }
    fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    cur_bytes_ = 0;
    ++rotations_;
}

void EventLogWriter::flush() {
    if (is_open()) {
        ::fsync(fd_);
    }
}

void EventLogWriter::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    if (!path_.empty()) {
        registry_release(path_);
        path_.clear();
    }
}

bool parse_event_line(std::string_view line, Event& out) {
    out = Event{};
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.remove_suffix(1);
    }
    if (line.empty()) {
        return false;
    }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(line.begin(), line.end());
    } catch (const nlohmann::json::exception&) {
        return false;
    }
    if (!j.is_object() || !j.contains("kind") || !j.at("kind").is_string()) {
        return false;
    }
    out.kind = j.at("kind").get<std::string>();
    if (j.contains("seq") && j.at("seq").is_number_unsigned()) {
        out.seq = j.at("seq").get<std::uint64_t>();
    } else if (j.contains("seq") && j.at("seq").is_number_integer()) {
        const auto v = j.at("seq").get<std::int64_t>();
        out.seq = v < 0 ? 0 : static_cast<std::uint64_t>(v);
    }
    if (j.contains("t_wall_ns") && j.at("t_wall_ns").is_number_integer()) {
        out.wall_ns = j.at("t_wall_ns").get<std::int64_t>();
    }
    if (j.contains("t_mono_us") && j.at("t_mono_us").is_number_integer()) {
        out.mono_us = j.at("t_mono_us").get<std::int64_t>();
    }

    // First pass: collect plain string fields. Second: overlay __b64 siblings so
    // byte-faithful values replace their lossy UTF-8 display forms.
    std::unordered_map<std::string, std::string> fields;
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& key = it.key();
        if (key == "seq" || key == "t_wall_ns" || key == "t_mono_us" || key == "kind") {
            continue;
        }
        if (key.size() > 5 && key.ends_with("__b64")) {
            continue;
        }
        if (it.value().is_string()) {
            fields[key] = it.value().get<std::string>();
        }
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& key = it.key();
        if (key.size() <= 5 || !key.ends_with("__b64") || !it.value().is_string()) {
            continue;
        }
        const std::string base = key.substr(0, key.size() - 5);
        std::string decoded;
        if (base64_decode(it.value().get<std::string>(), decoded)) {
            fields[base] = std::move(decoded);
        }
    }
    out.fields.reserve(fields.size());
    for (auto& [k, v] : fields) {
        out.fields.push_back({std::move(k), std::move(v)});
    }
    return true;
}

bool read_event_log(const std::string& path, std::vector<Event>& out, std::size_t& skipped,
                    std::string& error) {
    out.clear();
    skipped = 0;
    error.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open event log: " + path;
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        Event ev;
        if (!parse_event_line(line, ev)) {
            ++skipped;
            continue;
        }
        out.push_back(std::move(ev));
    }
    return true;
}

} // namespace lmp::platform
