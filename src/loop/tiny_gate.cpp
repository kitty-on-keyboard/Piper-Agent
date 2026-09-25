#include "src/loop/tiny_gate.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdint>

#include <nlohmann/json.hpp>

namespace lmp::loop {
namespace {

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

[[nodiscard]] bool is_hindsight_key(std::string_view key) noexcept {
    return key == "next" || key == "outcome" || key == "gold" || key == "label" ||
           key == "chosen" || key == "decision" || key == "answer" || key == "y_true" ||
           key == "y_pred";
}

void erase_hindsight_assignments(std::string& s) {
    std::size_t i = 0;
    while (i < s.size()) {
        if (!(std::isalpha(static_cast<unsigned char>(s[i])) || s[i] == '_')) {
            ++i;
            continue;
        }
        const std::size_t key_begin = i;
        while (i < s.size() &&
               (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) {
            ++i;
        }
        const std::string_view key(s.data() + key_begin, i - key_begin);
        if (i >= s.size() || s[i] != '=' || !is_hindsight_key(key)) {
            continue;
        }
        std::size_t val_end = i + 1;
        if (val_end < s.size() && (s[val_end] == '"' || s[val_end] == '\'')) {
            const char q = s[val_end];
            ++val_end;
            while (val_end < s.size() && s[val_end] != q && s[val_end] != '\n') {
                ++val_end;
            }
            if (val_end < s.size() && s[val_end] == q) {
                ++val_end;
            }
        } else {
            while (val_end < s.size() && !std::isspace(static_cast<unsigned char>(s[val_end])) &&
                   s[val_end] != ',' && s[val_end] != ';') {
                ++val_end;
            }
        }
        s.erase(key_begin, val_end - key_begin);
        i = key_begin;
    }
}

[[nodiscard]] std::optional<GateChoice> parse_choice_name(std::string_view name) noexcept {
    if (name == "force_tool") {
        return GateChoice::ForceTool;
    }
    if (name == "nudge") {
        return GateChoice::Nudge;
    }
    return std::nullopt;
}

[[nodiscard]] bool parse_url(std::string_view url, std::string& host, int& port,
                             std::string& path) {
    host.clear();
    path = "/v1/choose";
    port = 80;
    std::string_view rest = url;
    if (rest.starts_with("http://")) {
        rest.remove_prefix(7);
    } else if (rest.starts_with("https://")) {
        // Localhost gate is plain HTTP only; refuse TLS silently as transport error.
        return false;
    }
    const auto slash = rest.find('/');
    std::string_view authority = slash == std::string_view::npos ? rest : rest.substr(0, slash);
    if (slash != std::string_view::npos) {
        path.assign(rest.substr(slash));
    }
    const auto colon = authority.rfind(':');
    if (colon != std::string_view::npos) {
        host.assign(authority.substr(0, colon));
        const std::string port_s(authority.substr(colon + 1));
        char* end = nullptr;
        const long p = std::strtol(port_s.c_str(), &end, 10);
        if (end == port_s.c_str() || *end != '\0' || p <= 0 || p > 65535) {
            return false;
        }
        port = static_cast<int>(p);
    } else {
        host.assign(authority);
    }
    return !host.empty();
}

} // namespace

std::string format_gate_features(const GateFeatures& f) {
    std::ostringstream out;
    out << "consec=" << f.consec << " streak=" << f.streak << " prompt_tok=" << f.prompt_tok
        << " reread_max=" << f.reread_max << '\n'
        << "think=" << f.think << " text=" << f.text << " tool_tok=" << f.tool_tok
        << " why=" << (f.why.empty() ? "none" : f.why);
    return out.str();
}

std::string strip_gate_hindsight(std::string_view text) {
    std::string out(text);
    erase_hindsight_assignments(out);
    std::string cleaned;
    cleaned.reserve(out.size());
    std::size_t line_start = 0;
    while (line_start <= out.size()) {
        const std::size_t nl = out.find('\n', line_start);
        const std::size_t line_end = nl == std::string::npos ? out.size() : nl;
        std::string_view line(out.data() + line_start, line_end - line_start);
        std::size_t a = 0;
        std::size_t b = line.size();
        while (a < b && std::isspace(static_cast<unsigned char>(line[a]))) {
            ++a;
        }
        while (b > a && std::isspace(static_cast<unsigned char>(line[b - 1]))) {
            --b;
        }
        if (a < b) {
            if (!cleaned.empty() && cleaned.back() != '\n') {
                cleaned.push_back('\n');
            }
            cleaned.append(line.data() + a, b - a);
        }
        if (nl == std::string::npos) {
            break;
        }
        line_start = nl + 1;
    }
    return cleaned;
}

GateChoiceOrder shuffle_gate_t1_choices(std::uint64_t seed) noexcept {
    GateChoiceOrder order = {GateChoice::ForceTool, GateChoice::Nudge};
    std::uint64_t state = seed ^ 0x47415445ULL; // "GATE"
    if (state == 0) {
        state = 0xC0FFEEULL;
    }
    for (std::size_t i = kGateT1OptionCount; i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(splitmix64(state) % i);
        std::swap(order[i - 1], order[j]);
    }
    return order;
}

std::string gate_t1_forced_prefix(const GateChoiceOrder& order, const GateFeatures& features,
                                  std::string_view mission) {
    std::ostringstream out;
    out << "Pick exactly one harness next-step. Reply with only the code letter.\n\n";
    for (std::size_t i = 0; i < kGateT1OptionCount; ++i) {
        out << kGateT1Letters[i] << " = " << gate_choice_def(order[i]) << '\n';
    }
    out << "\nFeatures:\n" << format_gate_features(features);
    if (!mission.empty()) {
        out << "\nMission: " << mission;
    }
    out << "\n";
    return strip_gate_hindsight(out.str());
}

std::optional<GatePolicy> stage0_gate_policy(const GateFeatures& f) noexcept {
    if (f.consec >= 3 || f.streak >= 4) {
        return GatePolicy::Stall;
    }
    if (f.prompt_tok >= 16000 || f.reread_max >= 3) {
        return GatePolicy::Compact;
    }
    return std::nullopt;
}

GatePolicy apply_gate_policy(const GateMicroResult& result, float p_min) noexcept {
    if (!result.ok) {
        return GatePolicy::Fallback;
    }
    if (result.p_force >= p_min) {
        return GatePolicy::ForceTool;
    }
    return GatePolicy::Nudge;
}

std::string build_gate_request_json(std::string_view prompt, const GateChoiceOrder& order) {
    nlohmann::json choices = nlohmann::json::array();
    nlohmann::json letters = nlohmann::json::array();
    for (std::size_t i = 0; i < kGateT1OptionCount; ++i) {
        choices.push_back(std::string(gate_choice_name(order[i])));
        letters.push_back(std::string(1, kGateT1Letters[i]));
    }
    nlohmann::json req = {{"prompt", std::string(prompt)},
                          {"choices", std::move(choices)},
                          {"letters", std::move(letters)},
                          {"encoding", "letter"}};
    return req.dump();
}

GateMicroResult parse_gate_response_json(std::string_view body) {
    GateMicroResult r;
    r.encoding = "letter";
    if (body.empty()) {
        r.error = "gate: empty response";
        return r;
    }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(body);
    } catch (const std::exception& e) {
        r.error = std::string("gate: json parse: ") + e.what();
        return r;
    }
    if (!j.is_object()) {
        r.error = "gate: response not an object";
        return r;
    }
    if (j.contains("error") && j["error"].is_string()) {
        r.error = j["error"].get<std::string>();
    }
    if (j.contains("model") && j["model"].is_string()) {
        r.model = j["model"].get<std::string>();
    }
    if (j.contains("encoding") && j["encoding"].is_string()) {
        r.encoding = j["encoding"].get<std::string>();
    }
    if (j.contains("order") && j["order"].is_string()) {
        r.order = j["order"].get<std::string>();
    }
    if (j.contains("latency_ms") && j["latency_ms"].is_number()) {
        r.latency_ms = j["latency_ms"].get<double>();
    }
    const bool ok_flag = j.value("ok", false);
    if (!ok_flag) {
        if (r.error.empty()) {
            r.error = "gate: ok=false";
        }
        return r;
    }

    if (j.contains("p_vec") && j["p_vec"].is_array()) {
        for (const auto& v : j["p_vec"]) {
            if (v.is_number()) {
                r.p_vec.push_back(v.get<float>());
            }
        }
    }
    if (j.contains("p") && j["p"].is_number()) {
        r.p = j["p"].get<float>();
    }
    if (j.contains("p_force") && j["p_force"].is_number()) {
        r.p_force = j["p_force"].get<float>();
    }

    std::string choice_s;
    if (j.contains("choice") && j["choice"].is_string()) {
        choice_s = j["choice"].get<std::string>();
    }
    if (j.contains("letter") && j["letter"].is_string()) {
        const std::string let = j["letter"].get<std::string>();
        if (!let.empty()) {
            r.letter = let.front();
        }
    }

    if (const auto parsed = parse_choice_name(choice_s)) {
        r.choice = *parsed;
        r.choice_name = std::string(gate_choice_name(r.choice));
    } else if (!choice_s.empty()) {
        r.error = "gate: unknown choice '" + choice_s + "'";
        return r;
    } else {
        r.error = "gate: missing choice";
        return r;
    }

    // If helper omitted p_force, derive from order+p_vec when possible.
    if (!j.contains("p_force") && !r.p_vec.empty() && !r.order.empty()) {
        std::size_t idx = 0;
        std::size_t start = 0;
        while (start <= r.order.size()) {
            const std::size_t comma = r.order.find(',', start);
            const std::string_view part =
                comma == std::string::npos
                    ? std::string_view(r.order).substr(start)
                    : std::string_view(r.order).substr(start, comma - start);
            if (part == "force_tool" && idx < r.p_vec.size()) {
                r.p_force = r.p_vec[idx];
                break;
            }
            ++idx;
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }

    r.ok = true;
    r.error.clear();
    return r;
}

GateMicroResult query_gate_http(std::string_view base_url, std::string_view request_json,
                                int timeout_ms) {
    GateMicroResult fail;
    fail.error = "gate: http transport failed";

    std::string host;
    int port = 18765;
    std::string path = "/v1/choose";
    // Accept either full URL with path or a base like http://127.0.0.1:18765
    std::string url(base_url);
    if (url.find("/v1/") == std::string::npos) {
        if (!url.empty() && url.back() == '/') {
            url.pop_back();
        }
        url += "/v1/choose";
    }
    if (!parse_url(url, host, port, path)) {
        fail.error = "gate: bad url";
        return fail;
    }

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        fail.error = "gate: socket() failed";
        return fail;
    }

    timeval tv{};
    tv.tv_sec = std::max(1, timeout_ms / 1000);
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        // Fallback: localhost only — avoid blocking DNS in the agent hot path.
        if (host == "localhost") {
            (void)::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        } else {
            ::close(fd);
            fail.error = "gate: host must be IPv4 or localhost";
            return fail;
        }
    }

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        fail.error = std::string("gate: connect failed: ") + std::strerror(errno);
        return fail;
    }

    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << request_json.size() << "\r\n"
        << "Connection: close\r\n"
        << "\r\n"
        << request_json;
    const std::string req_s = req.str();
    std::size_t sent = 0;
    while (sent < req_s.size()) {
        const ssize_t n = ::send(fd, req_s.data() + sent, req_s.size() - sent, 0);
        if (n <= 0) {
            ::close(fd);
            fail.error = "gate: send failed";
            return fail;
        }
        sent += static_cast<std::size_t>(n);
    }

    std::string raw;
    char buf[4096];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            ::close(fd);
            fail.error = "gate: recv failed";
            return fail;
        }
        if (n == 0) {
            break;
        }
        raw.append(buf, static_cast<std::size_t>(n));
        if (raw.size() > 1U << 20) {
            ::close(fd);
            fail.error = "gate: response too large";
            return fail;
        }
    }
    ::close(fd);

    const auto hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos) {
        fail.error = "gate: malformed http response";
        return fail;
    }
    const std::string_view headers(raw.data(), hdr_end);
    if (headers.find(" 200 ") == std::string_view::npos &&
        headers.find(" 200\r") == std::string_view::npos) {
        // Still try to parse body if present; many helpers return 200 always.
        if (headers.find("HTTP/1.") == 0 && headers.find(" 2") == std::string_view::npos) {
            fail.error = "gate: http non-2xx";
            // Fall through to parse body for a structured error when possible.
        }
    }
    const std::string_view body(raw.data() + hdr_end + 4, raw.size() - (hdr_end + 4));
    GateMicroResult parsed = parse_gate_response_json(body);
    if (!parsed.ok && parsed.error == "gate: empty response" && !fail.error.empty() &&
        fail.error != "gate: http transport failed") {
        return fail;
    }
    return parsed;
}

} // namespace lmp::loop
