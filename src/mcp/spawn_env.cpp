#include "src/mcp/spawn_env.hpp"

#include <map>
#include <string>
#include <string_view>

namespace lmp::mcp {
namespace {

[[nodiscard]] char ascii_lower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

[[nodiscard]] bool ascii_ieq(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (ascii_lower(a[i]) != ascii_lower(b[i])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool ascii_istarts(std::string_view s, std::string_view prefix) noexcept {
    if (s.size() < prefix.size()) {
        return false;
    }
    return ascii_ieq(s.substr(0, prefix.size()), prefix);
}

[[nodiscard]] bool ascii_iends(std::string_view s, std::string_view suffix) noexcept {
    if (s.size() < suffix.size()) {
        return false;
    }
    return ascii_ieq(s.substr(s.size() - suffix.size()), suffix);
}

[[nodiscard]] bool denied_parent_key(std::string_view key) noexcept {
    static constexpr std::string_view kExact[] = {
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "GNUPGHOME",
        "KUBECONFIG",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "DATABASE_URL",
        "REDIS_URL",
        "NODE_AUTH_TOKEN",
        "NPM_TOKEN",
    };
    for (const std::string_view e : kExact) {
        if (ascii_ieq(key, e)) {
            return true;
        }
    }

    static constexpr std::string_view kPrefix[] = {
        "AWS_",  "AZURE_", "GCP_",     "OPENAI_", "ANTHROPIC_", "GEMINI_", "LMP_",
        "DYLD_", "LD_",    "SSH_",     "GITHUB_", "GITLAB_",    "NPM_",
    };
    for (const std::string_view p : kPrefix) {
        if (ascii_istarts(key, p)) {
            return true;
        }
    }

    // Exact name or trailing _NAME, so PATH does not match KEY and TOKENIZE does not
    // match TOKEN. CREDENTIALS covers GOOGLE_APPLICATION_CREDENTIALS if the exact
    // list is ever trimmed; CREDENTIAL covers the singular form.
    static constexpr std::string_view kSuffix[] = {
        "TOKEN", "SECRET", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS",
    };
    for (const std::string_view s : kSuffix) {
        if (ascii_ieq(key, s)) {
            return true;
        }
        if (key.size() > s.size() + 1 && key[key.size() - s.size() - 1] == '_' &&
            ascii_iends(key, s)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool allowlisted_parent_key(std::string_view key) noexcept {
    static constexpr std::string_view kExact[] = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "TZ",
        "TERM",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    };
    for (const std::string_view e : kExact) {
        if (key == e) {
            return true;
        }
    }
    return key.size() > 3 && (key.substr(0, 3) == "LC_" || key.substr(0, 4) == "XDG_");
}

[[nodiscard]] bool inherit_parent_key(std::string_view key) noexcept {
    return allowlisted_parent_key(key) && !denied_parent_key(key);
}

} // namespace

std::vector<std::string> build_child_environ(
    const char* const* parent,
    const std::vector<std::pair<std::string, std::string>>& extra) {
    std::map<std::string, std::string> out;
    if (parent != nullptr) {
        for (const char* const* e = parent; *e != nullptr; ++e) {
            const std::string_view kv(*e);
            const std::size_t eq = kv.find('=');
            if (eq == std::string_view::npos || eq == 0) {
                continue;
            }
            const std::string_view key = kv.substr(0, eq);
            if (!inherit_parent_key(key)) {
                continue;
            }
            out.insert_or_assign(std::string(key), std::string(kv.substr(eq + 1)));
        }
    }
    for (const auto& [k, v] : extra) {
        if (k.empty()) {
            continue;
        }
        out.insert_or_assign(k, v);
    }
    std::vector<std::string> envp;
    envp.reserve(out.size());
    for (const auto& [k, v] : out) {
        envp.push_back(k + "=" + v);
    }
    return envp;
}

} // namespace lmp::mcp
