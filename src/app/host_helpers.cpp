#include "src/app/host_helpers.hpp"

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/platform/event_log.hpp"
#include "src/platform/fs.hpp"

extern char** environ;

namespace lmp::app {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int hex_val(char c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool url_decode(std::string_view in, std::string& out) {
    out.clear();
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '%') {
            if (i + 2 >= in.size()) return false;
            const int hi = hex_val(in[i + 1]);
            const int lo = hex_val(in[i + 2]);
            if (hi < 0 || lo < 0) return false;
            const char ch = static_cast<char>((hi << 4) | lo);
            if (ch == '\0') return false;
            out.push_back(ch);
            i += 2;
        } else {
            if (in[i] == '\0') return false;
            out.push_back(in[i]);
        }
    }
    return true;
}

std::string mime_type_for_path(const fs::path& p) {
    std::string ext = p.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
    if (ext == ".js" || ext == ".mjs") return "application/javascript; charset=utf-8";
    if (ext == ".css") return "text/css; charset=utf-8";
    if (ext == ".json" || ext == ".map") return "application/json; charset=utf-8";
    if (ext == ".svg") return "image/svg+xml";
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".webp") return "image/webp";
    if (ext == ".ico") return "image/x-icon";
    if (ext == ".woff2") return "font/woff2";
    if (ext == ".woff") return "font/woff";
    if (ext == ".ttf") return "font/ttf";
    if (ext == ".txt") return "text/plain; charset=utf-8";
    return "application/octet-stream";
}

bool has_dot_dot_segment(std::string_view p) {
    std::size_t start = 0;
    while (start <= p.size()) {
        const std::size_t slash = p.find('/', start);
        const std::size_t len = (slash == std::string_view::npos) ? (p.size() - start) : (slash - start);
        if (p.substr(start, len) == "..") return true;
        if (slash == std::string_view::npos) break;
        start = slash + 1;
    }
    return false;
}

std::string lower_ascii(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

struct CmdResult {
    bool launched = false;
    int exit_code = -1;
    std::string stdout_bytes;
};

CmdResult run_git(const fs::path& root, const std::vector<std::string>& git_args,
                  std::size_t max_stdout = 8 * 1024 * 1024) {
    CmdResult res;
    int pipefd[2] = {-1, -1};
    if (::pipe(pipefd) != 0) return res;
    ::fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
    ::fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);

    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        return res;
    }
    struct ActionsGuard {
        posix_spawn_file_actions_t* a;
        ~ActionsGuard() { ::posix_spawn_file_actions_destroy(a); }
    } guard{&actions};

    ::posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    ::posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    ::posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    ::posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    ::posix_spawn_file_actions_addclose(&actions, pipefd[1]);

    std::vector<std::string> storage;
    storage.reserve(git_args.size() + 4);
    storage.emplace_back("git");
    storage.emplace_back("-C");
    storage.emplace_back(root.string());
    for (const auto& a : git_args) {
        storage.push_back(a);
    }

    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (auto& s : storage) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int rc = ::posix_spawnp(&pid, "git", &actions, nullptr, argv.data(), environ);
    ::close(pipefd[1]);
    if (rc != 0) {
        ::close(pipefd[0]);
        return res;
    }

    res.launched = true;
    char buf[8192];
    bool truncated = false;
    for (;;) {
        const ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
        if (n > 0) {
            const std::size_t count = static_cast<std::size_t>(n);
            if (!truncated) {
                const std::size_t room = max_stdout - res.stdout_bytes.size();
                if (count <= room) {
                    res.stdout_bytes.append(buf, count);
                } else {
                    res.stdout_bytes.append(buf, room);
                    truncated = true;
                }
            }
        } else if (n == 0) {
            break;
        } else if (errno != EINTR) {
            break;
        }
    }
    ::close(pipefd[0]);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (WIFEXITED(status)) {
        res.exit_code = WEXITSTATUS(status);
    }
    return res;
}

std::string trim_right_ws(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
}

std::size_t utf8_safe_prefix_len(std::string_view s) noexcept {
    if (s.empty()) return 0;
    std::size_t i = s.size();
    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0u) == 0x80u) {
        --i;
    }
    if (i == 0) return s.size();
    const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
    std::size_t expected = 1;
    if ((lead & 0xE0u) == 0xC0u) expected = 2;
    else if ((lead & 0xF0u) == 0xE0u) expected = 3;
    else if ((lead & 0xF8u) == 0xF0u) expected = 4;
    if (s.size() - (i - 1) < expected) {
        return i - 1;
    }
    return s.size();
}

bool contains_nul(std::string_view s) noexcept {
    return s.find('\0') != std::string_view::npos;
}

std::optional<json> read_config_json(const fs::path& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return std::nullopt;
    const fs::path cfg = dir / "config.json";
    if (!fs::is_regular_file(cfg, ec)) return std::nullopt;
    const platform::FileContents fc = platform::read_file_whole(cfg.string(), 4 * 1024 * 1024);
    if (!fc.ok()) return std::nullopt;
    try {
        return json::parse(fc.bytes);
    } catch (...) {
        return std::nullopt;
    }
}

bool is_usable_mtp(const fs::path& dir) {
    const auto cfg = read_config_json(dir);
    if (!cfg || !cfg->is_object()) return false;
    if (cfg->value("model_type", std::string{}) != "qwen3_5_mtp") return false;
    return cfg->value("block_size", 0) >= 2;
}

std::size_t shared_prefix_len(std::string_view a, std::string_view b) noexcept {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

std::uint64_t top_level_checkpoint_bytes(const fs::path& dir) {
    std::error_code ec;
    std::uint64_t total = 0;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string ext = lower_ascii(entry.path().extension().string());
        if (ext == ".safetensors" || ext == ".bin" || ext == ".json") {
            const auto sz = entry.file_size(ec);
            if (!ec) total += sz;
        }
    }
    return total;
}

bool is_ignored_walk_dir(std::string_view name) noexcept {
    return name == ".git" || name == "node_modules" || name == "build" || name == "build-asan" ||
           name == ".piper" || name == "dist" || name == "out" || name == ".cache";
}

struct ParsedFileDiff {
    std::string path;
    int additions = 0;
    int deletions = 0;
    std::string diff;
};

std::string extract_diff_path(std::string_view chunk) {
    // Expect lines like "diff --git a/foo b/foo\n..." or "+++ b/foo\n" / "--- a/foo\n".
    std::istringstream iss{std::string(chunk.substr(0, std::min<std::size_t>(chunk.size(), 4096)))};
    std::string line;
    std::string plus_path;
    std::string minus_path;
    std::string header_line;
    int seen = 0;
    while (std::getline(iss, line) && seen < 12) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (seen == 0) header_line = line;
        if (line.rfind("+++ b/", 0) == 0) {
            plus_path = line.substr(6);
        } else if (line.rfind("--- a/", 0) == 0) {
            minus_path = line.substr(6);
        }
        ++seen;
    }
    if (!plus_path.empty()) return plus_path;
    if (!minus_path.empty()) return minus_path;
    const std::size_t b_pos = header_line.rfind(" b/");
    if (b_pos != std::string::npos) return header_line.substr(b_pos + 3);
    return {};
}

void count_diff_lines(std::string_view chunk, int& adds, int& dels) {
    adds = 0;
    dels = 0;
    bool in_hunk = false;
    std::size_t pos = 0;
    while (pos < chunk.size()) {
        const std::size_t nl = chunk.find('\n', pos);
        const std::string_view line =
            (nl == std::string_view::npos) ? chunk.substr(pos) : chunk.substr(pos, nl - pos);
        if (line.rfind("@@ ", 0) == 0) {
            in_hunk = true;
        } else if (in_hunk && !line.empty()) {
            if (line[0] == '+' && line.rfind("+++ ", 0) != 0) ++adds;
            else if (line[0] == '-' && line.rfind("--- ", 0) != 0) ++dels;
        }
        if (nl == std::string_view::npos) break;
        pos = nl + 1;
    }
}

std::vector<ParsedFileDiff> parse_unified_diff(std::string_view raw) {
    std::vector<ParsedFileDiff> out;
    std::size_t pos = 0;
    while (pos < raw.size()) {
        if (raw.compare(pos, 11, "diff --git ") != 0) {
            const std::size_t next = raw.find("\ndiff --git ", pos);
            if (next == std::string_view::npos) break;
            pos = next + 1;
        }
        const std::size_t next = raw.find("\ndiff --git ", pos + 11);
        const std::size_t end = (next == std::string_view::npos) ? raw.size() : (next + 1);
        std::string_view chunk = raw.substr(pos, end - pos);
        ParsedFileDiff item;
        item.path = extract_diff_path(chunk);
        count_diff_lines(chunk, item.additions, item.deletions);
        constexpr std::size_t kMaxFileDiff = 262144;
        if (chunk.size() > kMaxFileDiff) {
            const std::size_t len = utf8_safe_prefix_len(chunk.substr(0, kMaxFileDiff));
            item.diff.assign(chunk.substr(0, len));
            item.diff += "\n... (diff truncated)\n";
        } else {
            item.diff.assign(chunk);
        }
        if (!item.path.empty()) out.push_back(std::move(item));
        pos = end;
    }
    return out;
}

std::string build_untracked_diff(const platform::WorkspaceFs& wfs, const std::string& rel_path,
                                 int& adds) {
    adds = 0;
    const platform::FileContents fc = wfs.read_file_whole(rel_path, 131072);
    if (!fc.ok()) {
        if (fc.status == platform::FsStatus::TooLarge) {
            return "--- /dev/null\n+++ b/" + rel_path + "\n@@ -0,0 +1 @@\n+[File too large to preview: " +
                   std::to_string(fc.actual_size) + " bytes]\n";
        }
        return {};
    }
    if (contains_nul(fc.bytes)) {
        return "--- /dev/null\n+++ b/" + rel_path + "\n@@ -0,0 +1 @@\n+[Binary file]\n";
    }
    std::string out = "--- /dev/null\n+++ b/" + rel_path + "\n";
    std::vector<std::string_view> lines;
    std::size_t pos = 0;
    while (pos < fc.bytes.size()) {
        const std::size_t nl = fc.bytes.find('\n', pos);
        if (nl == std::string::npos) {
            lines.emplace_back(fc.bytes.data() + pos, fc.bytes.size() - pos);
            break;
        }
        lines.emplace_back(fc.bytes.data() + pos, nl - pos);
        pos = nl + 1;
    }
    adds = static_cast<int>(lines.size());
    out += "@@ -0,0 +1," + std::to_string(std::max(adds, 1)) + " @@\n";
    for (const auto line : lines) {
        out.push_back('+');
        out.append(line);
        out.push_back('\n');
    }
    return out;
}

} // namespace

ResolvedAsset resolve_ui_asset(const fs::path& ui_root, std::string_view url_path) {
    ResolvedAsset out;
    std::error_code ec;
    const fs::path canon_root = fs::canonical(ui_root, ec);
    if (ec || !fs::is_directory(canon_root, ec)) {
        out.error = "ui root does not exist";
        return out;
    }

    // Strip query string and fragment before decoding.
    std::string_view raw = url_path;
    const std::size_t qpos = raw.find_first_of("?#");
    if (qpos != std::string_view::npos) raw = raw.substr(0, qpos);
    while (!raw.empty() && raw.front() == '/') raw.remove_prefix(1);

    std::string decoded;
    if (!url_decode(raw, decoded)) {
        out.error = "malformed url encoding";
        return out;
    }
    if (has_dot_dot_segment(decoded)) {
        out.error = "path traversal rejected";
        return out;
    }

    const fs::path rel = decoded.empty() ? fs::path("index.html") : fs::path(decoded);
    const fs::path candidate = fs::weakly_canonical(canon_root / rel, ec);
    if (ec || !platform::is_within(canon_root.string(), candidate.string())) {
        out.error = "resolved path outside ui root";
        return out;
    }

    if (fs::is_regular_file(candidate, ec)) {
        out.ok = true;
        out.path = candidate;
        out.mime_type = mime_type_for_path(candidate);
        return out;
    }

    // SPA client route fallback only for paths without a file extension.
    if (rel.extension().empty()) {
        const fs::path idx = canon_root / "index.html";
        if (fs::is_regular_file(idx, ec)) {
            out.ok = true;
            out.path = idx;
            out.mime_type = mime_type_for_path(idx);
            return out;
        }
    }

    out.error = "asset not found";
    return out;
}

bool is_allowed_dev_url(std::string_view url) {
    if (url.empty()) return false;
    for (const char c : url) {
        if (static_cast<unsigned char>(c) <= 0x20 || c == '@' || c == '\\') {
            return false;
        }
    }
    constexpr std::string_view kHttp = "http://";
    if (url.rfind(kHttp, 0) != 0) return false;
    const std::string_view rest = url.substr(kHttp.size());
    const std::size_t slash = rest.find('/');
    const std::string_view authority =
        (slash == std::string_view::npos) ? rest : rest.substr(0, slash);

    std::string_view host = authority;
    std::string_view port;
    const std::size_t colon = authority.find(':');
    if (colon != std::string_view::npos) {
        host = authority.substr(0, colon);
        port = authority.substr(colon + 1);
        if (port.empty()) return false;
        unsigned long port_val = 0;
        for (const char c : port) {
            if (c < '0' || c > '9') return false;
            port_val = port_val * 10 + static_cast<unsigned long>(c - '0');
            if (port_val > 65535) return false;
        }
        if (port_val == 0) return false;
    }
    return host == "localhost" || host == "127.0.0.1";
}

fs::path default_settings_path() {
    const char* home = std::getenv("HOME");
    fs::path base = (home && *home) ? fs::path(home) : fs::temp_directory_path();
    return base / "Library" / "Application Support" / "Piper" / "settings.json";
}

std::string load_settings_json(const fs::path& path) {
    const platform::FileContents fc = platform::read_file_whole(path.string(), 2 * 1024 * 1024);
    if (!fc.ok()) return "{}";
    try {
        const json parsed = json::parse(fc.bytes);
        if (parsed.is_object()) return parsed.dump();
    } catch (...) {
    }
    return "{}";
}

bool save_settings_json(const fs::path& path, std::string_view json_str, std::string& error_out) {
    error_out.clear();
    try {
        const json parsed = json::parse(json_str);
        if (!parsed.is_object()) {
            error_out = "settings JSON must be an object";
            return false;
        }
    } catch (const std::exception& e) {
        error_out = std::string("invalid settings JSON: ") + e.what();
        return false;
    }

    std::error_code ec;
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
        if (ec) {
            error_out = "failed to create settings directory: " + ec.message();
            return false;
        }
        ::chmod(parent.c_str(), 0700);
    }

    const fs::path tmp =
        path.string() + ".tmp." + std::to_string(static_cast<long>(::getpid()));
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        error_out = std::string("failed to open temp settings file: ") + std::strerror(errno);
        return false;
    }
    std::size_t off = 0;
    while (off < json_str.size()) {
        const ssize_t n = ::write(fd, json_str.data() + off, json_str.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            error_out = std::string("failed writing settings: ") + std::strerror(errno);
            ::close(fd);
            ::unlink(tmp.c_str());
            return false;
        }
        off += static_cast<std::size_t>(n);
    }
    ::fsync(fd);
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        error_out = std::string("failed renaming settings: ") + std::strerror(errno);
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

std::string_view to_string(CheckpointKind kind) noexcept {
    switch (kind) {
        case CheckpointKind::Dense: return "dense";
        case CheckpointKind::Moe: return "moe";
        case CheckpointKind::Mtp: return "mtp";
        case CheckpointKind::Unknown: return "unknown";
    }
    return "unknown";
}

CheckpointKind classify_checkpoint(const fs::path& dir) {
    const auto cfg = read_config_json(dir);
    if (!cfg || !cfg->is_object()) return CheckpointKind::Unknown;
    const std::string root_mt = cfg->value("model_type", std::string{});
    if (root_mt == "qwen3_5_mtp") return CheckpointKind::Mtp;

    std::string mt = root_mt;
    if (cfg->contains("text_config") && (*cfg)["text_config"].is_object()) {
        const std::string text_mt = (*cfg)["text_config"].value("model_type", std::string{});
        if (!text_mt.empty()) mt = text_mt;
    }
    if (mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text") return CheckpointKind::Moe;
    if (mt == "qwen3_5" || mt == "qwen3_5_text") return CheckpointKind::Dense;
    return CheckpointKind::Unknown;
}

fs::path find_sibling_mtp(const fs::path& dense_dir) {
    std::error_code ec;
    const fs::path parent = dense_dir.parent_path();
    const std::string base = dense_dir.filename().string();
    if (parent.empty() || base.empty() || !fs::is_directory(parent, ec)) return {};

    // Fast path for standard naming conventions.
    auto replace_once = [](std::string s, std::string_view from, std::string_view to) {
        const std::size_t p = s.find(from);
        if (p != std::string::npos) s.replace(p, from.size(), to);
        return s;
    };
    const std::vector<std::string> guesses = {
        replace_once(base, "-MLX-4bit", "-MTP-4bit"),
        replace_once(base, "-MLX-", "-MTP-"),
        replace_once(base, "MLX", "MTP"),
    };
    for (const auto& g : guesses) {
        if (g == base) continue;
        const fs::path full = parent / g;
        if (is_usable_mtp(full)) return full;
    }

    fs::path best;
    std::size_t best_prefix = 0;
    for (const auto& entry : fs::directory_iterator(parent, ec)) {
        if (ec) break;
        if (!entry.is_directory(ec)) continue;
        const std::string name = entry.path().filename().string();
        if (name == base) continue;
        if (!is_usable_mtp(entry.path())) continue;
        const std::size_t pref = shared_prefix_len(name, base);
        if (best.empty() || pref > best_prefix) {
            best = entry.path();
            best_prefix = pref;
        }
    }
    return best;
}

std::string scan_models_json(const std::vector<fs::path>& roots) {
    json models = json::array();
    std::set<std::string> seen;
    std::error_code ec;

    auto consider_dir = [&](const fs::path& dir) {
        const fs::path canon = fs::weakly_canonical(dir, ec);
        if (ec || !fs::is_directory(canon, ec)) return;
        const std::string key = canon.string();
        if (!seen.insert(key).second) return;
        const CheckpointKind kind = classify_checkpoint(canon);
        if (kind == CheckpointKind::Unknown || kind == CheckpointKind::Mtp) return;
        fs::path mtp;
        if (kind == CheckpointKind::Dense) {
            mtp = find_sibling_mtp(canon);
        }
        models.push_back({
            {"path", key},
            {"name", canon.filename().string()},
            {"kind", std::string(to_string(kind))},
            {"mtp_path", mtp.empty() ? "" : mtp.string()},
            {"size_bytes", top_level_checkpoint_bytes(canon)},
        });
    };

    for (const auto& root : roots) {
        if (root.empty()) continue;
        consider_dir(root);
        if (fs::is_directory(root, ec)) {
            for (const auto& entry : fs::directory_iterator(root, ec)) {
                if (ec) break;
                if (entry.is_directory(ec)) consider_dir(entry.path());
            }
        }
    }
    return json({{"models", std::move(models)}}).dump();
}

std::string workspace_diff_json(const fs::path& workspace_root) {
    const platform::WorkspaceFs wfs(workspace_root.string());
    if (!wfs.valid()) {
        return json({{"is_git", false}, {"branch", ""}, {"files", json::array()}, {"error", wfs.error()}})
            .dump();
    }

    const CmdResult inside = run_git(wfs.root(), {"rev-parse", "--is-inside-work-tree"}, 4096);
    if (!inside.launched || inside.exit_code != 0 ||
        trim_right_ws(inside.stdout_bytes) != "true") {
        return json({{"is_git", false}, {"branch", ""}, {"files", json::array()}}).dump();
    }

    const CmdResult branch_res = run_git(wfs.root(), {"rev-parse", "--abbrev-ref", "HEAD"}, 4096);
    const std::string branch =
        (branch_res.launched && branch_res.exit_code == 0) ? trim_right_ws(branch_res.stdout_bytes) : "";

    const CmdResult status_res = run_git(wfs.root(), {"status", "--porcelain=v1", "-z"}, 2 * 1024 * 1024);
    std::map<std::string, std::string> status_map;
    std::vector<std::string> status_order;
    if (status_res.launched && status_res.exit_code == 0) {
        const std::string& raw = status_res.stdout_bytes;
        std::size_t pos = 0;
        while (pos < raw.size()) {
            const std::size_t nul = raw.find('\0', pos);
            const std::string rec =
                (nul == std::string::npos) ? raw.substr(pos) : raw.substr(pos, nul - pos);
            pos = (nul == std::string::npos) ? raw.size() : (nul + 1);
            if (rec.size() < 4) continue;
            const std::string code = rec.substr(0, 2);
            const std::string path = rec.substr(3);
            if (!code.empty() && (code[0] == 'R' || code[0] == 'C' || code[1] == 'R' || code[1] == 'C')) {
                const std::size_t nul2 = raw.find('\0', pos);
                pos = (nul2 == std::string::npos) ? raw.size() : (nul2 + 1);
            }
            if (status_map.find(path) == status_map.end()) status_order.push_back(path);
            status_map[path] = code;
        }
    }

    std::string diff_text;
    const CmdResult head_check = run_git(wfs.root(), {"rev-parse", "--verify", "HEAD"}, 1024);
    if (head_check.launched && head_check.exit_code == 0) {
        diff_text = run_git(wfs.root(), {"diff", "--no-ext-diff", "HEAD"}).stdout_bytes;
    } else {
        diff_text = run_git(wfs.root(), {"diff", "--no-ext-diff", "--cached"}).stdout_bytes;
        diff_text += run_git(wfs.root(), {"diff", "--no-ext-diff"}).stdout_bytes;
    }

    const std::vector<ParsedFileDiff> parsed = parse_unified_diff(diff_text);
    std::map<std::string, ParsedFileDiff> by_path;
    for (const auto& p : parsed) {
        auto it = by_path.find(p.path);
        if (it == by_path.end()) {
            by_path.emplace(p.path, p);
        } else {
            it->second.additions += p.additions;
            it->second.deletions += p.deletions;
            it->second.diff += p.diff;
        }
    }

    json files = json::array();
    std::set<std::string> emitted;
    constexpr std::size_t kMaxFiles = 200;

    for (const auto& path : status_order) {
        if (files.size() >= kMaxFiles) break;
        const std::string code = status_map[path];
        if (code == "??") {
            int adds = 0;
            const std::string preview = build_untracked_diff(wfs, path, adds);
            files.push_back({
                {"path", path},
                {"status", "untracked"},
                {"additions", adds},
                {"deletions", 0},
                {"diff", preview},
            });
            emitted.insert(path);
            continue;
        }
        std::string friendly = "modified";
        if (code.find('D') != std::string::npos) friendly = "deleted";
        else if (code.find('A') != std::string::npos) friendly = "added";
        else if (code.find('R') != std::string::npos) friendly = "renamed";

        const auto it = by_path.find(path);
        if (it != by_path.end()) {
            files.push_back({
                {"path", path},
                {"status", friendly},
                {"additions", it->second.additions},
                {"deletions", it->second.deletions},
                {"diff", it->second.diff},
            });
        } else {
            files.push_back({
                {"path", path},
                {"status", friendly},
                {"additions", 0},
                {"deletions", 0},
                {"diff", ""},
            });
        }
        emitted.insert(path);
    }

    for (const auto& [path, item] : by_path) {
        if (files.size() >= kMaxFiles) break;
        if (emitted.count(path) != 0) continue;
        files.push_back({
            {"path", path},
            {"status", "modified"},
            {"additions", item.additions},
            {"deletions", item.deletions},
            {"diff", item.diff},
        });
    }

    return json({{"is_git", true}, {"branch", branch}, {"files", std::move(files)}}).dump();
}

std::string workspace_files_json(const fs::path& workspace_root, int limit) {
    const platform::WorkspaceFs wfs(workspace_root.string());
    if (!wfs.valid()) {
        return json({{"files", json::array()}, {"truncated", false}, {"error", wfs.error()}}).dump();
    }
    const int cap = std::max(1, std::min(limit, 10000));
    std::vector<std::string> files;
    bool truncated = false;

    const CmdResult inside = run_git(wfs.root(), {"rev-parse", "--is-inside-work-tree"}, 4096);
    if (inside.launched && inside.exit_code == 0 &&
        trim_right_ws(inside.stdout_bytes) == "true") {
        const CmdResult listed = run_git(
            wfs.root(), {"ls-files", "-z", "--cached", "--others", "--exclude-standard"}, 4 * 1024 * 1024);
        if (listed.launched && listed.exit_code == 0) {
            const std::string& raw = listed.stdout_bytes;
            std::size_t pos = 0;
            while (pos < raw.size()) {
                const std::size_t nul = raw.find('\0', pos);
                const std::string item =
                    (nul == std::string::npos) ? raw.substr(pos) : raw.substr(pos, nul - pos);
                pos = (nul == std::string::npos) ? raw.size() : (nul + 1);
                if (item.empty()) continue;
                if (static_cast<int>(files.size()) >= cap) {
                    truncated = true;
                    break;
                }
                files.push_back(item);
            }
            return json({{"files", files}, {"truncated", truncated}}).dump();
        }
    }

    std::error_code ec;
    const fs::path root_path(wfs.root());
    for (fs::recursive_directory_iterator it(root_path, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_symlink(ec)) {
            if (it->is_directory(ec)) it.disable_recursion_pending();
            continue;
        }
        const std::string name = it->path().filename().string();
        if (it->is_directory(ec)) {
            if (is_ignored_walk_dir(name)) it.disable_recursion_pending();
            continue;
        }
        if (!it->is_regular_file(ec)) continue;
        if (static_cast<int>(files.size()) >= cap) {
            truncated = true;
            break;
        }
        const fs::path rel = fs::relative(it->path(), root_path, ec);
        if (!ec) files.push_back(rel.generic_string());
    }
    std::sort(files.begin(), files.end());
    return json({{"files", files}, {"truncated", truncated}}).dump();
}

std::string read_file_preview_json(const fs::path& workspace_root, std::string_view rel_path,
                                   std::size_t max_bytes) {
    const platform::WorkspaceFs wfs(workspace_root.string());
    if (!wfs.valid()) {
        return json({{"ok", false}, {"path", std::string(rel_path)}, {"error", wfs.error()}}).dump();
    }
    const platform::ContainedPath contained = wfs.contained_path(rel_path);
    if (!contained.ok()) {
        return json({{"ok", false}, {"path", std::string(rel_path)}, {"error", contained.error}}).dump();
    }

    const platform::OpenedFile opened = wfs.open_file_readonly(contained.relative, false);
    if (!opened.ok()) {
        return json({{"ok", false}, {"path", contained.relative}, {"error", opened.error}}).dump();
    }

    const std::size_t cap = std::max<std::size_t>(1, max_bytes);
    std::string raw;
    raw.resize(cap + 1);
    std::size_t total = 0;
    while (total < raw.size()) {
        const ssize_t n = ::read(opened.fd, raw.data() + total, raw.size() - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return json({{"ok", false}, {"path", contained.relative}, {"error", std::strerror(errno)}})
                .dump();
        }
        if (n == 0) break;
        total += static_cast<std::size_t>(n);
    }
    const bool truncated = total > cap;
    raw.resize(truncated ? cap : total);

    if (contains_nul(raw)) {
        return json({
                        {"ok", true},
                        {"path", contained.relative},
                        {"content", ""},
                        {"truncated", truncated},
                        {"binary", true},
                    })
            .dump();
    }

    if (truncated) {
        raw.resize(utf8_safe_prefix_len(raw));
    }
    return json({
                    {"ok", true},
                    {"path", contained.relative},
                    {"content", raw},
                    {"truncated", truncated},
                    {"binary", false},
                })
        .dump();
}

std::string save_image_attachment_json(const fs::path& workspace_root, std::string_view ext,
                                       std::string_view base64_data) {
    const std::string ext_l = lower_ascii(ext);
    if (ext_l != "png" && ext_l != "jpg" && ext_l != "jpeg" && ext_l != "webp") {
        return json({{"ok", false}, {"error", "unsupported image extension"}}).dump();
    }
    constexpr std::size_t kMaxEncoded = 28 * 1024 * 1024;
    constexpr std::size_t kMaxDecoded = 20 * 1024 * 1024;
    if (base64_data.empty() || base64_data.size() > kMaxEncoded) {
        return json({{"ok", false}, {"error", "image attachment missing or exceeds 20 MB limit"}}).dump();
    }

    std::string decoded;
    if (!platform::base64_decode(base64_data, decoded) || decoded.empty() ||
        decoded.size() > kMaxDecoded) {
        return json({{"ok", false}, {"error", "invalid base64 image payload"}}).dump();
    }

    const platform::WorkspaceFs wfs(workspace_root.string());
    if (!wfs.valid()) {
        return json({{"ok", false}, {"error", wfs.error()}}).dump();
    }

    static std::atomic<std::uint64_t> counter{0};
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const std::string rel = ".piper/attachments/img_" + std::to_string(now_ms) + "_" +
                            std::to_string(++counter) + "." + ext_l;
    const platform::WriteResult wr = wfs.write_file_atomic(rel, decoded, true);
    if (!wr.ok()) {
        return json({{"ok", false}, {"error", wr.error}}).dump();
    }
    return json({{"ok", true}, {"rel_path", rel}}).dump();
}

std::string escape_js_string_arg(std::string_view raw) {
    const std::string dumped = json(raw).dump();
    std::string out;
    out.reserve(dumped.size());
    for (std::size_t i = 0; i < dumped.size(); ++i) {
        // Escape UTF-8 U+2028 (E2 80 A8) and U+2029 (E2 80 A9) which terminate JS lines in older specs.
        if (i + 2 < dumped.size() && static_cast<unsigned char>(dumped[i]) == 0xE2u &&
            static_cast<unsigned char>(dumped[i + 1]) == 0x80u &&
            (static_cast<unsigned char>(dumped[i + 2]) == 0xA8u ||
             static_cast<unsigned char>(dumped[i + 2]) == 0xA9u)) {
            out += (static_cast<unsigned char>(dumped[i + 2]) == 0xA8u) ? "\\u2028" : "\\u2029";
            i += 2;
            continue;
        }
        out.push_back(dumped[i]);
    }
    return out;
}

} // namespace lmp::app
