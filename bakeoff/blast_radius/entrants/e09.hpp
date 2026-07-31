#ifndef BLAST_RADIUS_HPP
#define BLAST_RADIUS_HPP

#include <cstdint>
#include <string_view>
#include <string>
#include <vector>

namespace blast_radius {

enum class ParseStatus : std::uint8_t {
    Parsed = 0,
    PartiallyParsed = 1,
    Unparseable = 2,
};

struct Capabilities {
    bool writes_outside_workspace = false;
    bool reads_outside_workspace = false;
    bool destroys_data = false;
    bool rewrites_vcs_history = false;
    bool network_access = false;
    bool spawns_unbounded_process = false;
    bool signals_foreign_process = false;
    bool escalates_privileges = false;
};

struct Verdict {
    Capabilities capabilities{};
    ParseStatus status = ParseStatus::Parsed;
};

struct CommandContext {
    std::string_view command;
    std::string_view workspace_root;
    std::string_view cwd;
};

namespace detail {

// forward decl for handle_dollar/handle_backtick
inline void parse_command_sequence(std::string_view script, Verdict& v, int depth, const CommandContext& ctx) noexcept;

inline void update_status(Verdict& v, ParseStatus s) noexcept {
    if (static_cast<std::uint8_t>(s) > static_cast<std::uint8_t>(v.status)) {
        v.status = s;
    }
}

inline bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

inline bool is_redirect(std::string_view script, size_t pos) noexcept {
    if (pos >= script.size()) return false;
    if (script[pos] == '<' || script[pos] == '>') return true;
    if (script[pos] >= '0' && script[pos] <= '9') {
        size_t i = pos + 1;
        while (i < script.size() && script[i] >= '0' && script[i] <= '9') ++i;
        if (i < script.size() && (script[i] == '<' || script[i] == '>')) return true;
    }
    if (script[pos] == '&' && pos + 1 < script.size() && script[pos+1] == '>') return true;
    return false;
}

inline bool is_word_char(char c, std::string_view script, size_t pos) noexcept {
    if (is_space(c)) return false;
    if (c == ';' || c == '|' || c == '(' || c == ')') return false;
    if (is_redirect(script, pos)) return false;
    if (c == '&' && (pos + 1 == script.size() || script[pos+1] != '>')) return false;
    return true;
}

inline void skip_until_matching_paren(std::string_view script, size_t& pos, int& parens, Verdict& v) noexcept {
    while (pos < script.size() && parens > 0) {
        char c = script[pos];
        if (c == '\\') {
            pos += 2;
        } else if (c == '\'') {
            pos++;
            while (pos < script.size() && script[pos] != '\'') pos++;
            if (pos < script.size()) pos++;
        } else if (c == '"') {
            pos++;
            while (pos < script.size() && script[pos] != '"') {
                if (script[pos] == '\\') pos += 2;
                else pos++;
            }
            if (pos < script.size()) pos++;
        } else if (c == '(') {
            parens++;
            pos++;
        } else if (c == ')') {
            parens--;
            pos++;
        } else {
            pos++;
        }
    }
    if (parens > 0) update_status(v, ParseStatus::Unparseable);
}

inline void handle_dollar(std::string_view script, size_t& pos, std::string& word, Verdict& v, int depth, const CommandContext& ctx) noexcept {
    update_status(v, ParseStatus::PartiallyParsed);
    pos++;
    if (pos >= script.size()) return;
    if (script[pos] == '(') {
        pos++;
        size_t start = pos;
        int parens = 1;
        skip_until_matching_paren(script, pos, parens, v);
        if (parens == 0 && pos > start) {
            std::string_view subcmd = script.substr(start, pos - start - 1);
            parse_command_sequence(subcmd, v, depth + 1, ctx);
        }
    } else if (script[pos] == '{') {
        pos++;
        while (pos < script.size() && script[pos] != '}') pos++;
        if (pos < script.size()) pos++;
        else update_status(v, ParseStatus::Unparseable);
    } else {
        while (pos < script.size() && ((script[pos] >= 'a' && script[pos] <= 'z') ||
               (script[pos] >= 'A' && script[pos] <= 'Z') ||
               (script[pos] >= '0' && script[pos] <= '9') || script[pos] == '_')) {
            pos++;
        }
    }
}

inline void handle_backtick(std::string_view script, size_t& pos, std::string& word, Verdict& v, int depth, const CommandContext& ctx) noexcept {
    update_status(v, ParseStatus::PartiallyParsed);
    pos++;
    size_t start = pos;
    while (pos < script.size() && script[pos] != '`') {
        if (script[pos] == '\\') pos += 2;
        else pos++;
    }
    std::string_view subcmd = script.substr(start, pos - start);
    if (pos < script.size()) pos++;
    else update_status(v, ParseStatus::Unparseable);
    parse_command_sequence(subcmd, v, depth + 1, ctx);
}

inline std::string next_word(std::string_view script, size_t& pos, Verdict& v, int depth, const CommandContext& ctx) noexcept {
    std::string word;
    while (pos < script.size() && is_space(script[pos])) pos++;

    while (pos < script.size()) {
        char c = script[pos];
        if (!is_word_char(c, script, pos)) break;

        if (c == '\\') {
            pos++;
            if (pos < script.size()) word += script[pos++];
        } else if (c == '\'') {
            pos++;
            while (pos < script.size() && script[pos] != '\'') word += script[pos++];
            if (pos < script.size()) pos++;
            else update_status(v, ParseStatus::Unparseable);
        } else if (c == '"') {
            pos++;
            while (pos < script.size() && script[pos] != '"') {
                if (script[pos] == '\\') {
                    pos++;
                    if (pos < script.size()) word += script[pos++];
                } else if (script[pos] == '$') {
                    handle_dollar(script, pos, word, v, depth, ctx);
                } else if (script[pos] == '`') {
                    handle_backtick(script, pos, word, v, depth, ctx);
                } else {
                    word += script[pos++];
                }
            }
            if (pos < script.size()) pos++;
            else update_status(v, ParseStatus::Unparseable);
        } else if (c == '$') {
            handle_dollar(script, pos, word, v, depth, ctx);
        } else if (c == '`') {
            handle_backtick(script, pos, word, v, depth, ctx);
        } else {
            word += script[pos++];
        }
    }
    return word;
}

// Path Resolution
inline std::vector<std::string_view> split_path(std::string_view path) noexcept {
    std::vector<std::string_view> parts;
    size_t start = 0;
    while (start < path.size()) {
        size_t end = path.find('/', start);
        if (end == std::string_view::npos) {
            std::string_view part = path.substr(start);
            if (!part.empty() && part != ".") parts.push_back(part);
            break;
        }
        std::string_view part = path.substr(start, end - start);
        if (!part.empty() && part != ".") parts.push_back(part);
        start = end + 1;
    }
    return parts;
}

inline std::string normalize_path(std::string_view path) noexcept {
    std::vector<std::string_view> parts = split_path(path);
    std::vector<std::string_view> resolved;
    for (auto part : parts) {
        if (part == "..") {
            if (!resolved.empty() && resolved.back() != "..") {
                resolved.pop_back();
            } else {
                resolved.push_back(part);
            }
        } else {
            resolved.push_back(part);
        }
    }

    std::string res;
    if (path.starts_with('/')) res += '/';
    for (size_t i = 0; i < resolved.size(); ++i) {
        res += resolved[i];
        if (i + 1 < resolved.size()) res += '/';
    }
    if (res.empty()) res = ".";
    return res;
}

inline bool is_outside_workspace(std::string_view target_path, const CommandContext& ctx) noexcept {
    if (ctx.workspace_root.empty()) return false;

    std::string combined;
    if (target_path.starts_with('/')) {
        combined = std::string(target_path);
    } else {
        combined = std::string(ctx.cwd) + "/" + std::string(target_path);
    }

    std::string norm_target = normalize_path(combined);
    std::string norm_ws = normalize_path(ctx.workspace_root);

    // Absolute path checks
    if (norm_target.starts_with('/')) {
        if (!norm_ws.starts_with('/')) return true; // Can't resolve absolute against relative reliably without FS, assume outside
        if (norm_target.size() < norm_ws.size()) return true;
        if (norm_target.substr(0, norm_ws.size()) != norm_ws) return true;
        if (norm_target.size() > norm_ws.size() && norm_target[norm_ws.size()] != '/' && norm_ws.back() != '/') return true;
        return false;
    }

    // Both relative
    if (norm_target.starts_with("..")) return true;
    return false;
}

struct RedirectOp {
    std::string op;
    std::string target;
};

inline void evaluate_command(const std::vector<std::string>& words, const std::vector<RedirectOp>& redirects, Verdict& v, int depth, const CommandContext& ctx) noexcept {
    // Process redirects first, even if words is empty (standalone redirect)
    for (const auto& r : redirects) {
        bool is_append = (r.op.find(">>") != std::string::npos);
        if (r.op.find('>') != std::string::npos && !is_append) {
            v.capabilities.destroys_data = true;
        }
        if (!r.target.empty()) {
            if (r.op.find('>') != std::string::npos && is_outside_workspace(r.target, ctx)) v.capabilities.writes_outside_workspace = true;
            if (r.op.find('<') != std::string::npos && is_outside_workspace(r.target, ctx)) v.capabilities.reads_outside_workspace = true;
        }
    }

    if (words.empty()) return;

    size_t cmd_idx = 0;
    while (cmd_idx < words.size()) {
        if (words[cmd_idx].find('=') != std::string::npos) {
            cmd_idx++;
        } else {
            break;
        }
    }

    if (cmd_idx >= words.size()) return;

    const std::string& cmd = words[cmd_idx];

    // File operations
    if (cmd == "rm" || cmd == "shred" || cmd == "dd" || cmd == "mv") {
        v.capabilities.destroys_data = true;
        for (size_t i = 1; i < words.size(); ++i) {
            if (!words[i].starts_with('-')) {
                if (is_outside_workspace(words[i], ctx)) v.capabilities.writes_outside_workspace = true;
            }
        }
    } else if (cmd == "cp" || cmd == "touch" || cmd == "mkdir" || cmd == "rmdir") {
        for (size_t i = 1; i < words.size(); ++i) {
            if (!words[i].starts_with('-')) {
                if (is_outside_workspace(words[i], ctx)) v.capabilities.writes_outside_workspace = true;
            }
        }
    } else if (cmd == "cat" || cmd == "grep" || cmd == "head" || cmd == "tail" || cmd == "less" || cmd == "more") {
        for (size_t i = 1; i < words.size(); ++i) {
            if (!words[i].starts_with('-')) {
                if (is_outside_workspace(words[i], ctx)) v.capabilities.reads_outside_workspace = true;
            }
        }
    }

    // Network
    else if (cmd == "curl" || cmd == "wget" || cmd == "nc" || cmd == "ssh" || cmd == "ping" || cmd == "nmap" || cmd == "ftp") {
        v.capabilities.network_access = true;
    }

    // VCS
    else if (cmd == "git") {
        for (size_t i = 1; i < words.size(); ++i) {
            if (words[i] == "reset") {
                if (i + 1 < words.size() && words[i+1] == "--hard") v.capabilities.rewrites_vcs_history = true;
            } else if (words[i] == "push") {
                for (size_t j = i + 1; j < words.size(); ++j) {
                    if (words[j] == "-f" || words[j] == "--force" || words[j] == "--force-with-lease") {
                        v.capabilities.rewrites_vcs_history = true;
                    }
                }
            } else if (words[i] == "rebase") {
                v.capabilities.rewrites_vcs_history = true;
            } else if (words[i] == "clean") {
                bool is_dry_run = false;
                for (size_t j = i + 1; j < words.size(); ++j) {
                    if (words[j] == "-n" || words[j] == "--dry-run") is_dry_run = true;
                }
                if (!is_dry_run) v.capabilities.destroys_data = true;
            } else if (words[i] == "fetch" || words[i] == "pull" || words[i] == "clone") {
                v.capabilities.network_access = true;
            }
        }
    }

    // Signals
    else if (cmd == "kill" || cmd == "pkill" || cmd == "killall") {
        v.capabilities.signals_foreign_process = true;
    }

    // Privileges
    else if (cmd == "sudo" || cmd == "su" || cmd == "doas" || cmd == "chown" || cmd == "chmod") {
        if (cmd == "sudo" || cmd == "su" || cmd == "doas") v.capabilities.escalates_privileges = true;
        if (cmd == "sudo" || cmd == "doas") {
            if (words.size() > 1) {
                std::vector<std::string> sub_words(words.begin() + 1, words.end());
                evaluate_command(sub_words, {}, v, depth + 1, ctx);
            }
        }
    }

    // Unbounded / Build tools
    else if (cmd == "make" || cmd == "npm" || cmd == "yarn" || cmd == "pnpm" || cmd == "cargo" || cmd == "gradle" || cmd == "mvn" || cmd == "xargs") {
        v.capabilities.spawns_unbounded_process = true;
        v.capabilities.network_access = true; // Assume build tools might fetch
        v.capabilities.writes_outside_workspace = true; // Build tools often write cache/global
        v.capabilities.reads_outside_workspace = true;
    }

    // Indirection
    else if (cmd == "sh" || cmd == "bash" || cmd == "zsh" || cmd == "csh" || cmd == "ksh") {
        for (size_t i = 1; i < words.size(); ++i) {
            if (words[i] == "-c" && i + 1 < words.size()) {
                parse_command_sequence(words[i+1], v, depth + 1, ctx);
                break;
            }
        }
    } else if (cmd == "eval") {
        for (size_t i = 1; i < words.size(); ++i) {
            parse_command_sequence(words[i], v, depth + 1, ctx);
        }
    } else if (cmd == "python" || cmd == "node" || cmd == "ruby" || cmd == "perl" || cmd == "php") {
        v.capabilities.spawns_unbounded_process = true;
    }
}

inline void parse_command_sequence(std::string_view script, Verdict& v, int depth, const CommandContext& ctx) noexcept {
    if (depth > 20) { // Limit depth to prevent stack overflow on adversarial inputs
        update_status(v, ParseStatus::PartiallyParsed);
        return;
    }

    size_t pos = 0;
    std::vector<std::string> current_words;
    std::vector<RedirectOp> current_redirects;

    while (pos < script.size()) {
        while (pos < script.size() && is_space(script[pos])) pos++;
        if (pos >= script.size()) break;

        char c = script[pos];
        if (is_redirect(script, pos)) {
            size_t start = pos;
            while (pos < script.size() && (script[pos] == '<' || script[pos] == '>' || script[pos] == '&' || (script[pos] >= '0' && script[pos] <= '9'))) pos++;
            std::string op = std::string(script.substr(start, pos - start));
            while (pos < script.size() && is_space(script[pos])) pos++;
            std::string target = next_word(script, pos, v, depth, ctx);
            current_redirects.push_back({op, target});
        } else if (c == ';' || c == '|' || c == '&') {
            evaluate_command(current_words, current_redirects, v, depth, ctx);
            current_words.clear();
            current_redirects.clear();
            pos++;
            if (pos < script.size() && (script[pos] == '|' || script[pos] == '&')) pos++;
        } else if (c == '(') {
            pos++;
            int parens = 1;
            size_t start = pos;
            skip_until_matching_paren(script, pos, parens, v);
            if (parens == 0 && pos > start) {
                parse_command_sequence(script.substr(start, pos - start - 1), v, depth + 1, ctx);
            }
        } else if (c == ')') {
            pos++;
            update_status(v, ParseStatus::Unparseable);
        } else {
            std::string word = next_word(script, pos, v, depth, ctx);
            if (!word.empty()) current_words.push_back(word);
        }
    }
    evaluate_command(current_words, current_redirects, v, depth, ctx);
}

} // namespace detail

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v;
    detail::parse_command_sequence(ctx.command, v, 0, ctx);
    return v;
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
