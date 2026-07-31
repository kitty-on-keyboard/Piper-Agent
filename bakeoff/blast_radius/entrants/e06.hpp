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

inline void merge_capabilities(Capabilities& a, const Capabilities& b) noexcept {
    a.writes_outside_workspace |= b.writes_outside_workspace;
    a.reads_outside_workspace |= b.reads_outside_workspace;
    a.destroys_data |= b.destroys_data;
    a.rewrites_vcs_history |= b.rewrites_vcs_history;
    a.network_access |= b.network_access;
    a.spawns_unbounded_process |= b.spawns_unbounded_process;
    a.signals_foreign_process |= b.signals_foreign_process;
    a.escalates_privileges |= b.escalates_privileges;
}

inline bool is_path_outside_workspace(std::string_view path, std::string_view cwd, std::string_view workspace_root) noexcept {
    if (path.empty()) return false;

    std::string full_path;
    if (path[0] == '/') {
        full_path = std::string(path);
    } else if (path[0] == '~') {
        return true;
    } else {
        full_path = std::string(cwd) + "/" + std::string(path);
    }

    std::vector<std::string> parts;
    size_t start = 0;
    while (start < full_path.size()) {
        size_t end = full_path.find('/', start);
        if (end == std::string::npos) end = full_path.size();

        std::string part = full_path.substr(start, end - start);
        if (part == "..") {
            if (!parts.empty()) parts.pop_back();
        } else if (!part.empty() && part != ".") {
            parts.push_back(part);
        }

        start = end + 1;
    }

    std::string normalized = "/";
    for (size_t i = 0; i < parts.size(); ++i) {
        normalized += parts[i];
        if (i < parts.size() - 1) normalized += "/";
    }

    std::string ws_root = std::string(workspace_root);
    if (ws_root.empty() || ws_root[0] != '/') {
        ws_root = "/" + ws_root;
    }
    while (ws_root.size() > 1 && ws_root.back() == '/') {
        ws_root.pop_back();
    }
    if (ws_root.empty()) ws_root = "/";

    if (ws_root == "/") return false;

    if (normalized == ws_root) return false;
    if (normalized.size() > ws_root.size() &&
        normalized.compare(0, ws_root.size(), ws_root) == 0 &&
        normalized[ws_root.size()] == '/') {
        return false;
    }

    return true;
}

enum class TokenType {
    Word,
    RedirIn,
    RedirOut,
    RedirAppend,
    Sep,
    Eof
};

struct Token {
    TokenType type;
    std::string value;
};

struct SimpleCommand {
    std::vector<std::string> words;
    std::vector<std::pair<TokenType, std::string>> redirects;
};

// Forward declaration
[[nodiscard]] Verdict classify_internal(std::string_view command, const CommandContext& ctx, int recursion_depth) noexcept;

inline void analyze_command(const SimpleCommand& cmd, Verdict& verdict, const CommandContext& ctx, int recursion_depth) noexcept {
    if (cmd.words.empty() && cmd.redirects.empty()) return;

    for (const auto& redir : cmd.redirects) {
        if (redir.first == TokenType::RedirOut) {
            verdict.capabilities.destroys_data = true;
        }

        bool outside = is_path_outside_workspace(redir.second, ctx.cwd, ctx.workspace_root);
        if (outside) {
            if (redir.first == TokenType::RedirIn) verdict.capabilities.reads_outside_workspace = true;
            else verdict.capabilities.writes_outside_workspace = true;
        }
    }

    if (cmd.words.empty()) return;

    size_t cmd_idx = 0;
    while (cmd_idx < cmd.words.size() && cmd.words[cmd_idx].find('=') != std::string::npos && cmd.words[cmd_idx].find('=') > 0) {
        cmd_idx++;
    }
    if (cmd_idx >= cmd.words.size()) return;

    auto process_command = [&](size_t idx, auto& process_ref) -> void {
        if (idx >= cmd.words.size()) return;
        std::string name = cmd.words[idx];

        if (name == "sudo" || name == "su" || name == "doas" || name == "pkexec") {
            verdict.capabilities.escalates_privileges = true;
            for (size_t i = idx + 1; i < cmd.words.size(); ++i) {
                if (cmd.words[i].empty() || cmd.words[i][0] == '-') continue;
                if (cmd.words[i].find('=') != std::string::npos && cmd.words[i].find('=') > 0) continue;
                process_ref(i, process_ref);
            }
            return;
        }

        if (name == "env" || name == "xargs" || name == "time" || name == "nice" || name == "nohup" || name == "stdbuf") {
            if (name == "nohup") verdict.capabilities.spawns_unbounded_process = true;
            for (size_t i = idx + 1; i < cmd.words.size(); ++i) {
                if (cmd.words[i].empty() || cmd.words[i][0] == '-') continue;
                if (cmd.words[i].find('=') != std::string::npos && cmd.words[i].find('=') > 0) continue;
                process_ref(i, process_ref);
            }
            return;
        }

        if (name == "sh" || name == "bash" || name == "zsh" || name == "csh" || name == "dash" || name == "ash" || name == "tcsh") {
            for (size_t i = idx + 1; i < cmd.words.size(); ++i) {
                if (cmd.words[i] == "-c" && i + 1 < cmd.words.size()) {
                    std::string_view inner = cmd.words[i + 1];
                    Verdict inner_v = classify_internal(inner, ctx, recursion_depth + 1);
                    merge_capabilities(verdict.capabilities, inner_v.capabilities);
                    if (inner_v.status == ParseStatus::Unparseable) verdict.status = ParseStatus::Unparseable;
                    else if (inner_v.status == ParseStatus::PartiallyParsed && verdict.status == ParseStatus::Parsed) verdict.status = ParseStatus::PartiallyParsed;
                    break;
                }
            }
            return;
        }

        if (name == "make" || name == "npm" || name == "yarn" || name == "pnpm" || name == "cargo") {
            if (verdict.status == ParseStatus::Parsed) verdict.status = ParseStatus::PartiallyParsed;
            return;
        }

        if (name == "rm" || name == "shred" || name == "wipe") {
            verdict.capabilities.destroys_data = true;
        }
        else if (name == "git") {
            if (idx + 1 < cmd.words.size()) {
                std::string sub = cmd.words[idx + 1];
                if (sub == "reset") {
                    for (size_t i = idx + 2; i < cmd.words.size(); ++i) {
                        if (cmd.words[i] == "--hard") {
                            verdict.capabilities.destroys_data = true;
                            verdict.capabilities.rewrites_vcs_history = true;
                        }
                    }
                } else if (sub == "push") {
                    for (size_t i = idx + 2; i < cmd.words.size(); ++i) {
                        if (cmd.words[i] == "-f" || cmd.words[i] == "--force" || cmd.words[i] == "--force-with-lease") {
                            verdict.capabilities.rewrites_vcs_history = true;
                        }
                    }
                } else if (sub == "rebase" || sub == "filter-branch" || sub == "commit") {
                    if (sub == "commit") {
                        for (size_t i = idx + 2; i < cmd.words.size(); ++i) {
                            if (cmd.words[i] == "--amend") verdict.capabilities.rewrites_vcs_history = true;
                        }
                    } else {
                        verdict.capabilities.rewrites_vcs_history = true;
                    }
                } else if (sub == "clean") {
                    for (size_t i = idx + 2; i < cmd.words.size(); ++i) {
                        if (cmd.words[i].find('f') != std::string::npos && !cmd.words[i].empty() && cmd.words[i][0] == '-') {
                            verdict.capabilities.destroys_data = true;
                        }
                    }
                }
            }
        }
        else if (name == "curl" || name == "wget" || name == "nc" || name == "ping" || name == "ssh" || name == "scp" || name == "ftp" || name == "nmap" || name == "netcat") {
            verdict.capabilities.network_access = true;
        }
        else if (name == "yes" || name == "daemon") {
            verdict.capabilities.spawns_unbounded_process = true;
        }
        else if (name == "kill" || name == "pkill" || name == "killall") {
            verdict.capabilities.signals_foreign_process = true;
        }
        else if (name == "cp" || name == "mv" || name == "cat" || name == "echo" || name == "tee" || name == "touch" || name == "mkdir") {
            for (size_t i = idx + 1; i < cmd.words.size(); ++i) {
                if (cmd.words[i].empty() || cmd.words[i][0] != '-') {
                    if (is_path_outside_workspace(cmd.words[i], ctx.cwd, ctx.workspace_root)) {
                        if (name == "cp" && i == cmd.words.size() - 1) verdict.capabilities.writes_outside_workspace = true;
                        else if (name == "mv" && i == cmd.words.size() - 1) verdict.capabilities.writes_outside_workspace = true;
                        else if (name == "tee" || name == "touch" || name == "mkdir") verdict.capabilities.writes_outside_workspace = true;
                        else verdict.capabilities.reads_outside_workspace = true;
                    }
                }
            }
        } else {
            for (size_t i = idx + 1; i < cmd.words.size(); ++i) {
                if (!cmd.words[i].empty() && cmd.words[i][0] != '-' && (cmd.words[i].find('/') != std::string::npos || cmd.words[i] == "..")) {
                    if (is_path_outside_workspace(cmd.words[i], ctx.cwd, ctx.workspace_root)) {
                        verdict.capabilities.reads_outside_workspace = true;
                    }
                }
            }
        }
    };

    process_command(cmd_idx, process_command);
}

inline Verdict classify_internal(std::string_view str, const CommandContext& ctx, int recursion_depth) noexcept {
    Verdict verdict;
    if (recursion_depth > 20) {
        verdict.status = ParseStatus::Unparseable;
        return verdict;
    }

    size_t pos = 0;

    auto is_space = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\0';
    };

    auto extract_dollar_paren = [&]() -> std::string_view {
        pos++;
        int depth = 1;
        size_t start = pos;
        while (pos < str.size() && depth > 0) {
            if (str[pos] == '\\') { pos += 2; continue; }
            if (str[pos] == '\'') {
                pos++;
                while (pos < str.size() && str[pos] != '\'') pos++;
                if (pos < str.size()) pos++;
                continue;
            }
            if (str[pos] == '"') {
                pos++;
                while (pos < str.size() && str[pos] != '"') {
                    if (str[pos] == '\\') pos += 2;
                    else pos++;
                }
                if (pos < str.size()) pos++;
                continue;
            }
            if (str[pos] == '(') depth++;
            else if (str[pos] == ')') depth--;

            if (depth > 0) pos++;
        }
        std::string_view inner = str.substr(start, pos > start ? pos - start : 0);
        if (pos < str.size()) pos++;
        return inner;
    };

    auto handle_dollar = [&]() {
        if (verdict.status == ParseStatus::Parsed) verdict.status = ParseStatus::PartiallyParsed;
        pos++;
        if (pos >= str.size()) return;
        if (str[pos] == '(') {
            std::string_view inner = extract_dollar_paren();
            Verdict inner_v = classify_internal(inner, ctx, recursion_depth + 1);
            merge_capabilities(verdict.capabilities, inner_v.capabilities);
            if (inner_v.status == ParseStatus::Unparseable) verdict.status = ParseStatus::Unparseable;
        } else if (str[pos] == '{') {
            pos++;
            while (pos < str.size() && str[pos] != '}') pos++;
            if (pos < str.size()) pos++;
        } else {
            while (pos < str.size() && ((str[pos] >= 'a' && str[pos] <= 'z') || (str[pos] >= 'A' && str[pos] <= 'Z') || (str[pos] >= '0' && str[pos] <= '9') || str[pos] == '_')) {
                pos++;
            }
        }
    };

    auto handle_backtick = [&]() {
        if (verdict.status == ParseStatus::Parsed) verdict.status = ParseStatus::PartiallyParsed;
        pos++;
        size_t start = pos;
        while (pos < str.size() && str[pos] != '`') {
            if (str[pos] == '\\') pos += 2;
            else pos++;
        }
        std::string_view inner = str.substr(start, pos > start ? pos - start : 0);
        if (pos < str.size()) pos++;
        Verdict inner_v = classify_internal(inner, ctx, recursion_depth + 1);
        merge_capabilities(verdict.capabilities, inner_v.capabilities);
        if (inner_v.status == ParseStatus::Unparseable) verdict.status = ParseStatus::Unparseable;
    };

    auto next_token = [&]() -> Token {
        while (pos < str.size() && is_space(str[pos])) pos++;
        if (pos >= str.size()) return {TokenType::Eof, ""};

        char c = str[pos];
        if (c == ';' || c == '(' || c == ')') { pos++; return {TokenType::Sep, ""}; }
        if (c == '&') {
            pos++;
            if (pos < str.size() && str[pos] == '&') { pos++; return {TokenType::Sep, ""}; }
            if (pos < str.size() && str[pos] == '>') { pos++; return {TokenType::RedirOut, ""}; }
            return {TokenType::Sep, ""};
        }
        if (c == '|') {
            pos++;
            if (pos < str.size() && str[pos] == '|') { pos++; return {TokenType::Sep, ""}; }
            if (pos < str.size() && str[pos] == '&') { pos++; return {TokenType::Sep, ""}; }
            return {TokenType::Sep, ""};
        }
        if (c == '<') {
            pos++;
            if (pos < str.size() && str[pos] == '<') {
                pos++;
                if (pos < str.size() && str[pos] == '-') pos++;
                else if (pos < str.size() && str[pos] == '<') pos++;
            }
            return {TokenType::RedirIn, ""};
        }
        if (c == '>') {
            pos++;
            if (pos < str.size() && str[pos] == '>') { pos++; return {TokenType::RedirAppend, ""}; }
            if (pos < str.size() && str[pos] == '&') { pos++; return {TokenType::RedirOut, ""}; }
            return {TokenType::RedirOut, ""};
        }

        std::string word;
        auto append = [&](char ch) {
            if (word.size() < 16384) word += ch;
        };

        while (pos < str.size()) {
            c = str[pos];
            if (is_space(c)) break;
            if (c == ';' || c == '&' || c == '|' || c == '<' || c == '>' || c == '(' || c == ')') break;

            if (c == '\\') {
                pos++;
                if (pos < str.size()) append(str[pos++]);
            } else if (c == '\'') {
                pos++;
                while (pos < str.size() && str[pos] != '\'') {
                    append(str[pos++]);
                }
                if (pos < str.size()) pos++;
            } else if (c == '"') {
                pos++;
                while (pos < str.size() && str[pos] != '"') {
                    if (str[pos] == '\\') {
                        pos++;
                        if (pos < str.size()) {
                            char esc = str[pos];
                            if (esc == '$' || esc == '`' || esc == '"' || esc == '\\' || esc == '\n') {
                                append(str[pos++]);
                            } else {
                                append('\\');
                                append(str[pos++]);
                            }
                        }
                    } else if (str[pos] == '$') {
                        handle_dollar();
                    } else if (str[pos] == '`') {
                        handle_backtick();
                    } else {
                        append(str[pos++]);
                    }
                }
                if (pos < str.size()) pos++;
            } else if (c == '$') {
                handle_dollar();
            } else if (c == '`') {
                handle_backtick();
            } else {
                append(str[pos++]);
            }
        }
        return {TokenType::Word, word};
    };

    SimpleCommand cmd;
    while (true) {
        Token t = next_token();
        if (t.type == TokenType::Word) {
            cmd.words.push_back(t.value);
        } else if (t.type == TokenType::RedirIn || t.type == TokenType::RedirOut || t.type == TokenType::RedirAppend) {
            Token target = next_token();
            if (target.type == TokenType::Word) {
                cmd.redirects.push_back({t.type, target.value});
            } else if (target.type != TokenType::Eof) {
                // If it's a separator instead of word, maybe syntax error, skip
                if (verdict.status == ParseStatus::Parsed) verdict.status = ParseStatus::PartiallyParsed;
            }
        } else {
            analyze_command(cmd, verdict, ctx, recursion_depth);
            cmd = SimpleCommand();

            if (t.type == TokenType::Eof) break;
        }
    }

    return verdict;
}

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    try {
        return classify_internal(ctx.command, ctx, 0);
    } catch (...) {
        Verdict v;
        v.status = ParseStatus::Unparseable;
        return v;
    }
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
