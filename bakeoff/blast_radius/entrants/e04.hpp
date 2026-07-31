#ifndef BLAST_RADIUS_HPP
#define BLAST_RADIUS_HPP

#include <cstdint>
#include <string_view>
#include <vector>
#include <string>

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

enum class TokenType : std::uint8_t {
    Word,
    Semi,
    Pipe,
    LogicalAnd,
    LogicalOr,
    Background,
    RedirectOut,
    RedirectAppend,
    RedirectIn,
};

struct Token {
    TokenType type;
    std::string value;
};

enum class Ctx : std::uint8_t {
    Normal,
    DoubleQuote,
    SingleQuote,
    Backtick,
    VarSubst,
    Paren
};

inline std::vector<std::string> resolve_path(std::string_view path, std::string_view cwd) {
    std::vector<std::string> parts;
    bool is_absolute = (!path.empty() && path[0] == '/');

    auto add_parts = [&](std::string_view p) {
        size_t start = 0;
        while (start < p.size()) {
            size_t end = p.find('/', start);
            if (end == std::string_view::npos) end = p.size();
            if (end > start) {
                std::string_view part = p.substr(start, end - start);
                if (part == "..") {
                    if (!parts.empty() && parts.back() != "..") parts.pop_back();
                    else parts.push_back("..");
                } else if (part != ".") {
                    parts.push_back(std::string(part));
                }
            }
            start = end + 1;
        }
    };

    if (is_absolute) {
        add_parts(path);
    } else {
        add_parts(cwd);
        add_parts(path);
    }
    return parts;
}

inline bool is_outside_workspace(std::string_view path, std::string_view workspace_root, std::string_view cwd) {
    auto path_parts = resolve_path(path, cwd);
    auto ws_parts = resolve_path(workspace_root, "");

    if (ws_parts.size() > path_parts.size()) return true;
    for (size_t i = 0; i < ws_parts.size(); ++i) {
        if (ws_parts[i] != path_parts[i]) return true;
    }

    if (!path_parts.empty() && path_parts[0] == "..") return true;

    return false;
}

inline std::vector<Token> tokenize(std::string_view input, ParseStatus& status) {
    std::vector<Token> tokens;
    std::vector<Ctx> stack;
    stack.push_back(Ctx::Normal);
    std::string current_word;

    auto emit_word = [&]() {
        if (!current_word.empty()) {
            tokens.push_back({TokenType::Word, current_word});
            current_word.clear();
        }
    };

    auto emit_op = [&](TokenType type) {
        emit_word();
        tokens.push_back({type, ""});
    };

    size_t i = 0;
    while (i < input.size()) {
        char c = input[i];
        Ctx ctx = stack.back();

        if (c == '\0') {
            i++;
            continue;
        }

        if (c == '\\') {
            if (i + 1 < input.size()) {
                if (ctx != Ctx::SingleQuote) {
                    current_word += input[i + 1];
                    i += 2;
                    continue;
                }
            } else {
                i++;
                continue;
            }
        }

        if (ctx == Ctx::VarSubst) {
            if (c == '}') {
                stack.pop_back();
            } else {
                current_word += c;
            }
            i++;
            continue;
        }

        if (c == '$') {
            if (status == ParseStatus::Parsed) status = ParseStatus::PartiallyParsed;
            if (i + 1 < input.size()) {
                if (input[i+1] == '(') {
                    emit_op(TokenType::Semi);
                    if (stack.size() < 256) stack.push_back(Ctx::Paren);
                    else status = ParseStatus::Unparseable;
                    i += 2;
                    continue;
                } else if (input[i+1] == '{') {
                    if (stack.size() < 256) stack.push_back(Ctx::VarSubst);
                    else status = ParseStatus::Unparseable;
                    current_word += "${";
                    i += 2;
                    continue;
                } else {
                    current_word += '$';
                    i++;
                    continue;
                }
            } else {
                current_word += '$';
                i++;
                continue;
            }
        }

        if (c == '`') {
            if (status == ParseStatus::Parsed) status = ParseStatus::PartiallyParsed;
            if (ctx == Ctx::Backtick) {
                stack.pop_back();
                emit_op(TokenType::Semi);
            } else {
                emit_op(TokenType::Semi);
                if (stack.size() < 256) stack.push_back(Ctx::Backtick);
                else status = ParseStatus::Unparseable;
            }
            i++;
            continue;
        }

        if (c == '\'') {
            if (ctx == Ctx::SingleQuote) {
                stack.pop_back();
            } else if (ctx != Ctx::DoubleQuote && ctx != Ctx::SingleQuote) {
                if (stack.size() < 256) stack.push_back(Ctx::SingleQuote);
                else status = ParseStatus::Unparseable;
            } else {
                current_word += c;
            }
            i++;
            continue;
        }

        if (c == '"') {
            if (ctx == Ctx::DoubleQuote) {
                stack.pop_back();
            } else if (ctx != Ctx::SingleQuote) {
                if (stack.size() < 256) stack.push_back(Ctx::DoubleQuote);
                else status = ParseStatus::Unparseable;
            } else {
                current_word += c;
            }
            i++;
            continue;
        }

        if (ctx == Ctx::DoubleQuote) {
            current_word += c;
            i++;
            continue;
        }

        if (c == '(') {
            emit_op(TokenType::Semi);
            if (stack.size() < 256) stack.push_back(Ctx::Paren);
            else status = ParseStatus::Unparseable;
            i++;
            continue;
        }

        if (c == ')') {
            if (ctx == Ctx::Paren) {
                stack.pop_back();
                emit_op(TokenType::Semi);
            } else {
                emit_op(TokenType::Semi);
            }
            i++;
            continue;
        }

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (c == '\n') {
                emit_op(TokenType::Semi);
            } else {
                emit_word();
            }
            i++;
            continue;
        }

        if (c == ';') { emit_op(TokenType::Semi); i++; continue; }
        if (c == '|') {
            if (i + 1 < input.size() && input[i+1] == '|') {
                emit_op(TokenType::LogicalOr); i += 2;
            } else {
                emit_op(TokenType::Pipe); i++;
            }
            continue;
        }
        if (c == '&') {
            if (i + 1 < input.size() && input[i+1] == '&') {
                emit_op(TokenType::LogicalAnd); i += 2;
            } else {
                emit_op(TokenType::Background); i++;
            }
            continue;
        }
        if (c == '>') {
            if (i + 1 < input.size() && input[i+1] == '>') {
                emit_op(TokenType::RedirectAppend); i += 2;
            } else {
                emit_op(TokenType::RedirectOut); i++;
            }
            continue;
        }
        if (c == '<') { emit_op(TokenType::RedirectIn); i++; continue; }

        current_word += c;
        i++;
    }

    emit_word();

    if (stack.size() > 1 && status == ParseStatus::Parsed) {
        status = ParseStatus::PartiallyParsed;
    }

    return tokens;
}

inline void analyze_tokens(const std::vector<Token>& tokens, std::string_view ws_root, std::string_view cwd, Verdict& v) {
    std::vector<std::string> args;

    auto process_command = [&]() {
        if (args.empty()) return;

        size_t cmd_idx = 0;
        while (cmd_idx < args.size()) {
            const std::string& cmd = args[cmd_idx];
            if (cmd == "sudo" || cmd == "su" || cmd == "doas") {
                v.capabilities.escalates_privileges = true;
                cmd_idx++;
            } else if (cmd == "env" || cmd == "stdbuf" || cmd == "nohup" || cmd == "time" || cmd == "exec") {
                cmd_idx++;
                while (cmd_idx < args.size() && args[cmd_idx].find('=') != std::string::npos) {
                    cmd_idx++;
                }
            } else {
                break;
            }
        }
        if (cmd_idx >= args.size()) return;
        std::string cmd = args[cmd_idx];

        if (cmd == "curl" || cmd == "wget" || cmd == "ssh" || cmd == "nc" || cmd == "ping" || cmd == "ftp" || cmd == "scp") {
            v.capabilities.network_access = true;
        }
        if (cmd == "rm" || cmd == "dd" || cmd == "shred" || cmd == "wipe") {
            v.capabilities.destroys_data = true;
        }
        if (cmd == "kill" || cmd == "killall" || cmd == "pkill" || cmd == "killall5") {
            v.capabilities.signals_foreign_process = true;
        }

        if (cmd == "make" || cmd == "npm" || cmd == "xargs" || cmd == "yarn" || cmd == "pnpm" || cmd == "docker" || cmd == "cargo" || cmd == "bazel" || cmd == "ninja") {
            v.capabilities.spawns_unbounded_process = true;
        }

        if (cmd == "git") {
            bool has_clean = false, has_reset = false, has_push = false, has_rebase = false, has_commit = false;
            bool force = false, hard = false, amend = false;
            for (size_t i = cmd_idx + 1; i < args.size(); ++i) {
                if (args[i] == "clean") has_clean = true;
                if (args[i] == "reset") has_reset = true;
                if (args[i] == "push") has_push = true;
                if (args[i] == "rebase") has_rebase = true;
                if (args[i] == "commit") has_commit = true;

                if (args[i].length() > 1 && args[i][0] == '-' && args[i][1] != '-') {
                    if (args[i].find('f') != std::string::npos) force = true;
                } else if (args[i] == "--force") force = true;

                if (args[i] == "--hard") hard = true;
                if (args[i] == "--amend") amend = true;
            }
            if ((has_clean && force) || (has_reset && hard)) v.capabilities.destroys_data = true;
            if ((has_push && force) || has_rebase || (has_commit && amend)) v.capabilities.rewrites_vcs_history = true;
        }

        if (cmd == "sh" || cmd == "bash" || cmd == "zsh" || cmd == "ksh" || cmd == "csh") {
            v.capabilities.spawns_unbounded_process = true;
            for (size_t i = cmd_idx + 1; i < args.size(); ++i) {
                if (args[i] == "-c" && i + 1 < args.size()) {
                    ParseStatus temp = ParseStatus::Parsed;
                    auto sub_tokens = tokenize(args[i+1], temp);
                    if (temp > v.status) v.status = temp;
                    analyze_tokens(sub_tokens, ws_root, cwd, v);
                    break;
                }
            }
        }

        for (size_t i = cmd_idx + 1; i < args.size(); ++i) {
            if (!args[i].empty() && args[i][0] != '-') {
                if (is_outside_workspace(args[i], ws_root, cwd)) {
                    if (cmd == "touch" || cmd == "mkdir" || cmd == "rm" || cmd == "rmdir") {
                        v.capabilities.writes_outside_workspace = true;
                    } else if (cmd == "cp" || cmd == "mv" || cmd == "ln") {
                        if (i == args.size() - 1) v.capabilities.writes_outside_workspace = true;
                        else v.capabilities.reads_outside_workspace = true;
                    } else {
                        v.capabilities.reads_outside_workspace = true;
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i].type == TokenType::Word) {
            args.push_back(tokens[i].value);
        } else if (tokens[i].type == TokenType::RedirectOut || tokens[i].type == TokenType::RedirectAppend || tokens[i].type == TokenType::RedirectIn) {
            if (tokens[i].type == TokenType::RedirectOut) {
                v.capabilities.destroys_data = true;
            }
            if (i + 1 < tokens.size() && tokens[i+1].type == TokenType::Word) {
                std::string target = tokens[i+1].value;
                if (is_outside_workspace(target, ws_root, cwd)) {
                    if (tokens[i].type == TokenType::RedirectIn) v.capabilities.reads_outside_workspace = true;
                    else v.capabilities.writes_outside_workspace = true;
                }
                i++;
            }
        } else {
            process_command();
            args.clear();
        }
    }
    process_command();
}

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v;
    try {
        ParseStatus status = ParseStatus::Parsed;
        auto tokens = tokenize(ctx.command, status);
        v.status = status;
        analyze_tokens(tokens, ctx.workspace_root, ctx.cwd, v);
    } catch (...) {
        v.status = ParseStatus::Unparseable;
    }
    return v;
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
