#ifndef BLAST_RADIUS_HPP
#define BLAST_RADIUS_HPP

#include <cstdint>
#include <string_view>
#include <vector>
#include <string>
#include <cstddef>
#include <cctype>

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

inline bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

inline bool is_operator_start(char c) {
    return c == '|' || c == '&' || c == ';' || c == '<' || c == '>' || c == '(' || c == ')';
}

inline size_t match_operator(std::string_view s) {
    if (s.size() >= 3) {
        std::string_view op3 = s.substr(0, 3);
        if (op3 == "<<-") return 3;
    }
    if (s.size() >= 2) {
        std::string_view op2 = s.substr(0, 2);
        if (op2 == "||" || op2 == "&&" || op2 == ";;" || op2 == "<<" || op2 == ">>" || op2 == "<&" || op2 == ">&" || op2 == ">|") return 2;
    }
    if (s.size() >= 1) {
        char c = s[0];
        if (c == '|' || c == '&' || c == ';' || c == '<' || c == '>' || c == '(' || c == ')') return 1;
    }
    return 0;
}

struct Token {
    enum class Kind {
        Word,
        Operator,
        Eof,
        Error
    };
    Kind kind;
    std::string_view text;
    bool partially_parsed;
    std::string unquoted_text;
};

class Lexer {
    std::string_view input;
    size_t pos = 0;

    enum class Ctx {
        Normal,
        DoubleQuote,
        Backtick,
        DollarParen,
        SingleQuote
    };

    Ctx stack[256];
    size_t depth = 0;

    bool push(Ctx c) {
        if (depth >= 256) return false;
        stack[depth++] = c;
        return true;
    }

    bool pop(Ctx expected) {
        if (depth == 0) return false;
        if (stack[depth - 1] != expected) return false;
        depth--;
        return true;
    }

    Ctx current_ctx() const {
        if (depth == 0) return Ctx::Normal;
        return stack[depth - 1];
    }

public:
    Lexer(std::string_view c) : input(c) {}

    Token next_token() {
        while (pos < input.size() && is_space(input[pos]) && current_ctx() == Ctx::Normal) {
            pos++;
        }
        if (pos >= input.size()) {
            return depth > 0 ? Token{Token::Kind::Error, "", false, ""} : Token{Token::Kind::Eof, "", false, ""};
        }

        size_t start = pos;
        bool partially_parsed = false;
        bool is_word = false;
        std::string unquoted_text;

        while (pos < input.size()) {
            if (input[pos] == '\0') {
                return Token{Token::Kind::Error, "", false, ""};
            }

            if (input[pos] == '\\') {
                if (pos + 1 < input.size()) {
                    unquoted_text += input[pos + 1];
                    pos += 2;
                } else {
                    unquoted_text += input[pos];
                    pos++;
                }
                is_word = true;
                continue;
            }

            Ctx ctx = current_ctx();
            char c = input[pos];

            if (ctx == Ctx::SingleQuote) {
                if (c == '\'') {
                    if (!pop(Ctx::SingleQuote)) return Token{Token::Kind::Error, "", false, ""};
                    pos++;
                } else {
                    unquoted_text += c;
                    pos++;
                }
                is_word = true;
                continue;
            }

            if (ctx == Ctx::DoubleQuote) {
                if (c == '"') {
                    if (!pop(Ctx::DoubleQuote)) return Token{Token::Kind::Error, "", false, ""};
                    pos++;
                    is_word = true;
                    continue;
                }
                if (c == '$' && pos + 1 < input.size() && input[pos+1] == '(') {
                    if (!push(Ctx::DollarParen)) return Token{Token::Kind::Error, "", false, ""};
                    unquoted_text += "$(";
                    pos += 2;
                    is_word = true;
                    partially_parsed = true;
                    continue;
                }
                if (c == '`') {
                    if (!push(Ctx::Backtick)) return Token{Token::Kind::Error, "", false, ""};
                    unquoted_text += "`";
                    pos++;
                    is_word = true;
                    partially_parsed = true;
                    continue;
                }
                if (c == '$') {
                    partially_parsed = true;
                    if (pos + 1 < input.size() && input[pos+1] == '{') {
                        unquoted_text += "${";
                        pos += 2;
                        while (pos < input.size() && input[pos] != '}') {
                            unquoted_text += input[pos];
                            pos++;
                        }
                        if (pos < input.size()) {
                            unquoted_text += input[pos];
                            pos++;
                        }
                    } else {
                        unquoted_text += input[pos];
                        pos++;
                        while (pos < input.size() && (std::isalnum(input[pos]) || input[pos] == '_')) {
                            unquoted_text += input[pos];
                            pos++;
                        }
                    }
                    is_word = true;
                    continue;
                }
                unquoted_text += c;
                pos++;
                is_word = true;
                continue;
            }

            if (ctx == Ctx::Backtick) {
                if (c == '`') {
                    if (!pop(Ctx::Backtick)) return Token{Token::Kind::Error, "", false, ""};
                    unquoted_text += "`";
                    pos++;
                } else {
                    unquoted_text += c;
                    pos++;
                }
                is_word = true;
                continue;
            }

            if (ctx == Ctx::DollarParen) {
                if (c == ')') {
                    if (!pop(Ctx::DollarParen)) return Token{Token::Kind::Error, "", false, ""};
                    unquoted_text += ")";
                    pos++;
                    is_word = true;
                    continue;
                }
                if (c == '"') {
                    if (!push(Ctx::DoubleQuote)) return Token{Token::Kind::Error, "", false, ""};
                    unquoted_text += "\"";
                    pos++;
                    is_word = true;
                    continue;
                }
                if (c == '\'') {
                    if (!push(Ctx::SingleQuote)) return Token{Token::Kind::Error, "", false, ""};
                    unquoted_text += "'";
                    pos++;
                    is_word = true;
                    continue;
                }
                if (c == '$' && pos + 1 < input.size() && input[pos+1] == '(') {
                    if (!push(Ctx::DollarParen)) return Token{Token::Kind::Error, "", false, ""};
                    unquoted_text += "$(";
                    pos += 2;
                    is_word = true;
                    continue;
                }
                if (c == '`') {
                    if (!push(Ctx::Backtick)) return Token{Token::Kind::Error, "", false, ""};
                    unquoted_text += "`";
                    pos++;
                    is_word = true;
                    continue;
                }
                if (c == '$') {
                    partially_parsed = true;
                    unquoted_text += "$";
                    pos++;
                    is_word = true;
                    continue;
                }
                unquoted_text += c;
                pos++;
                is_word = true;
                continue;
            }

            // Normal context
            if (is_space(c)) {
                if (is_word) break;
                pos++;
                start = pos;
                continue;
            }

            if (is_operator_start(c)) {
                if (is_word) break;
                size_t op_len = match_operator(input.substr(pos));
                if (op_len > 0) {
                    std::string_view op = input.substr(pos, op_len);
                    pos += op_len;
                    if (op == "(") {
                        pos -= op_len;
                    } else {
                        return Token{Token::Kind::Operator, op, false, std::string(op)};
                    }
                    pos = start + op_len;
                    return Token{Token::Kind::Operator, op, false, std::string(op)};
                }
            }

            if (c == '\'') {
                if (!push(Ctx::SingleQuote)) return Token{Token::Kind::Error, "", false, ""};
                pos++;
                is_word = true;
                continue;
            }
            if (c == '"') {
                if (!push(Ctx::DoubleQuote)) return Token{Token::Kind::Error, "", false, ""};
                pos++;
                is_word = true;
                continue;
            }
            if (c == '$' && pos + 1 < input.size() && input[pos+1] == '(') {
                if (!push(Ctx::DollarParen)) return Token{Token::Kind::Error, "", false, ""};
                unquoted_text += "$(";
                pos += 2;
                is_word = true;
                partially_parsed = true;
                continue;
            }
            if (c == '`') {
                if (!push(Ctx::Backtick)) return Token{Token::Kind::Error, "", false, ""};
                unquoted_text += "`";
                pos++;
                is_word = true;
                partially_parsed = true;
                continue;
            }
            if (c == '$') {
                partially_parsed = true;
                if (pos + 1 < input.size() && input[pos+1] == '{') {
                    unquoted_text += "${";
                    pos += 2;
                    while (pos < input.size() && input[pos] != '}') {
                        unquoted_text += input[pos];
                        pos++;
                    }
                    if (pos < input.size()) {
                        unquoted_text += input[pos];
                        pos++;
                    }
                } else {
                    unquoted_text += input[pos];
                    pos++;
                    while (pos < input.size() && (std::isalnum(input[pos]) || input[pos] == '_')) {
                        unquoted_text += input[pos];
                        pos++;
                    }
                }
                is_word = true;
                continue;
            }

            unquoted_text += c;
            pos++;
            is_word = true;
        }

        if (depth > 0) {
             return Token{Token::Kind::Error, "", false, ""};
        }

        if (is_word) {
            return Token{Token::Kind::Word, input.substr(start, pos - start), partially_parsed, unquoted_text};
        }

        return Token{Token::Kind::Eof, "", false, ""};
    }
};

inline std::string normalize_path(std::string_view path, std::string_view cwd) {
    if (path.empty()) return "";
    std::string full_path;
    if (path[0] == '/') {
        full_path = std::string(path);
    } else {
        full_path = std::string(cwd) + "/" + std::string(path);
    }

    std::vector<std::string> parts;
    size_t start = 0;
    while (start < full_path.size()) {
        while (start < full_path.size() && full_path[start] == '/') start++;
        if (start == full_path.size()) break;
        size_t end = start;
        while (end < full_path.size() && full_path[end] != '/') end++;

        std::string_view part = std::string_view(full_path).substr(start, end - start);
        if (part == "..") {
            if (!parts.empty()) parts.pop_back();
        } else if (part != ".") {
            parts.push_back(std::string(part));
        }
        start = end;
    }

    std::string resolved = "/";
    for (size_t i = 0; i < parts.size(); ++i) {
        resolved += parts[i];
        if (i + 1 < parts.size()) resolved += "/";
    }
    return resolved;
}

inline bool is_path_outside_workspace(std::string_view path, std::string_view cwd, std::string_view workspace_root) {
    if (path.empty() || workspace_root.empty()) return true; // If undefined, assume worst
    std::string normalized_path = normalize_path(path, cwd);
    std::string normalized_ws = normalize_path(workspace_root, "/"); // Ensure absolute looking workspace

    if (normalized_path == normalized_ws) return false;

    std::string prefix = normalized_ws;
    if (prefix != "/" && prefix.back() != '/') {
        prefix += "/";
    }

    return !normalized_path.starts_with(prefix);
}

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v;
    Lexer lexer(ctx.command);
    std::vector<Token> tokens;

    while (true) {
        Token t = lexer.next_token();
        if (t.kind == Token::Kind::Error) {
            v.status = ParseStatus::Unparseable;
            return v;
        }
        if (t.kind == Token::Kind::Eof) {
            break;
        }
        if (t.partially_parsed) {
            v.status = ParseStatus::PartiallyParsed;
        }
        tokens.push_back(t);
    }

    struct SimpleCommand {
        std::vector<std::string> args;
        std::vector<std::string> redirects_out;
        std::vector<std::string> redirects_in;
    };

    std::vector<SimpleCommand> commands;
    SimpleCommand current_cmd;

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i].kind == Token::Kind::Operator) {
            if (tokens[i].text == ">" || tokens[i].text == ">>" || tokens[i].text == ">|") {
                if (tokens[i].text == ">" || tokens[i].text == ">|") {
                    v.capabilities.destroys_data = true;
                }
                if (i + 1 < tokens.size() && tokens[i+1].kind == Token::Kind::Word) {
                    current_cmd.redirects_out.push_back(tokens[i+1].unquoted_text);
                    i++;
                }
            } else if (tokens[i].text == "<") {
                if (i + 1 < tokens.size() && tokens[i+1].kind == Token::Kind::Word) {
                    current_cmd.redirects_in.push_back(tokens[i+1].unquoted_text);
                    i++;
                }
            } else if (tokens[i].text == "|" || tokens[i].text == ";" || tokens[i].text == "&&" || tokens[i].text == "||" || tokens[i].text == "(" || tokens[i].text == ")") {
                if (!current_cmd.args.empty() || !current_cmd.redirects_out.empty() || !current_cmd.redirects_in.empty()) {
                    commands.push_back(current_cmd);
                    current_cmd = SimpleCommand();
                }
            } else if (tokens[i].text == "&") {
                v.capabilities.spawns_unbounded_process = true; // Background task
                if (!current_cmd.args.empty() || !current_cmd.redirects_out.empty() || !current_cmd.redirects_in.empty()) {
                    commands.push_back(current_cmd);
                    current_cmd = SimpleCommand();
                }
            }
        } else {
            current_cmd.args.push_back(tokens[i].unquoted_text);
        }
    }
    if (!current_cmd.args.empty() || !current_cmd.redirects_out.empty() || !current_cmd.redirects_in.empty()) {
        commands.push_back(current_cmd);
    }

    auto check_path_write = [&](const std::string& path) {
        if (is_path_outside_workspace(path, ctx.cwd, ctx.workspace_root)) {
            v.capabilities.writes_outside_workspace = true;
        }
    };

    auto check_path_read = [&](const std::string& path) {
        if (is_path_outside_workspace(path, ctx.cwd, ctx.workspace_root)) {
            v.capabilities.reads_outside_workspace = true;
        }
    };

    for (const auto& cmd : commands) {
        for (const auto& path : cmd.redirects_out) {
            check_path_write(path);
        }
        for (const auto& path : cmd.redirects_in) {
            check_path_read(path);
        }

        if (cmd.args.empty()) continue;

        std::string prog = cmd.args[0];

        // Remove path if present (e.g. /usr/bin/curl -> curl)
        size_t last_slash = prog.find_last_of('/');
        if (last_slash != std::string::npos) {
            prog = prog.substr(last_slash + 1);
        }

        size_t arg_idx = 0;
        if (prog == "sudo" || prog == "su" || prog == "doas" || prog == "pkexec") {
            v.capabilities.escalates_privileges = true;
            if (cmd.args.size() > 1) {
                arg_idx = 1;
                prog = cmd.args[1];
                size_t lslash = prog.find_last_of('/');
                if (lslash != std::string::npos) prog = prog.substr(lslash + 1);
            }
        }
        if (prog == "xargs" || prog == "time" || prog == "env" || prog == "nohup") {
            v.capabilities.spawns_unbounded_process = true;
            if (cmd.args.size() > arg_idx + 1) {
                for (size_t k = arg_idx + 1; k < cmd.args.size(); ++k) {
                    if (cmd.args[k].empty() || cmd.args[k][0] == '-') continue;
                    prog = cmd.args[k];
                    size_t lslash = prog.find_last_of('/');
                    if (lslash != std::string::npos) prog = prog.substr(lslash + 1);
                    arg_idx = k;
                    break;
                }
            }
        }

        if (prog == "curl" || prog == "wget" || prog == "nc" || prog == "ping" || prog == "ssh" || prog == "ftp" || prog == "scp" || prog == "rsync") {
            v.capabilities.network_access = true;
        }

        if (prog == "rm" || prog == "shred" || prog == "dd" || prog == "wipe") {
            v.capabilities.destroys_data = true;
        }

        if (prog == "kill" || prog == "killall" || prog == "pkill") {
            v.capabilities.signals_foreign_process = true;
        }

        if (prog == "make" || prog == "npm" || prog == "yarn" || prog == "pnpm" || prog == "cargo" || prog == "sh" || prog == "bash" || prog == "zsh" || prog == "fish" || prog == "python" || prog == "python3" || prog == "node" || prog == "xargs") {
            v.capabilities.spawns_unbounded_process = true;
        }

        if (prog == "git") {
            if (cmd.args.size() > 1) {
                std::string sub = cmd.args[1];
                if (sub == "push" || sub == "fetch" || sub == "pull" || sub == "clone") {
                    v.capabilities.network_access = true;
                }
                if (sub == "reset" || sub == "rebase" || sub == "commit" || sub == "checkout" || sub == "clean") {
                    for (size_t i = 2; i < cmd.args.size(); ++i) {
                        if (sub == "reset" && cmd.args[i] == "--hard") v.capabilities.destroys_data = true;
                        if (sub == "clean" && cmd.args[i].find('f') != std::string::npos && cmd.args[i][0] == '-') v.capabilities.destroys_data = true; // -fdx
                        if (sub == "checkout" && cmd.args[i] == "--") {} // Just path parsing
                        if (cmd.args[i] == "--amend") v.capabilities.rewrites_vcs_history = true;
                    }
                    if (sub == "rebase") v.capabilities.rewrites_vcs_history = true;
                }
                if (sub == "push") {
                     for (size_t i = 2; i < cmd.args.size(); ++i) {
                        if (cmd.args[i] == "-f" || cmd.args[i] == "--force" || cmd.args[i] == "--force-with-lease") {
                             v.capabilities.rewrites_vcs_history = true;
                        }
                     }
                }
            }
        }

        // Simple heuristic for reads/writes by checking for files outside workspace in arguments
        for (size_t i = 1; i < cmd.args.size(); ++i) {
            const std::string& arg = cmd.args[i];
            if (!arg.empty() && arg[0] != '-') {
                if (is_path_outside_workspace(arg, ctx.cwd, ctx.workspace_root)) {
                    // It's very hard to know if it's read or write without knowing the command.
                    // If it's cp, mv, cat, etc.
                    if (prog == "cp" || prog == "mv" || prog == "install") {
                         if (i == cmd.args.size() - 1) { // Assume last is target
                             v.capabilities.writes_outside_workspace = true;
                         } else {
                             v.capabilities.reads_outside_workspace = true;
                         }
                    } else if (prog == "rm" || prog == "touch" || prog == "mkdir" || prog == "rmdir") {
                         v.capabilities.writes_outside_workspace = true;
                    } else if (prog == "cat" || prog == "grep" || prog == "head" || prog == "tail" || prog == "less" || prog == "more" || prog == "awk" || prog == "sed") {
                         v.capabilities.reads_outside_workspace = true;
                    }
                }
            }
        }
    }

    return v;
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
