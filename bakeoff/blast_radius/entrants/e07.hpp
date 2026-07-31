#ifndef BLAST_RADIUS_HPP
#define BLAST_RADIUS_HPP

#include <cstdint>
#include <string_view>
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

namespace detail {

struct Word {
    char buf[256];
    size_t len = 0;

    void append(char c, ParseStatus& status) noexcept {
        if (len < sizeof(buf)) {
            buf[len++] = c;
        } else {
            status = ParseStatus::Unparseable;
        }
    }
    std::string_view view() const noexcept {
        return {buf, len}; // len is guaranteed <= 256
    }
    void clear() noexcept {
        len = 0;
    }
};

enum class TokenType {
    Eof,
    Word,
    Pipe,       // |
    Or,         // ||
    And,        // &&
    Ampersand,  // &
    Semicolon,  // ;
    Less,       // <
    Greater,    // >
    GreaterGreater, // >>
    LParen,     // (
    RParen,     // )
    Newline     // \n
};

struct Token {
    TokenType type;
    Word word;
};

static void parse_script(std::string_view script, std::string_view ws, std::string_view cwd, Capabilities& caps, ParseStatus& status, int depth) noexcept;

class Lexer {
    std::string_view cmd;
    size_t i = 0;
    std::string_view ws;
    std::string_view cwd;
    Capabilities& caps;
    ParseStatus& status;
    int depth;

public:
    Lexer(std::string_view c, std::string_view w, std::string_view cw, Capabilities& ca, ParseStatus& st, int d) noexcept
        : cmd(c), ws(w), cwd(cw), caps(ca), status(st), depth(d) {}

    Token next() noexcept {
        Token tok;
        tok.type = TokenType::Eof;

        while (i < cmd.size() && (cmd[i] == ' ' || cmd[i] == '\t' || cmd[i] == '\r')) {
            i++;
        }
        if (i >= cmd.size()) return tok;

        char c = cmd[i];
        if (c == '|') {
            i++;
            if (i < cmd.size() && cmd[i] == '|') {
                i++; tok.type = TokenType::Or; return tok;
            }
            tok.type = TokenType::Pipe; return tok;
        }
        if (c == '&') {
            i++;
            if (i < cmd.size() && cmd[i] == '&') {
                i++; tok.type = TokenType::And; return tok;
            }
            tok.type = TokenType::Ampersand; return tok;
        }
        if (c == ';') {
            i++; tok.type = TokenType::Semicolon; return tok;
        }
        if (c == '<') {
            i++; tok.type = TokenType::Less; return tok;
        }
        if (c == '>') {
            i++;
            if (i < cmd.size() && cmd[i] == '>') {
                i++; tok.type = TokenType::GreaterGreater; return tok;
            }
            tok.type = TokenType::Greater; return tok;
        }
        if (c == '(') {
            i++; tok.type = TokenType::LParen; return tok;
        }
        if (c == ')') {
            i++; tok.type = TokenType::RParen; return tok;
        }
        if (c == '\n') {
            i++; tok.type = TokenType::Newline; return tok;
        }

        tok.type = TokenType::Word;
        parse_word(tok.word);
        return tok;
    }

private:
    void parse_word(Word& word) noexcept {
        while (i < cmd.size()) {
            char c = cmd[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                c == '|' || c == '&' || c == ';' || c == '<' || c == '>' ||
                c == '(' || c == ')') {
                break;
            }

            if (c == '\\') {
                i++;
                if (i < cmd.size()) {
                    word.append(cmd[i], status);
                    i++;
                } else {
                    status = ParseStatus::Unparseable;
                }
            } else if (c == '\'') {
                i++; // skip '
                while (i < cmd.size() && cmd[i] != '\'') {
                    word.append(cmd[i], status);
                    i++;
                }
                if (i < cmd.size()) i++; // skip closing '
                else status = ParseStatus::Unparseable;
            } else if (c == '"') {
                i++; // skip "
                while (i < cmd.size() && cmd[i] != '"') {
                    if (cmd[i] == '\\') {
                        i++;
                        if (i < cmd.size()) {
                            if (cmd[i] == '$' || cmd[i] == '`' || cmd[i] == '"' || cmd[i] == '\\' || cmd[i] == '\n') {
                                word.append(cmd[i], status);
                            } else {
                                word.append('\\', status);
                                word.append(cmd[i], status);
                            }
                            i++;
                        }
                    } else if (cmd[i] == '$') {
                        handle_substitution();
                    } else if (cmd[i] == '`') {
                        handle_backticks();
                    } else {
                        word.append(cmd[i], status);
                        i++;
                    }
                }
                if (i < cmd.size()) i++; // skip closing "
                else status = ParseStatus::Unparseable;
            } else if (c == '$') {
                handle_substitution();
            } else if (c == '`') {
                handle_backticks();
            } else {
                word.append(c, status);
                i++;
            }
        }
    }

    void handle_substitution() noexcept {
        if (status == ParseStatus::Unparseable) return;
        if (status == ParseStatus::Parsed) status = ParseStatus::PartiallyParsed;

        i++; // skip $
        if (i >= cmd.size()) return;

        if (cmd[i] == '{') {
            i++;
            while (i < cmd.size() && cmd[i] != '}') {
                if (cmd[i] == '\\') i += 2;
                else if (cmd[i] == '\'') {
                    i++; while(i < cmd.size() && cmd[i] != '\'') i++; if(i < cmd.size()) i++;
                } else if (cmd[i] == '"') {
                    i++; while(i < cmd.size() && cmd[i] != '"') {
                        if (cmd[i] == '\\') i += 2; else i++;
                    }
                    if(i < cmd.size()) i++;
                } else if (cmd[i] == '$') {
                    i++;
                } else {
                    i++;
                }
            }
            if (i < cmd.size()) i++;
            else status = ParseStatus::Unparseable;
        } else if (cmd[i] == '(') {
            i++;
            size_t start = i;
            skip_balanced('(', ')');
            if (status != ParseStatus::Unparseable) {
                size_t end = i - 1; // before the closing )
                if (depth < 32) {
                    parse_script(cmd.substr(start, end - start), ws, cwd, caps, status, depth + 1);
                } else {
                    status = ParseStatus::Unparseable;
                }
            }
        } else {
            // $VAR
            while (i < cmd.size() && (std::isalnum(static_cast<unsigned char>(cmd[i])) || cmd[i] == '_')) {
                i++;
            }
        }
    }

    void handle_backticks() noexcept {
        if (status == ParseStatus::Unparseable) return;
        if (status == ParseStatus::Parsed) status = ParseStatus::PartiallyParsed;

        i++; // skip `
        size_t start = i;
        while (i < cmd.size() && cmd[i] != '`') {
            if (cmd[i] == '\\') {
                i += 2;
            } else {
                i++;
            }
        }
        if (status != ParseStatus::Unparseable) {
            size_t end = i;
            if (i < cmd.size()) {
                i++;
                if (depth < 32) {
                    parse_script(cmd.substr(start, end - start), ws, cwd, caps, status, depth + 1);
                } else {
                    status = ParseStatus::Unparseable;
                }
            } else {
                status = ParseStatus::Unparseable;
            }
        }
    }

    void skip_balanced(char open, char close) noexcept {
        int d = 1;
        while (i < cmd.size() && d > 0) {
            char c = cmd[i];
            if (c == '\\') {
                i += 2;
                continue;
            } else if (c == '\'') {
                i++;
                while (i < cmd.size() && cmd[i] != '\'') i++;
                if (i < cmd.size()) i++;
                else status = ParseStatus::Unparseable;
                continue;
            } else if (c == '"') {
                i++;
                while (i < cmd.size() && cmd[i] != '"') {
                    if (cmd[i] == '\\') i += 2;
                    else i++;
                }
                if (i < cmd.size()) i++;
                else status = ParseStatus::Unparseable;
                continue;
            } else if (c == '`') {
                i++;
                while (i < cmd.size() && cmd[i] != '`') {
                    if (cmd[i] == '\\') i += 2;
                    else i++;
                }
                if (i < cmd.size()) i++;
                else status = ParseStatus::Unparseable;
                continue;
            }

            if (c == open) d++;
            else if (c == close) d--;
            i++;
        }
        if (d > 0) status = ParseStatus::Unparseable;
    }
};

static bool is_path_escaping(std::string_view path, std::string_view ws, std::string_view cwd) noexcept {
    if (path.empty()) return false;

    // Handle absolute paths
    if (path[0] == '/') {
        if (!ws.empty() && ws[0] == '/') {
            // Need to check if path starts with ws
            if (path.starts_with(ws)) {
                auto tail = path.substr(ws.size());
                if (tail.empty() || tail[0] == '/') {
                    // Safe so far, need to check if trailing part resolves outside
                } else {
                    return true; // Not in workspace
                }
            } else {
                return true; // Absolute path outside workspace
            }
        } else {
            return true; // Assuming any absolute path escapes if no valid ws
        }
    }

    // Simplified model: count depth.
    // If it's absolute, starting point is ws (depth = 0).
    // If it's relative, starting point is cwd. Let's calculate cwd depth from ws.

    int depth = 0;
    size_t i = 0;

    if (path[0] != '/') {
        if (!ws.empty() && cwd.starts_with(ws)) {
            auto cwd_tail = cwd.substr(ws.size());
            size_t ci = 0;
            while (ci < cwd_tail.size()) {
                while (ci < cwd_tail.size() && cwd_tail[ci] == '/') ci++;
                if (ci >= cwd_tail.size()) break;
                size_t c_start = ci;
                while (ci < cwd_tail.size() && cwd_tail[ci] != '/') ci++;
                std::string_view c_seg = cwd_tail.substr(c_start, ci - c_start);
                if (c_seg == "..") {
                    if (depth > 0) depth--;
                } else if (c_seg != ".") {
                    depth++;
                }
            }
        } else if (!ws.empty() && !cwd.starts_with(ws)) {
             // If cwd is not in ws, any relative path could be considered escaping,
             // but let's assume it escaped.
             return true;
        }
    } else {
        // Absolute path inside ws, start scanning after ws
        i = ws.size();
    }

    // Path segment parsing
    bool escaped = false;
    while (i < path.size()) {
        while (i < path.size() && path[i] == '/') i++; // skip slashes
        if (i >= path.size()) break;

        size_t start = i;
        while (i < path.size() && path[i] != '/') i++;
        std::string_view seg = path.substr(start, i - start);

        if (seg == ".") {
            continue;
        } else if (seg == "..") {
            depth--;
            if (depth < 0) escaped = true;
        } else {
            depth++;
        }
    }

    return escaped;
}

static void evaluate_command(const Word* args, size_t argc, std::string_view ws, std::string_view cwd, Capabilities& caps, ParseStatus& status, int current_depth) noexcept {
    if (argc == 0) return;
    std::string_view cmd = args[0].view();

    if (cmd == "rm" || cmd == "unlink") {
        caps.destroys_data = true;
    } else if (cmd == "git") {
        for (size_t i = 1; i < argc; ++i) {
            std::string_view arg = args[i].view();
            if (arg == "reset" || arg == "rebase" || arg == "commit" || arg == "push" || arg == "clean") {
                caps.rewrites_vcs_history = true;
            }
            if (arg == "clean") {
                for (size_t j = i + 1; j < argc; ++j) {
                    if (args[j].view().starts_with('-')) {
                        if (args[j].view().find('f') != std::string_view::npos || args[j].view().find('x') != std::string_view::npos) {
                            caps.destroys_data = true;
                        }
                    }
                }
            }
            if (arg == "reset") {
                for (size_t j = i + 1; j < argc; ++j) {
                    if (args[j].view() == "--hard") {
                        caps.destroys_data = true;
                    }
                }
            }
        }
    } else if (cmd == "curl" || cmd == "wget" || cmd == "nc" || cmd == "ssh" || cmd == "ping") {
        caps.network_access = true;
    } else if (cmd == "sudo" || cmd == "su") {
        caps.escalates_privileges = true;
        if (argc > 1) {
            evaluate_command(args + 1, argc - 1, ws, cwd, caps, status, current_depth);
        }
    } else if (cmd == "kill" || cmd == "killall" || cmd == "pkill") {
        caps.signals_foreign_process = true;
    } else if (cmd == "yes" || cmd == "cat" && argc == 1) {
        // Not perfectly accurate but captures unbounded process idea
        caps.spawns_unbounded_process = true;
    } else if (cmd == "sh" || cmd == "bash" || cmd == "zsh") {
        for (size_t i = 1; i < argc; ++i) {
            if (args[i].view() == "-c" && i + 1 < argc) {
                if (current_depth < 32) {
                    parse_script(args[i+1].view(), ws, cwd, caps, status, current_depth + 1);
                } else {
                    status = ParseStatus::Unparseable;
                }
            }
        }
    } else if (cmd == "xargs") {
        if (argc > 1) {
             evaluate_command(args + 1, argc - 1, ws, cwd, caps, status, current_depth);
        }
    } else if (cmd == "make" || cmd == "npm" || cmd == "yarn") {
        // Build tools often do many things, mark as unbounded/spawns for safety or just note they are indirect
        // Depends on what we want to flag, partially parsed might be better if they run arbitrary scripts
        status = ParseStatus::PartiallyParsed;
    }

    // Check paths for escaping
    for (size_t i = 1; i < argc; ++i) {
        if (args[i].view().starts_with("-")) continue;
        if (is_path_escaping(args[i].view(), ws, cwd)) {
            // Simplistic: assume it's read/write escaping depending on command.
            if (cmd == "cat" || cmd == "grep") {
                caps.reads_outside_workspace = true;
            }
            if (cmd == "cp") {
                // If it's the last argument of cp, it's a write.
                // Otherwise it's a read.
                if (i == argc - 1 && argc > 2) {
                    caps.writes_outside_workspace = true;
                } else {
                    caps.reads_outside_workspace = true;
                }
            }
            if (cmd == "mv" || cmd == "rm" || cmd == "touch") {
                caps.writes_outside_workspace = true;
            }
        }
    }
}

static void parse_script(std::string_view script, std::string_view ws, std::string_view cwd, Capabilities& caps, ParseStatus& status, int depth) noexcept {
    Lexer lexer(script, ws, cwd, caps, status, depth);

    Word args[32];
    size_t argc = 0;

    auto process_command = [&]() {
        if (argc > 0) {
            evaluate_command(args, argc, ws, cwd, caps, status, depth);
            argc = 0;
        }
    };

    while (status != ParseStatus::Unparseable) {
        Token tok = lexer.next();
        if (tok.type == TokenType::Eof) {
            process_command();
            break;
        }

        if (tok.type == TokenType::Word) {
            if (argc < 32) {
                args[argc++] = tok.word;
            } else {
                status = ParseStatus::Unparseable;
            }
        } else if (tok.type == TokenType::Greater || tok.type == TokenType::GreaterGreater) {
            caps.destroys_data = true;

            // Advance to target
            Token target = lexer.next();
            if (target.type == TokenType::Word) {
                if (is_path_escaping(target.word.view(), ws, cwd)) {
                    caps.writes_outside_workspace = true;
                }
            } else {
                status = ParseStatus::Unparseable;
            }
        } else if (tok.type == TokenType::Less) {
            Token target = lexer.next();
            if (target.type == TokenType::Word) {
                if (is_path_escaping(target.word.view(), ws, cwd)) {
                    caps.reads_outside_workspace = true;
                }
            } else {
                status = ParseStatus::Unparseable;
            }
        } else if (tok.type == TokenType::Pipe || tok.type == TokenType::And ||
                   tok.type == TokenType::Or || tok.type == TokenType::Semicolon ||
                   tok.type == TokenType::Newline || tok.type == TokenType::Ampersand) {
            if (tok.type == TokenType::Ampersand) {
                caps.spawns_unbounded_process = true; // Background process
            }
            process_command();
        } else if (tok.type == TokenType::LParen) {
            // Handled subshell inline or rely on lexer substituting
        } else if (tok.type == TokenType::RParen) {
            // Syntax error if not handled
        }
    }
}

} // namespace detail

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v;
    detail::parse_script(ctx.command, ctx.workspace_root, ctx.cwd, v.capabilities, v.status, 0);
    return v;
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
