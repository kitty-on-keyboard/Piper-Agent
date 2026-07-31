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

enum class TokenType {
    Word,
    Operator,
    SubshellStart,
    SubshellEnd,
    Eof,
    Error
};

struct Token {
    TokenType type;
    std::string value;
};

class Lexer {
    std::string_view input;
    size_t pos = 0;
    ParseStatus status = ParseStatus::Parsed;

public:
    Lexer(std::string_view input) : input(input) {}

    ParseStatus get_status() const { return status; }
    void set_status(ParseStatus s) { if (s > status) status = s; }

    Token next_token() {
        while (pos < input.size() && (input[pos] == ' ' || input[pos] == '\t' || input[pos] == '\n' || input[pos] == '\r')) {
            pos++;
        }

        if (pos >= input.size()) {
            return {TokenType::Eof, ""};
        }

        if (input[pos] == '\0') {
            set_status(ParseStatus::Unparseable);
            return {TokenType::Error, ""};
        }

        char c = input[pos];
        if (c == '(') {
            pos++;
            return {TokenType::SubshellStart, "("};
        }
        if (c == ')') {
            pos++;
            return {TokenType::SubshellEnd, ")"};
        }
        if (c == ';' || c == '|' || c == '&' || c == '<' || c == '>') {
            std::string op;
            op += c;
            pos++;
            if (pos < input.size()) {
                char nc = input[pos];
                if ((c == '&' && nc == '&') || (c == '|' && nc == '|') || (c == '>' && nc == '>')) {
                    op += nc;
                    pos++;
                } else if (c == '>' && nc == '&') {
                    op += nc;
                    pos++;
                }
            }
            return {TokenType::Operator, op};
        }

        std::string word;
        bool in_single = false;
        bool in_double = false;

        while (pos < input.size()) {
            c = input[pos];
            if (c == '\0') {
                set_status(ParseStatus::Unparseable);
                break;
            }

            if (!in_single && !in_double) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                    c == ';' || c == '|' || c == '&' || c == '<' || c == '>' ||
                    c == '(' || c == ')') {
                    break;
                }
            }

            if (c == '\\') {
                if (in_single) {
                    word += c;
                    pos++;
                } else {
                    pos++;
                    if (pos < input.size()) {
                        word += input[pos];
                        pos++;
                    }
                }
            } else if (c == '\'') {
                if (!in_double) {
                    in_single = !in_single;
                } else {
                    word += c;
                }
                pos++;
            } else if (c == '"') {
                if (!in_single) {
                    in_double = !in_double;
                } else {
                    word += c;
                }
                pos++;
            } else if (c == '$' && !in_single) {
                set_status(ParseStatus::PartiallyParsed);
                pos++;
                if (pos < input.size() && input[pos] == '(') {
                    pos++;
                    int depth = 1;
                    while (pos < input.size() && depth > 0) {
                        if (input[pos] == '\\') {
                            pos += 2;
                        } else if (input[pos] == '(') {
                            depth++;
                            pos++;
                        } else if (input[pos] == ')') {
                            depth--;
                            pos++;
                        } else {
                            pos++;
                        }
                    }
                    if (depth > 0) set_status(ParseStatus::Unparseable);
                } else if (pos < input.size() && input[pos] == '{') {
                    pos++;
                    while (pos < input.size() && input[pos] != '}') {
                        if (input[pos] == '\\') pos += 2;
                        else pos++;
                    }
                    if (pos < input.size() && input[pos] == '}') pos++;
                    else set_status(ParseStatus::Unparseable);
                } else {
                    while (pos < input.size() && (std::isalnum(input[pos]) || input[pos] == '_')) {
                        pos++;
                    }
                }
                word += "$VAR";
            } else if (c == '`') {
                set_status(ParseStatus::PartiallyParsed);
                pos++;
                while (pos < input.size() && input[pos] != '`') {
                    if (input[pos] == '\\') pos += 2;
                    else pos++;
                }
                if (pos < input.size() && input[pos] == '`') pos++;
                else set_status(ParseStatus::Unparseable);
                word += "`CMD`";
            } else {
                word += c;
                pos++;
            }
        }

        if (in_single || in_double) {
            set_status(ParseStatus::Unparseable);
        }

        if (word.empty() && status == ParseStatus::Unparseable) {
            return {TokenType::Error, ""};
        }

        if (word.size() > 1024 * 1024) {
            set_status(ParseStatus::Unparseable);
        }

        return {TokenType::Word, word};
    }
};

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v;
    try {
        Lexer lexer(ctx.command);
        std::vector<Token> tokens;
        while (true) {
            Token t = lexer.next_token();
            if (t.type == TokenType::Eof || t.type == TokenType::Error) break;
            tokens.push_back(t);
            if (tokens.size() > 100000) {
                v.status = ParseStatus::Unparseable;
                return v;
            }
        }

        if (lexer.get_status() > v.status) {
            v.status = lexer.get_status();
        }

        auto normalize_path = [](std::string_view p) -> std::string {
            std::vector<std::string> parts;
            size_t start = 0;
            while (start < p.size()) {
                size_t end = p.find('/', start);
                if (end == std::string_view::npos) end = p.size();
                std::string_view part = p.substr(start, end - start);
                if (part == ".." && !parts.empty() && parts.back() != "..") {
                    parts.pop_back();
                } else if (part != "." && !part.empty()) {
                    parts.push_back(std::string(part));
                } else if (part == ".." && parts.empty()) {
                    parts.push_back("..");
                }
                start = end + 1;
            }
            std::string res = p.empty() ? "" : (p[0] == '/' ? "/" : "");
            for (size_t i = 0; i < parts.size(); ++i) {
                if (i > 0 && res != "/") res += "/";
                res += parts[i];
            }
            if (res.empty()) res = ".";
            return res;
        };

        auto is_outside = [&](std::string_view path) {
            std::string full_path;
            if (!path.empty() && path[0] == '/') {
                full_path = std::string(path);
            } else {
                full_path = std::string(ctx.cwd) + "/" + std::string(path);
            }
            std::string norm = normalize_path(full_path);
            std::string norm_root = normalize_path(ctx.workspace_root);
            if (norm_root.empty() || norm_root.back() != '/') norm_root += '/';
            if (norm.empty() || norm.back() != '/') norm += '/';

            if (norm.size() < norm_root.size()) return true;
            return norm.compare(0, norm_root.size(), norm_root) != 0;
        };

        size_t i = 0;
        bool expect_command = true;

        while (i < tokens.size()) {
            if (tokens[i].type == TokenType::Operator) {
                if (tokens[i].value == ">" || tokens[i].value == ">>") {
                    if (tokens[i].value == ">") {
                        v.capabilities.destroys_data = true;
                    }
                    if (i + 1 < tokens.size() && tokens[i+1].type == TokenType::Word) {
                        if (is_outside(tokens[i+1].value)) {
                            v.capabilities.writes_outside_workspace = true;
                        }
                    }
                } else if (tokens[i].value == "<") {
                    if (i + 1 < tokens.size() && tokens[i+1].type == TokenType::Word) {
                        if (is_outside(tokens[i+1].value)) {
                            v.capabilities.reads_outside_workspace = true;
                        }
                    }
                } else if (tokens[i].value == "&") {
                    v.capabilities.spawns_unbounded_process = true;
                }
                expect_command = true;
                i++;
                continue;
            }
            if (tokens[i].type == TokenType::SubshellStart || tokens[i].type == TokenType::SubshellEnd) {
                expect_command = (tokens[i].type == TokenType::SubshellStart);
                i++;
                continue;
            }

            if (expect_command && tokens[i].type == TokenType::Word) {
                std::string cmd = tokens[i].value;
                if (cmd == "sudo" || cmd == "su" || cmd == "doas") {
                    v.capabilities.escalates_privileges = true;
                    expect_command = true;
                } else if (cmd == "nohup" || cmd == "disown") {
                    v.capabilities.spawns_unbounded_process = true;
                    expect_command = true;
                } else if (cmd == "xargs") {
                    v.status = ParseStatus::PartiallyParsed;
                    expect_command = true;
                } else if (cmd == "curl" || cmd == "wget" || cmd == "nc" || cmd == "ping" || cmd == "ssh") {
                    v.capabilities.network_access = true;
                    expect_command = false;
                } else if (cmd == "rm" || cmd == "shred" || cmd == "wipe") {
                    v.capabilities.destroys_data = true;
                    // Check if rm args go outside
                    for (size_t j = i + 1; j < tokens.size() && tokens[j].type == TokenType::Word; ++j) {
                        if (tokens[j].value[0] != '-') {
                            if (is_outside(tokens[j].value)) {
                                v.capabilities.writes_outside_workspace = true;
                            }
                        }
                    }
                    expect_command = false;
                } else if (cmd == "kill" || cmd == "killall" || cmd == "pkill") {
                    v.capabilities.signals_foreign_process = true;
                    expect_command = false;
                } else if (cmd == "sh" || cmd == "bash" || cmd == "python" || cmd == "node" || cmd == "make" || cmd == "npm") {
                    v.capabilities.spawns_unbounded_process = true;
                    if (cmd == "sh" || cmd == "bash") {
                        if (i + 1 < tokens.size() && tokens[i+1].value == "-c") {
                            v.status = ParseStatus::PartiallyParsed;
                        }
                    }
                    expect_command = false;
                } else if (cmd == "git") {
                    bool is_push = false;
                    bool is_force = false;
                    for (size_t j = i + 1; j < tokens.size() && tokens[j].type == TokenType::Word; ++j) {
                        if (tokens[j].value == "reset" || tokens[j].value == "rebase") {
                            v.capabilities.rewrites_vcs_history = true;
                        } else if (tokens[j].value == "push") {
                            is_push = true;
                        } else if (tokens[j].value == "--force" || tokens[j].value == "-f") {
                            is_force = true;
                        } else if (tokens[j].value == "clean") {
                            for (size_t k = j + 1; k < tokens.size() && tokens[k].type == TokenType::Word; ++k) {
                                if (tokens[k].value.length() > 1 && tokens[k].value[0] == '-' && tokens[k].value[1] != '-') {
                                    if (tokens[k].value.find('f') != std::string::npos && tokens[k].value.find('d') != std::string::npos && tokens[k].value.find('x') != std::string::npos) {
                                        v.capabilities.destroys_data = true;
                                    }
                                }
                            }
                        }
                    }
                    if (is_push && is_force) v.capabilities.rewrites_vcs_history = true;
                    expect_command = false;
                } else {
                    for (size_t j = i + 1; j < tokens.size() && tokens[j].type == TokenType::Word; ++j) {
                        if (tokens[j].value[0] != '-' && !tokens[j].value.empty()) {
                            if (is_outside(tokens[j].value)) {
                                if (cmd == "cat" || cmd == "grep" || cmd == "ls" || cmd == "cp") {
                                    v.capabilities.reads_outside_workspace = true;
                                }
                                if (cmd == "cp" || cmd == "mv" || cmd == "touch" || cmd == "mkdir") {
                                    v.capabilities.writes_outside_workspace = true;
                                }
                            }
                        }
                    }
                    expect_command = false;
                }
            } else if (tokens[i].type == TokenType::Word) {
                // Not expect command, but it's a word. We can still check if it's an outside path if it might be used by a default command...
                // But we handled path checks in the command parsing above.
            }
            i++;
        }

    } catch (...) {
        v.status = ParseStatus::Unparseable;
    }
    return v;
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
