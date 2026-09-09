#include "src/surface/worker.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace lmp::surface::worker {
namespace {

std::string expand_user(const std::string& path) {
    if (path.empty()) return path;
    if (path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            if (path.size() == 1) return std::string(home);
            if (path[1] == '/') return std::string(home) + path.substr(1);
        }
    }
    return path;
}

bool run_cmd(const std::string& cmd, const std::string& cwd, double timeout_s,
             int& exit_code, std::string& output) {
    output.clear();
    exit_code = -1;

    int pipefd[2];
    if (::pipe(pipefd) != 0) {
        output = "pipe() failed";
        return false;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        output = "fork() failed";
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        return false;
    }

    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);

        if (!cwd.empty()) {
            if (::chdir(cwd.c_str()) != 0) {
                ::_exit(126);
            }
        }
        ::setpgid(0, 0);
        ::execlp("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
        ::_exit(127);
    }

    ::close(pipefd[1]);
    int flags = ::fcntl(pipefd[0], F_GETFL, 0);
    ::fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

    const auto start = std::chrono::steady_clock::now();
    char buf[4096];
    bool timed_out = false;

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start).count();
        if (timeout_s > 0 && elapsed >= timeout_s) {
            timed_out = true;
            ::kill(-pid, SIGKILL);
            break;
        }

        int poll_timeout_ms = 100;
        if (timeout_s > 0) {
            double remaining = (timeout_s - elapsed) * 1000.0;
            if (remaining < 100.0) poll_timeout_ms = std::max(1, static_cast<int>(remaining));
        }

        struct pollfd pfd{};
        pfd.fd = pipefd[0];
        pfd.events = POLLIN | POLLHUP | POLLERR;

        int ret = ::poll(&pfd, 1, poll_timeout_ms);
        if (ret > 0 && (pfd.revents & POLLIN)) {
            ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
            if (n > 0) {
                if (output.size() < 1024 * 1024) {
                    output.append(buf, static_cast<size_t>(n));
                }
            } else if (n == 0) {
                break;
            }
        } else if (ret > 0 && (pfd.revents & (POLLHUP | POLLERR))) {
            // Drain remaining bytes
            while (true) {
                ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
                if (n > 0) {
                    if (output.size() < 1024 * 1024) {
                        output.append(buf, static_cast<size_t>(n));
                    }
                } else {
                    break;
                }
            }
            break;
        }
    }

    ::close(pipefd[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);

    if (timed_out) {
        exit_code = -1;
        output += "\n[timeout after " + std::to_string(timeout_s) + "s]";
        return false;
    }

    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        exit_code = 128 + WTERMSIG(status);
    }
    return true;
}

} // namespace

std::optional<TaskPacket> load_packet(const std::string& path_in, std::string& error) {
    error.clear();
    std::string path = expand_user(path_in);

    if (path.empty()) {
        error = "task path is empty";
        return std::nullopt;
    }

    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        std::filesystem::path cand = std::filesystem::path(path) / "task.json";
        if (!std::filesystem::is_regular_file(cand, ec)) {
            error = "no task.json in directory " + path;
            return std::nullopt;
        }
        path = cand.string();
    } else if (!std::filesystem::exists(path, ec)) {
        error = "task packet not found: " + path;
        return std::nullopt;
    }

    std::filesystem::path abs_task = std::filesystem::absolute(path, ec);
    std::string task_path = abs_task.string();
    std::string task_dir = abs_task.parent_path().string();

    std::ifstream fh(task_path);
    if (!fh.is_open()) {
        error = "cannot read task packet: " + task_path;
        return std::nullopt;
    }
    std::stringstream ss;
    ss << fh.rdbuf();
    std::string raw = ss.str();

    auto j = nlohmann::json::parse(raw, nullptr, false);
    if (j.is_discarded()) {
        error = "task.json is not valid JSON";
        return std::nullopt;
    }
    if (!j.is_object()) {
        error = "task.json must be a JSON object";
        return std::nullopt;
    }

    TaskPacket packet;
    packet.task_path = task_path;
    packet.task_dir = task_dir;

    if (!j.contains("id") || !j["id"].is_string() || j["id"].get<std::string>().empty()) {
        error = "task.json missing non-empty string `id`";
        return std::nullopt;
    }
    packet.id = j["id"].get<std::string>();

    if (!j.contains("cwd") || !j["cwd"].is_string() || j["cwd"].get<std::string>().empty()) {
        error = "task.json missing `cwd`";
        return std::nullopt;
    }
    std::string raw_cwd = expand_user(j["cwd"].get<std::string>());
    std::filesystem::path p_cwd(raw_cwd);
    if (!p_cwd.is_absolute()) {
        error = "cwd must be an absolute path, got '" + raw_cwd + "'";
        return std::nullopt;
    }
    if (!std::filesystem::is_directory(p_cwd, ec)) {
        error = "cwd is not a directory: " + raw_cwd;
        return std::nullopt;
    }
    packet.cwd = std::filesystem::canonical(p_cwd, ec).string();
    if (ec) packet.cwd = p_cwd.string();

    std::string prompt;
    if (j.contains("prompt") && j["prompt"].is_string()) {
        prompt = j["prompt"].get<std::string>();
    }
    if (prompt.empty()) {
        std::filesystem::path prompt_md = std::filesystem::path(task_dir) / "prompt.md";
        if (std::filesystem::is_regular_file(prompt_md, ec)) {
            std::ifstream pf(prompt_md.string());
            if (pf.is_open()) {
                std::stringstream pss;
                pss << pf.rdbuf();
                prompt = pss.str();
            }
        }
    }
    if (prompt.empty()) {
        error = "task.json missing `prompt` (or sibling prompt.md)";
        return std::nullopt;
    }
    packet.prompt = prompt;

    std::string model_dir;
    if (j.contains("model_dir") && j["model_dir"].is_string() && !j["model_dir"].get<std::string>().empty()) {
        model_dir = j["model_dir"].get<std::string>();
    } else {
        const char* env_dir = std::getenv("LMP_QWEN_DIR");
        if (env_dir) model_dir = env_dir;
    }
    if (model_dir.empty()) {
        error = "missing model_dir (set task.json `model_dir` or LMP_QWEN_DIR)";
        return std::nullopt;
    }
    std::filesystem::path p_model(expand_user(model_dir));
    if (!std::filesystem::is_directory(p_model, ec)) {
        error = "model_dir is not a directory: " + model_dir;
        return std::nullopt;
    }
    packet.model_dir = std::filesystem::canonical(p_model, ec).string();
    if (ec) packet.model_dir = p_model.string();

    if (j.contains("mode")) {
        if (!j["mode"].is_string()) {
            error = "mode must be a string";
            return std::nullopt;
        }
        std::string mode = j["mode"].get<std::string>();
        if (mode != "agent" && mode != "plan" && mode != "debug") {
            error = "mode must be one of ('plan', 'debug', 'agent'), got '" + mode + "'";
            return std::nullopt;
        }
        packet.mode = mode;
    }

    if (j.contains("timeout_s")) {
        if (j["timeout_s"].is_boolean() || !j["timeout_s"].is_number()) {
            error = "timeout_s must be a number";
            return std::nullopt;
        }
        double t = j["timeout_s"].get<double>();
        if (t <= 0.0) {
            error = "timeout_s must be positive";
            return std::nullopt;
        }
        packet.timeout_s = t;
    }

    auto parse_bool = [](const nlohmann::json& v, bool fallback) -> bool {
        if (v.is_boolean()) return v.get<bool>();
        if (v.is_number()) return v.get<double>() != 0.0;
        if (v.is_string()) {
            std::string s = v.get<std::string>();
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            if (s == "0" || s == "false" || s == "no" || s == "off") return false;
            if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
        }
        return fallback;
    };

    if (j.contains("auto_approve_exec")) {
        packet.auto_approve_exec = parse_bool(j["auto_approve_exec"], true);
    }
    if (j.contains("auto_approve_writes")) {
        packet.auto_approve_writes = parse_bool(j["auto_approve_writes"], true);
    }
    if (j.contains("auto_approve_irreversible")) {
        packet.auto_approve_irreversible = parse_bool(j["auto_approve_irreversible"], false);
    }
    if (j.contains("commit_think")) {
        packet.commit_think = parse_bool(j["commit_think"], true);
    }
    if (j.contains("shadow_compact")) {
        packet.shadow_compact = parse_bool(j["shadow_compact"], true);
    }

    if (j.contains("result_path") && j["result_path"].is_string() && !j["result_path"].get<std::string>().empty()) {
        std::string rp = expand_user(j["result_path"].get<std::string>());
        std::filesystem::path p_res(rp);
        if (!p_res.is_absolute()) {
            p_res = std::filesystem::path(task_dir) / p_res;
        }
        packet.result_path = p_res.string();
    } else {
        packet.result_path = (std::filesystem::path(task_dir) / "result.json").string();
    }

    if (j.contains("check") && j["check"].is_string()) {
        packet.check_command = j["check"].get<std::string>();
    } else if (j.contains("check_command") && j["check_command"].is_string()) {
        packet.check_command = j["check_command"].get<std::string>();
    }

    if (j.contains("check_timeout_s") && j["check_timeout_s"].is_number()) {
        packet.check_timeout_s = j["check_timeout_s"].get<double>();
    }

    if (j.contains("trust_mcp")) {
        if (!j["trust_mcp"].is_array()) {
            error = "trust_mcp must be an array of server names";
            return std::nullopt;
        }
        for (const auto& item : j["trust_mcp"]) {
            if (!item.is_string()) {
                error = "trust_mcp elements must be strings";
                return std::nullopt;
            }
            packet.trust_mcp.push_back(item.get<std::string>());
        }
    }

    if (!packet.trust_mcp.empty()) {
        std::filesystem::path mcp_json_path = std::filesystem::path(packet.cwd) / ".mcp.json";
        if (!std::filesystem::exists(mcp_json_path, ec)) {
            error = "trust_mcp names server(s), but .mcp.json was not found in " + packet.cwd;
            return std::nullopt;
        }
        std::ifstream mcp_file(mcp_json_path.string());
        if (!mcp_file.is_open()) {
            error = "cannot open .mcp.json in " + packet.cwd;
            return std::nullopt;
        }
        std::stringstream ss_mcp;
        ss_mcp << mcp_file.rdbuf();
        auto mcp_j = nlohmann::json::parse(ss_mcp.str(), nullptr, false);
        if (mcp_j.is_discarded() || !mcp_j.is_object() || !mcp_j.contains("mcpServers") || !mcp_j["mcpServers"].is_object()) {
            error = "failed to parse .mcp.json in " + packet.cwd + " (missing or invalid 'mcpServers')";
            return std::nullopt;
        }
        const auto& servers = mcp_j["mcpServers"];
        for (const std::string& name : packet.trust_mcp) {
            if (!servers.contains(name) || !servers[name].is_object()) {
                error = "trust_mcp names server '" + name + "', but it is absent from " + mcp_json_path.string();
                return std::nullopt;
            }
            const auto& s = servers[name];
            if (!s.contains("command") || !s["command"].is_string() || s["command"].get<std::string>().empty()) {
                error = "trust_mcp server '" + name + "' in .mcp.json has missing or empty 'command'";
                return std::nullopt;
            }
        }
    }

    if (j.contains("orch_webhook") && j["orch_webhook"].is_string()) {
        packet.orch_webhook = j["orch_webhook"].get<std::string>();
    } else {
        const char* env_hook = std::getenv("LMP_ORCH_WEBHOOK");
        if (env_hook && *env_hook != '\0') {
            packet.orch_webhook = env_hook;
        }
    }

    return packet;
}

std::string build_start_message(const TaskPacket& packet, const std::string& request_id) {
    nlohmann::json settings = {
        {"model_dir", packet.model_dir},
        {"workspace_root", packet.cwd},
        {"mode", packet.mode},
        {"max_iterations", 30},
        {"wall_clock_seconds", static_cast<int>(packet.timeout_s)},
        {"auto_approve_exec", packet.auto_approve_exec},
        {"auto_approve_writes", packet.auto_approve_writes},
        {"auto_approve_irreversible", packet.auto_approve_irreversible},
        {"require_approval", false},
        {"commit_think", packet.commit_think},
        {"shadow_compact", packet.shadow_compact}
    };
    if (!packet.check_command.empty()) {
        settings["verify_contract"] = packet.check_command;
    }
    const char* draft = std::getenv("LMP_DRAFT_DIR");
    if (draft && *draft) {
        settings["draft_model_dir"] = draft;
    }

    if (!packet.trust_mcp.empty()) {
        std::filesystem::path mcp_json_path = std::filesystem::path(packet.cwd) / ".mcp.json";
        std::ifstream mcp_file(mcp_json_path.string());
        if (mcp_file.is_open()) {
            std::stringstream ss_mcp;
            ss_mcp << mcp_file.rdbuf();
            auto mcp_j = nlohmann::json::parse(ss_mcp.str(), nullptr, false);
            if (!mcp_j.is_discarded() && mcp_j.is_object() && mcp_j.contains("mcpServers") && mcp_j["mcpServers"].is_object()) {
                const auto& servers = mcp_j["mcpServers"];
                nlohmann::json mcp_servers_list = nlohmann::json::array();
                for (const std::string& name : packet.trust_mcp) {
                    if (servers.contains(name) && servers[name].is_object()) {
                        const auto& s = servers[name];
                        nlohmann::json s_obj;
                        s_obj["name"] = name;
                        s_obj["command"] = s.value("command", "");

                        nlohmann::json args_arr = nlohmann::json::array();
                        if (s.contains("args") && s["args"].is_array()) {
                            for (const auto& a : s["args"]) {
                                if (a.is_string()) args_arr.push_back(a.get<std::string>());
                            }
                        }
                        s_obj["args"] = args_arr;

                        nlohmann::json env_arr = nlohmann::json::array();
                        if (s.contains("env")) {
                            if (s["env"].is_object()) {
                                for (const auto& [k, v] : s["env"].items()) {
                                    if (v.is_string()) {
                                        env_arr.push_back(k + "=" + v.get<std::string>());
                                    }
                                }
                            } else if (s["env"].is_array()) {
                                for (const auto& e : s["env"]) {
                                    if (e.is_string()) env_arr.push_back(e.get<std::string>());
                                }
                            }
                        }
                        s_obj["env"] = env_arr;
                        s_obj["trusted"] = true;

                        mcp_servers_list.push_back(s_obj);
                    }
                }
                settings["mcp_servers"] = mcp_servers_list;
            }
        }
    }

    nlohmann::json msg = {
        {"jsonrpc", "2.0"},
        {"id", request_id},
        {"method", "lmp/start"},
        {"params", {
            {"mission", packet.prompt},
            {"settings", settings}
        }}
    };
    return msg.dump();
}

std::string strip_think_leak(const std::string& answer) {
    std::string out = answer;

    while (true) {
        auto start = out.find("<think>");
        if (start == std::string::npos) break;
        auto end = out.find("</think>", start);
        if (end == std::string::npos) {
            out.erase(start);
            break;
        } else {
            out.erase(start, (end + 8) - start);
        }
    }

    size_t first = out.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    out = out.substr(first);

    static const std::vector<std::string> preambles = {
        "let me ", "i will now ", "i'll now ", "i will ", "okay, ", "sure, "
    };
    size_t nl = out.find('\n');
    if (nl != std::string::npos && nl < 200) {
        std::string first_line = out.substr(0, nl);
        std::string lower = first_line;
        for (char& c : lower) c = static_cast<char>(std::tolower(c));
        for (const auto& p : preambles) {
            if (lower.rfind(p, 0) == 0) {
                size_t next_text = out.find_first_not_of(" \t\r\n", nl);
                if (next_text != std::string::npos) {
                    out = out.substr(next_text);
                }
                break;
            }
        }
    }

    size_t last = out.find_last_not_of(" \t\r\n");
    if (last != std::string::npos) {
        out = out.substr(0, last + 1);
    }
    if (out.size() > 2000) {
        out.resize(2000);
    }
    return out;
}

std::vector<std::string> collect_files_touched(const std::string& log_path, const std::string& cwd) {
    std::vector<std::string> ordered;
    std::unordered_set<std::string> seen;
    if (log_path.empty() || !std::filesystem::exists(log_path)) {
        return ordered;
    }
    std::ifstream file(log_path);
    if (!file.is_open()) {
        return ordered;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded() || !j.is_object()) continue;
        if (!j.contains("kind") || j["kind"] != "write") continue;
        if (j.contains("changed")) {
            auto& ch = j["changed"];
            if (ch.is_string() && ch.get<std::string>() == "0") continue;
            if (ch.is_number() && ch.get<int64_t>() == 0) continue;
        }
        std::string path;
        if (j.contains("path") && j["path"].is_string()) {
            path = j["path"].get<std::string>();
        } else if (j.contains("normalised") && j["normalised"].is_string()) {
            path = j["normalised"].get<std::string>();
        }
        if (path.empty()) continue;

        std::string rel = path;
        if (!cwd.empty() && std::filesystem::path(path).is_absolute()) {
            try {
                auto r = std::filesystem::relative(path, cwd);
                std::string r_str = r.string();
                if (r_str.rfind("..", 0) != 0) {
                    rel = r_str;
                }
            } catch (...) {}
        }
        std::replace(rel.begin(), rel.end(), '\\', '/');
        if (seen.insert(rel).second) {
            ordered.push_back(rel);
        }
    }
    return ordered;
}

void collect_git(const std::string& cwd, const std::string& out_dir, RunResult& result) {
    result.diff_stat = {0, 0, 0};
    result.git_diff_path.clear();

    int code = -1;
    std::string probe_out;
    if (!run_cmd("git rev-parse --is-inside-work-tree", cwd, 10.0, code, probe_out)) {
        return;
    }
    while (!probe_out.empty() && (probe_out.back() == '\n' || probe_out.back() == '\r' || probe_out.back() == ' ')) {
        probe_out.pop_back();
    }
    if (code != 0 || probe_out != "true") {
        return;
    }

    int insertions = 0;
    int deletions = 0;
    int files = 0;

    std::string numstat_out;
    if (run_cmd("git diff --numstat", cwd, 30.0, code, numstat_out) && code == 0) {
        std::stringstream ss(numstat_out);
        std::string line;
        while (std::getline(ss, line)) {
            if (line.empty()) continue;
            std::stringstream lss(line);
            std::string ins_str, del_str, file_str;
            if (lss >> ins_str >> del_str) {
                files++;
                if (ins_str != "-") {
                    try { insertions += std::stoi(ins_str); } catch (...) {}
                }
                if (del_str != "-") {
                    try { deletions += std::stoi(del_str); } catch (...) {}
                }
            }
        }
    }

    std::string unified_out;
    run_cmd("git diff", cwd, 30.0, code, unified_out);

    // Also include untracked new files
    std::string status_out;
    if (run_cmd("git status --porcelain", cwd, 30.0, code, status_out) && code == 0) {
        std::stringstream ss(status_out);
        std::string line;
        while (std::getline(ss, line)) {
            if (line.size() > 3 && line[0] == '?' && line[1] == '?') {
                std::string rel_file = line.substr(3);
                while (!rel_file.empty() && (rel_file.back() == '\r' || rel_file.back() == '\n')) {
                    rel_file.pop_back();
                }
                // Strip quotes if git quoted the path
                if (rel_file.size() >= 2 && rel_file.front() == '"' && rel_file.back() == '"') {
                    rel_file = rel_file.substr(1, rel_file.size() - 2);
                }
                std::filesystem::path full = std::filesystem::path(cwd) / rel_file;
                std::error_code ec;
                if (std::filesystem::is_regular_file(full, ec)) {
                    std::ifstream uf(full.string());
                    if (uf.is_open()) {
                        files++;
                        int line_count = 0;
                        std::string u_line;
                        std::string content_diff;
                        while (std::getline(uf, u_line)) {
                            line_count++;
                            content_diff += "+" + u_line + "\n";
                        }
                        insertions += line_count;

                        unified_out += "diff --git a/" + rel_file + " b/" + rel_file + "\n";
                        unified_out += "new file mode 100644\n";
                        unified_out += "--- /dev/null\n";
                        unified_out += "+++ b/" + rel_file + "\n";
                        unified_out += "@@ -0,0 +1," + std::to_string(line_count) + " @@\n";
                        unified_out += content_diff;
                    }
                }
            }
        }
    }

    result.diff_stat = {insertions, deletions, files};

    if (!unified_out.empty()) {
        std::filesystem::path diff_path = std::filesystem::path(out_dir) / "git.diff";
        std::error_code ec;
        std::filesystem::create_directories(diff_path.parent_path(), ec);
        std::ofstream df(diff_path.string());
        if (df.is_open()) {
            df << unified_out;
            result.git_diff_path = diff_path.string();
        }
    }
}

void run_check(const TaskPacket& packet, RunResult& result) {
    if (packet.check_command.empty()) {
        result.test.ran = false;
        result.test.exit_code = -1;
        result.test.command = "";
        result.test.output_tail = "";
        return;
    }

    result.test.ran = true;
    result.test.command = packet.check_command;

    int code = -1;
    std::string out;
    run_cmd(packet.check_command, packet.cwd, packet.check_timeout_s, code, out);

    result.test.exit_code = code;
    if (out.size() > 2000) {
        out = out.substr(out.size() - 2000);
    }
    result.test.output_tail = out;

    if (code != 0) {
        if (result.status == "ok") {
            result.status = "error";
            result.error = "check command failed (exit code " + std::to_string(code) + ")";
        }
    }
}

void write_result(const std::string& path, const RunResult& result) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    nlohmann::json diff_stat = {
        {"insertions", result.diff_stat.insertions},
        {"deletions", result.diff_stat.deletions},
        {"files", result.diff_stat.files}
    };

    nlohmann::json test = {
        {"ran", result.test.ran},
        {"exit_code", result.test.ran ? nlohmann::json(result.test.exit_code) : nlohmann::json(nullptr)},
        {"command", result.test.ran ? nlohmann::json(result.test.command) : nlohmann::json(nullptr)},
        {"output_tail", result.test.ran ? nlohmann::json(result.test.output_tail) : nlohmann::json(nullptr)}
    };

    nlohmann::json j = {
        {"task_id", result.task_id},
        {"status", result.status},
        {"message", result.message},
        {"cwd", result.cwd},
        {"model_dir", result.model_dir},
        {"wall_seconds", result.wall_seconds},
        {"turns", result.turns},
        {"generated_tokens", result.generated_tokens},
        {"files_touched", result.files_touched},
        {"diff_stat", diff_stat},
        {"git_diff_path", result.git_diff_path.empty() ? nlohmann::json(nullptr) : nlohmann::json(result.git_diff_path)},
        {"test", test},
        {"log_path", result.log_path.empty() ? nlohmann::json(nullptr) : nlohmann::json(result.log_path)},
        {"error", result.error.empty() ? nlohmann::json(nullptr) : nlohmann::json(result.error)}
    };

    std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path);
        out << j.dump(2) << "\n";
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        std::filesystem::copy_file(tmp_path, path, std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tmp_path, ec);
    }
}

std::optional<AwaitingUserInfo> find_last_ask_user(const std::string& log_path, const std::string& run_id) {
    if (log_path.empty() || !std::filesystem::exists(log_path)) {
        return std::nullopt;
    }
    std::ifstream file(log_path);
    if (!file.is_open()) {
        return std::nullopt;
    }
    std::string line;
    std::optional<AwaitingUserInfo> last_ask;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded() || !j.is_object()) continue;
        if (!j.contains("kind") || j["kind"] != "ask_user") continue;
        AwaitingUserInfo info;
        info.run_id = run_id;
        if (j.contains("question") && j["question"].is_string()) {
            info.question = j["question"].get<std::string>();
        }
        if (j.contains("options") && j["options"].is_string()) {
            info.options = j["options"].get<std::string>();
        }
        if (j.contains("seq") && j["seq"].is_number()) {
            info.seq = j["seq"].get<uint64_t>();
        }
        last_ask = std::move(info);
    }
    return last_ask;
}

void write_awaiting_user(const std::string& path, const AwaitingUserInfo& info) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    nlohmann::json j = {
        {"question", info.question},
        {"options", info.options},
        {"run_id", info.run_id},
        {"seq", info.seq}
    };
    std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path);
        out << j.dump(2) << "\n";
    }
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        std::filesystem::copy_file(tmp_path, path, std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tmp_path, ec);
    }
}

std::optional<std::string> read_and_consume_answer(const std::string& answer_path, const std::string& awaiting_path) {
    std::error_code ec;
    if (!std::filesystem::exists(answer_path, ec) || ec) {
        return std::nullopt;
    }
    auto sz = std::filesystem::file_size(answer_path, ec);
    if (ec || sz == 0) {
        return std::nullopt;
    }
    std::ifstream file(answer_path);
    if (!file.is_open()) {
        return std::nullopt;
    }
    std::stringstream buf;
    buf << file.rdbuf();
    std::string raw = buf.str();
    file.close();

    size_t first_non_ws = raw.find_first_not_of(" \t\r\n");
    if (first_non_ws == std::string::npos) {
        return std::nullopt;
    }

    std::string answer;
    auto j = nlohmann::json::parse(raw, nullptr, false);
    if (!j.is_discarded()) {
        if (j.is_object()) {
            if (j.contains("text") && j["text"].is_string()) {
                answer = j["text"].get<std::string>();
            } else if (j.contains("answer") && j["answer"].is_string()) {
                answer = j["answer"].get<std::string>();
            } else {
                answer = j.dump();
            }
        } else if (j.is_string()) {
            answer = j.get<std::string>();
        } else {
            answer = raw;
        }
    } else {
        answer = raw;
    }

    size_t last = answer.find_last_not_of(" \t\r\n");
    if (last != std::string::npos) {
        answer = answer.substr(0, last + 1);
    }

    std::filesystem::remove(answer_path, ec);
    if (!awaiting_path.empty()) {
        std::filesystem::remove(awaiting_path, ec);
    }

    return answer;
}

bool post_orch_webhook(const std::string& webhook_url,
                       const WebhookPayload& payload,
                       double timeout_s) {
    if (webhook_url.empty()) {
        return false;
    }

    nlohmann::json j = {
        {"kind", payload.kind},
        {"task_id", payload.task_id},
        {"run_id", payload.run_id},
        {"cwd", payload.cwd},
        {"result_path", payload.result_path},
        {"seq", payload.seq},
        {"status", payload.status},
        {"question", payload.question}
    };
    std::string body = j.dump();

    int in_pipe[2];
    if (::pipe(in_pipe) != 0) {
        std::fprintf(stderr, "piper: warning: pipe() failed for webhook POST: %s\n", std::strerror(errno));
        return false;
    }

    int err_pipe[2];
    if (::pipe(err_pipe) != 0) {
        std::fprintf(stderr, "piper: warning: pipe() failed for webhook POST: %s\n", std::strerror(errno));
        ::close(in_pipe[0]);
        ::close(in_pipe[1]);
        return false;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        std::fprintf(stderr, "piper: warning: fork() failed for webhook POST: %s\n", std::strerror(errno));
        ::close(in_pipe[0]);
        ::close(in_pipe[1]);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);
        return false;
    }

    if (pid == 0) {
        ::close(in_pipe[1]);
        ::close(err_pipe[0]);

        ::dup2(in_pipe[0], STDIN_FILENO);
        ::close(in_pipe[0]);

        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(err_pipe[1]);

        int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDOUT_FILENO);
            ::close(devnull);
        }

        ::setpgid(0, 0);

        int timeout_sec = std::max(1, static_cast<int>(std::ceil(timeout_s)));
        std::string timeout_str = std::to_string(timeout_sec);

        ::execlp("curl", "curl", "-f", "-s", "-S",
                 "--max-time", timeout_str.c_str(),
                 "-X", "POST",
                 "-H", "Content-Type: application/json",
                 "--data-binary", "@-",
                 webhook_url.c_str(),
                 static_cast<char*>(nullptr));
        ::_exit(127);
    }

    ::close(in_pipe[0]);
    ::close(err_pipe[1]);

    const char* data = body.data();
    size_t remaining = body.size();
    while (remaining > 0) {
        ssize_t n = ::write(in_pipe[1], data, remaining);
        if (n <= 0) break;
        data += n;
        remaining -= static_cast<size_t>(n);
    }
    ::close(in_pipe[1]);

    int flags = ::fcntl(err_pipe[0], F_GETFL, 0);
    ::fcntl(err_pipe[0], F_SETFL, flags | O_NONBLOCK);

    std::string err_out;
    const auto start = std::chrono::steady_clock::now();
    bool timed_out = false;

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start).count();
        if (timeout_s > 0 && elapsed >= timeout_s) {
            timed_out = true;
            ::kill(-pid, SIGKILL);
            break;
        }

        int poll_timeout_ms = 100;
        if (timeout_s > 0) {
            double rem_ms = (timeout_s - elapsed) * 1000.0;
            if (rem_ms < 100.0) poll_timeout_ms = std::max(1, static_cast<int>(rem_ms));
        }

        struct pollfd pfd{};
        pfd.fd = err_pipe[0];
        pfd.events = POLLIN | POLLHUP | POLLERR;

        int ret = ::poll(&pfd, 1, poll_timeout_ms);
        if (ret > 0 && (pfd.revents & POLLIN)) {
            char buf[1024];
            ssize_t n = ::read(err_pipe[0], buf, sizeof(buf));
            if (n > 0) {
                err_out.append(buf, static_cast<size_t>(n));
            }
        }

        int status = 0;
        pid_t w = ::waitpid(pid, &status, WNOHANG);
        if (w == pid) {
            char buf[1024];
            while (true) {
                ssize_t n = ::read(err_pipe[0], buf, sizeof(buf));
                if (n <= 0) break;
                err_out.append(buf, static_cast<size_t>(n));
            }
            ::close(err_pipe[0]);
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                return true;
            }
            while (!err_out.empty() && (err_out.back() == '\n' || err_out.back() == '\r')) {
                err_out.pop_back();
            }
            std::string reason = err_out.empty() ? ("exit code " + std::to_string(WEXITSTATUS(status))) : err_out;
            std::fprintf(stderr, "piper: warning: failed to post '%s' wake event to %s: %s\n",
                         payload.kind.c_str(), webhook_url.c_str(), reason.c_str());
            return false;
        } else if (w < 0 && errno == ECHILD) {
            ::close(err_pipe[0]);
            return false;
        }
    }

    ::close(err_pipe[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (timed_out) {
        std::fprintf(stderr, "piper: warning: timed out posting '%s' wake event to %s (%.1fs)\n",
                     payload.kind.c_str(), webhook_url.c_str(), timeout_s);
    }
    return false;
}

bool stdin_is_devnull() {
    struct stat s0{}, sn{};
    if (::fstat(STDIN_FILENO, &s0) != 0 || ::stat("/dev/null", &sn) != 0) {
        return false;
    }
    return (s0.st_dev == sn.st_dev && s0.st_ino == sn.st_ino);
}

bool stdout_is_regular_file() {
    struct stat s1{};
    if (::fstat(STDOUT_FILENO, &s1) != 0) {
        return false;
    }
    return S_ISREG(s1.st_mode);
}

bool is_detached_launch(bool cli_detach) {
    if (cli_detach) {
        return true;
    }
    const char* flag = std::getenv("LMP_DAEMONIZE");
    std::string f = flag ? flag : "";
    if (f == "1") {
        return true;
    }
    if (f != "0" && (stdin_is_devnull() || stdout_is_regular_file())) {
        return true;
    }
    return false;
}

bool parse_approval_answer(const std::string& raw) {
    std::string text = raw;
    auto j = nlohmann::json::parse(raw, nullptr, false);
    if (!j.is_discarded()) {
        if (j.is_object()) {
            if (j.contains("allow") && j["allow"].is_boolean()) {
                return j["allow"].get<bool>();
            }
            if (j.contains("approved") && j["approved"].is_boolean()) {
                return j["approved"].get<bool>();
            }
            if (j.contains("answer") && j["answer"].is_boolean()) {
                return j["answer"].get<bool>();
            }
            if (j.contains("text") && j["text"].is_string()) {
                text = j["text"].get<std::string>();
            } else if (j.contains("answer") && j["answer"].is_string()) {
                text = j["answer"].get<std::string>();
            }
        } else if (j.is_boolean()) {
            return j.get<bool>();
        } else if (j.is_string()) {
            text = j.get<std::string>();
        }
    }

    size_t start = text.find_first_not_of(" \t\r\n'\"`");
    if (start == std::string::npos) {
        return false;
    }
    size_t end = text.find_last_not_of(" \t\r\n'\"`");
    std::string s = text.substr(start, end - start + 1);
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (s == "allow" || s == "allowed" || s == "approve" || s == "approved" ||
        s == "yes" || s == "y" || s == "true" || s == "1") {
        return true;
    }
    return false;
}

std::string format_irreversible_question(const std::string& tool, const std::string& command_or_preview) {
    return tool + " " + command_or_preview + ": this call is irreversible and the orchestrator must allow or deny it";
}

IrreversibleAskResult handle_irreversible_ask(const IrreversibleAskParams& params) {
    std::string question = format_irreversible_question(params.tool, params.command_or_preview);

    AwaitingUserInfo info;
    info.question = question;
    info.options = "allow,deny";
    info.run_id = params.run_id;
    info.seq = params.seq;
    write_awaiting_user(params.awaiting_path, info);

    if (!params.orch_webhook.empty()) {
        WebhookPayload hook_payload;
        hook_payload.kind = "ask";
        hook_payload.task_id = params.task_id;
        hook_payload.run_id = params.run_id;
        hook_payload.cwd = params.cwd;
        hook_payload.result_path = params.result_path;
        hook_payload.seq = info.seq;
        hook_payload.status = "";
        hook_payload.question = info.question;
        post_orch_webhook(params.orch_webhook, hook_payload);
    } else {
        std::fprintf(stderr,
                     "piper: awaiting_user.json written for irreversible tool '%s' (%s). "
                     "No webhook configured; execution paused waiting for answer.json...\n",
                     params.tool.c_str(), params.command_or_preview.c_str());
        std::fflush(stderr);
    }

    auto last_heartbeat = std::chrono::steady_clock::now();
    while (true) {
        if (params.is_cancelled && params.is_cancelled()) {
            return IrreversibleAskResult::Cancelled;
        }
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - params.wall_start).count();
        if (params.timeout_s > 0.0 && elapsed >= params.timeout_s) {
            return IrreversibleAskResult::Timeout;
        }

        auto ans = read_and_consume_answer(params.answer_path, params.awaiting_path);
        if (ans.has_value()) {
            if (params.on_answer_received) {
                params.on_answer_received(*ans);
            }
            return parse_approval_answer(*ans) ? IrreversibleAskResult::Allowed
                                               : IrreversibleAskResult::Denied;
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= 5) {
            last_heartbeat = now;
            std::fprintf(stderr,
                         "piper: awaiting_user.json written for irreversible tool '%s' (%s). "
                         "Waiting for answer.json...\n",
                         params.tool.c_str(), params.command_or_preview.c_str());
            std::fflush(stderr);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

namespace {

constexpr std::string_view kAgentWakeStandard = R"(# Piper Local Worker — Orchestrator Guide & Parent Runbook

## Overview
**Core Principle:** Cloud directs, local writes, cloud verifies, repeat.

Piper is a fast headless coding worker running locally on Apple Silicon (via MLX). It executes edits, shell commands, and file operations inside `cwd` with zero cloud output token cost.

The cloud orchestrator (Cursor, Claude, Gemini, Antigravity, or custom script) acts as the high-level brain:
- Maintains the long-horizon plan and acceptance criteria.
- Decomposes complex tasks into bounded packets (1–3 files per packet).
- Directs Piper by writing task packets (`task.json`).
- Verifies outcomes (diffs, test execution, acceptance checks).
- Loops until the entire mission is verified complete.

## The Long-Horizon Execution Loop
```
┌─────────────────────────┐         task.json            ┌──────────────────────────┐
│   Cloud Orchestrator    │ ───────────────────────────► │       Piper Worker       │
│  (Cursor, Claude, etc.) │                              │  (MLX on Apple Silicon)  │
│                         │ ◄─────────────────────────── │                          │
│ plan · review · verify  │     result.json + diff       │  edits · tools · loop    │
└─────────────────────────┘                              └──────────────────────────┘
             │                                                         │
             └────────── repeat until horizon acceptance passes ───────┘
```

### Roles

| Who | Owns | Does not own |
|---|---|---|
| **Cloud Orchestrator** | Goal decomposition, file-level direction, acceptance criteria, high-level review, troubleshooting, "are we done?" | Bulk code generation tokens, local tool thrash |
| **Piper Worker** | Edits, tool execution, test commands, local iteration inside `cwd` | Long-horizon judgment, multi-repo strategy |

Local models work best on scoped packets: **packets must be specific**, and **every turn gets an orchestrator review**. Trust outcomes (diff + tests + `result.json`), not vibes.

### Step-by-Step Procedure

1. **Frame the Horizon**
   Define the overarching objective and an acceptance checklist (e.g. unit tests pass, new command works, UI renders).

2. **Slice into Discrete Packets**
   Pick the smallest incremental step towards the goal.
   - Scope each slice to 1–3 files.
   - Explicitly list which files to EDIT, CREATE, and DO NOT TOUCH.

3. **Write `task.json`**
   Write a task packet in the workspace or a task directory:
   ```json
   {
     "id": "slice-001",
     "cwd": "/absolute/path/to/workspace",
     "mode": "agent",
     "model_dir": "/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit",
     "prompt": "## Horizon Context\nBuilding feature X.\n\n## This Slice Only\nAdd validator in src/validator.cpp and test in tests/test_validator.cpp.\n\n## Files\n- EDIT: src/validator.cpp\n- CREATE: tests/test_validator.cpp\n- DO NOT TOUCH: src/core.cpp\n\n## Done When\n- Unit test passes with `ctest -R test_validator`",
     "auto_approve_exec": true,
     "auto_approve_writes": true,
     "auto_approve_irreversible": true,
     "timeout_s": 600,
     "result_path": "/absolute/path/to/result.json"
   }
   ```

4. **Dispatch Piper**
   Run the CLI command:
   ```bash
   piper run --task /path/to/task.json
   # or equivalently:
   piper worker run --task /path/to/task.json
   # For autonomous unattended execution without prompts or pauses:
   piper run --task /path/to/task.json --auto-approve-irreversible
   # Or approve all (exec + writes + irreversible):
   piper run --task /path/to/task.json --auto-approve-all
   ```
   - **Attached mode (standard)**: Process waits and exits when the slice completes.
     - `0`: Completed normally.
     - `1`: Worker error.
     - `2`: Execution timed out (`timeout_s`).
     - `3`: Invalid task packet or missing wake URL for detached run.
   - **Detached mode**: If launching in the background (nohup, screen), you MUST pass `--orch-webhook <URL>`. Silent background launches without a webhook are refused.
   - **Irreversible tools**: Destructive tools or project managers (like `godot_project`, `delete_file`, or whole-file overwrites) escalate to `gate: irreversible`. Set `"auto_approve_irreversible": true` or pass `--auto-approve-irreversible` / `--auto-approve-all` for unattended runs; otherwise Piper pauses and writes `awaiting_user.json` for `answer.json`.

5. **Review `result.json` & Inspect Changes**
   Piper writes a structured result upon completion:
   ```json
   {
     "task_id": "slice-001",
     "status": "ok",
     "message": "Implemented validation logic and verified with unit test.",
     "files_touched": ["src/validator.cpp", "tests/test_validator.cpp"],
     "diff_stat": "+52 -2",
     "git_diff_path": "/path/to/slice.diff"
   }
   ```

   **Review Rubric (Keep it cheap):**
   - **Status**: Is `status == "ok"`? If `"error"` or `"stalled"`, inspect the message.
   - **Files Touched**: Are changes confined to expected paths? Reject drive-by edits.
   - **Diff**: Skim git diff for regressions or unnecessary churn.
   - **Acceptance**: Run verification commands or tests to validate the slice.

6. **Iterate or Complete**
   - **Pass**: If acceptance criteria for the slice pass, dispatch the next slice.
   - **Fail**: Send a narrowed/clarified packet, or perform that specific edit yourself.
   - **Done**: When all acceptance checklist items are verified, complete the mission.

---

## Piper worker wake standard

Any agent that starts Piper is the parent. Piper does not come find you.
A human must not copy a URL from a panel. A timer that checks the folder is
not the wake. It is a fallback, and it is how a finished or stalled run sits
until someone asks.

## Who owns the wake

The agent that launches `lmp_sidecar --worker` or `piper worker run` already
has a way to be woken, or it stays attached to the process. Piper does not
know who that agent is. There is no default host. Grok, Gemini, a script,
and a human CI job each pass their own URL.

Turn-based agents cannot stay attached. They pass a wake URL or they do not
detach.

## How to launch

Stay attached only if your process waits and reads the exit.

Otherwise, before start:

- `--orch-webhook URL`, or
- task field `orch_webhook`, or
- env `LMP_ORCH_WEBHOOK`

Detached with no URL: the CLI exits before the sidecar starts. `--help`
says this in one paragraph. Read that before the first launch. The help
text is the contract, not this chat.

## Events

One short JSON POST. No prompt bodies. Failed POST does not fail the
mission. No URL means no POST.

| kind | when | parent does |
| --- | --- | --- |
| `ask` | `awaiting_user.json` written, or an irreversible call is paused | write `answer.json` as `{"text":"..."}`. Do not restart. `allow` or `deny` for an irreversible call. Guidance for a real question. |
| `done` | `result.json` written and the slice completed | read the files. Send the next slice or stop. |
| `stalled` | `result.json` written and the harness stopped the run (`stalled`, `max_turns`, not completed) | read what landed. Do not treat it as success. Next slice or stop. |
| `died` | process exited and no `result.json` was written | launch parent sends this. Tell the user. Do not relaunch blindly. |

`stalled` is its own kind. Do not hide it inside `done` with `status: error`.
A parent that only handles `done` will miss a stall, which is the bug this
standard exists to kill.

Body:

```json
{
  "kind": "stalled",
  "task_id": "mc-live-2",
  "run_id": "r-...",
  "cwd": "/absolute/path",
  "result_path": "/absolute/path/result.json",
  "seq": 0,
  "status": "error",
  "question": ""
}
```

`ask` fills `question` and `seq`. `done` and `stalled` fill `status`.
`died` fills `cwd` and `task_id`.

Do not POST every turn. Do not POST tool output.

## What the parent must not do

- Do not poll Piper, and do not poll the cloud from Piper.
- Do not ask the human if the sidecar is still there. The event is the notice.
- Do not bake another agent's webhook into the binary.
- Do not start a silent screen or nohup without the URL.

A folder watch is allowed only as a backup when the platform has no wake
URL yet. It is not the standard, and it must not be the path a second
agent copies.

## Done when

- `--help` names this standard in one paragraph.
- A detached launch with no URL exits before the sidecar starts.
- A run that writes `result.json` POSTs `done` if it completed, `stalled`
  if it did not.
- The launch parent POSTs `died` if the sidecar exits with no result.
- An irreversible call and `ask_user` both POST `ask` and wait.
)";

constexpr std::string_view kCursorParentRule = R"(---
description: Piper local worker orchestration and parent contract
alwaysApply: true
---

piper: you are the parent. You have access to `piper`, a fast local coding worker CLI running on Apple Silicon (MLX).
Philosophy: Cloud directs, local writes, cloud verifies, repeat.
Use Piper to execute small, bounded slices of long-horizon tasks until the greater task is complete.

## How to Orchestrate with Piper

1. **Decompose into Small Slices**:
   Break the horizon goal into small slices touching 1–3 files per slice.
   Never ask Piper to solve an entire complex task in a single prompt.

2. **Prepare a Task Packet (`task.json`)**:
   Write a packet file with this structure:
   ```json
   {
     "id": "slice-001",
     "cwd": "/absolute/path/to/workspace",
     "prompt": "## Goal\nImplement X.\n\n## Files\n- EDIT: src/a.cpp\n- CREATE: tests/test_a.cpp\n- DO NOT TOUCH: other files\n\n## Done When\n- Tests compile and pass.",
     "auto_approve_exec": true,
     "auto_approve_irreversible": true,
     "timeout_s": 600
   }
   ```

3. **Dispatch Piper CLI**:
   Run and stay attached to the process:
   `piper run --task path/to/task.json` (or `piper worker run --task ...`)
   - For unattended execution (no pauses on irreversible tools): pass `--auto-approve-irreversible` or `--auto-approve-all`.
   - Exit code `0` = success, non-zero = error / timeout.
   - If running detached/background, pass `--orch-webhook <URL>`.

4. **Review Results**:
   Inspect `result.json` written by Piper:
   - Check `status` ("ok" vs "error"/"stalled").
   - Review `files_touched` and `git_diff` — verify changes match instructions.
   - Run slice acceptance tests.

5. **Loop Until Horizon Complete**:
   - If slice passed: dispatch the next slice.
   - If slice failed: write a narrower prompt or fix minor issues directly.
   - Repeat until all acceptance criteria for the greater task pass.

6. **Parent Contract & Events**:
   - Stay attached and read exit code + `result.json`.
   - Events: `ask` (write `answer.json`), `done`, `stalled` (not success; harness stopped), `died` (crashed).
   - See `PIPER.md` for full specification and review rubric.
)";

} // namespace

int init_project(const std::string& target_dir_in) {
    std::error_code ec;
    std::string dir_str = expand_user(target_dir_in.empty() ? "." : target_dir_in);
    std::filesystem::path target_dir = std::filesystem::absolute(dir_str, ec);
    std::filesystem::create_directories(target_dir, ec);

    // 1. Write PIPER.md (atomic tmp replace)
    std::filesystem::path piper_md = target_dir / "PIPER.md";
    std::filesystem::path piper_tmp = target_dir / "PIPER.md.tmp";
    {
        std::ofstream f(piper_tmp.string());
        if (!f.is_open()) {
            std::fprintf(stderr, "piper init: failed to write %s: %s\n",
                         piper_md.string().c_str(), std::strerror(errno));
            return kExitError;
        }
        f << kAgentWakeStandard;
    }
    std::filesystem::rename(piper_tmp, piper_md, ec);
    if (ec) {
        std::filesystem::copy_file(piper_tmp, piper_md, std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(piper_tmp, ec);
    }

    // 2. Write .cursor/rules/piper-parent.mdc
    std::filesystem::path cursor_dir = target_dir / ".cursor" / "rules";
    std::filesystem::create_directories(cursor_dir, ec);
    std::filesystem::path cursor_rule = cursor_dir / "piper-parent.mdc";
    std::filesystem::path cursor_tmp = cursor_dir / "piper-parent.mdc.tmp";
    {
        std::ofstream f(cursor_tmp.string());
        if (!f.is_open()) {
            std::fprintf(stderr, "piper init: failed to write %s: %s\n",
                         cursor_rule.string().c_str(), std::strerror(errno));
            return kExitError;
        }
        f << kCursorParentRule;
    }
    std::filesystem::rename(cursor_tmp, cursor_rule, ec);
    if (ec) {
        std::filesystem::copy_file(cursor_tmp, cursor_rule, std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(cursor_tmp, ec);
    }

    std::fprintf(stdout,
                 "piper init: PIPER.md and .cursor/rules/piper-parent.mdc. "
                 "Godoer briefs left alone. Stay attached, or pass --orch-webhook.\n");
    std::fflush(stdout);
    return kExitOk;
}

} // namespace lmp::surface::worker

