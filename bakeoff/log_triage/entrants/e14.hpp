#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <istream>
#include <mutex>

namespace log_triage {

enum class Priority {
    Low = 0,
    Medium = 1,
    High = 2
};

inline Priority match_priority(std::string_view line) {
    if (line.find("error:") != std::string_view::npos ||
        line.find("Error:") != std::string_view::npos ||
        line.find("fatal error:") != std::string_view::npos ||
        line.find("CMake Error") != std::string_view::npos ||
        line.find("error C") != std::string_view::npos ||
        line.find("fatal error C") != std::string_view::npos ||
        line.find("error LNK") != std::string_view::npos) {
        return Priority::High;
    }
    if (line.find("warning:") != std::string_view::npos ||
        line.find("Warning:") != std::string_view::npos ||
        line.find("CMake Warning") != std::string_view::npos ||
        line.find("warning C") != std::string_view::npos ||
        line.find("warning LNK") != std::string_view::npos) {
        return Priority::Medium;
    }
    return Priority::Low;
}

inline std::string format_elision(size_t count) {
    return "... " + std::to_string(count) + " line" + (count == 1 ? "" : "s") + " elided ...\n";
}

class Compactor {
    struct Node {
        bool in_use = false;
        bool is_elision = false;
        std::string text;
        Priority prio = Priority::Low;
        size_t elided_count = 0;

        int prev = -1;
        int next = -1;

        int q_prev = -1;
        int q_next = -1;
    };

    std::vector<Node> nodes;
    int head = -1;
    int tail = -1;

    int free_head = -1;

    int q_head[3] = {-1, -1, -1};
    int q_tail[3] = {-1, -1, -1};

    size_t current_bytes = 0;
    size_t max_bytes = 0;
    std::mutex mtx;

    size_t node_size(int idx) const {
        if (idx == -1) return 0;
        const auto& n = nodes[idx];
        if (n.is_elision) {
            return format_elision(n.elided_count).size();
        } else {
            return n.text.size();
        }
    }

    int alloc_node() {
        if (free_head != -1) {
            int idx = free_head;
            free_head = nodes[idx].next;
            nodes[idx] = Node();
            nodes[idx].in_use = true;
            return idx;
        }
        int idx = nodes.size();
        nodes.push_back(Node());
        nodes[idx].in_use = true;
        return idx;
    }

    void free_node(int idx) {
        nodes[idx].in_use = false;
        nodes[idx].next = free_head;
        free_head = idx;
    }

    void list_push_back(int idx) {
        nodes[idx].prev = tail;
        nodes[idx].next = -1;
        if (tail != -1) nodes[tail].next = idx;
        else head = idx;
        tail = idx;
    }

    void list_remove(int idx) {
        if (nodes[idx].prev != -1) nodes[nodes[idx].prev].next = nodes[idx].next;
        else head = nodes[idx].next;

        if (nodes[idx].next != -1) nodes[nodes[idx].next].prev = nodes[idx].prev;
        else tail = nodes[idx].prev;
    }

    void q_push_back(int p, int idx) {
        nodes[idx].q_prev = q_tail[p];
        nodes[idx].q_next = -1;
        if (q_tail[p] != -1) nodes[q_tail[p]].q_next = idx;
        else q_head[p] = idx;
        q_tail[p] = idx;
    }

    void q_remove(int p, int idx) {
        if (nodes[idx].q_prev != -1) nodes[nodes[idx].q_prev].q_next = nodes[idx].q_next;
        else q_head[p] = nodes[idx].q_next;

        if (nodes[idx].q_next != -1) nodes[nodes[idx].q_next].q_prev = nodes[idx].q_prev;
        else q_tail[p] = nodes[idx].q_prev;
    }

    void drop_node(int idx) {
        size_t old_size = node_size(idx);
        current_bytes -= old_size;

        nodes[idx].is_elision = true;
        nodes[idx].elided_count = 1;
        nodes[idx].text.clear();

        int prev_idx = nodes[idx].prev;
        if (prev_idx != -1 && nodes[prev_idx].is_elision) {
            current_bytes -= node_size(prev_idx);
            nodes[idx].elided_count += nodes[prev_idx].elided_count;
            list_remove(prev_idx);
            free_node(prev_idx);
        }

        int next_idx = nodes[idx].next;
        if (next_idx != -1 && nodes[next_idx].is_elision) {
            current_bytes -= node_size(next_idx);
            nodes[idx].elided_count += nodes[next_idx].elided_count;
            list_remove(next_idx);
            free_node(next_idx);
        }

        current_bytes += node_size(idx);
    }

public:
    Compactor(size_t max_bytes) : max_bytes(max_bytes) {}

    void add_line(std::string_view line) {
        std::lock_guard<std::mutex> lock(mtx);

        Priority p = match_priority(line);

        int idx = alloc_node();
        nodes[idx].is_elision = false;
        nodes[idx].text = std::string(line);
        if (nodes[idx].text.empty() || nodes[idx].text.back() != '\n') {
            nodes[idx].text += '\n';
        }
        nodes[idx].prio = p;
        nodes[idx].elided_count = 0;

        list_push_back(idx);
        q_push_back(static_cast<int>(p), idx);

        current_bytes += node_size(idx);

        while (current_bytes > max_bytes) {
            int best_p = -1;
            for (int i = 0; i <= 2; ++i) {
                if (q_head[i] != -1) {
                    best_p = i;
                    break;
                }
            }
            if (best_p == -1) {
                // If only elisions remain and we are still over budget
                while (current_bytes > max_bytes && tail != -1) {
                    current_bytes -= node_size(tail);
                    int tmp = tail;
                    list_remove(tmp);
                    free_node(tmp);
                }
                break;
            }
            int drop_idx = q_head[best_p];
            q_remove(best_p, drop_idx);
            drop_node(drop_idx);
        }
    }

    std::string get_result() const {
        std::string res;
        int curr = head;
        while (curr != -1) {
            if (nodes[curr].is_elision) {
                res += format_elision(nodes[curr].elided_count);
            } else {
                res += nodes[curr].text;
            }
            curr = nodes[curr].next;
        }
        return res;
    }
};

inline std::string compact(std::istream& in, size_t max_bytes) {
    Compactor c(max_bytes);
    std::string line;
    while (std::getline(in, line)) {
        c.add_line(line);
    }
    return c.get_result();
}

inline std::string compact(std::string_view in, size_t max_bytes) {
    Compactor c(max_bytes);
    size_t pos = 0;
    while (pos < in.size()) {
        size_t next = in.find('\n', pos);
        if (next == std::string_view::npos) {
            c.add_line(in.substr(pos));
            break;
        } else {
            c.add_line(in.substr(pos, next - pos));
            pos = next + 1;
        }
    }
    return c.get_result();
}

} // namespace log_triage
