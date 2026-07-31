#pragma once

#include <string>
#include <deque>
#include <queue>
#include <vector>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cstdint>

namespace log_triage {

    struct Block {
        int priority;
        size_t sequence_id;
        std::string text;

        bool operator>(const Block& other) const {
            if (priority != other.priority) return priority > other.priority;
            return sequence_id > other.sequence_id;
        }
    };

    inline int default_priority_func(const std::string& line) {
        if (line.find("FATAL") != std::string::npos) return 4;
        if (line.find("CRITICAL") != std::string::npos) return 3;
        if (line.find("ERROR") != std::string::npos) return 2;
        if (line.find("WARN") != std::string::npos) return 1;
        return 0;
    }

    inline std::string compact(std::istream& in,
                               size_t budget_bytes,
                               size_t context_window_size = 5,
                               std::function<int(const std::string&)> get_priority = default_priority_func) {
        std::deque<std::string> window;
        std::priority_queue<Block, std::vector<Block>, std::greater<Block>> pq;
        size_t current_pq_bytes = 0;
        size_t seq_id = 0;

        std::string line;
        while (std::getline(in, line)) {
            int p = get_priority(line);
            if (p > 0) {
                std::ostringstream block_stream;
                for (const auto& ctx : window) {
                    block_stream << ctx << "\n";
                }
                block_stream << line << "\n";

                std::string block_text = block_stream.str();
                Block b{p, seq_id++, block_text};

                pq.push(b);
                current_pq_bytes += b.text.size();

                while (current_pq_bytes > budget_bytes && pq.size() > 1) {
                    current_pq_bytes -= pq.top().text.size();
                    pq.pop();
                }

                window.clear();
            } else {
                window.push_back(line);
                if (window.size() > context_window_size) {
                    window.pop_front();
                }
            }
        }

        std::vector<Block> final_blocks;
        while (!pq.empty()) {
            final_blocks.push_back(pq.top());
            pq.pop();
        }

        std::sort(final_blocks.begin(), final_blocks.end(), [](const Block& a, const Block& b) {
            return a.sequence_id < b.sequence_id;
        });

        std::string result;
        for (const auto& b : final_blocks) {
            if (result.size() + b.text.size() <= budget_bytes) {
                result += b.text;
            } else {
                size_t remaining = budget_bytes - result.size();
                if (remaining > 0) {
                    result += b.text.substr(0, remaining);
                }
                break;
            }
        }

        return result;
    }

    inline std::string compact(const std::string& log_data,
                               size_t budget_bytes,
                               size_t context_window_size = 5,
                               std::function<int(const std::string&)> get_priority = default_priority_func) {
        std::istringstream iss(log_data);
        return compact(iss, budget_bytes, context_window_size, get_priority);
    }

} // namespace log_triage
