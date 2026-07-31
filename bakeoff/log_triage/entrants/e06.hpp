#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <cmath>

namespace log_triage {

enum class Toolchain {
    Unknown,
    GCC,
    Clang,
    MSVC,
    Python,
    Rust
};

struct ToolchainProfile {
    Toolchain type;
    std::vector<std::string_view> error_signatures;
    std::vector<std::string_view> warning_signatures;
    std::vector<std::string_view> context_signatures;
};

inline Toolchain detect_toolchain(std::string_view log) {
    if (log.find("error[E") != std::string_view::npos || log.find("cargo build") != std::string_view::npos || log.find("rustc ") != std::string_view::npos) return Toolchain::Rust;
    if (log.find("Traceback (most recent call last):") != std::string_view::npos) return Toolchain::Python;
    if (log.find(" MSVC ") != std::string_view::npos || log.find(" cl.exe ") != std::string_view::npos || log.find(" error C") != std::string_view::npos || log.find(" warning C") != std::string_view::npos) return Toolchain::MSVC;
    if (log.find("clang") != std::string_view::npos) return Toolchain::Clang;
    if (log.find("gcc") != std::string_view::npos || log.find("g++") != std::string_view::npos || log.find(": error: ") != std::string_view::npos || log.find(": warning: ") != std::string_view::npos) return Toolchain::GCC;
    return Toolchain::Unknown;
}

inline ToolchainProfile get_profile(Toolchain tc) {
    switch(tc) {
        case Toolchain::GCC:
        case Toolchain::Clang:
            return {tc, {": error:", " error:", " fatal error:"}, {": warning:", " warning:"}, {"note: "}};
        case Toolchain::MSVC:
            return {tc, {" error C", ": error C", " fatal error C"}, {" warning C", ": warning C"}, {"note: "}};
        case Toolchain::Python:
            return {tc, {"Traceback (most recent call last):", "Error:"}, {"Warning:"}, {"  File \""}};
        case Toolchain::Rust:
            return {tc, {"error[E", "error:"}, {"warning:"}, {" --> "}};
        default:
            return {tc, {"error", "Error", "ERROR", "Exception"}, {"warning", "Warning", "WARNING"}, {}};
    }
}

struct LogLine {
    std::string_view text;
    size_t weight;
    double value;
    size_t original_index;
    bool selected = false;
};

inline std::string compact(std::string_view log, size_t budget) {
    if (log.size() <= budget) {
        return std::string(log);
    }

    Toolchain tc = detect_toolchain(log);
    ToolchainProfile profile = get_profile(tc);

    std::vector<LogLine> lines;
    size_t start = 0;
    size_t index = 0;
    while (start < log.size()) {
        size_t end = log.find('\n', start);
        if (end == std::string_view::npos) {
            end = log.size();
        } else {
            end++; // Include newline in the weight
        }
        std::string_view line = log.substr(start, end - start);
        lines.push_back({line, line.size(), 0.0, index++, false});
        start = end;
    }

    if (lines.empty()) return "";

    const size_t num_lines = lines.size();
    const size_t head_pin = std::min<size_t>(3, num_lines);
    const size_t tail_pin = std::min<size_t>(3, num_lines);

    for (size_t i = 0; i < num_lines; ++i) {
        double density_multiplier = 1.0;

        if (i < head_pin || i >= num_lines - tail_pin) {
            density_multiplier += 10.0; // Pin head and tail
        }

        // Apply toolchain heuristics
        for (const auto& sig : profile.error_signatures) {
            if (lines[i].text.find(sig) != std::string_view::npos) {
                density_multiplier += 10000.0;
                break;
            }
        }
        for (const auto& sig : profile.warning_signatures) {
            if (lines[i].text.find(sig) != std::string_view::npos) {
                density_multiplier += 1000.0;
                break;
            }
        }
        for (const auto& sig : profile.context_signatures) {
            if (lines[i].text.find(sig) != std::string_view::npos) {
                density_multiplier += 100.0;
                break;
            }
        }

        // Give slightly higher value to lines closer to the end, often where the real error is.
        density_multiplier += static_cast<double>(i) / static_cast<double>(num_lines) * 2.0;

        // Multiply weight by density_multiplier to get value, so density will be exactly density_multiplier
        lines[i].value = static_cast<double>(lines[i].weight) * density_multiplier;
    }

    std::vector<size_t> indices(num_lines);
    for (size_t i = 0; i < num_lines; ++i) indices[i] = i;

    // Greedy Knapsack: sort by value density (value / weight) descending
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        double density_a = lines[a].value / static_cast<double>(lines[a].weight);
        double density_b = lines[b].value / static_cast<double>(lines[b].weight);
        if (density_a != density_b) return density_a > density_b;
        return a > b; // Prefer later lines on tie
    });

    size_t current_weight = 0;

    for (size_t idx : indices) {
        if (current_weight + lines[idx].weight <= budget) {
            current_weight += lines[idx].weight;
            lines[idx].selected = true;
        }
    }

    std::string result;
    result.reserve(current_weight);

    for (size_t i = 0; i < num_lines; ++i) {
        if (lines[i].selected) {
            result.append(lines[i].text);
        }
    }

    return result;
}

} // namespace log_triage
