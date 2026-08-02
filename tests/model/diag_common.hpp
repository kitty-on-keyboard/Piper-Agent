// Shared scaffolding for the attribution driver (S19.3), used by both translation units:
// the MoE subcommands live in diag_moe.cpp and everything else in diag_main.cpp.
#ifndef LMP_TESTS_MODEL_DIAG_COMMON_HPP
#define LMP_TESTS_MODEL_DIAG_COMMON_HPP

#include <chrono>
#include <cstdlib>

namespace lmp::diag {

using Clock = std::chrono::steady_clock;

inline double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

inline const char* qwen_dir() {
    const char* v = std::getenv("LMP_QWEN_DIR");
    return v != nullptr
               ? v
               : "/Users/dev/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit";
}

#if LMP_HAVE_MLX
int cmd_blocks(int T);
int cmd_moe(int T);
int cmd_moe_stream(int iters);
#endif

} // namespace lmp::diag

#endif // LMP_TESTS_MODEL_DIAG_COMMON_HPP
