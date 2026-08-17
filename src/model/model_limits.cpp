#include "src/model/model_limits.hpp"

#include <limits>

#include "src/model/mlx/qwen35_moe_config.hpp"

namespace lmp::model {

int load_max_position_embeddings(const std::string& model_dir) {
    return mlxl::load_max_position_embeddings(model_dir);
}

std::size_t kv_bytes_per_token(const std::string& model_dir) {
    return mlxl::kv_bytes_per_token(model_dir);
}

// The two constants are the whole of the policy, and both are fitted to the run that died
// rather than chosen for tidiness.
//
// kCacheOverheadPercent is MEASURED and kSafeWorkingSetPercent is CALIBRATED, and the
// difference between those two words is the whole design of this function.
//
// kCacheOverheadPercent -- MLX's allocator cache is not free space. It tracked the KV
// working set at roughly 1.6x for the whole run (KV 8.21 GB against 13.4 GB of cache), and
// while it IS reclaimable, counting it as headroom is exactly what made 38 GB look
// survivable. So a token is charged for its KV plus 160% of it. That number comes off the
// event log and is not a knob to turn for a nicer answer.
//
// kSafeWorkingSetPercent -- the device's max_recommended_working_set_size is not a safe
// TOTAL on a machine also running two editors and a game engine. The death landed at
// 38.0 GB of 40.2 GB, i.e. 94.5%, so the safe share is definitely below that.
//
// It is set to 88 rather than anything tighter because of a second piece of evidence that
// outranks arithmetic: 96,000 is the budget the extension has shipped as its default and
// has run on this machine for months without a memory death. A guard whose first act is to
// clamp a known-good default is wrong about the default, not right about the danger -- and
// it would fire a `context_budget` clamp on every ordinary run, which is how a real signal
// gets tuned out. 85% put the limit at 93,296 and did precisely that.
//
// On this host: (0.88 * 40.2 - 16.29) GB / (72 KB * 2.6) ~= 99,600 tokens. It admits the
// 96,000 default, refuses the 112,088 the process actually died at, and refuses the
// 245,760 that was configured. Those three verdicts are pinned in test_model_bounds.
//
// One host is one data point. mlx_backend.cpp is emphatic that memory fixes must be judged
// against a real run's BEHAVIOUR and not only its bytes -- a cache cap that balanced
// perfectly on paper made runs measurably worse and was reverted. If this clamps something
// that demonstrably worked, the evidence wins and the constant moves.
int max_affordable_context_tokens(std::size_t kv_per_token, std::size_t weights_bytes,
                                  std::size_t device_working_set_bytes) {
    constexpr std::size_t kSafeWorkingSetPercent = 88;
    constexpr std::size_t kCacheOverheadPercent = 160;
    if (kv_per_token == 0 || device_working_set_bytes == 0) {
        return 0; // cannot tell; the caller leaves the budget alone
    }
    const std::size_t safe_total =
        device_working_set_bytes / 100 * kSafeWorkingSetPercent;
    if (safe_total <= weights_bytes) {
        return 0; // the weights alone are over the line; not this function's call to make
    }
    const std::size_t for_kv = safe_total - weights_bytes;
    const std::size_t charged_per_token =
        kv_per_token + kv_per_token / 100 * kCacheOverheadPercent;
    if (charged_per_token == 0) {
        return 0;
    }
    const std::size_t tokens = for_kv / charged_per_token;
    return tokens > static_cast<std::size_t>(std::numeric_limits<int>::max())
               ? std::numeric_limits<int>::max()
               : static_cast<int>(tokens);
}

bool checkpoint_declares_vision(const std::string& model_dir) {
    mlxl::Qwen35VisionConfig cfg;
    return mlxl::load_qwen35_vision_config(model_dir, cfg) && cfg.present;
}

} // namespace lmp::model
