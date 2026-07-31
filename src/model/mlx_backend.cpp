#include "src/model/mlx_backend.hpp"

#include <algorithm>
#include <chrono>

#include "src/model/sampler.hpp"

#ifdef LMP_HAVE_MLX
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "src/model/mlx/qwen35_moe_model.hpp"
#endif

namespace lmp::model {

#ifdef LMP_HAVE_MLX

namespace mx = mlx::core;

struct MlxBackend::Impl {
    mlxl::Qwen35MoeModel model;
};

MlxBackend::MlxBackend(const platform::Clock& clock) : clock_(clock) {}
MlxBackend::~MlxBackend() = default;

LoadStatus MlxBackend::load(const MlxBackendConfig& config) {
    if (loaded_) {
        return {false, "MlxBackend: already loaded; one model load per process (S5.11)"};
    }
    if (config.model_dir.empty()) {
        return {false, "model_dir is required and empty (S7.5)"};
    }
    impl_ = std::make_unique<Impl>();
    if (!impl_->model.load(config.model_dir)) {
        impl_.reset();
        return {false, config.model_dir + ": model load failed (missing config.json or "
                       "safetensors)"};
    }
    loaded_ = true;
    return {true, {}};
}

void MlxBackend::reset_cache() {
    if (impl_) {
        impl_->model.reset_cache();
    }
    ledger_.clear();
}

namespace {

// One logits row -> CPU floats. The single sync point per decode step (S5.11).
void logits_to_host(const mx::array& logits, std::vector<float>& out) {
    mx::array row = mx::astype(mx::reshape(logits, {-1}), mx::float32);
    mx::eval(row);
    const auto* data = row.data<float>();
    out.assign(data, data + row.size());
}

double ms_between(platform::MonoTime a, platform::MonoTime b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

} // namespace

GenResult MlxBackend::generate(const InferenceTask& task, TokenSink& sink,
                               const CancelToken& cancel) {
    GenResult r;
    if (!loaded_) {
        r.error = "MlxBackend: no model loaded";
        return r;
    }
    if (task.prompt.empty()) {
        r.error = "MlxBackend: empty prompt";
        return r;
    }

    // --- verified prefix reuse (S5.10) -------------------------------------
    const ReuseDecision reuse = ledger_.plan_reuse(task.prompt);
    if (reuse.divergent) {
        // Stale context is never decoded past. One honest full re-prefill.
        impl_->model.reset_cache();
        ledger_.clear();
    }
    const std::size_t start = reuse.divergent ? 0 : reuse.reusable;

    const auto t0 = clock_.mono();

    // --- chunked prefill ----------------------------------------------------
    constexpr std::size_t kPrefillChunk = 512;
    std::vector<float> logits_host;
    const std::size_t prompt_n = task.prompt.size();
    // Everything before the last token is pure prefill; the last token's forward pass
    // produces the first sampling distribution.
    for (std::size_t at = start; at < prompt_n; at += kPrefillChunk) {
        if (cancel.cancelled()) {
            r.status = GenStatus::Cancelled;
            return r;
        }
        const std::size_t end = std::min(at + kPrefillChunk, prompt_n);
        mx::array ids = mx::array(task.prompt.data() + at,
                                  {1, static_cast<int>(end - at)}, mx::int32);
        mx::array logits = impl_->model.forward_logits(ids);
        impl_->model.eval_caches();
        for (std::size_t i = at; i < end; ++i) {
            ledger_.append(task.prompt[i]);
        }
        if (end == prompt_n) {
            logits_to_host(logits, logits_host);
        }
    }
    const auto t_prefill = clock_.mono();
    const double prefill_ms = ms_between(t0, t_prefill);
    const auto prefilled = static_cast<double>(prompt_n - start);
    r.prefill_tok_per_s = prefill_ms > 0 ? prefilled / (prefill_ms / 1000.0) : 0.0;

    // --- decode -------------------------------------------------------------
    Sampler sampler(task.sampling);
    std::vector<TokenId> recent;
    bool first_token = true;
    auto t_decode_start = clock_.mono();

    while (r.tokens_generated < task.max_new_tokens) {
        if (cancel.cancelled()) {
            r.status = GenStatus::Cancelled;
            return r;
        }
        const SampleResult pick = sampler.sample(logits_host, task.mask, recent);
        if (pick.no_legal_token) {
            r.status = GenStatus::BackendError;
            r.error = "constrained decode: no legal token -- the grammar and the "
                      "vocabulary disagree, which is a build defect";
            return r;
        }
        if (first_token) {
            r.ttft_ms = ms_between(t0, clock_.mono());
            t_decode_start = clock_.mono();
            first_token = false;
        }
        ++r.tokens_generated;
        recent.push_back(pick.id);
        if (recent.size() > 64) {
            recent.erase(recent.begin());
        }
        ledger_.append(pick.id);

        const bool keep_going = sink.on_token(pick.id);
        if (!keep_going) {
            r.status = GenStatus::Complete;
            break;
        }

        mx::array ids = mx::array(&pick.id, {1, 1}, mx::int32);
        mx::array logits = impl_->model.forward_logits(ids);
        logits_to_host(logits, logits_host);
    }
    if (r.status != GenStatus::Complete) {
        r.status = GenStatus::LengthCapped;
    }
    const double decode_ms = ms_between(t_decode_start, clock_.mono());
    r.decode_tok_per_s =
        decode_ms > 0 ? static_cast<double>(r.tokens_generated) / (decode_ms / 1000.0) : 0.0;
    return r;
}

#else // !LMP_HAVE_MLX

struct MlxBackend::Impl {};

MlxBackend::MlxBackend(const platform::Clock& clock) : clock_(clock) {}
MlxBackend::~MlxBackend() = default;

LoadStatus MlxBackend::load(const MlxBackendConfig& config) {
    (void)config;
    return {false,
            "MLX is not compiled into this build. Install the mlx pip package (it ships "
            "the C++ SDK) and reconfigure; src/model/CMakeLists.txt probes "
            "`python3 -m mlx --cmake-dir` automatically. ScriptedBackend and "
            "ReplayBackend drive the identical loop in the meantime."};
}

void MlxBackend::reset_cache() { ledger_.clear(); }

GenResult MlxBackend::generate(const InferenceTask& task, TokenSink& sink,
                               const CancelToken& cancel) {
    (void)task;
    (void)sink;
    (void)cancel;
    GenResult r;
    r.status = GenStatus::BackendError;
    r.error = "MlxBackend: MLX not compiled in";
    return r;
}

#endif // LMP_HAVE_MLX

} // namespace lmp::model
