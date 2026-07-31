#include "src/model/mlx_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <variant>

#include "src/model/sampler.hpp"

#ifdef LMP_HAVE_MLX
#include <mlx/array.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/device.h>
#include <mlx/memory.h>
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

// MLX's wired limit defaults to 0: nothing is kept resident, so a 19 GB checkpoint is
// re-made-resident by the OS around GPU dispatches. That is invisible to any op-level
// benchmark -- a microbenchmark touches a small hot set and never pays it -- but it
// throttles a real decode step, where every layer walks a different 8-of-256 slice of
// the expert weights. mlx-lm wires the whole working set in generate.py's wired_limit()
// context manager, which is why the same ops on the same MLX decode faster there.
//
// max_recommended_working_set_size is what mlx-lm passes, and a value above the system
// wired limit is an error, so clamp to it rather than to the model size.
void wire_working_set() {
    if (!mx::metal::is_available()) {
        return;
    }
    // mx::metal::device_info() is declared in the headers but no longer exported by the
    // 0.31.2 dylib; mx::device_info() is the current spelling and carries the same keys.
    const auto& info = mx::device_info();
    const auto it = info.find("max_recommended_working_set_size");
    if (it == info.end()) {
        return;
    }
    if (const auto* limit = std::get_if<std::size_t>(&it->second)) {
        mx::set_wired_limit(*limit);
    }
}

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
    wire_working_set();
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

// One logits row -> CPU floats. The sync point per decode step (S5.11).
//
// The float32 cast is deliberately issued AFTER the forward has been evaluated, not
// folded into its graph. Folding it in reads better -- one round-trip instead of two --
// and measures worse: 84.8 -> 83.9 tok/s, reproduced across three runs. As a separate
// tiny dispatch it costs nothing; on the end of the step's graph it extends the critical
// path. Do not "simplify" this without re-running `lmp_diag bench`.
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
        const auto t_s0 = clock_.mono();
        // ONE mask lookup per step, not one predicate call per vocabulary id.
        const TokenMask* mask = task.mask != nullptr ? &task.mask->mask() : nullptr;
        const SampleResult pick = sampler.sample(logits_host, mask, recent);
        r.sample_ms += ms_between(t_s0, clock_.mono());
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

        const auto t_f0 = clock_.mono();
        mx::array ids = mx::array(&pick.id, {1, 1}, mx::int32);
        mx::array logits = impl_->model.forward_logits(ids);
        mx::eval(logits);
        const auto t_f1 = clock_.mono();
        logits_to_host(logits, logits_host);
        const auto t_f2 = clock_.mono();
        r.forward_ms += ms_between(t_f0, t_f1);
        r.logits_copy_ms += ms_between(t_f1, t_f2);
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
