#include "src/model/mlx_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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
    // The turn checkpoint: the caches as they stood at the end of the last prompt's
    // STABLE prefix. Held here rather than in the header because CacheCheckpoint is
    // MLX-shaped and mlx headers stay out of mlx_backend.hpp.
    struct TurnCheckpoint {
        mlxl::Qwen35MoeModel::CacheCheckpoint cp{};
        std::size_t len = 0;
        bool valid = false;
    } ckpt;
};

// Tokens per prefill eval. Each chunk ends in a full synchronous barrier
// (`eval_caches`), so the chunk size sets how often prefill drains the GPU and rebuilds
// a 48-layer graph from the host. 512 made prefill roughly flat in prompt length
// (1118 tok/s at 547 tokens, 1170 at 8240) because the per-chunk cost dominated; 2048 --
// which is also mlx-lm's `prefill_step_size` default -- measures 1317 and 1684 on the
// same prompts. Above 2048 it turns over again (1528 at 8240), so this is the knee.
// It costs peak memory only on long prompts, where the bigger activation is live:
// unchanged at 19.00 GB for a 547-token prompt, 19.08 -> 20.41 GB at 8240, against the
// 20.18 GB mlx-lm peaks at on this checkpoint. Overridable so the sweep can be re-run
// without a rebuild.
std::size_t prefill_chunk() {
    if (const char* s = std::getenv("LMP_PREFILL_CHUNK")) {
        const int v = std::atoi(s);
        if (v > 0) {
            return static_cast<std::size_t>(v);
        }
    }
    return 2048;
}

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
    spec_ = config.speculative;
    loaded_ = true;
    return {true, {}};
}

void MlxBackend::reset_cache() {
    if (impl_) {
        impl_->model.reset_cache();
        impl_->ckpt = {};
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

// Speculation on without a rebuild, so the two paths can be compared on one binary --
// which matters because the only number that settles this is a real end-to-end run.
bool speculative_env_override(bool from_config) {
    if (const char* s = std::getenv("LMP_SPECULATIVE")) {
        return std::atoi(s) != 0;
    }
    return from_config;
}

// The model, as src/model/speculative.hpp needs to see it. Everything MLX-shaped lives
// here so the block algebra itself stays testable without a GPU.
class MlxSpecForward final : public SpecForward {
  public:
    MlxSpecForward(mlxl::Qwen35MoeModel& model, KvCacheLedger& ledger)
        : model_(model), ledger_(ledger) {}

    void forward_all(std::span<const TokenId> tokens,
                     std::vector<std::vector<float>>& rows) override {
        rows.clear();
        if (tokens.empty()) {
            return;
        }
        mx::array ids = mx::array(tokens.data(), {1, static_cast<int>(tokens.size())}, mx::int32);
        // The opt-in all-position path: verification needs a row per drafted position,
        // and forward_logits() would slice all but the last away.
        mx::array logits = mx::astype(impl_forward_all(ids), mx::float32);
        mx::eval(logits);
        const int seq = static_cast<int>(logits.shape()[1]);
        const int vocab = static_cast<int>(logits.shape()[2]);
        const float* data = logits.data<float>();
        rows.reserve(static_cast<std::size_t>(seq));
        for (int s = 0; s < seq; ++s) {
            rows.emplace_back(data + static_cast<std::ptrdiff_t>(s) * vocab,
                              data + static_cast<std::ptrdiff_t>(s + 1) * vocab);
        }
        // The ledger tracks what the MODEL consumed, drafts included -- restore() puts it
        // back in step with the caches.
        ledger_.append(tokens);
    }

    void forward_last(std::span<const TokenId> tokens, std::vector<float>& row) override {
        if (tokens.empty()) {
            return;
        }
        mx::array ids = mx::array(tokens.data(), {1, static_cast<int>(tokens.size())}, mx::int32);
        mx::array logits = impl_forward_last(ids);
        mx::eval(logits);
        logits_to_host(logits, row);
        ledger_.append(tokens);
    }

    void checkpoint() override {
        mark_ = model_.checkpoint();
        ledger_mark_ = ledger_.size();
    }

    void restore() override {
        model_.restore(mark_);
        // Exactly the same position the caches went back to. A ledger that disagreed with
        // the caches is the silent-stale-context failure S5.10 exists to prevent.
        ledger_.truncate_to(ledger_mark_);
    }

  private:
    mx::array impl_forward_all(const mx::array& ids) { return model_.forward_logits_all(ids); }
    mx::array impl_forward_last(const mx::array& ids) { return model_.forward_logits(ids); }

    mlxl::Qwen35MoeModel& model_;
    KvCacheLedger& ledger_;
    mlxl::Qwen35MoeModel::CacheCheckpoint mark_{};
    std::size_t ledger_mark_ = 0;
};

// The speculative decode loop. Separate from generate() so the plain path keeps the shape
// it was tuned in, and so this stays under the function-size ratchet.
GenResult decode_speculative(mlxl::Qwen35MoeModel& model, KvCacheLedger& ledger,
                             const InferenceTask& task, TokenSink& sink,
                             const CancelToken& cancel, const platform::Clock& clock,
                             std::vector<float>& logits_host, GenResult r,
                             platform::MonoTime t0, SpecConfig cfg) {
    SpeculativeDecoder decoder(task.sampling, cfg);
    MlxSpecForward fwd(model, ledger);
    // The proposer matches against history, so it has to have seen the prompt: an agent
    // turn reuses the tool output and the code it just read, which is the whole premise.
    decoder.observe(std::span<const TokenId>(task.prompt));

    const auto is_special = [&task](TokenId id) {
        return task.mask != nullptr && task.mask->is_block_boundary(id);
    };

    std::vector<TokenId> recent;
    bool first_token = true;
    auto t_decode_start = clock.mono();
    bool stop = false;

    while (!stop && r.tokens_generated < task.max_new_tokens) {
        if (cancel.cancelled()) {
            r.status = GenStatus::Cancelled;
            return r;
        }
        const TokenMask* mask = task.mask != nullptr ? &task.mask->mask() : nullptr;
        const bool may_speculate = task.mask == nullptr || task.mask->mask_is_block_stable();

        const auto t_s0 = clock.mono();
        // ledger.ids() is the exact history the model consumed, which is what the
        // proposer must match against -- not the prompt, and not the emitted text.
        SpecStep st = decoder.step(logits_host, mask, recent,
                                   std::span<const TokenId>(ledger.ids()), may_speculate,
                                   is_special, fwd);
        r.forward_ms += ms_between(t_s0, clock.mono());

        if (st.no_legal_token) {
            r.status = GenStatus::BackendError;
            r.error = "constrained decode: no legal token -- the grammar and the "
                      "vocabulary disagree, which is a build defect";
            return r;
        }

        for (TokenId id : st.committed) {
            if (first_token) {
                r.ttft_ms = ms_between(t0, clock.mono());
                t_decode_start = clock.mono();
                first_token = false;
            }
            ++r.tokens_generated;
            recent.push_back(id);
            if (recent.size() > 64) {
                recent.erase(recent.begin());
            }
            // The sink is the authority on when a turn ends (S5.5). A block may commit
            // several tokens at once, and the grammar must not be advanced past the one
            // that accepted -- so stop feeding at the first refusal and drop the rest.
            if (!sink.on_token(id)) {
                r.status = GenStatus::Complete;
                stop = true;
                break;
            }
            if (r.tokens_generated >= task.max_new_tokens) {
                stop = true;
                break;
            }
        }
        decoder.observe(std::span<const TokenId>(st.committed));
        logits_host = std::move(st.next_logits);
    }

    if (r.status != GenStatus::Complete) {
        r.status = GenStatus::LengthCapped;
    }
    const double decode_ms = ms_between(t_decode_start, clock.mono());
    r.decode_tok_per_s =
        decode_ms > 0 ? static_cast<double>(r.tokens_generated) / (decode_ms / 1000.0) : 0.0;
    const SpecStats& s = decoder.stats();
    // Printed, not silently accumulated: a speculative run whose acceptance rate is on the
    // floor is slower than not speculating, and that has to be visible without a profiler.
    std::fprintf(stderr,
                 "  [spec] blocks=%llu drafted=%llu accepted=%llu (%.1f%%) committed=%llu "
                 "fallbacks=%llu\n",
                 static_cast<unsigned long long>(s.blocks),
                 static_cast<unsigned long long>(s.drafted),
                 static_cast<unsigned long long>(s.accepted_drafts),
                 100.0 * s.acceptance_rate(),
                 static_cast<unsigned long long>(s.committed),
                 static_cast<unsigned long long>(s.fallbacks));
    return r;
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
    //
    // Three outcomes, decided by a pure function so the algebra is gate-testable with no
    // GPU. Restore is the one that is new: between turns the prompt always diverges
    // mid-ledger (a new turn record is inserted before the live-state block), and the old
    // code answered that with a full reset -- so every turn re-prefilled the whole
    // context and the most-stable-first prompt layout bought nothing.
    const TurnReuse plan =
        plan_turn_reuse(ledger_, task.prompt, impl_->ckpt.len, impl_->ckpt.valid);
    switch (plan.mode) {
        case ReuseMode::Extend:
            break;
        case ReuseMode::Restore:
            impl_->model.restore(impl_->ckpt.cp);
            // Exactly the position the caches went back to. A ledger that disagreed with
            // the caches is the silent-stale-context failure S5.10 exists to prevent.
            ledger_.truncate_to(impl_->ckpt.len);
            break;
        case ReuseMode::Reset:
            // Stale context is never decoded past. One honest full re-prefill.
            impl_->model.reset_cache();
            ledger_.clear();
            impl_->ckpt = {};
            break;
    }
    const std::size_t start = plan.prefill_from;
    r.prefill_reused_tokens = start;

    const auto t0 = clock_.mono();

    // --- chunked prefill ----------------------------------------------------
    const std::size_t kPrefillChunk = prefill_chunk();
    std::vector<float> logits_host;
    const std::size_t prompt_n = task.prompt.size();
    // The stable boundary is made a CHUNK EDGE so the snapshot lands exactly on it. One
    // extra edge per turn, against a full GPU barrier every 512 tokens anyway.
    const std::size_t boundary =
        task.checkpoint_at > start && task.checkpoint_at <= prompt_n ? task.checkpoint_at
                                                                     : 0;
    // Everything before the last token is pure prefill; the last token's forward pass
    // produces the first sampling distribution.
    // A while loop, not `at += kPrefillChunk`: the boundary shortens a chunk, so the step
    // is whatever was actually consumed.
    std::size_t at = start;
    while (at < prompt_n) {
        if (cancel.cancelled()) {
            r.status = GenStatus::Cancelled;
            return r;
        }
        std::size_t end = std::min(at + kPrefillChunk, prompt_n);
        if (boundary > at && boundary < end) {
            end = boundary;
        }
        mx::array ids = mx::array(task.prompt.data() + static_cast<std::ptrdiff_t>(at),
                                  {1, static_cast<int>(end - at)}, mx::int32);
        mx::array logits = impl_->model.forward_logits(ids);
        impl_->model.eval_caches();
        for (std::size_t i = at; i < end; ++i) {
            ledger_.append(task.prompt[i]);
        }
        if (end == boundary) {
            // Exactly one checkpoint is held, and it is overwritten each turn -- which is
            // what bounds the memory the 30 gated-delta snapshots cost.
            impl_->ckpt.cp = impl_->model.checkpoint();
            impl_->ckpt.len = ledger_.size();
            impl_->ckpt.valid = true;
        }
        if (end == prompt_n) {
            logits_to_host(logits, logits_host);
        }
        at = end;
    }
    const auto t_prefill = clock_.mono();
    const double prefill_ms = ms_between(t0, t_prefill);
    const auto prefilled = static_cast<double>(prompt_n - start);
    r.prefill_tok_per_s = prefill_ms > 0 ? prefilled / (prefill_ms / 1000.0) : 0.0;

    // --- decode -------------------------------------------------------------
    //
    // Two loops, not one with branches in it. The plain path below is tuned and is the
    // reference: when speculation is off it must run the code it ran before speculation
    // existed, not a version of it carrying an `if` per step.
    if (speculative_env_override(spec_.enabled)) {
        SpecConfig cfg = spec_;
        cfg.enabled = true;
        return decode_speculative(impl_->model, ledger_, task, sink, cancel, clock_,
                                  logits_host, r, t0, cfg);
    }

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
