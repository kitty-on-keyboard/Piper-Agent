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
#include <mlx/stream.h>
#include <mlx/transforms.h>

#include "src/model/mlx/qwen35_moe_model.hpp"
#endif

namespace lmp::model {

#ifdef LMP_HAVE_MLX

namespace mx = mlx::core;

struct MlxBackend::Impl {
    mlxl::Qwen35MoeModel model;
    // What wire_working_set() displaced, so unloading can put it back. Held here rather
    // than in the header because the no-MLX build never sets it, and an unread private
    // member is a -Wunused-private-field error in this tree.
    std::size_t prev_wired_limit = 0;
    bool wired = false;
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
// Returns the displaced limit through `previous`, and whether it wired anything at all:
// undoing this on unload is what gives the memory back (see ~MlxBackend), and undoing a
// limit that was never raised would pin the WRONG value.
bool wire_working_set(std::size_t& previous) {
    if (!mx::metal::is_available()) {
        return false;
    }
    // mx::metal::device_info() is declared in the headers but no longer exported by the
    // 0.31.2 dylib; mx::device_info() is the current spelling and carries the same keys.
    const auto& info = mx::device_info();
    const auto it = info.find("max_recommended_working_set_size");
    if (it == info.end()) {
        return false;
    }
    if (const auto* limit = std::get_if<std::size_t>(&it->second)) {
        previous = mx::set_wired_limit(*limit);
        return true;
    }
    return false;
}

// No ceiling is put on MLX's allocator cache. One WAS, on 2026-08-08, and it is the
// reason this comment exists rather than a call.
//
// The problem it addressed is real and still open: `unload_model` has repeatedly reported
// ~18 GB of allocator cache on top of a ~20 GB WIRED model, and a run died with no crash
// report at ~38 GB resident. Capping retention at a derived 4.75 GB fixed that arithmetic
// and measurably held (active 19.51 GB, cache 0.89 GB).
//
// It also made runs WORSE, which the arithmetic did not predict and which is the only
// thing that decides. The plan run immediately after it shipped got further before the
// change than after: prefill slowed (turns at 5-6 s time-to-first-token where the same
// shape had been sub-500 ms), and the model stopped calling tools mid-exploration. The
// mechanism is not established -- reclaiming buffers cannot change a number -- but the
// correlation was direct and the user observed it across runs.
//
// So it is out, and the memory problem goes back to open. What stays is the measurement
// that found it: `model_memory` at load, `unload_model` at the end. Whatever fixes this
// next needs to be checked against a real run's BEHAVIOUR, not only against its bytes --
// this was verified by loading a model and reading two numbers, and that was not enough.
std::size_t g_cache_limit = 0;

// PUT MLX'S SAFETY VALVE BELOW THE CLIFF INSTEAD OF ABOVE IT.
//
// MLX's memory limit defaults to 1.5x the device's max recommended working set
// (mlx/memory.h). On this 48 GB machine that is ~60 GB -- MORE THAN THE MACHINE HAS. The
// limit exists so that an allocation which cannot be satisfied raises instead of running
// the system out of memory, and at 60 GB it can never do that: the OS reaches its own
// limit first and kills the process, which is uncatchable, unreportable, and exactly what
// has been happening.
//
// Setting it to the working set the device actually reports moves the valve to the right
// side of the cliff. An over-large allocation now throws, generate() catches it and the
// turn fails with MLX's own message naming the size it wanted; the model stays loaded and
// the run reports instead of vanishing.
//
// THIS IS NOT THE CACHE CAP THAT WAS REVERTED. That one bounded what the allocator
// RETAINS -- 4.75 GB against 18 GB observed -- and forced reclaim on every allocation in
// steady state, which is where the slowdown came from. This changes nothing in steady
// state: the cache limit follows the memory limit and lands at ~40 GB, far above the
// ~18 GB high-water mark ever measured, so nothing is reclaimed that would not have been.
// Only the edge changes, and only from "killed" to "reported".
void set_memory_ceiling() {
    if (!mx::metal::is_available()) {
        return;
    }
    const auto& info = mx::device_info();
    const auto it = info.find("max_recommended_working_set_size");
    if (it == info.end()) {
        return;
    }
    if (const auto* working_set = std::get_if<std::size_t>(&it->second)) {
        mx::set_memory_limit(*working_set);
        mx::set_cache_limit(*working_set);
        g_cache_limit = *working_set;
    }
}

MemoryReport mlx_memory_report() {
    return {mx::get_active_memory(), mx::get_cache_memory(), mx::get_peak_memory()};
}

std::size_t mlx_cache_limit() { return g_cache_limit; }

MlxBackend::MlxBackend(const platform::Clock& clock) : clock_(clock) {}

// UNLOADING HAS TO SAY SO TO MLX. Destroying the arrays is necessary and not sufficient:
// MLX's allocator returns their Metal buffers to its OWN cache, not to the OS, and the
// working set is still wired resident from load. So `lmp/unload_model` replied
// `{"unloaded":true}`, the surface said "unloaded", and all ~19 GB stayed resident until
// the process exited -- which is the whole reason unload existed.
//
// Three steps, in this order:
//   synchronize   -- pending GPU work holds references to the very buffers being freed
//   destroy       -- arrays go, their buffers land in MLX's cache
//   clear_cache   -- the cache gives them back to the OS. This is the step that was
//                    missing; without it the two above only move the memory sideways.
//   unwire        -- restore the limit load() displaced, so nothing stays pinned
//
// This also runs on the RELOAD path, where load_model() resets the backend before
// reading new weights precisely so the peak is one checkpoint and not two.
MlxBackend::~MlxBackend() {
    if (!impl_) {
        return;
    }
    const std::size_t prev_wired = impl_->prev_wired_limit;
    const bool wired = impl_->wired;
    mx::synchronize();
    impl_.reset();
    mx::clear_cache();
    if (wired) {
        mx::set_wired_limit(prev_wired);
    }
}

LoadStatus MlxBackend::load(const MlxBackendConfig& config) {
    if (loaded_) {
        return {false, "MlxBackend: already loaded; one model load per process (S5.11)"};
    }
    if (config.model_dir.empty()) {
        return {false, "model_dir is required and empty (S7.5)"};
    }
    impl_ = std::make_unique<Impl>();
    // load() reports absence by returning false, but it can also THROW: mx::load_safetensors
    // on a dtype this MLX cannot parse, or WeightStore::get on a checkpoint whose tensor
    // names we do not know. generate() has always caught its own throws; this path did not,
    // so an unfamiliar checkpoint terminated the sidecar with no report at all -- the one
    // failure mode that tells the operator nothing. Report it like any other load failure.
    bool ok = false;
    std::string threw;
    try {
        ok = impl_->model.load(config.model_dir);
    } catch (const std::exception& e) {
        threw = e.what();
    } catch (...) {
        threw = "unknown exception";
    }
    if (!ok) {
        const std::string why = impl_->model.load_error();
        // A load that got part way still allocated, and those buffers are in MLX's cache
        // exactly as a successful load's would be. Giving them back here matters most on
        // the reload path, where the caller is about to try a different checkpoint.
        impl_.reset();
        mx::clear_cache();
        if (!threw.empty()) {
            return {false, config.model_dir + ": model load failed: " + threw};
        }
        if (!why.empty()) {
            return {false, config.model_dir + ": " + why};
        }
        return {false, config.model_dir + ": model load failed (missing config.json or "
                       "safetensors)"};
    }
    spec_ = config.speculative;
    if (!config.draft_model_dir.empty()) {
        // Upstream reports MTP crashing against the MoE target (mlx-vlm #1317), and the
        // head we ship is split from the dense 27B. Refuse the pairing rather than inherit
        // a failure that surfaces at first generation.
        if (!impl_->model.is_dense()) {
            const std::string why = config.draft_model_dir +
                                    ": an MTP draft head is only supported on the dense "
                                    "target (model_type qwen3_5 / qwen3_5_text)";
            impl_.reset();
            mx::clear_cache();
            return {false, why};
        }
        // A hard failure, not a warning. The operator asked for a drafter; running without
        // one would be slower than requested and say nothing about why.
        if (!impl_->model.load_mtp(config.draft_model_dir)) {
            const std::string why =
                config.draft_model_dir +
                ": not a usable MTP draft head (needs config.json with model_type "
                "qwen3_5_mtp and block_size >= 2, plus fc / norm / layers.0 tensors)";
            impl_.reset();
            mx::clear_cache();
            return {false, why};
        }
        spec_.mtp_block_size = static_cast<std::size_t>(impl_->model.mtp_block_size());
        // A LOADED DRAFT HEAD MEANS SPECULATION IS ON. There is no other reason to name
        // one: it is 253 MB of resident weights whose only consumer is this path, and
        // leaving `enabled` at its default made asking for a head cost memory and deliver
        // nothing. Nobody would have found that from the outside -- the head loads, the
        // log says so, and decode is simply unchanged.
        //
        // LMP_SPECULATIVE still wins over this, in both directions, which is what keeps
        // the A/B available on one binary: 0 forces the plain path back on with the head
        // still loaded.
        spec_.enabled = true;
    }
    impl_->wired = wire_working_set(impl_->prev_wired_limit);
    set_memory_ceiling();
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

// How many recent tokens the repetition penalty sees.
//
// WHAT THIS NUMBER CAN AND CANNOT DO, because it was an undocumented 64 and the obvious
// reading of a repetition bug is "the window is too small". It usually is not.
//
// The penalty is applied ONCE PER UNIQUE ID (apply_repetition_penalty, HF semantics; it was
// made non-compounding after per-occurrence division wrote "idlePer cent" into a real
// agent's file). At the pinned Qwen default of 1.05 that turns a winning logit of ~15 into
// ~14.3. A model that is near-certain of its next token stays near-certain, so this cannot
// break a confident degenerate cycle AT ANY WINDOW SIZE -- widening it trades a real risk
// to generated code, where identifiers legitimately recur, for a fraction of a logit.
//
// MEASURED: a Qwen3.6 4-bit run repeated one ~70-token paragraph about fifty times. 64 was
// shorter than the cycle, so no copy was ever in the window beside its predecessor and the
// penalty saw no repetition at all. Raising it to 256 would have made the window longer than
// the cycle and still not stopped it, because 1.05 per unique id is not enough force.
//
// 256 anyway: it is the scale modern stacks use for a recent-context penalty (mlx-lm
// defaults to 20, HF applies over the whole sequence and is the version known to damage
// code), it is comfortably longer than the loops actually observed, and at per-unique-id
// 1.05 the cost to code is small. It is a better default, not a fix.
//
// The fix for a cycle is loop::LoopBreaker, which detects the repeat and ends the turn.
// Qwen's own guidance for endless repetition in long generations is a PRESENCE penalty, not
// a longer repetition window -- if distribution-level pressure is ever wanted here, that is
// the knob to add, and it needs its own measurement.
constexpr std::size_t kPenaltyWindow = 256;

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

// Where a speculative block's wall time actually goes. `lmp_diag verify` prices the
// verification pass on its own; this prices everything wrapped around it, because the
// first end-to-end measurement of speculation on this dense checkpoint came in BELOW
// plain decode (16.2 vs 17.4 tok/s) while the verification pass was only 1.25x of a
// single-token step at 65% acceptance. Those two facts cannot both be the whole story,
// and a phase breakdown is the only thing that says which one is lying.
struct SpecPhases {
    double checkpoint_ms = 0;
    double restore_ms = 0;
    double verify_ms = 0;
    double forward_last_ms = 0;
    std::uint64_t forward_last_positions = 0;
    double hidden_ms = 0;
    double mtp_step_ms = 0;
    double mtp_logits_ms = 0;
    std::uint64_t mtp_logits_calls = 0;
};

// The model, as src/model/speculative.hpp needs to see it. Everything MLX-shaped lives
// here so the block algebra itself stays testable without a GPU.
class MlxSpecForward final : public SpecForward {
  public:
    MlxSpecForward(mlxl::Qwen35MoeModel& model, KvCacheLedger& ledger)
        : model_(model), ledger_(ledger) {}

    [[nodiscard]] const SpecPhases& phases() const noexcept { return phases_; }

    void forward_all(std::span<const TokenId> tokens,
                     std::vector<std::vector<float>>& rows) override {
        rows.clear();
        if (tokens.empty()) {
            return;
        }
        Phase _p(phases_.verify_ms);
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
        Phase _p(phases_.forward_last_ms);
        phases_.forward_last_positions += tokens.size();
        mx::array ids = mx::array(tokens.data(), {1, static_cast<int>(tokens.size())}, mx::int32);
        mx::array logits = impl_forward_last(ids);
        mx::eval(logits);
        logits_to_host(logits, row);
        ledger_.append(tokens);
    }

    void checkpoint() override {
        Phase _p(phases_.checkpoint_ms);
        mark_ = model_.checkpoint();
        ledger_mark_ = ledger_.size();
    }

    void restore() override {
        Phase _p(phases_.restore_ms);
        model_.restore(mark_);
        // Exactly the same position the caches went back to. A ledger that disagreed with
        // the caches is the silent-stale-context failure S5.10 exists to prevent.
        ledger_.truncate_to(ledger_mark_);
    }

    // ---- the MTP head ----------------------------------------------------------------
    //
    // These cross the GPU/host boundary once per drafted token, because the seam is
    // float-vector shaped so the bookkeeping above it can be proved without a GPU. It is a
    // ~20 KB round trip against a step that reads ~14 GB of weights, so the trade is
    // heavily in favour of being able to test the part that fails silently.

    [[nodiscard]] bool has_mtp() const override { return model_.has_mtp(); }

    void last_hidden(std::vector<std::vector<float>>& rows) override {
        rows.clear();
        const std::optional<mx::array>& h = model_.last_hidden();
        if (!h.has_value()) {
            return;
        }
        Phase _p(phases_.hidden_ms);
        mx::array host = mx::astype(*h, mx::float32);
        mx::eval(host);
        const int seq = static_cast<int>(host.shape()[1]);
        const int width = static_cast<int>(host.shape()[2]);
        const float* data = host.data<float>();
        rows.reserve(static_cast<std::size_t>(seq));
        for (int s = 0; s < seq; ++s) {
            rows.emplace_back(data + static_cast<std::ptrdiff_t>(s) * width,
                              data + static_cast<std::ptrdiff_t>(s + 1) * width);
        }
    }

    void mtp_step(TokenId tok, std::span<const float> hidden,
                  std::vector<float>& out_hidden) override {
        out_hidden.clear();
        if (!model_.has_mtp() || hidden.empty()) {
            return;
        }
        Phase _p(phases_.mtp_step_ms);
        const std::array<TokenId, 1> ids_buf{tok};
        mx::array ids = mx::array(ids_buf.data(), {1, 1}, mx::int32);
        mx::array h = mx::array(hidden.data(), {1, 1, static_cast<int>(hidden.size())},
                                mx::float32);
        mx::array next = mx::astype(model_.mtp_forward(ids, h), mx::float32);
        mx::eval(next);
        const float* data = next.data<float>();
        out_hidden.assign(data, data + next.size());
    }

    void mtp_logits(std::span<const float> hidden, std::vector<float>& row) override {
        row.clear();
        if (!model_.has_mtp() || hidden.empty()) {
            return;
        }
        Phase _p(phases_.mtp_logits_ms);
        ++phases_.mtp_logits_calls;
        mx::array h = mx::array(hidden.data(), {1, 1, static_cast<int>(hidden.size())},
                                mx::float32);
        // The TARGET's head. The MTP checkpoint ships none, which is also why a drafted
        // token is not free -- this projection is the most expensive tensor either model
        // touches, ~0.7 GB at 4-bit against the head body's ~0.15 GB.
        mx::array logits = mx::astype(model_.logits_from_hidden_public(h), mx::float32);
        mx::eval(logits);
        const float* data = logits.data<float>();
        row.assign(data, data + logits.size());
    }

    void mtp_trim(std::size_t n) override { model_.mtp_trim(static_cast<int>(n)); }
    void mtp_reset() override { model_.mtp_reset(); }

  private:
    mx::array impl_forward_all(const mx::array& ids) { return model_.forward_logits_all(ids); }
    mx::array impl_forward_last(const mx::array& ids) { return model_.forward_logits(ids); }

    // Accumulates into one phase counter for the lifetime of the scope. Wall time, not
    // GPU time: every one of these phases ends in an mx::eval, so the barrier is inside
    // the scope and the host clock sees the real cost.
    struct Phase {
        double& sink;
        std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
        explicit Phase(double& s) : sink(s) {}
        ~Phase() {
            sink += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        }
    };

    mlxl::Qwen35MoeModel& model_;
    KvCacheLedger& ledger_;
    mlxl::Qwen35MoeModel::CacheCheckpoint mark_{};
    std::size_t ledger_mark_ = 0;
    SpecPhases phases_{};
};

// The speculative decode loop. Separate from generate() so the plain path keeps the exact
// shape it was tuned and measured in: the two share a prefill and diverge completely
// after it, and interleaving them would put a branch in the hot decode step for the
// benefit of whichever mode is off.
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
    // The prefill row. Handed over once: from here the decoder owns the row, because a
    // block that defers its forward has no row to hand back.
    decoder.seed(std::move(logits_host));

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
        // Block-stable, or checkpointable AND opted in.
        //
        // The second arm puts tool-call bodies in scope -- their mask moves per token, so
        // the decoder walks the grammar over the draft instead of reusing one snapshot --
        // and 47% of this agent's generated tokens are inside a call, so it is worth
        // roughly a third of what speculation can ever deliver.
        //
        // It shipped a regression the first time and is back on with the cause fixed: a
        // drafted token could walk the automaton into a state no token in the vocabulary
        // can leave, and the block then shaped an empty distribution there and reported
        // it as a build defect. walk_masks() now refuses to draft into a position with no
        // legal token, and a block that still cannot answer is abandoned for one ordinary
        // token rather than ending the run. See SpecStats::abandoned.
        //
        // LMP_SPEC_IN_TOOLCALLS=0 forces it back off on one binary, which is the control
        // for any measurement of what it is worth.
        const bool spec_in_tool_calls = [] {
            const char* s = std::getenv("LMP_SPEC_IN_TOOLCALLS");
            return s == nullptr || std::atoi(s) != 0;
        }();
        const bool may_speculate = task.mask == nullptr ||
                                   task.mask->mask_is_block_stable() ||
                                   (spec_in_tool_calls && task.mask->can_checkpoint());

        const auto t_s0 = clock.mono();
        // ledger.ids() is the exact history the model consumed, which is what the
        // proposer must match against -- not the prompt, and not the emitted text.
        SpecStep st = decoder.step(task.mask, recent, std::span<const TokenId>(ledger.ids()),
                                   may_speculate, is_special, fwd);
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
            // The SAME window as the plain path. Two copies of this number is how the
            // speculative and non-speculative paths start sampling from different
            // distributions -- which would make speculation observable, and S5.12 says it
            // must not be.
            if (recent.size() > kPenaltyWindow) {
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
    }

    if (r.status != GenStatus::Complete) {
        r.status = GenStatus::LengthCapped;
    }
    const double decode_ms = ms_between(t_decode_start, clock.mono());
    r.decode_tok_per_s =
        decode_ms > 0 ? static_cast<double>(r.tokens_generated) / (decode_ms / 1000.0) : 0.0;
    const SpecStats& s = decoder.stats();
    // Carried on the result, not only printed to stderr: stderr is a diagnostic driver's
    // channel, and under the editor it goes nowhere anyone reads. These three reach the
    // event log, which is the only place a REAL run can be asked whether the draft head
    // was doing anything.
    r.spec_blocks = s.blocks;
    r.spec_drafted = s.drafted;
    r.spec_accepted = s.accepted_drafts;
    r.spec_abandoned = s.abandoned;
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
    // Per BLOCK, not per run: the question a phase breakdown has to answer is "what does
    // one block cost and where", and a run total hides that behind the block count.
    const SpecPhases& p = fwd.phases();
    const double nb = s.blocks > 0 ? static_cast<double>(s.blocks) : 1.0;
    std::fprintf(stderr,
                 "  [spec] per block: verify=%.1fms forward_last=%.1fms (%.1f pos) "
                 "mtp_logits=%.1fms (%.1f calls) mtp_step=%.1fms hidden=%.1fms "
                 "checkpoint=%.1fms restore=%.1fms\n",
                 p.verify_ms / nb, p.forward_last_ms / nb,
                 static_cast<double>(p.forward_last_positions) / nb, p.mtp_logits_ms / nb,
                 static_cast<double>(p.mtp_logits_calls) / nb, p.mtp_step_ms / nb,
                 p.hidden_ms / nb, p.checkpoint_ms / nb, p.restore_ms / nb);
    return r;
}

} // namespace

GenResult MlxBackend::generate(const InferenceTask& task, TokenSink& sink,
                               const CancelToken& cancel) {
    // AN MLX EXCEPTION MUST NOT TAKE THE PROCESS DOWN WITH IT.
    //
    // MLX throws when an allocation cannot be satisfied ("allocations will result in an
    // exception", mlx/memory.h) and NOTHING in this tree caught it -- not here, not in the
    // loop, not in main. An uncaught throw is std::terminate, which kills the sidecar and
    // takes ~20 GB of loaded weights with it, ends the run with no run_end, and leaves the
    // operator with a dead process and no sentence explaining it.
    //
    // A turn that cannot allocate is a turn that failed. That is an ordinary GenResult and
    // the loop already knows how to carry one: BackendError ends the run with the reason
    // in the report, the model stays loaded, and the next request still has somewhere to
    // go. Catching by reference and reporting what() also puts MLX's own words -- which
    // name the size it could not get -- in front of whoever reads the trace.
    try {
        return generate_impl(task, sink, cancel);
    } catch (const std::exception& e) {
        GenResult r;
        r.status = GenStatus::BackendError;
        r.error = std::string("MLX threw during generation: ") + e.what();
        return r;
    } catch (...) {
        GenResult r;
        r.status = GenStatus::BackendError;
        r.error = "MLX threw a non-standard exception during generation";
        return r;
    }
}

GenResult MlxBackend::generate_impl(const InferenceTask& task, TokenSink& sink,
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
        if (recent.size() > kPenaltyWindow) {
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

// Nothing is allocated without MLX, so there is nothing to report, nothing to free, and
// no allocator cache to put a ceiling on.
MemoryReport mlx_memory_report() { return {}; }
std::size_t mlx_cache_limit() { return 0; }

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
    // Not decoration. Without MLX nothing in this translation unit reads clock_, and
    // -Wunused-private-field is an error here -- so the no-MLX build did not compile at
    // all. Suppressed HERE rather than with [[maybe_unused]] on the member, so the
    // warning stays live for the MLX path, where an unread clock would be a real finding.
    (void)clock_;
    GenResult r;
    r.status = GenStatus::BackendError;
    r.error = "MlxBackend: MLX not compiled in";
    return r;
}

#endif // LMP_HAVE_MLX

} // namespace lmp::model
