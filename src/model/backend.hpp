#pragma once
//
// InferenceBackend -- the model seam (spec S5.12, S2.1.1).
//
// "The single largest multiplier in the project, and it must exist before the loop
// does." Three implementations:
//
//   MlxBackend      the real thing (phase 3; compiled only where MLX exists)
//   ScriptedBackend deterministic, and it RECORDS WHAT THE HARNESS SENT -- so "did the
//                   model actually receive this?" is an assertion, not a guess
//   ReplayBackend   byte-faithful trace playback; every hand-found bug becomes a test
//
// The last two need no GPU and run in CI.
//
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "src/model/qwen_tokenizer.hpp"

namespace lmp::model {

// The one mutable global the spec permits besides the event log writer (S3). Passed by
// reference; nothing reaches for it ambiently.
class CancelToken {
  public:
    void cancel() noexcept { flag_.store(true, std::memory_order_release); }
    [[nodiscard]] bool cancelled() const noexcept {
        return flag_.load(std::memory_order_acquire);
    }
    void reset() noexcept { flag_.store(false, std::memory_order_release); }

  private:
    std::atomic<bool> flag_{false};
};

struct SamplingParams {
    // Qwen3 defaults, pinned (S5.9). Changing them must move a benchmark first.
    float temperature = 0.6F;
    float top_p = 0.95F;
    std::int32_t top_k = 20;
    float min_p = 0.0F;
    float repetition_penalty = 1.05F;
    std::uint64_t seed = 0;
};

struct InferenceTask {
    std::vector<TokenId> prompt;
    SamplingParams sampling;
    std::int32_t max_new_tokens = 4096;
    // Constrained decoding (S5.6): true iff `id` may be emitted next. Stateful through
    // the caller's grammar -- the sink's on_token advances it, so by the time the next
    // step consults this, it answers for the new state. Null means unconstrained.
    // ScriptedBackend and ReplayBackend ignore it: their tokens are the script's
    // business, and the loop's sink still classifies them.
    std::function<bool(TokenId)> mask;
};

// Receives each sampled id as it is produced. Returns false to stop generation -- this is
// how the grammar's accepting state ends a turn (S5.5): the SINK decides, the backend
// never inspects text.
class TokenSink {
  public:
    TokenSink() = default;
    TokenSink(const TokenSink&) = delete;
    TokenSink& operator=(const TokenSink&) = delete;
    TokenSink(TokenSink&&) = delete;
    TokenSink& operator=(TokenSink&&) = delete;
    virtual ~TokenSink() = default;
    [[nodiscard]] virtual bool on_token(TokenId id) = 0;
};

enum class GenStatus : std::uint8_t {
    Complete,      // the sink said stop -- grammar accept
    LengthCapped,  // max_new_tokens reached; NOT complete, and callers must not blur this
    Cancelled,
    BackendError,
};

struct GenResult {
    GenStatus status = GenStatus::BackendError;
    std::int32_t tokens_generated = 0;
    std::string error;
    // Always-on instrumentation (S5.11): a performance claim without these numbers is
    // not accepted into the repo.
    double ttft_ms = 0.0;
    double prefill_tok_per_s = 0.0;
    double decode_tok_per_s = 0.0;
};

class InferenceBackend {
  public:
    InferenceBackend() = default;
    InferenceBackend(const InferenceBackend&) = delete;
    InferenceBackend& operator=(const InferenceBackend&) = delete;
    InferenceBackend(InferenceBackend&&) = delete;
    InferenceBackend& operator=(InferenceBackend&&) = delete;
    virtual ~InferenceBackend() = default;

    [[nodiscard]] virtual GenResult generate(const InferenceTask& task, TokenSink& sink,
                                             const CancelToken& cancel) = 0;
};

// --- ScriptedBackend -------------------------------------------------------

class ScriptedBackend final : public InferenceBackend {
  public:
    // Each call to generate() consumes the next script entry in order.
    void enqueue_response(std::vector<TokenId> ids) {
        script_.push_back(std::move(ids));
    }

    [[nodiscard]] GenResult generate(const InferenceTask& task, TokenSink& sink,
                                     const CancelToken& cancel) override;

    // What the harness sent, verbatim. The reason this class exists: v1 reported
    // "0 of 17 tool calls completed" as a model failure when the model had gone 17/17
    // and the harness was the thing eating them. Asserting on received_ is how that
    // misattribution becomes impossible to write.
    [[nodiscard]] const std::vector<InferenceTask>& received() const noexcept {
        return received_;
    }
    [[nodiscard]] std::size_t responses_remaining() const noexcept {
        return script_.size() - next_;
    }

  private:
    std::vector<std::vector<TokenId>> script_;
    std::vector<InferenceTask> received_;
    std::size_t next_ = 0;
};

// --- ReplayBackend ---------------------------------------------------------

// Plays back token streams captured in an event log. Construction is explicit from
// decoded events rather than a file path so the log-parsing stays in one place (L0).
class ReplayBackend final : public InferenceBackend {
  public:
    explicit ReplayBackend(std::vector<std::vector<TokenId>> turns)
        : turns_(std::move(turns)) {}

    [[nodiscard]] GenResult generate(const InferenceTask& task, TokenSink& sink,
                                     const CancelToken& cancel) override;

  private:
    std::vector<std::vector<TokenId>> turns_;
    std::size_t next_ = 0;
};

} // namespace lmp::model
