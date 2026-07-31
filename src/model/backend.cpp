#include "src/model/backend.hpp"

namespace lmp::model {
namespace {

GenResult play(const std::vector<TokenId>& ids, TokenSink& sink, const CancelToken& cancel,
               std::int32_t max_new_tokens) {
    GenResult r;
    for (TokenId id : ids) {
        if (cancel.cancelled()) {
            r.status = GenStatus::Cancelled;
            return r;
        }
        if (r.tokens_generated >= max_new_tokens) {
            r.status = GenStatus::LengthCapped;
            return r;
        }
        ++r.tokens_generated;
        if (!sink.on_token(id)) {
            r.status = GenStatus::Complete;
            return r;
        }
    }
    // Script exhausted without the sink stopping: the scripted "model" ran out of
    // tokens, which maps to the length cap, not to completion. Blurring these two is
    // exactly the one-turn-two-outcomes bug class (S9.1).
    r.status = GenStatus::LengthCapped;
    return r;
}

} // namespace

GenResult ScriptedBackend::generate(const InferenceTask& task, TokenSink& sink,
                                    const CancelToken& cancel) {
    received_.push_back(task);
    if (next_ >= script_.size()) {
        GenResult r;
        r.status = GenStatus::BackendError;
        r.error = "ScriptedBackend: no response scripted for call " +
                  std::to_string(received_.size());
        return r;
    }
    return play(script_[next_++], sink, cancel, task.max_new_tokens);
}

GenResult ReplayBackend::generate(const InferenceTask& task, TokenSink& sink,
                                  const CancelToken& cancel) {
    if (next_ >= turns_.size()) {
        GenResult r;
        r.status = GenStatus::BackendError;
        r.error = "ReplayBackend: trace has no turn " + std::to_string(turns_.size() + 1);
        return r;
    }
    return play(turns_[next_++], sink, cancel, task.max_new_tokens);
}

} // namespace lmp::model
