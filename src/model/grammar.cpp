#include "src/model/grammar.hpp"

namespace lmp::model {

TurnGrammar::TurnGrammar(const QwenTokenizer& tok,
                         const std::vector<parsephony::ToolSpec>& tools)
    : tok_(tok), tools_(tools) {
    reset();
}

void TurnGrammar::reset() {
    phase_ = TurnPhase::Think;
    think_.clear();
    text_.clear();
    saw_tool_call_ = false;
    guard_ = tools_.empty() ? nullptr
                            : std::make_unique<parsephony::ToolCallGuard>(tools_);
}

bool TurnGrammar::is_structural(TokenId id) const noexcept {
    const SpecialIds& s = tok_.specials();
    return id == s.im_start || id == s.im_end || id == s.tool_call_open ||
           id == s.tool_call_close || id == s.tool_response_open ||
           id == s.tool_response_close || id == s.think_open || id == s.think_close ||
           id == s.endoftext;
}

Advance TurnGrammar::advance_think(TokenId id) {
    if (id == tok_.specials().think_close) {
        phase_ = TurnPhase::Text;
        return Advance::Ok;
    }
    if (is_structural(id)) {
        // No nested <think>, no <|im_end|> mid-thought, no tool call inside reasoning.
        return Advance::Rejected;
    }
    think_.push_back(id);
    return Advance::Ok;
}

Advance TurnGrammar::advance_text(TokenId id) {
    const SpecialIds& s = tok_.specials();
    if (id == s.im_end) {
        phase_ = TurnPhase::Done;
        return Advance::Accepted;
    }
    if (id == s.tool_call_open) {
        if (saw_tool_call_ || guard_ == nullptr) {
            // One call per turn -- one turn, one outcome (S9.1). And with no registry,
            // a tool call has nowhere to go: rejected, not silently narrated.
            return Advance::Rejected;
        }
        phase_ = TurnPhase::ToolCall;
        guard_->reset();
        // The guard's grammar includes the <tool_call>\n framing itself.
        if (guard_->feed("<tool_call>\n") != parsephony::Error::Ok) {
            phase_ = TurnPhase::Text;
            return Advance::Rejected;
        }
        return Advance::Ok;
    }
    if (is_structural(id)) {
        return Advance::Rejected;
    }
    text_.push_back(id);
    return Advance::Ok;
}

Advance TurnGrammar::advance_tool_call(TokenId id) {
    // </tool_call> is a single vocab id; the guard expects its bytes and completes on
    // them. Everything inside the call is ordinary tokens fed as bytes.
    std::string_view bytes = tok_.token_bytes(id);
    if (id == tok_.specials().tool_call_close) {
        if (guard_->feed(bytes) != parsephony::Error::Ok || !guard_->complete()) {
            return Advance::Rejected;
        }
        saw_tool_call_ = true;
        phase_ = TurnPhase::Done;
        return Advance::Accepted;
    }
    if (is_structural(id)) {
        return Advance::Rejected;
    }
    if (guard_->feed(bytes) != parsephony::Error::Ok) {
        return Advance::Rejected;
    }
    return Advance::Ok;
}

Advance TurnGrammar::advance(TokenId id) {
    switch (phase_) {
        case TurnPhase::Think:
            return advance_think(id);
        case TurnPhase::Text:
            return advance_text(id);
        case TurnPhase::ToolCall:
            return advance_tool_call(id);
        case TurnPhase::Done:
            return Advance::Rejected;
    }
    return Advance::Rejected;
}

bool TurnGrammar::permitted(TokenId id) const {
    if (phase_ == TurnPhase::ToolCall) {
        // Probe through the guard without mutating it: copy, mute, feed. parsephony's
        // ToolCallGuard is copyable for exactly this purpose (its mask engine does the
        // same, cached by state signature).
        if (is_structural(id) && id != tok_.specials().tool_call_close) {
            return false;
        }
        parsephony::ToolCallGuard probe(*guard_);
        probe.mute();
        const std::string_view bytes = tok_.token_bytes(id);
        for (const char c : bytes) {
            if (probe.probe_byte(static_cast<unsigned char>(c)) !=
                parsephony::Error::Ok) {
                return false;
            }
        }
        if (id == tok_.specials().tool_call_close) {
            return probe.complete();
        }
        return true;
    }
    // Outside a call the state is a few ids; a copy-probe is cheap and cannot drift
    // from advance() because it IS advance().
    TurnGrammar probe(tok_, tools_);
    probe.phase_ = phase_;
    probe.saw_tool_call_ = saw_tool_call_;
    return probe.advance(id) != Advance::Rejected;
}

} // namespace lmp::model
