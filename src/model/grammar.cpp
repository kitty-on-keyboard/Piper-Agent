#include "src/model/grammar.hpp"

#include <array>
#include <unordered_map>

namespace lmp::model {

// Everything the mask path needs to answer in constant time, built lazily.
struct TurnGrammar::MaskCache {
    // Key: phase (Think/Text/Done) plus whether a call has already been made, which is
    // the only other bit advance_text() branches on.
    std::unordered_map<std::uint32_t, TokenMask> structural;
    // Built only if a tool call actually starts.
    std::unique_ptr<parsephony::Vocab> vocab;
    std::unique_ptr<parsephony::TokenMaskT<parsephony::ToolCallGuard>> engine;
    TokenMask tool_call;
};

TurnGrammar::TurnGrammar(const QwenTokenizer& tok,
                         const std::vector<parsephony::ToolSpec>& tools)
    : tok_(tok), tools_(tools), cache_(std::make_unique<MaskCache>()) {
    reset();
}

TurnGrammar::~TurnGrammar() = default;

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

std::vector<TokenId> TurnGrammar::structural_ids() const {
    const SpecialIds& s = tok_.specials();
    return {s.im_start,          s.im_end,     s.tool_call_open,      s.tool_call_close,
            s.tool_response_open, s.think_open, s.tool_response_close, s.think_close,
            s.endoftext};
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

// --- bulk mask --------------------------------------------------------------

namespace {

std::uint32_t state_key(TurnPhase phase, bool saw_tool_call) {
    return (static_cast<std::uint32_t>(phase) << 1U) | (saw_tool_call ? 1U : 0U);
}

} // namespace

void TurnGrammar::build_structural_mask() const {
    TokenMask m(tok_.vocab_size());
    // In Think and Text, advance() consumes ANY non-structural id and returns Ok --
    // there is no per-token work to do and no reason to walk 248k ids to discover
    // that. So: allow everything, then ask permitted() about the nine structural ids
    // one at a time. The answer for those comes from the real predicate, so the fast
    // path cannot drift from the slow one. Done accepts nothing.
    if (phase_ != TurnPhase::Done) {
        m.allow_all();
        for (TokenId id : structural_ids()) {
            if (!permitted(id)) {
                m.deny(id);
            }
        }
    }
    cache_->structural.insert_or_assign(state_key(phase_, saw_tool_call_), std::move(m));
}

void TurnGrammar::build_tool_call_mask() const {
    if (!cache_->engine) {
        const std::size_t n = tok_.vocab_size();
        cache_->vocab = std::make_unique<parsephony::Vocab>();
        cache_->vocab->tokens.resize(n);
        cache_->vocab->special.assign(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const auto id = static_cast<TokenId>(i);
            cache_->vocab->tokens[i] = std::string(tok_.token_bytes(id));
            // ONLY the structural ids are withheld from the engine. Marking every
            // added token special instead would deny ids that permitted() allows --
            // the fast mask would be stricter than the grammar, and a mask that
            // disagrees with the walk is a mask that lies to the sampler.
            cache_->vocab->special[i] = is_structural(id) ? std::uint8_t{1} : std::uint8_t{0};
        }
        cache_->engine =
            std::make_unique<parsephony::TokenMaskT<parsephony::ToolCallGuard>>(*cache_->vocab);
    }
    cache_->tool_call.adopt(cache_->engine->compute(*guard_), tok_.vocab_size());
    // </tool_call> is structural, so the engine never offers it; whether it is legal
    // right here is the guard's own completion question, which permitted() asks.
    const TokenId close = tok_.specials().tool_call_close;
    if (permitted(close)) {
        cache_->tool_call.allow(close);
    } else {
        cache_->tool_call.deny(close);
    }
}

const TokenMask& TurnGrammar::mask() const {
    if (phase_ == TurnPhase::ToolCall) {
        build_tool_call_mask();
        return cache_->tool_call;
    }
    const std::uint32_t key = state_key(phase_, saw_tool_call_);
    auto it = cache_->structural.find(key);
    if (it == cache_->structural.end()) {
        build_structural_mask();
        it = cache_->structural.find(key);
    }
    return it->second;
}

} // namespace lmp::model
