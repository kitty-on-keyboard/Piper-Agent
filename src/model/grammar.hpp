#pragma once
//
// TurnGrammar -- structural constraint over TOKEN IDS (spec S5.4-S5.6).
//
// Replaces three v1 files totalling 2,177 lines:
//   stop_heuristics.hpp (543)        -> stopping is "grammar accepting AND terminal id
//                                       emitted". Nothing else. No text matching.
//   tool_call_extractor.hpp (1,403)  -> ONE syntax, enforced at decode time, extractor
//                                       built into the automaton -- no second pass
//   streaming_token_decoder.hpp (231)-> frankentok owns decode (S5.3)
//
// THE SYNTAX IS QWEN 3.6's OWN, WHICH IS XML, NOT JSON. The build spec's S5.6 pins the
// JSON form ({"name": ..., "arguments": ...}); that was written for Qwen3, and the
// model this project actually loads emits -- per its own chat_template.jinja, verified
// by parsephone -- the XML framing:
//
//     <tool_call>
//     <function=get_weather>
//     <parameter=city>
//     Denver
//     </parameter>
//     </function>
//     </tool_call>
//
// S19.6 (re-measure before acting on any document, including the spec) resolves the
// conflict in favour of the model's template. Enforcing the spec's JSON form with a
// mask would force the model off its trained distribution on every call.
//
// The enforcement inside <tool_call> is parsephony::ToolCallGuard: schema-aware, byte
// by byte -- the model cannot name an unregistered tool, misspell or repeat a
// parameter, close </function> with a required parameter missing, or emit a malformed
// typed value. Measured in its own repo: 1000/1000 valid constrained generations,
// 17.7 ns steady-state mask cost.
//
// Turn shape:  <think> ... </think>  text  [ tool_call ]  <|im_end|>
//
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <parsephony/mask.hpp>
#include <parsephony/toolcall.hpp>

#include "src/model/qwen_tokenizer.hpp"
#include "src/model/token_mask.hpp"

namespace lmp::model {

enum class TurnPhase : std::uint8_t {
    Think,     // inside <think> ... </think>
    Text,      // after </think>, before any structure
    ToolCall,  // inside <tool_call>, ToolCallGuard is the authority
    Done,      // accepted
};

enum class Advance : std::uint8_t {
    Ok,        // consumed, still generating
    Accepted,  // consumed, turn complete -- STOP now, not a token later
    Rejected,  // not legal here (reachable only when generation is unmasked)
};

class TurnGrammar final : public MaskSource {
  public:
    // `tools` must outlive the grammar (ToolCallGuard keeps a reference). An empty
    // registry means a text-only turn: <tool_call> itself is rejected.
    TurnGrammar(const QwenTokenizer& tok, const std::vector<parsephony::ToolSpec>& tools);
    ~TurnGrammar() final;

    void reset();

    [[nodiscard]] Advance advance(TokenId id);

    // True if `id` may follow the current state. This is the DEFINITION of the mask --
    // it is advance() minus the mutation -- and it is what mask() is tested against
    // (test_grammar_realmodel). It is not what the sampler calls: at one call per id it
    // costs 22.8 ms/token.
    [[nodiscard]] bool permitted(TokenId id) const;

    // The same answer for the whole vocabulary at once, which is what the sampler
    // consults (S5.6). Outside a tool call the legal set is "everything except a
    // handful of structural ids", so it is a cached bitset per phase with no
    // vocabulary walk at all. Inside one it is parsephony's TokenMaskT over
    // ToolCallGuard: masks cached by state signature, candidates bucketed by first
    // byte, string-safe tokens pre-classified.
    [[nodiscard]] const TokenMask& mask() const final;

    // Think and Text mask by PHASE, not by history: outside a tool call the legal set is
    // a cached bitset of "everything except a handful of structural ids", and only a
    // structural (special) token moves the phase. Inside a tool call parsephony's mask is
    // state-dependent per token, so a block-wide snapshot would be wrong -- speculation
    // falls back to one-at-a-time there. See src/model/speculative.hpp.
    [[nodiscard]] bool mask_is_block_stable() const final {
        return phase_ == TurnPhase::Think || phase_ == TurnPhase::Text;
    }

    // Only a structural id moves the phase, so only a structural id ends the block.
    [[nodiscard]] bool is_block_boundary(TokenId id) const final { return is_structural(id); }

    [[nodiscard]] TurnPhase phase() const noexcept { return phase_; }
    [[nodiscard]] const std::vector<TokenId>& think_ids() const noexcept { return think_; }
    [[nodiscard]] const std::vector<TokenId>& text_ids() const noexcept { return text_; }
    // One turn may carry several calls (S9.1 amended): the model batches independent
    // reads, and serialising them cost a full prefill+decode round-trip each. The guard
    // is reset between calls, so each is parsed by the same automaton -- parsing is not
    // relaxed by allowing more than one.
    struct ParsedCall {
        std::string name;
        std::vector<parsephony::ToolCallGuard::Param> params;
    };

    // Bounded so a stuck model cannot emit calls forever inside one turn.
    static constexpr std::size_t kMaxCallsPerTurn = 4;

    [[nodiscard]] bool has_tool_call() const noexcept { return !calls_.empty(); }
    [[nodiscard]] const std::vector<ParsedCall>& tool_calls() const noexcept {
        return calls_;
    }

    // The FIRST call. Valid once has_tool_call(); copied out of the automaton at the
    // moment it completed, because the guard is reset before the next one.
    [[nodiscard]] const std::string& tool_name() const { return calls_.front().name; }
    [[nodiscard]] const std::vector<parsephony::ToolCallGuard::Param>& tool_params() const {
        return calls_.front().params;
    }

  private:
    [[nodiscard]] Advance advance_think(TokenId id);
    [[nodiscard]] Advance advance_text(TokenId id);
    [[nodiscard]] Advance advance_tool_call(TokenId id);
    [[nodiscard]] bool is_structural(TokenId id) const noexcept;
    [[nodiscard]] std::vector<TokenId> structural_ids() const;
    void build_structural_mask() const;
    void build_tool_call_mask() const;

    const QwenTokenizer& tok_;
    const std::vector<parsephony::ToolSpec>& tools_;
    std::unique_ptr<parsephony::ToolCallGuard> guard_;
    TurnPhase phase_ = TurnPhase::Think;
    std::vector<TokenId> think_;
    std::vector<TokenId> text_;
    std::vector<ParsedCall> calls_;

    // The mask outside a call turns on exactly one bit: may another <tool_call> open
    // here? That keeps the cache key a bool, as it was when the answer was "have we
    // seen one yet".
    [[nodiscard]] bool at_call_cap() const noexcept {
        return calls_.size() >= kMaxCallsPerTurn;
    }

    // --- mask caches ---------------------------------------------------------
    // Outside a call there are only a few distinct states (phase x saw_tool_call_),
    // and each one's answer is the same bitset every time it recurs. The engine and
    // its vocabulary copy are built on first entry into a tool call and not before:
    // a text-only turn never pays for them.
    struct MaskCache;
    std::unique_ptr<MaskCache> cache_;
};

} // namespace lmp::model
