#pragma once
//
// TokenStreamer -- per-token text to the surface, off the generation thread (S5.7).
//
// Before this, `Observer::on_token` was named per token and fired ONCE PER TURN, after
// generate() returned, carrying the whole decoded block. The loop measured TTFT and then
// showed the user nothing until the turn ended.
//
// WHY A THREAD AND NOT A DIRECT CALL. The cost of decoding one token is microseconds
// against a ~12 ms decode step, so this is not about CPU. It is about who we block on:
// the sidecar's write_line() does fwrite + fflush to stdout, a pipe whose reader is the
// extension. Calling that inline would put a blocking pipe write, flushed every token,
// on the critical path of the GPU decode loop, and a slow or paused reader would throttle
// generation itself. The queue absorbs that; at 80 tok/s a 1024-slot channel is twelve
// seconds of reader stall before the producer ever notices.
//
// WHY IDS AND NOT TEXT CROSS THE QUEUE. The decoder is stateful -- 944 of the 248k tokens
// are byte fragments, so a token can end mid-codepoint -- and keeping that state on the
// consumer keeps the generation thread doing strictly less work.
//
// WHY TWO DECODERS. One per channel, each fed only its own channel's ids in order. That
// makes the streamed concatenation byte-identical to the batch `decode(think_ids())` /
// `decode(text_ids())` the turn still reports, which is asserted in test_token_stream.
// A single shared decoder would have to skip the other channel's tokens mid-stream and
// could split a codepoint across the skip.
//
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include "src/model/backend.hpp"
#include "src/model/grammar.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/spsc_channel.hpp"

namespace lmp::loop {

// Which surface channel a token's text belongs to. Reasoning is surfaced separately and
// never inlined into the answer (S5.7).
enum class StreamChannel : std::uint8_t { Thinking, Answer };

[[nodiscard]] const char* channel_name(StreamChannel c) noexcept;

class TokenStreamer {
  public:
    using Emit = std::function<void(const std::string& channel, const std::string& text)>;

    // `tok` must outlive the streamer. `emit` is called ON THE STREAMER'S THREAD, so it
    // must be safe to call from a thread other than the one that constructed it -- the
    // sidecar's notify() takes a stdout lock for exactly this reason.
    TokenStreamer(const model::QwenTokenizer& tok, Emit emit);

    // Joins the worker. finish() is the ordered shutdown; this is the backstop.
    ~TokenStreamer();

    TokenStreamer(const TokenStreamer&) = delete;
    TokenStreamer& operator=(const TokenStreamer&) = delete;

    // Producer (generation thread) only. Blocks with a yield if the queue is full, which
    // is backpressure from a reader that has stopped draining -- dropping the token
    // instead would silently lose the user's text, and text loss is not a perf trade.
    void push(model::TokenId id, StreamChannel channel);

    // Flushes both decoders, drains the queue, joins the worker. Idempotent. Must be
    // called before reading the turn's text so the surface and the transcript agree.
    void finish();

  private:
    struct Event {
        model::TokenId id = model::kInvalidToken;
        StreamChannel channel = StreamChannel::Answer;
    };

    void run();

    const model::QwenTokenizer& tok_;
    Emit emit_;
    platform::SpscChannel<Event> queue_;
    std::atomic<bool> finished_{false};
    std::thread worker_;
};

// Collects tokens, walks the grammar, and stops the instant the grammar accepts -- not a
// token later (S5.5). With a streamer attached it also routes each token to the surface as
// it is produced.
//
// The routing does NOT re-derive "is this reasoning or answer text". It ASKS the grammar
// what it just did, by observing which id list grew. advance_think/advance_text own that
// decision -- a `</think>` closes the phase without being content, `<tool_call>` opens a
// phase whose tokens are structure rather than text -- and a second copy of those rules
// here would be free to drift from them. Observing the effect cannot.
// Stops a turn that has started repeating itself verbatim.
//
// WHY THIS IS NOT A SAMPLING FIX. The repetition penalty cannot reach this failure, and
// tuning it is a dead end worth documenting so nobody re-tries it. The penalty is applied
// ONCE PER UNIQUE ID (HF semantics -- see apply_repetition_penalty, which was made
// non-compounding after per-occurrence division wrote "idlePer cent" into a real file), so
// at the pinned Qwen default of 1.05 a winning logit of ~15 becomes ~14.3. A model that is
// near-certain of its next token is still near-certain. That is true at ANY window size, so
// the sliding window's length -- 64 tokens, and see kPenaltyWindow for what it is actually
// worth -- is not the reason this run looped.
//
// MEASURED: a Qwen3.6-35B-A3B 4-bit run emitted one 281-character paragraph roughly fifty
// times inside a single thinking block and stopped only when it hit the 4096-token cap,
// having produced nothing. The repeated unit was ~70 tokens long -- longer than the whole
// penalty window -- so each copy fell out of `recent` before the next one began and the
// penalty never saw a repeat at all. Endless repetition is a known failure of quantized
// Qwen3 in long generations; the vendor's own advice for it is a presence penalty, not a
// bigger repetition window.
//
// PROSE ONLY, AND THAT IS A SAFETY PROPERTY, NOT A HEURISTIC. This counts a token only when
// the grammar routed it into think_ or text_. File content is written inside a `tool_call`
// phase, whose tokens go to the parsephony guard and enter NEITHER buffer -- so this cannot
// truncate a file the model is in the middle of writing, which would be far worse than the
// loop it exists to stop. It is structurally incapable of it, rather than tuned not to.
class LoopBreaker {
  public:
    // A repeat this long is not style. Legitimate prose and legitimate code both repeat
    // short runs constantly -- `}`, `return nil`, a boilerplate line -- and both stay far
    // below 32 tokens of EXACT match. The observed loop was ~70, so a cycle contains many
    // whole windows of itself and trips this well before a second full copy lands.
    static constexpr std::size_t kWindow = 32;
    // Three, because two can be honest. A model legitimately restates a signature or a path
    // twice; the third verbatim recurrence of the same 32 tokens is a cycle. The run this
    // was built from produced fifty.
    static constexpr std::size_t kMaxRepeats = 3;

    // Returns true when the turn should be cut. Feed it PROSE tokens only.
    [[nodiscard]] bool saw(model::TokenId id);

    [[nodiscard]] std::size_t repeats() const noexcept { return worst_; }

  private:
    std::vector<model::TokenId> window_;
    // Rolling hash -> how many times that exact window has been seen. A hash collision
    // costs a spuriously-cut turn, never a wrong answer, and at 64 bits over a few thousand
    // windows it is not a real risk.
    std::vector<std::pair<std::uint64_t, std::size_t>> seen_;
    std::size_t worst_ = 0;
};

class GrammarSink final : public model::TokenSink {
  public:
    // `streamer` may be null, which is the non-streaming path (no observer attached).
    // `max_think_tokens` 0 means no separate think cap (legacy / unconstrained tests).
    GrammarSink(model::TurnGrammar& g, TokenStreamer* streamer,
                std::size_t max_think_tokens = 0)
        : g_(g), streamer_(streamer), max_think_tokens_(max_think_tokens) {}

    bool on_token(model::TokenId id) override;

    model::Advance last = model::Advance::Ok;
    // Set when the turn was cut for looping rather than by the grammar. The agent reports
    // this as its own ending -- a turn that was stopped is not a turn that finished, and
    // blurring the two is how the LengthCapped case used to read as completion.
    bool looped = false;
    std::size_t loop_repeats = 0;
    // True when force_end_think() closed the Think phase because the think budget hit.
    bool think_capped = false;

  private:
    model::TurnGrammar& g_;
    TokenStreamer* streamer_;
    LoopBreaker breaker_;
    std::size_t max_think_tokens_ = 0;
};

} // namespace lmp::loop
