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
class GrammarSink final : public model::TokenSink {
  public:
    // `streamer` may be null, which is the non-streaming path (no observer attached).
    GrammarSink(model::TurnGrammar& g, TokenStreamer* streamer) : g_(g), streamer_(streamer) {}

    bool on_token(model::TokenId id) override;

    model::Advance last = model::Advance::Ok;

  private:
    model::TurnGrammar& g_;
    TokenStreamer* streamer_;
};

} // namespace lmp::loop
