#include "src/loop/token_stream.hpp"

#include <utility>

namespace lmp::loop {
namespace {

// Twelve seconds of reader stall at 80 tok/s before a producer ever blocks. Sized to
// absorb a paused extension, not to hide a permanently dead one.
constexpr std::size_t kQueueSlots = 1024;

} // namespace

const char* channel_name(StreamChannel c) noexcept {
    switch (c) {
        case StreamChannel::Thinking:
            return "thinking";
        case StreamChannel::Answer:
            return "answer";
    }
    return "answer";
}

TokenStreamer::TokenStreamer(const model::QwenTokenizer& tok, Emit emit)
    : tok_(tok), emit_(std::move(emit)), queue_(kQueueSlots) {
    worker_ = std::thread([this] { run(); });
}

TokenStreamer::~TokenStreamer() { finish(); }

void TokenStreamer::push(model::TokenId id, StreamChannel channel) {
    Event ev{id, channel};
    // try_push only consumes the value on success, so retrying with `ev` is sound.
    while (!queue_.try_push(std::move(ev))) {
        std::this_thread::yield();
    }
}

void TokenStreamer::finish() {
    if (finished_.exchange(true)) {
        return;
    }
    queue_.close();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void TokenStreamer::run() {
    model::QwenTokenizer::Stream thinking(tok_);
    model::QwenTokenizer::Stream answer(tok_);

    std::string pending;
    StreamChannel pending_channel = StreamChannel::Answer;
    bool have_pending = false;

    const auto flush_pending = [&] {
        if (have_pending && !pending.empty()) {
            emit_(channel_name(pending_channel), pending);
        }
        pending.clear();
        have_pending = false;
    };

    Event ev;
    while (true) {
        if (queue_.try_pop(ev)) {
            if (have_pending && pending_channel != ev.channel) {
                flush_pending();
            }
            model::QwenTokenizer::Stream& decoder =
                ev.channel == StreamChannel::Thinking ? thinking : answer;
            pending += decoder.push(ev.id);
            pending_channel = ev.channel;
            have_pending = true;

            // Coalesce only what is ALREADY queued, never wait for more. In steady state
            // the queue is empty between tokens, so this emits per token; it batches only
            // when a backlog exists, which is exactly when the extra flushes would hurt
            // and when nothing is gained by delivering them separately.
            if (queue_.empty()) {
                flush_pending();
            }
            continue;
        }
        if (queue_.drained()) {
            break;
        }
        std::this_thread::yield();
    }
    flush_pending();

    // Any trailing bytes the decoders were holding for a codepoint that never completed.
    // Emitted in channel order, which is turn order: reasoning precedes the answer.
    const std::string think_tail = thinking.flush();
    if (!think_tail.empty()) {
        emit_(channel_name(StreamChannel::Thinking), think_tail);
    }
    const std::string answer_tail = answer.flush();
    if (!answer_tail.empty()) {
        emit_(channel_name(StreamChannel::Answer), answer_tail);
    }
}

bool LoopBreaker::saw(model::TokenId id) {
    window_.push_back(id);
    if (window_.size() < kWindow) {
        return false;
    }
    if (window_.size() > kWindow) {
        window_.erase(window_.begin());
    }
    // FNV-1a over the window. Order matters -- a bag of the same tokens in a different
    // arrangement is a different sentence, and treating them alike would cut turns that are
    // merely on-topic.
    std::uint64_t h = 1469598103934665603ULL;
    for (const model::TokenId t : window_) {
        const auto u = static_cast<std::uint64_t>(static_cast<std::uint32_t>(t));
        for (int b = 0; b < 4; ++b) {
            h ^= (u >> (b * 8)) & 0xFFULL;
            h *= 1099511628211ULL;
        }
    }
    for (auto& [key, count] : seen_) {
        if (key == h) {
            ++count;
            worst_ = std::max(worst_, count);
            return count >= kMaxRepeats;
        }
    }
    seen_.emplace_back(h, 1);
    worst_ = std::max(worst_, std::size_t{1});
    return false;
}

bool GrammarSink::on_token(model::TokenId id) {
    const std::size_t think_before = g_.think_ids().size();
    const std::size_t text_before = g_.text_ids().size();

    last = g_.advance(id);

    const bool is_think = g_.think_ids().size() > think_before;
    const bool is_text = g_.text_ids().size() > text_before;
    if (streamer_ != nullptr) {
        if (is_think) {
            streamer_->push(id, StreamChannel::Thinking);
        } else if (is_text) {
            streamer_->push(id, StreamChannel::Answer);
        }
    }
    if (last != model::Advance::Ok) {
        return false;
    }
    // Cap thinking separately so tool XML keeps a guaranteed remainder of max_new_tokens.
    // Force the phase transition rather than stopping generation: a LengthCapped mid-think
    // turn produces nothing the loop can act on.
    if (max_think_tokens_ > 0 && g_.phase() == model::TurnPhase::Think &&
        g_.think_ids().size() >= max_think_tokens_) {
        if (g_.force_end_think()) {
            think_capped = true;
        }
    }
    // Only what the grammar filed as PROSE. A tool call's tokens grow neither buffer, so a
    // file being written cannot be cut short here -- see LoopBreaker.
    if ((is_think || is_text) && breaker_.saw(id)) {
        looped = true;
        loop_repeats = breaker_.repeats();
        return false;
    }
    return true;
}

} // namespace lmp::loop
