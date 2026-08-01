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

bool GrammarSink::on_token(model::TokenId id) {
    const std::size_t think_before = g_.think_ids().size();
    const std::size_t text_before = g_.text_ids().size();

    last = g_.advance(id);

    if (streamer_ != nullptr) {
        if (g_.think_ids().size() > think_before) {
            streamer_->push(id, StreamChannel::Thinking);
        } else if (g_.text_ids().size() > text_before) {
            streamer_->push(id, StreamChannel::Answer);
        }
    }
    return last == model::Advance::Ok;
}

} // namespace lmp::loop
