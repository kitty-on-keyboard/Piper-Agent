// TokenStreamer (S5.7): needs a vocabulary, never the weights.
//
// REGISTERED TWICE (R1), like tests/model/test_grammar.cpp. The `gate` variant runs on
// the generated miniature vocabulary and so runs on CI; the `realmodel` variant runs on
// the real checkpoint. The fixture is if anything the harsher of the two here: its merges
// are ASCII-only, so EVERY token of a multi-byte string is a byte fragment, where the real
// checkpoint merely has 944 such tokens. naive_invalid_chunks() below asserts the input
// actually exercises the hazard, so this is checked rather than assumed on both.
//
// Two properties, and the first one is not the obvious one. Concatenating raw per-token
// bytes DOES eventually reassemble the text -- byte-level BPE splits a multi-byte
// character into byte fragments that add back up -- so "does the total match" passes even
// for a naive implementation and proves nothing. What actually breaks when you stream is
// the INDIVIDUAL message: a fragment is not valid UTF-8 on its own, and an invalid string
// cannot be JSON-encoded, so it reaches the extension as replacement characters or not at
// all. Measured on this checkpoint: every token of a pure multi-byte string is such a
// fragment.
//
// So this asserts both: every emitted chunk is independently valid UTF-8, AND the
// concatenation is byte-identical to the batch decode the transcript is built from.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/context/context.hpp"
#include "src/loop/agent.hpp"
#include "src/loop/token_stream.hpp"
#include "src/model/backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
#include "src/tools/registry.hpp"

#include "tests/check.hpp"

using lmp::loop::StreamChannel;
using lmp::loop::TokenStreamer;
using lmp::model::Family;
using lmp::model::LoadStatus;
using lmp::model::QwenTokenizer;
using lmp::model::TokenId;

namespace {

std::string tokenizer_path() {
#ifdef LMP_MINI_VOCAB_JSON
    return LMP_MINI_VOCAB_JSON;
#else
    const char* v = std::getenv("LMP_QWEN_DIR");
    return (v != nullptr
                ? std::string(v)
                : std::string("")) +
           "/tokenizer.json";
#endif
}

const QwenTokenizer& loaded_tokenizer() {
    static QwenTokenizer tok;
    static LoadStatus st = tok.load(tokenizer_path(), Family::Qwen3);
    if (!st.ok) {
        static bool reported = false;
        if (!reported) {
            lmp::test::record_failure(__FILE__, __LINE__,
                                      "tokenizer load (" + tokenizer_path() + "): " + st.error);
            reported = true;
        }
    }
    return tok;
}

bool is_valid_utf8(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b = static_cast<unsigned char>(s[i]);
        std::size_t need = 0;
        if (b < 0x80U) {
            need = 0;
        } else if ((b & 0xE0U) == 0xC0U) {
            need = 1;
        } else if ((b & 0xF0U) == 0xE0U) {
            need = 2;
        } else if ((b & 0xF8U) == 0xF0U) {
            need = 3;
        } else {
            return false; // lone continuation byte, or an invalid lead
        }
        if (i + need >= s.size() && need > 0) {
            return false; // truncated sequence -- the streaming hazard
        }
        for (std::size_t k = 1; k <= need; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0U) != 0x80U) {
                return false;
            }
        }
        i += need + 1;
    }
    return true;
}

// Everything the surface was handed, kept as separate messages -- because "was each
// message valid" is the question, and joining them first would destroy the evidence.
struct Captured {
    std::vector<std::string> chunks;
    std::string thinking;
    std::string answer;

    [[nodiscard]] bool every_chunk_valid_utf8() const {
        for (const std::string& c : chunks) {
            if (!is_valid_utf8(c)) {
                return false;
            }
        }
        return true;
    }
};

// How many chunks a naive per-token emitter would have gotten wrong on this input. Used
// to prove the input exercises the hazard, so a green result below means something.
int naive_invalid_chunks(const QwenTokenizer& tok, const std::vector<TokenId>& ids) {
    int bad = 0;
    for (TokenId id : ids) {
        if (!is_valid_utf8(tok.decode({id}))) {
            ++bad;
        }
    }
    return bad;
}

} // namespace

TEST(every_streamed_chunk_is_valid_utf8_on_its_own) {
    const QwenTokenizer& tok = loaded_tokenizer();
    REQUIRE(tok.loaded());

    const std::string reasoning =
        "Let me check \xE4\xB8\xAD\xE6\x96\x87 and \xF0\x9F\x8E\x89 first.";
    const std::string answer = "The answer is \xF0\x9F\x91\x8D caf\xC3\xA9 \xE2\x80\x94 done.";

    const std::vector<TokenId> think_ids = tok.encode_content(reasoning);
    const std::vector<TokenId> text_ids = tok.encode_content(answer);

    // The input must actually contain byte fragments, or this test is green for the wrong
    // reason and would stay green against an implementation that does no buffering at all.
    CHECK(naive_invalid_chunks(tok, think_ids) > 0);
    CHECK(naive_invalid_chunks(tok, text_ids) > 0);

    Captured got;
    {
        TokenStreamer streamer(tok, [&got](const std::string& channel, const std::string& text) {
            got.chunks.push_back(text);
            (channel == "thinking" ? got.thinking : got.answer) += text;
        });
        for (TokenId id : think_ids) {
            streamer.push(id, StreamChannel::Thinking);
        }
        for (TokenId id : text_ids) {
            streamer.push(id, StreamChannel::Answer);
        }
        streamer.finish();
    }

    // The property the naive path fails.
    CHECK(got.every_chunk_valid_utf8());
    // And nothing was lost or reordered getting there.
    CHECK_EQ(got.thinking, tok.decode(think_ids));
    CHECK_EQ(got.answer, tok.decode(text_ids));
    CHECK_EQ(got.thinking, reasoning);
    CHECK_EQ(got.answer, answer);
}

TEST(channels_do_not_bleed_into_each_other) {
    const QwenTokenizer& tok = loaded_tokenizer();
    REQUIRE(tok.loaded());

    // Interleaved, which the turn machine never does but the queue must survive: each
    // channel carries its own decoder, so neither may absorb the other's bytes. Multi-byte
    // on both sides so a shared decoder would visibly cross-contaminate.
    const std::vector<TokenId> a = tok.encode_content("alpha \xE4\xB8\xAD\xE6\x96\x87");
    const std::vector<TokenId> b = tok.encode_content("beta \xF0\x9F\x8E\x89");

    Captured got;
    {
        TokenStreamer streamer(tok, [&got](const std::string& channel, const std::string& text) {
            got.chunks.push_back(text);
            (channel == "thinking" ? got.thinking : got.answer) += text;
        });
        const std::size_t n = a.size() > b.size() ? a.size() : b.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (i < a.size()) {
                streamer.push(a[i], StreamChannel::Thinking);
            }
            if (i < b.size()) {
                streamer.push(b[i], StreamChannel::Answer);
            }
        }
        streamer.finish();
    }

    CHECK(got.every_chunk_valid_utf8());
    CHECK_EQ(got.thinking, tok.decode(a));
    CHECK_EQ(got.answer, tok.decode(b));
}

// The end-to-end claim: the AGENT streams, not just the streamer. Uses ScriptedBackend,
// so this needs the vocabulary but never the weights -- no GPU, no 19 GB load.
TEST(the_agent_streams_a_turn_token_by_token) {
    const QwenTokenizer& tok = loaded_tokenizer();
    REQUIRE(tok.loaded());

    // A whole turn as ids. The turn OPENS in the Think phase -- `<think>` is emitted by
    // the chat template, not generated -- so the script starts with reasoning content and
    // the first structural token it carries is the `</think>` that closes it. Feeding a
    // leading `<think>` here is rejected as structural and ends the turn immediately.
    const auto& sp = tok.specials();
    std::vector<TokenId> script;
    for (TokenId id : tok.encode_content("weighing the options")) {
        script.push_back(id);
    }
    script.push_back(sp.think_close);
    const std::vector<TokenId> answer_ids = tok.encode_content("Here is the answer.");
    for (TokenId id : answer_ids) {
        script.push_back(id);
    }
    script.push_back(sp.im_end);

    lmp::model::ScriptedBackend backend;
    backend.enqueue_response(script);

    lmp::tools::WorkspaceContext ws;
    ws.root = "/tmp";
    ws.max_read_bytes = 1 << 20;
    ws.max_model_read_bytes = 16384;
    ws.max_result_bytes = 1 << 16;
    ws.spool_dir = "/tmp";
    ws.shell_wall_clock_seconds = 5;
    lmp::tools::Registry registry(ws);
    lmp::context::ContextStore ctx("stream check");
    lmp::platform::EventLogWriter log;
    lmp::platform::SystemClock clock;
    lmp::loop::AgentConfig config;

    lmp::loop::Agent agent(tok, backend, registry, ctx, log, clock, config);

    std::vector<std::pair<std::string, std::string>> messages;
    lmp::loop::Observer obs;
    obs.on_token = [&messages](const std::string& channel, const std::string& text) {
        messages.emplace_back(channel, text);
    };
    agent.set_observer(obs);

    const lmp::model::CancelToken cancel;
    const lmp::loop::TurnResult turn = agent.step(cancel);

    // NOT asserting a message count here. ScriptedBackend hands over every token with no
    // delay, so the queue always has a backlog and the streamer's coalescing merges the
    // run into one message per channel -- which is the designed behaviour, not a fault.
    // Incremental delivery is guaranteed only relative to a consumer that keeps up, and
    // that is asserted in one_message_per_token_when_the_consumer_keeps_up, where the
    // pacing is controlled. What this test owns is that the streamed bytes and the
    // recorded bytes are the same, through the real Agent.
    CHECK(messages.size() >= 2);
    CHECK_EQ(messages.front().first, std::string("thinking"));
    CHECK_EQ(messages.back().first, std::string("answer"));

    std::string streamed_thinking;
    std::string streamed_answer;
    for (const auto& m : messages) {
        (m.first == "thinking" ? streamed_thinking : streamed_answer) += m.second;
        CHECK(is_valid_utf8(m.second));
    }

    // What the surface saw is exactly what the transcript kept.
    CHECK_EQ(streamed_answer, turn.assistant_text);
    CHECK_EQ(streamed_thinking, turn.reasoning);
    CHECK_EQ(streamed_answer, std::string("Here is the answer."));
}

// The incrementality guarantee, stated precisely: when the consumer keeps up, each token
// becomes its own message. Coalescing is a backlog behaviour and must never fire here.
TEST(one_message_per_token_when_the_consumer_keeps_up) {
    const QwenTokenizer& tok = loaded_tokenizer();
    REQUIRE(tok.loaded());

    // Pure ASCII: every token decodes to a complete string, so no message can be withheld
    // waiting for a codepoint to finish and the count is exactly the token count.
    const std::vector<TokenId> ids = tok.encode_content("one two three four five six");
    REQUIRE(ids.size() >= 6);

    std::atomic<std::size_t> seen{0};
    std::vector<std::string> chunks;
    {
        TokenStreamer streamer(tok, [&](const std::string&, const std::string& text) {
            chunks.push_back(text);
            seen.fetch_add(1, std::memory_order_release);
        });
        for (std::size_t i = 0; i < ids.size(); ++i) {
            streamer.push(ids[i], StreamChannel::Answer);
            // Let the consumer drain before the next push, so the queue is empty and the
            // coalescing branch cannot merge this token with the next.
            while (seen.load(std::memory_order_acquire) <= i) {
                std::this_thread::yield();
            }
        }
        streamer.finish();
    }

    CHECK_EQ(chunks.size(), ids.size());
    std::string joined;
    for (const std::string& c : chunks) {
        joined += c;
    }
    CHECK_EQ(joined, tok.decode(ids));
}

TEST(finish_is_idempotent_and_survives_a_full_queue) {
    const QwenTokenizer& tok = loaded_tokenizer();
    REQUIRE(tok.loaded());

    // More tokens than the queue has slots, so the producer is forced through the
    // full-queue backpressure path rather than the happy path.
    std::string big;
    for (int i = 0; i < 400; ++i) {
        big += "chunk " + std::to_string(i) + " ";
    }
    const std::vector<TokenId> ids = tok.encode_content(big);
    CHECK(ids.size() > 1024);

    std::string got;
    {
        TokenStreamer streamer(
            tok, [&got](const std::string&, const std::string& text) { got += text; });
        for (TokenId id : ids) {
            streamer.push(id, StreamChannel::Answer);
        }
        streamer.finish();
        streamer.finish(); // second call must be a no-op, not a double join
    }
    CHECK_EQ(got, tok.decode(ids));
}
