// TurnGrammar over a synthetic tokenizer-free path is impossible -- the grammar reads
// token bytes -- so these tests need a vocabulary. The structural machine (think/text
// phases, the per-turn call cap) is what they pin; ToolCallGuard's own 1000/1000 corpus
// lives in its source repo.
//
// THIS SOURCE IS REGISTERED TWICE (R1). The `gate` variant compiles with
// LMP_MINI_VOCAB_JSON and runs on the generated miniature vocabulary -- no GPU, no 19 GB
// -- so it runs on CI. The `realmodel` variant runs on the real checkpoint, which keeps
// the actual merges under test. Only the second one used to exist, and that is exactly
// how commit 4300a3c changed what closing a tool call means and left this file asserting
// the old contract, red, for two days: the label hid it from every automated run.

#include <cstdlib>
#include <string>

#include "src/model/grammar.hpp"
#include "src/model/qwen_tokenizer.hpp"

#include "tests/check.hpp"

using namespace lmp::model;

namespace {

std::string tokenizer_path() {
#ifdef LMP_MINI_VOCAB_JSON
    return LMP_MINI_VOCAB_JSON;
#else
    const char* v = std::getenv("LMP_QWEN_DIR");
    return std::string(v != nullptr ? v
                                    : "/Users/dev/.lmstudio/models/lmstudio-community/"
                                      "Qwen3.6-35B-A3B-MLX-4bit") +
           "/tokenizer.json";
#endif
}

const QwenTokenizer& tok() {
    static QwenTokenizer t;
    static LoadStatus st = t.load(tokenizer_path(), Family::Qwen3);
    // Reported rather than discarded: REQUIRE(loaded()) below would fail either way, but
    // a bad fixture path and a bad vocabulary are very different problems.
    if (!st.ok) {
        static bool reported = false;
        if (!reported) {
            lmp::test::record_failure(__FILE__, __LINE__,
                                      "tokenizer load (" + tokenizer_path() + "): " + st.error);
            reported = true;
        }
    }
    return t;
}

std::vector<parsephony::ToolSpec> one_tool() {
    std::vector<parsephony::ToolSpec> tools;
    parsephony::ToolSpec spec;
    spec.name = "read_file";
    parsephony::ParamSpec p;
    p.name = "path";
    p.type = parsephony::ParamType::Text;
    p.required = true;
    spec.params.push_back(p);
    tools.push_back(spec);
    return tools;
}

// Drives the grammar with the token encoding of `text` (content path -- ordinary ids).
Advance feed_text(TurnGrammar& g, const std::string& text) {
    Advance last = Advance::Ok;
    for (TokenId id : tok().encode_content(text)) {
        last = g.advance(id);
        if (last != Advance::Ok) {
            return last;
        }
    }
    return last;
}

} // namespace

TEST(turn_walks_think_text_accept) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    CHECK(g.phase() == TurnPhase::Think);

    CHECK(feed_text(g, "let me think about this") == Advance::Ok);
    CHECK(g.advance(tok().specials().think_close) == Advance::Ok);
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(feed_text(g, "The answer is 4.") == Advance::Ok);
    CHECK(g.advance(tok().specials().im_end) == Advance::Accepted);
    CHECK(g.phase() == TurnPhase::Done);
    CHECK(!g.has_tool_call());
    CHECK_EQ(tok().decode(g.text_ids()), std::string("The answer is 4."));
    CHECK_EQ(tok().decode(g.think_ids()), std::string("let me think about this"));
}

TEST(force_end_think_opens_text_without_close_token) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    CHECK(feed_text(g, "ruminate") == Advance::Ok);
    CHECK(g.phase() == TurnPhase::Think);
    CHECK(g.force_end_think());
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(!g.force_end_think()); // already out of Think
    CHECK(feed_text(g, "now tools") == Advance::Ok);
    CHECK(g.advance(tok().specials().im_end) == Advance::Accepted);
}

TEST(structure_is_rejected_inside_think) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    CHECK(g.advance(tok().specials().im_end) == Advance::Rejected);
    CHECK(g.advance(tok().specials().tool_call_open) == Advance::Rejected);
    CHECK(!g.permitted(tok().specials().im_end));
    CHECK(g.permitted(tok().specials().think_close));
}

TEST(a_valid_tool_call_is_parsed_by_the_automaton) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    CHECK(g.advance(tok().specials().think_close) == Advance::Ok);
    CHECK(g.advance(tok().specials().tool_call_open) == Advance::Ok);
    CHECK(g.phase() == TurnPhase::ToolCall);

    CHECK(feed_text(g, "<function=read_file>\n<parameter=path>\nsrc/main.cpp\n"
                       "</parameter>\n</function>\n") == Advance::Ok);
    // Ok and back to Text, not Accepted and Done: since batching landed, a closed call
    // returns the turn to text so another may follow. The turn ends on <|im_end|>.
    CHECK(g.advance(tok().specials().tool_call_close) == Advance::Ok);
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(g.advance(tok().specials().im_end) == Advance::Accepted);
    CHECK(g.phase() == TurnPhase::Done);
    REQUIRE(g.has_tool_call());
    // Extraction IS the automaton -- no second pass over decoded text (S5.6).
    CHECK_EQ(g.tool_name(), std::string("read_file"));
    REQUIRE(g.tool_params().size() == 1);
    CHECK_EQ(g.tool_params()[0].name, std::string("path"));
    CHECK_EQ(g.tool_params()[0].value, std::string("src/main.cpp"));
}

TEST(an_unregistered_tool_is_unrepresentable) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    (void)g.advance(tok().specials().think_close);
    (void)g.advance(tok().specials().tool_call_open);
    // "write_file" is not in the registry; the guard rejects at the first byte that
    // diverges from every registered name.
    CHECK(feed_text(g, "<function=write_file>") == Advance::Rejected);
}

TEST(closing_with_a_required_parameter_missing_is_unrepresentable) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    (void)g.advance(tok().specials().think_close);
    (void)g.advance(tok().specials().tool_call_open);
    CHECK(feed_text(g, "<function=read_file>\n") == Advance::Ok);
    // </function> before the required `path`: the guard refuses the byte sequence.
    CHECK(feed_text(g, "</function>\n") == Advance::Rejected);
}

TEST(a_turn_batches_up_to_the_call_cap_and_no_further) {
    // This replaces `a_second_tool_call_in_one_turn_is_rejected`, which asserted the
    // pre-batching contract. Batching (commit 4300a3c) made a second call legal and
    // bounded it instead: kMaxCallsPerTurn, then the open is rejected. The old test kept
    // asserting the old rule for two days because it carries the `realmodel` label and
    // the gate never runs it.
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    (void)g.advance(tok().specials().think_close);

    const auto one_call = [&] {
        const Advance opened = g.advance(tok().specials().tool_call_open);
        if (opened != Advance::Ok) {
            return opened;
        }
        (void)feed_text(g,
                        "<function=read_file>\n<parameter=path>\nx\n</parameter>\n</function>\n");
        return g.advance(tok().specials().tool_call_close);
    };

    for (std::size_t i = 0; i < TurnGrammar::kMaxCallsPerTurn; ++i) {
        CHECK(one_call() == Advance::Ok);
        CHECK(g.phase() == TurnPhase::Text);
    }
    CHECK_EQ(g.tool_calls().size(), TurnGrammar::kMaxCallsPerTurn);

    // At the cap the open is rejected rather than silently narrated, and the mask must
    // say the same thing the walk does.
    CHECK(g.advance(tok().specials().tool_call_open) == Advance::Rejected);
    CHECK(!g.permitted(tok().specials().tool_call_open));
}

TEST(with_no_registry_a_tool_call_cannot_start) {
    REQUIRE(tok().loaded());
    const std::vector<parsephony::ToolSpec> none;
    TurnGrammar g(tok(), none);
    (void)g.advance(tok().specials().think_close);
    CHECK(g.advance(tok().specials().tool_call_open) == Advance::Rejected);
    CHECK(!g.permitted(tok().specials().tool_call_open));
}

TEST(the_mask_and_the_walk_agree) {
    // permitted() must be advance() minus the mutation. Walk a call and at each step
    // assert the taken token was permitted -- a divergence here is the mask lying to
    // the sampler about the grammar.
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);

    std::vector<TokenId> walk;
    walk.push_back(tok().specials().think_close);
    walk.push_back(tok().specials().tool_call_open);
    for (TokenId id : tok().encode_content(
             "<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n")) {
        walk.push_back(id);
    }
    walk.push_back(tok().specials().tool_call_close);

    for (TokenId id : walk) {
        if (!g.permitted(id)) {
            lmp::test::record_failure(__FILE__, __LINE__,
                                      "permitted() denied a token advance() accepts: " +
                                          std::string(tok().token_bytes(id)));
        }
        ++lmp::test::reg().checks;
        REQUIRE(g.advance(id) != Advance::Rejected);
    }
    // The walk ends at </tool_call>, which since batching returns the turn to Text so a
    // second call may follow. Done is reached on <|im_end|>, not on the close.
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(g.advance(tok().specials().im_end) == Advance::Accepted);
    CHECK(g.phase() == TurnPhase::Done);
}

TEST(the_bulk_mask_and_the_predicate_agree_over_the_whole_vocabulary) {
    // mask() is what the sampler consults; permitted() is what the grammar means. This
    // is the test that keeps them the same function. It compares them for EVERY id at
    // EVERY state of a real tool call -- including inside <tool_call>, where mask()
    // stops being a cached denylist and becomes parsephony's TokenMaskT.
    //
    // The fast path is worth roughly 22.8 ms/token, which is more than enough motive to
    // let it drift; nothing but this loop would notice.
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    const auto vocab = static_cast<TokenId>(tok().vocab_size());

    std::vector<TokenId> walk;
    walk.push_back(tok().specials().think_close);
    walk.push_back(tok().specials().tool_call_open);
    for (TokenId id : tok().encode_content(
             "<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n")) {
        walk.push_back(id);
    }
    walk.push_back(tok().specials().tool_call_close);

    std::size_t disagreements = 0;
    std::string first;
    for (std::size_t step = 0; step <= walk.size(); ++step) {
        const TokenMask& m = g.mask();
        REQUIRE(m.size() == tok().vocab_size());
        for (TokenId id = 0; id < vocab; ++id) {
            if (m.allows(id) == g.permitted(id)) {
                continue;
            }
            ++disagreements;
            if (first.empty()) {
                first = "step " + std::to_string(step) + " phase " +
                        std::to_string(static_cast<int>(g.phase())) + " id " +
                        std::to_string(id) + " mask=" + (m.allows(id) ? "1" : "0") +
                        " permitted=" + (g.permitted(id) ? "1" : "0") + " bytes='" +
                        std::string(tok().token_bytes(id)) + "'";
            }
        }
        ++lmp::test::reg().checks;
        if (step < walk.size()) {
            REQUIRE(g.advance(walk[step]) != Advance::Rejected);
        }
    }
    if (disagreements != 0) {
        lmp::test::record_failure(__FILE__, __LINE__,
                                  std::to_string(disagreements) +
                                      " mask/predicate disagreements; first: " + first);
    }
    // As above: </tool_call> returns the turn to Text since batching landed.
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(g.advance(tok().specials().im_end) == Advance::Accepted);
    CHECK(g.phase() == TurnPhase::Done);
}
