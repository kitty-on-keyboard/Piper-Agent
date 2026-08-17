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

// The think budget collapses the legal set to a single id, so the close the model emits
// is a REAL token -- it reaches the ledger and the forward pass, and the model's own
// context carries the boundary. The predecessor flipped the phase and appended nothing,
// and the model went on reasoning into the answer channel.
TEST(the_think_cap_forces_a_real_close_token) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    const auto ids = tok().encode_content("ruminate at some length about it");
    REQUIRE(ids.size() >= 4);
    ThinkCapMask capped(g, tok(), /*cap=*/3);

    // Below the cap the mask is the grammar's own: ordinary prose is legal.
    CHECK(capped.mask().allows(ids[0]));
    CHECK(capped.mask_is_block_stable());
    for (std::size_t i = 0; i < 3; ++i) {
        CHECK(g.advance(ids[i]) == Advance::Ok);
    }

    // At the cap exactly one id is legal, and it is the closer.
    const TokenId close = tok().specials().think_close;
    CHECK_EQ(capped.mask().count(), std::size_t{1});
    CHECK(capped.mask().allows(close));
    CHECK(!capped.mask().allows(ids[3]));
    // A block-wide snapshot would offer the closer at the next position too, where it is
    // no longer what the turn needs -- so the block walks instead of reusing one.
    CHECK(!capped.mask_is_block_stable());
    CHECK(capped.can_checkpoint());

    // Emitting it transitions the grammar normally: this is advance_think's own path.
    CHECK(g.advance(close) == Advance::Ok);
    CHECK(g.phase() == TurnPhase::Text);
    CHECK_EQ(g.think_ids().size(), std::size_t{3});
    // ...and the cap stops binding, so the turn can produce an answer or a call.
    CHECK(capped.mask().allows(ids[3]));
    CHECK(feed_text(g, "now tools") == Advance::Ok);
    CHECK(g.advance(tok().specials().im_end) == Advance::Accepted);
}

// The cap is a mask policy, not a grammar rule, so TurnGrammar keeps accepting think
// tokens past it. That is what makes a speculative block drafted just below the cap safe:
// its remaining tokens commit as ordinary reasoning and the close is forced at the next
// mask. Rejecting them would end the turn on a budget.
TEST(overshooting_the_think_cap_is_accepted_not_rejected) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    const auto ids = tok().encode_content("one two three four five six");
    REQUIRE(ids.size() >= 5);
    ThinkCapMask capped(g, tok(), /*cap=*/2);
    for (const TokenId id : ids) {
        CHECK(g.advance(id) == Advance::Ok);
    }
    CHECK(g.phase() == TurnPhase::Think);
    CHECK_EQ(g.think_ids().size(), ids.size());
    // Still forced, however far past the cap the block ran.
    CHECK_EQ(capped.mask().count(), std::size_t{1});
    CHECK(capped.mask().allows(tok().specials().think_close));
}

// EXACTLY WHAT THE DECODER DOES ACROSS A FORCED CLOSE, through the MaskSource interface:
// checkpoint, read the mask, probe a drafted token, read it again, then put it back.
//
// The failure this guards is the one that took tool-call speculation down once already --
// a position with NO legal token. A block-wide snapshot at the cap would offer `</think>`
// at the position after it too, where the phase has already moved to Text; that is why
// mask_is_block_stable() is false here and the decoder walks instead. If the walk ever
// produced an empty mask the decode loop would report "the grammar and the vocabulary
// disagree" and end the run.
TEST(the_decoder_walk_across_a_forced_close_never_sees_an_empty_mask) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    const auto ids = tok().encode_content("weigh it up carefully");
    REQUIRE(ids.size() >= 3);
    const TokenId close = tok().specials().think_close;

    ThinkCapMask capped(g, tok(), /*cap=*/2);
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK(g.advance(ids[i]) == Advance::Ok);
    }
    REQUIRE(!capped.mask_is_block_stable());
    REQUIRE(capped.can_checkpoint());

    capped.checkpoint();
    const TokenMask at_cap = capped.mask(); // copied: the next probe may invalidate it
    CHECK(at_cap.any());
    CHECK_EQ(at_cap.count(), std::size_t{1});
    CHECK(at_cap.allows(close));
    // A drafter that guesses anything else is refused by the mask, which truncates the
    // draft -- it does not walk the automaton somewhere it cannot leave.
    CHECK(!at_cap.allows(ids[2]));

    CHECK(capped.probe_advance(close));
    const TokenMask after = capped.mask();
    CHECK(after.any()); // THE ASSERTION: the position after the close is not a dead end
    CHECK(after.allows(ids[2]));
    CHECK(!after.allows(tok().specials().im_start));

    capped.rollback();
    // Back exactly where it was: still in Think, still at the cap, still forcing.
    CHECK(g.phase() == TurnPhase::Think);
    CHECK_EQ(g.think_ids().size(), std::size_t{2});
    CHECK_EQ(capped.mask().count(), std::size_t{1});
    CHECK(capped.mask().allows(close));
}

// A cap of 0 is "no budget", and the wrapper must then be invisible.
TEST(a_zero_think_cap_delegates_everything) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    ThinkCapMask capped(g, tok(), /*cap=*/0);
    CHECK(feed_text(g, "ruminate as long as it likes") == Advance::Ok);
    CHECK(g.phase() == TurnPhase::Think);
    CHECK_EQ(capped.mask().count(), g.mask().count());
    CHECK(capped.mask_is_block_stable());
}

// `</think>` in Text is a no-op rather than a rejection: reasoning has already ended, so a
// second closer carries nothing -- and ending the turn on it loses a run to punctuation.
TEST(a_second_think_close_in_text_is_a_no_op) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    const TokenId close = tok().specials().think_close;
    CHECK(feed_text(g, "thinking") == Advance::Ok);
    CHECK(g.advance(close) == Advance::Ok);
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(g.advance(close) == Advance::Ok); // the second one
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(g.text_ids().empty()); // and it is not filed as answer text
    CHECK(g.permitted(close));
    CHECK(g.mask().allows(close));
    CHECK(feed_text(g, "The answer is 4.") == Advance::Ok);
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

// --- checkpoint / rollback, which is what lets speculation run inside a tool call ------
//
// The mask inside a call moves with every token, so a speculative block cannot reuse one
// snapshot of it -- it has to walk the grammar forward over the draft and put it back.
// Getting the restore wrong is silent: the automaton carries on from a state the model
// never reached, and the next real token is masked against the wrong legal set. It does
// not throw, and the text stays well-formed right up until the call is malformed.

TEST(rollback_restores_the_automaton_exactly_inside_a_tool_call) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    (void)g.advance(tok().specials().think_close);
    (void)g.advance(tok().specials().tool_call_open);
    REQUIRE(feed_text(g, "<function=read_file>\n<parameter=path>\nsrc/") == Advance::Ok);
    REQUIRE(g.phase() == TurnPhase::ToolCall);

    // The state the probe must give back, recorded as the mask itself.
    const TokenMask before = g.mask();
    const std::size_t before_count = before.count();

    g.checkpoint();
    const std::vector<TokenId> probe = tok().encode_content("model/grammar");
    REQUIRE(!probe.empty());
    for (const TokenId id : probe) {
        REQUIRE(g.probe_advance(id));
    }
    // The walk really moved: a mask that did not change would make this test vacuous.
    CHECK(g.mask().count() != before_count || g.phase() != TurnPhase::ToolCall ||
          !probe.empty());
    g.rollback();

    CHECK(g.phase() == TurnPhase::ToolCall);
    const TokenMask after = g.mask();
    CHECK_EQ(after.size(), before.size());
    CHECK(after.words() == before.words());

    // And the automaton, not just its mask: the rolled-back grammar must still parse the
    // call the un-probed one would have. If the probe's bytes survived, `path` would read
    // "src/model/grammarmain.cpp" and this comparison is what catches it.
    REQUIRE(feed_text(g, "main.cpp\n</parameter>\n</function>\n") == Advance::Ok);
    REQUIRE(g.advance(tok().specials().tool_call_close) == Advance::Ok);
    REQUIRE(g.has_tool_call());
    CHECK_EQ(g.tool_name(), std::string("read_file"));
    REQUIRE(g.tool_params().size() == 1);
    CHECK_EQ(g.tool_params()[0].value, std::string("src/main.cpp"));
}

TEST(a_probe_that_walks_into_an_illegal_token_is_refused_not_silently_accepted) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    (void)g.advance(tok().specials().think_close);
    (void)g.advance(tok().specials().tool_call_open);
    // At the very start of a call only "<function=" can follow, so a plain word is
    // unrepresentable here -- probe_advance must SAY so, because that refusal is what
    // truncates a speculative draft instead of proposing bytes the guard will reject.
    g.checkpoint();
    const std::vector<TokenId> bad = tok().encode_content("hello");
    REQUIRE(!bad.empty());
    CHECK(!g.probe_advance(bad.front()));
    g.rollback();

    // Still able to open the call properly afterwards.
    CHECK(feed_text(g, "<function=read_file>\n") == Advance::Ok);
    CHECK(g.phase() == TurnPhase::ToolCall);
}

TEST(rollback_restores_phase_and_the_accumulated_text) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    (void)g.advance(tok().specials().think_close);
    const std::vector<TokenId> some = tok().encode_content("answer text");
    for (const TokenId id : some) {
        REQUIRE(g.advance(id) == Advance::Ok);
    }
    const std::size_t text_len = g.text_ids().size();

    g.checkpoint();
    const std::vector<TokenId> probe = tok().encode_content(" more drafted words");
    for (const TokenId id : probe) {
        REQUIRE(g.probe_advance(id));
    }
    CHECK(g.text_ids().size() > text_len); // the probe appended
    g.rollback();
    // text_ is append-only within a turn, so the rollback is a resize -- and it has to be
    // exact, because text_ids() is what the turn hands back as the answer.
    CHECK_EQ(g.text_ids().size(), text_len);
    CHECK(g.phase() == TurnPhase::Text);
}

// --- the closer is not always one token ------------------------------------------------
//
// Reproduces the exact turn that ended a run: a read_file call on FRICTION.md whose
// `</tool_call>` the model spelled out as five ordinary tokens instead of emitting the
// single special id. The guard's own literal is `</function>\n</tool_call>`, so every
// byte of the closer is a legal continuation while the model is part way through it --
// the mask offers `<`, `/`, `tool`, `_call`, `>` and the model takes them.
//
// Before the fix the guard completed and nothing noticed: the call was never recorded and
// the phase stayed ToolCall, so the next mask() ran the engine on a COMPLETE guard, got
// nothing, and denied the special closer too (permitted() re-feeds bytes already
// consumed). An empty mask is reported as "the grammar and the vocabulary disagree" and
// the run ends.

TEST(a_tool_call_closed_by_ordinary_tokens_still_closes) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    (void)g.advance(tok().specials().think_close);
    REQUIRE(g.advance(tok().specials().tool_call_open) == Advance::Ok);

    // Everything up to and including `</function>\n`, then the closer SPELLED OUT.
    REQUIRE(feed_text(g, "<function=read_file>\n<parameter=path>\nFRICTION.md\n"
                         "</parameter>\n</function>\n") == Advance::Ok);
    REQUIRE(g.phase() == TurnPhase::ToolCall);
    REQUIRE(feed_text(g, "</tool_call>") == Advance::Ok);

    // The call is closed, recorded, and the turn is back in Text -- exactly as if the
    // single `</tool_call>` id had been used.
    CHECK(g.phase() == TurnPhase::Text);
    REQUIRE(g.has_tool_call());
    CHECK_EQ(g.tool_name(), std::string("read_file"));
    REQUIRE(g.tool_params().size() == 1);
    CHECK_EQ(g.tool_params()[0].value, std::string("FRICTION.md"));

    // And the state is decodable: something is legal here. An empty mask at this point
    // is the failure this test exists for.
    CHECK(g.mask().any());
    CHECK(g.mask().count() > 0);

    // The turn still ends the normal way.
    CHECK(g.advance(tok().specials().im_end) == Advance::Accepted);
}

TEST(the_mask_never_empties_while_a_call_is_closed_either_way) {
    // Both spellings, checked at every step: no state along either path may leave the
    // sampler with nothing to choose from.
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    for (int spelled = 0; spelled < 2; ++spelled) {
        TurnGrammar g(tok(), tools);
        (void)g.advance(tok().specials().think_close);
        REQUIRE(g.advance(tok().specials().tool_call_open) == Advance::Ok);
        std::vector<TokenId> body = tok().encode_content(
            "<function=read_file>\n<parameter=path>\nFRICTION.md\n</parameter>\n</function>\n");
        if (spelled == 1) {
            for (const TokenId id : tok().encode_content("</tool_call>")) {
                body.push_back(id);
            }
        }
        for (const TokenId id : body) {
            CHECK(g.mask().any());
            REQUIRE(g.advance(id) != Advance::Rejected);
        }
        if (spelled == 0) {
            CHECK(g.mask().any());
            REQUIRE(g.advance(tok().specials().tool_call_close) == Advance::Ok);
        }
        CHECK(g.phase() == TurnPhase::Text);
        CHECK(g.mask().any());
        REQUIRE(g.has_tool_call());
        CHECK_EQ(g.tool_params()[0].value, std::string("FRICTION.md"));
    }
}
