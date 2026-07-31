// TurnGrammar over a synthetic tokenizer-free path is impossible -- the grammar reads
// token bytes -- so these tests run against the REAL tokenizer.json when present and
// are labelled realmodel. The structural machine (think/text phases, one-call rule) is
// what they pin; ToolCallGuard's own 1000/1000 corpus lives in its source repo.

#include <cstdlib>
#include <string>

#include "src/model/grammar.hpp"
#include "src/model/qwen_tokenizer.hpp"

#include "tests/check.hpp"

using namespace lmp::model;

namespace {

const QwenTokenizer& tok() {
    static QwenTokenizer t;
    static LoadStatus st = t.load(
        std::string(std::getenv("LMP_QWEN_DIR") != nullptr
                        ? std::getenv("LMP_QWEN_DIR")
                        : "/Users/dev/.lmstudio/models/lmstudio-community/"
                          "Qwen3.6-35B-A3B-MLX-4bit") +
            "/tokenizer.json",
        Family::Qwen3);
    (void)st;
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
    CHECK(g.advance(tok().specials().tool_call_close) == Advance::Accepted);
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

TEST(a_second_tool_call_in_one_turn_is_rejected) {
    REQUIRE(tok().loaded());
    const auto tools = one_tool();
    TurnGrammar g(tok(), tools);
    (void)g.advance(tok().specials().think_close);
    (void)g.advance(tok().specials().tool_call_open);
    (void)feed_text(g, "<function=read_file>\n<parameter=path>\nx\n</parameter>\n</function>\n");
    REQUIRE(g.advance(tok().specials().tool_call_close) == Advance::Accepted);
    // One turn, one outcome (S9.1): the turn is Done; nothing more is consumable.
    CHECK(g.advance(tok().specials().tool_call_open) == Advance::Rejected);
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
    CHECK(g.phase() == TurnPhase::Done);
}
