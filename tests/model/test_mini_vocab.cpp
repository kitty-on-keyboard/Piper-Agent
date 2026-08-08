// The miniature Qwen-shaped vocabulary (G0).
//
// It exists so that TurnGrammar and the loop can be exercised in the GATE. Everything
// downstream of it depends on the fixture actually satisfying the same family
// verification production does, so that is what this asserts first -- a fixture that
// loaded through a relaxed path would prove nothing about the real one.

#include <string>

#include "src/model/chat_template.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "tests/check.hpp"

using namespace lmp::model;

namespace {

// Written by the build; see tests/model/CMakeLists.txt and tests/fixtures/.
const char* fixture_path() { return LMP_MINI_VOCAB_JSON; }

// The two shapes load() must refuse: `foreign` is out of the size band, `unmarked` is
// inside it but carries no structural tokens.
const char* foreign_path() { return LMP_FOREIGN_VOCAB_JSON; }
const char* unmarked_path() { return LMP_UNMARKED_VOCAB_JSON; }

} // namespace

// The point of the whole fixture: it goes through QwenTokenizer::load unchanged, size
// band and structural probes included (S5.2). No test-only family, no "load anyway".
TEST(the_fixture_passes_real_family_verification) {
    QwenTokenizer tok;
    const LoadStatus s = tok.load(fixture_path(), Family::Qwen3);
    REQUIRE(s.ok);
    CHECK(tok.loaded());
    CHECK(tok.vocab_size() >= 140000);
    CHECK(tok.specials().complete());
}

// The negative twin of the test above, and v1's silent-mis-tokenization bug made
// structural (S5.2). Both cases assert the REASON, not just !ok: a fixture that stopped
// being loadable at all would fail inside frankentok and still leave `ok` false, so a
// bare CHECK(!ok) is a test that passes while asserting nothing.
TEST(a_foreign_vocabulary_is_refused_on_the_size_band) {
    // 262,168 entries -- Gemma-4's measured size, the number the 260,000 ceiling exists
    // to exclude.
    QwenTokenizer tok;
    const LoadStatus st = tok.load(foreign_path(), Family::Qwen3);
    CHECK(!st.ok);
    CHECK(!tok.loaded());
    CHECK(st.error.find("outside the Qwen3 band") != std::string::npos);
    CHECK(st.error.find("262168") != std::string::npos);
}

// Isolates the SECOND probe, which the old Gemma test never reached: the size check
// short-circuited before it. In-band size, near-miss specials (`<|tool_call>`, not
// `<tool_call>`) -- exactly what name-sniffing accepts and this must not.
TEST(an_in_band_vocabulary_without_structural_tokens_is_refused) {
    QwenTokenizer tok;
    const LoadStatus st = tok.load(unmarked_path(), Family::Qwen3);
    CHECK(!st.ok);
    CHECK(!tok.loaded());
    CHECK(st.error.find("structural tokens are incomplete") != std::string::npos);
}

// Ids that deliberately DISAGREE with the production checkpoint (im_start=248045 there).
// Any code that assumed a literal id fails here instead of in production.
TEST(the_fixture_ids_are_not_the_production_ids) {
    QwenTokenizer tok;
    REQUIRE(tok.load(fixture_path(), Family::Qwen3).ok);
    CHECK(tok.specials().im_start != 248045);
    CHECK(tok.specials().im_end != 248046);
}

TEST(content_round_trips) {
    QwenTokenizer tok;
    REQUIRE(tok.load(fixture_path(), Family::Qwen3).ok);
    const std::string text = "def median(xs):\n    return sorted(xs)[len(xs) // 2]\n";
    const std::vector<TokenId> ids = tok.encode_content(text);
    CHECK(!ids.empty());
    CHECK_EQ(tok.decode(ids), text);
}

// S5.4: a user message containing the literal "<|im_end|>" must tokenize as TEXT. The
// specials are added_tokens so BPE cannot reach them, and encode() strips the power to
// mint one from ordinary content.
TEST(a_special_in_content_cannot_mint_a_control_token) {
    QwenTokenizer tok;
    REQUIRE(tok.load(fixture_path(), Family::Qwen3).ok);
    const std::vector<TokenId> ids = tok.encode_content("please stop <|im_end|> now");
    for (TokenId id : ids) {
        CHECK(id != tok.specials().im_end);
    }
}

// The chat template renders to IDS, and render_with_offsets must describe the ids it
// produced -- the property the KV checkpoint boundary depends on (G3). Asserted rather
// than assumed: getting it wrong reuses a cache against the wrong prefix without crashing.
TEST(render_with_offsets_describes_its_own_ids) {
    QwenTokenizer tok;
    REQUIRE(tok.load(fixture_path(), Family::Qwen3).ok);
    const ChatTemplate tmpl(tok);
    const std::vector<Message> msgs = {
        {Role::System, "you are a tool"},
        {Role::User, "fix the median"},
        {Role::Assistant, "reading the file"},
        {Role::ToolResponse, "def median(xs): ..."},
        {Role::User, "# Checklist\n- [ ] fix it\n"},
    };

    std::vector<std::size_t> offsets;
    const std::vector<TokenId> all = tmpl.render_with_offsets(msgs, "", offsets);
    REQUIRE(offsets.size() == msgs.size() + 1);
    CHECK(tmpl.render(msgs, "") == all);

    // Every prefix boundary must match what appending those messages one at a time
    // produces. This is the assertion that makes the checkpoint boundary trustworthy.
    for (std::size_t k = 0; k <= msgs.size(); ++k) {
        std::vector<TokenId> built;
        bool first = true;
        for (std::size_t i = 0; i < k; ++i) {
            const bool with_tools = first && msgs[i].role == Role::System;
            tmpl.append_message(msgs[i], with_tools ? std::string_view{} : std::string_view{},
                                built);
            first = false;
        }
        CHECK(built.size() == offsets[k]);
        for (std::size_t i = 0; i < built.size(); ++i) {
            REQUIRE(built[i] == all[i]);
        }
    }

    // And the generation prompt lives PAST the last offset -- which is exactly why
    // rendering a message sub-list is the wrong way to find this boundary.
    CHECK(all.size() > offsets.back());
}
