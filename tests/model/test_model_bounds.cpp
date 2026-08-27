// Checkpoint sequence ceiling + think-budget sink behaviour (P0 §4).

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "src/loop/token_stream.hpp"
#include "src/model/backend.hpp"
#include "src/model/family_traits.hpp"
#include "src/model/grammar.hpp"
#include "src/model/mlx/qwen35_moe_config.hpp"
#include "src/model/model_limits.hpp"
#include "src/model/qwen_tokenizer.hpp"

#include <algorithm>
#include <stdexcept>
#include "src/model/chat_template.hpp"
#include "tests/check.hpp"

using namespace lmp::model;

namespace {

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_bounds_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

std::string tokenizer_path() {
#ifdef LMP_MINI_VOCAB_JSON
    return LMP_MINI_VOCAB_JSON;
#else
    const char* v = std::getenv("LMP_QWEN_DIR");
    return std::string(v != nullptr ? v
                                    : "") +
           "/tokenizer.json";
#endif
}

const QwenTokenizer& tok() {
    static QwenTokenizer t;
    static LoadStatus st = t.load(tokenizer_path(), Family::Qwen3);
    (void)st;
    return t;
}

} // namespace

TEST(max_position_embeddings_is_read_from_text_config) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    {
        std::ofstream out(root + "/config.json");
        out << R"({
  "model_type": "qwen3_5_moe",
  "text_config": {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "vocab_size": 256,
    "max_position_embeddings": 4096
  }
})";
    }
    CHECK_EQ(load_max_position_embeddings(root), 4096);
    CHECK_EQ(load_max_position_embeddings(root + "/missing"), 0);
}

// A stock HF export puts rope_theta at the config level; newer ones nest it under
// rope_parameters. Reading only the nested form left the base at the hardcoded 1e7 with
// no error -- and 1e7 is what our own checkpoints happen to ship, so nothing showed it.
// A checkpoint on a different base would rotate positions wrong and decode fluent text
// that rots with sequence length, which no other assertion in this suite would catch.
TEST(rope_theta_is_read_when_it_sits_at_the_config_level) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    {
        std::ofstream out(root + "/config.json");
        out << R"({
  "text_config": {
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "vocab_size": 256,
    "rope_theta": 1000000.0,
    "partial_rotary_factor": 1.0
  }
})";
    }
    mlxl::Qwen35MoeConfig cfg;
    REQUIRE(mlxl::load_qwen35_moe_config(root, cfg));
    CHECK(cfg.rope_theta == 1000000.0F);
    CHECK(cfg.partial_rotary_factor == 1.0F);
}

// The FFN is the one axis the 3.5/3.6 generation's two checkpoints differ on, and picking
// the wrong graph is not a crash -- it is a missing-weight throw at the first token, after
// ~16 GB is resident. An allowlist rather than a substring test, so a checkpoint nobody
// has seen is REFUSED instead of being run through whichever branch matches loosely.
TEST(ffn_kind_classifies_known_model_types_and_refuses_the_rest) {
    using lmp::model::mlxl::FfnKind;
    using lmp::model::mlxl::ffn_kind_for;

    // Nested text_config spellings (what the loader actually reads) and root spellings.
    CHECK(ffn_kind_for("qwen3_5_moe_text") == FfnKind::Moe);
    CHECK(ffn_kind_for("qwen3_5_moe") == FfnKind::Moe);
    CHECK(ffn_kind_for("qwen3_5_text") == FfnKind::Dense);
    CHECK(ffn_kind_for("qwen3_5") == FfnKind::Dense);

    CHECK(ffn_kind_for("llama") == FfnKind::Unknown);
    CHECK(ffn_kind_for("") == FfnKind::Unknown);
    // Substring matching would call these Moe/Dense; the allowlist must not.
    CHECK(ffn_kind_for("qwen3_5_moe_text_v2") == FfnKind::Unknown);
    CHECK(ffn_kind_for("qwen4_5") == FfnKind::Unknown);
}

// A config with no model_type at all must keep behaving exactly as it did before dispatch
// existed -- i.e. as the MoE -- or every checkpoint predating this change changes graph.
TEST(absent_model_type_still_selects_the_moe_graph) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    {
        std::ofstream out(root + "/config.json");
        out << R"({"text_config":{"hidden_size":128,"num_hidden_layers":2,"vocab_size":256}})";
    }
    mlxl::Qwen35MoeConfig cfg;
    REQUIRE(mlxl::load_qwen35_moe_config(root, cfg));
    CHECK(cfg.ffn_kind() == mlxl::FfnKind::Moe);
}

// The real 27B ships model_type "qwen3_5_text" nested under text_config, and the real
// 35B-A3B ships "qwen3_5_moe_text". Assert both end-to-end through the parser.
TEST(nested_model_type_selects_the_matching_graph) {
    for (const auto& [mt, want] :
         std::vector<std::pair<std::string, mlxl::FfnKind>>{
             {"qwen3_5_text", mlxl::FfnKind::Dense},
             {"qwen3_5_moe_text", mlxl::FfnKind::Moe}}) {
        const std::string root = temp_dir();
        REQUIRE(!root.empty());
        {
            std::ofstream out(root + "/config.json");
            out << R"({"text_config":{"model_type":")" << mt
                << R"(","hidden_size":5120,"num_hidden_layers":64,"vocab_size":248320,)"
                << R"("intermediate_size":17408}})";
        }
        mlxl::Qwen35MoeConfig cfg;
        REQUIRE(mlxl::load_qwen35_moe_config(root, cfg));
        CHECK(cfg.model_type == mt);
        CHECK(cfg.ffn_kind() == want);
        CHECK_EQ(cfg.intermediate_size, 17408);
    }
}

// The MTP head is a SEPARATE directory from the target, so the obvious operator mistake
// is pointing draftModelDir at the target. That must be refused by inspection: accepting
// it would merge 16 GB of the wrong tensors under an mtp. prefix and fail much later,
// after the load appeared to succeed.
TEST(mtp_block_size_accepts_only_a_real_mtp_checkpoint) {
    const auto write_cfg = [](const std::string& body) {
        const std::string root = temp_dir();
        std::ofstream out(root + "/config.json");
        out << body;
        return root;
    };

    CHECK_EQ(mlxl::load_mtp_block_size(
                 write_cfg(R"({"model_type":"qwen3_5_mtp","block_size":3})")),
             3);

    // The target's own config -- the mistake this guard exists for.
    CHECK_EQ(mlxl::load_mtp_block_size(
                 write_cfg(R"({"model_type":"qwen3_5","text_config":{"hidden_size":5120}})")),
             0);
    CHECK_EQ(mlxl::load_mtp_block_size(
                 write_cfg(R"({"model_type":"qwen3_5_moe","block_size":3})")),
             0);
    // block_size 1 would draft nothing; treat it as unusable rather than load weights.
    CHECK_EQ(mlxl::load_mtp_block_size(
                 write_cfg(R"({"model_type":"qwen3_5_mtp","block_size":1})")),
             0);
    CHECK_EQ(mlxl::load_mtp_block_size(write_cfg(R"({"model_type":"qwen3_5_mtp"})")), 0);
    CHECK_EQ(mlxl::load_mtp_block_size(write_cfg(R"(not json)")), 0);
    CHECK_EQ(mlxl::load_mtp_block_size("/nonexistent/mtp"), 0);
}

TEST(rope_parameters_still_overrides_the_config_level) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    {
        std::ofstream out(root + "/config.json");
        out << R"({
  "text_config": {
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "vocab_size": 256,
    "rope_theta": 1000000.0,
    "rope_parameters": { "rope_theta": 5000000.0, "partial_rotary_factor": 0.5 }
  }
})";
    }
    mlxl::Qwen35MoeConfig cfg;
    REQUIRE(mlxl::load_qwen35_moe_config(root, cfg));
    CHECK(cfg.rope_theta == 5000000.0F);
    CHECK(cfg.partial_rotary_factor == 0.5F);
}

TEST(think_budget_leaves_room_for_tools_by_forcing_a_close) {
    REQUIRE(tok().loaded());
    std::vector<parsephony::ToolSpec> tools;
    parsephony::ToolSpec spec;
    spec.name = "read_file";
    parsephony::ParamSpec p;
    p.name = "path";
    p.type = parsephony::ParamType::Text;
    p.required = true;
    spec.params.push_back(p);
    tools.push_back(spec);

    TurnGrammar g(tok(), tools);
    lmp::loop::GrammarSink sink(g, nullptr);
    ThinkCapMask capped(g, tok(), /*cap=*/3);

    // Mini vocab is sparse; pad until encode_content yields enough ordinary ids.
    std::string prose = "a b c d e f g h i j";
    auto ids = tok().encode_content(prose);
    while (ids.size() < 6) {
        prose += " k";
        ids = tok().encode_content(prose);
    }

    // Decode as the backend does: consult the mask, then feed what it permitted. Three
    // think tokens in, the mask leaves the sampler one choice.
    for (std::size_t i = 0; i < 3; ++i) {
        CHECK(capped.mask().allows(ids[i]));
        CHECK(sink.on_token(ids[i]));
    }
    CHECK_EQ(capped.mask().count(), std::size_t{1});
    CHECK(capped.mask().allows(tok().specials().think_close));

    // The forced token is an ordinary emission: the sink sees it, the phase moves, and
    // whatever budget remains is now spendable on answer text and tool XML.
    CHECK(sink.on_token(tok().specials().think_close));
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(g.think_ids().size() == 3);
    for (std::size_t i = 3; i < 6; ++i) {
        CHECK(capped.mask().allows(ids[i]));
        CHECK(sink.on_token(ids[i]));
    }
    CHECK(g.text_ids().size() == 3);
}

// --- image framing in the chat template ----------------------------------------
//
// The pad OFFSET is the number the splice addresses, and nothing downstream can check it:
// point it one token early and the splice overwrites real text, leaving the model reading
// a sentence with a hole in it -- fluent, and silent. So the template computes it and
// these pin it.

TEST(an_image_is_framed_and_its_pad_run_is_located) {
    REQUIRE(tok().loaded());
    if (!tok().specials().has_vision()) {
        // The miniature vocabulary carries the vision specials only if the generator was
        // asked for them; say so rather than passing green on an untested path.
        lmp::test::record_failure(__FILE__, __LINE__,
                                  "mini vocab has no vision specials, so the image "
                                  "framing is untested here");
        return;
    }
    const ChatTemplate tmpl(tok());
    const SpecialIds& s = tok().specials();

    Message user;
    user.role = Role::User;
    user.content = "what is this?";
    user.images.push_back({4});
    const std::vector<Message> messages = {{Role::System, "sys"}, user};

    std::vector<std::size_t> offsets;
    std::vector<ImagePlacement> places;
    const std::vector<TokenId> ids = tmpl.render_with_offsets(messages, {}, offsets, &places);

    REQUIRE(places.size() == 1);
    CHECK_EQ(places[0].message_index, std::size_t{1});
    CHECK_EQ(places[0].tokens, 4);

    // The run is exactly where it says, is exactly as long as it says, and is bracketed.
    const std::size_t at = places[0].pad_offset;
    REQUIRE(at + 4 < ids.size());
    CHECK_EQ(ids[at - 1], s.vision_start);
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(ids[at + static_cast<std::size_t>(i)], s.image_pad);
    }
    CHECK_EQ(ids[at + 4], s.vision_end);

    // ...and exactly four pads exist in the whole prompt, so nothing else minted one.
    CHECK_EQ(std::count(ids.begin(), ids.end(), s.image_pad), 4);
}

TEST(images_outside_a_user_message_are_refused) {
    REQUIRE(tok().loaded());
    if (!tok().specials().has_vision()) {
        return; // reported by the test above
    }
    const ChatTemplate tmpl(tok());
    Message sys;
    sys.role = Role::System;
    sys.content = "sys";
    sys.images.push_back({2});
    std::vector<std::size_t> offsets;
    bool threw = false;
    try {
        (void)tmpl.render_with_offsets({sys}, {}, offsets, nullptr);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

// Two images in one message keep their order and do not overlap.
TEST(two_images_in_one_message_are_placed_in_order) {
    REQUIRE(tok().loaded());
    if (!tok().specials().has_vision()) {
        return;
    }
    const ChatTemplate tmpl(tok());
    Message user;
    user.role = Role::User;
    user.content = "compare these";
    user.images.push_back({3});
    user.images.push_back({5});
    std::vector<std::size_t> offsets;
    std::vector<ImagePlacement> places;
    const std::vector<TokenId> ids =
        tmpl.render_with_offsets({user}, {}, offsets, &places);
    REQUIRE(places.size() == 2);
    CHECK_EQ(places[0].image_index, std::size_t{0});
    CHECK_EQ(places[1].image_index, std::size_t{1});
    CHECK(places[0].pad_offset + 3 <= places[1].pad_offset);
    CHECK_EQ(std::count(ids.begin(), ids.end(), tok().specials().image_pad), 8);
}

// --- can the machine hold this context budget? -------------------------------
//
// context_budget_tokens was only ever checked against max_position_embeddings, which says
// what the MODEL can address and nothing about what the MACHINE can hold. A run died with
// no run_end and no crash report because 245,760 passed that check and needed ~18 GB of KV
// beside 16.3 GB of weights on a 48 GB host. The numbers below are that run's, so this
// test fails if the policy ever stops refusing the budget that killed it.

TEST(kv_cost_counts_only_the_layers_that_actually_hold_a_cache) {
    // The shape of Qwen3.8-27B-MLX-4bit, built here rather than read from a checkpoint so
    // the gate needs no weights.
    lmp::model::mlxl::Qwen35MoeConfig cfg;
    cfg.num_hidden_layers = 64;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.full_attention_interval = 4;

    // 16 full-attention layers, not 64: the linear layers carry a fixed-size state that
    // does not grow with the sequence. Counting all of them overstates KV by 4x.
    int full = 0;
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        if (!cfg.is_linear_layer(i)) {
            ++full;
        }
    }
    CHECK_EQ(full, 16);

    // 2 (K and V) * 16 * 4 kv heads * 256 wide * 2 bytes (bf16 -- NOT the 4 bits the
    // weights are quantised to) = 64 KB per token.
    const std::size_t per_token =
        std::size_t{2} * 16 * 4 * 256 * 2;
    CHECK_EQ(per_token, std::size_t{65536});
}

TEST(a_context_budget_the_device_cannot_hold_is_refused) {
    using lmp::model::max_affordable_context_tokens;
    // The crashed run, exactly: 72 KB/token (27B + its MTP draft head), 16.29 GB of
    // weights resident, a 40.20 GB device working set.
    const std::size_t kv = 72 * 1024;
    const std::size_t weights = 16290000000ULL;
    const std::size_t working_set = 40200000000ULL;

    const int affordable = max_affordable_context_tokens(kv, weights, working_set);

    // THE ASSERTION. The configured 245,760 must not survive, and neither may the 112,088
    // the process actually died at -- a limit that still permits the observed death would
    // be a limit in name only.
    CHECK(affordable > 0);
    CHECK(affordable < 245760);
    CHECK(affordable < 112088);
    // ...and the 96,000 the extension SHIPS AS ITS DEFAULT must survive untouched. That
    // budget has run on this machine for months with no memory death, so a guard that
    // clamps it is wrong about the default rather than right about the danger -- and it
    // would fire a clamp event on every ordinary run, which is how a real signal gets
    // tuned out. This is the assertion that keeps the constants honest in both directions.
    CHECK(affordable >= 96000);

    // Unknowable inputs mean "leave the operator's setting alone", never "clamp to zero".
    // A no-MLX build reports a 0 working set, and a checkpoint we cannot parse reports 0
    // bytes per token; both must be inert rather than silently capping every run.
    CHECK_EQ(max_affordable_context_tokens(kv, weights, 0), 0);
    CHECK_EQ(max_affordable_context_tokens(0, weights, working_set), 0);
    // Weights alone over the safe share is not this function's call to make.
    CHECK_EQ(max_affordable_context_tokens(kv, working_set, working_set), 0);

    // More device, more context: the policy has to be monotonic in the memory available,
    // or a bigger machine would buy nothing.
    CHECK(max_affordable_context_tokens(kv, weights, working_set * 2) > affordable);
}

// The three-level thinking control. Both halves of it are guessable-and-wrong, which is
// the reason each has a test: `high` reads as an obvious fourth level and the reference
// template raises on it, and the two checkpoints on this machine are indistinguishable by
// Family and differ on whether they have levels at all.
TEST(reasoning_effort_accepts_only_the_three_levels_the_template_defines) {
    CHECK(parse_reasoning_effort("low").has_value());
    CHECK(parse_reasoning_effort("medium").has_value());
    CHECK(parse_reasoning_effort("xhigh").has_value());
    // Empty is "leave the checkpoint alone", and is the one non-level that is accepted.
    CHECK(parse_reasoning_effort("").has_value());
    CHECK_EQ(static_cast<int>(*parse_reasoning_effort("")),
             static_cast<int>(ReasoningEffort::Default));

    // THE ONE THAT MATTERS. Every summary of this feature lists `high` among the levels;
    // the checkpoint's own template validates against ('xhigh','medium','low') and raises
    // on everything else. Accepting it here would send an instruction the model never saw
    // in training, and the run would look fine.
    CHECK(!parse_reasoning_effort("high").has_value());
    CHECK(!parse_reasoning_effort("XHIGH").has_value()); // case is the template's, not ours
    CHECK(!parse_reasoning_effort("xhigh ").has_value());
    CHECK(!parse_reasoning_effort("none").has_value());
}

TEST(medium_is_the_level_that_instructs_nothing) {
    // Not an omission -- the template sets reasoning_instructions for xhigh and low only.
    // This is also why `medium` is the harness default: it is byte-identical to the prompt
    // every run produced before the setting existed.
    CHECK(reasoning_instructions_for(ReasoningEffort::Medium).empty());
    CHECK(reasoning_instructions_for(ReasoningEffort::Default).empty());
    CHECK(!reasoning_instructions_for(ReasoningEffort::Low).empty());
    CHECK(!reasoning_instructions_for(ReasoningEffort::XHigh).empty());
    // Quoted from the checkpoint, not paraphrased: a better-worded instruction is a
    // different instruction, and these are the words the model was tuned against.
    CHECK(reasoning_instructions_for(ReasoningEffort::XHigh)
              .find("Reasoning effort is set to xhigh.") == 0);
    CHECK(reasoning_instructions_for(ReasoningEffort::Low)
              .find("Reasoning effort is set to low.") == 0);
}

TEST(reasoning_effort_support_is_read_from_the_template_not_the_name) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    // A template that mentions it.
    {
        std::ofstream out(root + "/chat_template.jinja");
        out << "{%- set resolved = reasoning_effort|default('xhigh') %}";
    }
    CHECK(supports_reasoning_effort(root));

    // A template that does not. The directory name never changes, which is the point:
    // Qwen3.6-35B-A3B and Qwen3.8-27B both load as Family::Qwen3 and only one has levels.
    const std::string bare = temp_dir();
    REQUIRE(!bare.empty());
    {
        std::ofstream out(bare + "/chat_template.jinja");
        out << "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
    }
    CHECK(!supports_reasoning_effort(bare));

    // The older packaging: no .jinja, template embedded in tokenizer_config.json.
    const std::string embedded = temp_dir();
    REQUIRE(!embedded.empty());
    {
        std::ofstream out(embedded + "/tokenizer_config.json");
        out << R"({"chat_template": "{%- set e = reasoning_effort|default('xhigh') %}"})";
    }
    CHECK(supports_reasoning_effort(embedded));

    // Nothing readable at all answers false, so an unreadable checkpoint keeps its own
    // default rather than being handed an instruction it may not understand.
    CHECK(!supports_reasoning_effort(root + "/missing"));
}
