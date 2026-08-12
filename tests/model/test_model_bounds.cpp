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
#include "src/model/grammar.hpp"
#include "src/model/mlx/qwen35_moe_config.hpp"
#include "src/model/model_limits.hpp"
#include "src/model/qwen_tokenizer.hpp"

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
                                    : "/Users/dev/.lmstudio/models/lmstudio-community/"
                                      "Qwen3.6-35B-A3B-MLX-4bit") +
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

TEST(think_budget_force_transitions_so_tools_keep_room) {
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
    lmp::loop::GrammarSink sink(g, nullptr, /*max_think_tokens=*/3);

    // Mini vocab is sparse; pad until encode_content yields enough ordinary ids.
    std::string prose = "a b c d e f g h i j";
    auto ids = tok().encode_content(prose);
    while (ids.size() < 6) {
        prose += " k";
        ids = tok().encode_content(prose);
    }
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK(sink.on_token(ids[i]));
    }
    CHECK(sink.think_capped);
    CHECK(g.phase() == TurnPhase::Text);
    CHECK(g.think_ids().size() == 3);
    // Remaining tokens land as answer text, not more think -- the reserved tool budget
    // path depends on this phase flip.
    CHECK(g.text_ids().size() == 3);
}
