// Checkpoint sequence ceiling + think-budget sink behaviour (P0 §4).

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/loop/token_stream.hpp"
#include "src/model/backend.hpp"
#include "src/model/grammar.hpp"
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
