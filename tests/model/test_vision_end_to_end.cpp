// THE TEST THAT ACTUALLY ANSWERS "CAN IT SEE?".
//
// Everything else in the vision path checks a property: shapes, finiteness, patch order,
// cache tags. None of them can tell a correct tower from a plausible one -- there is no
// reference implementation on this machine to diff activations against (no transformers,
// no torch, no mlx-lm, no mlx-vlm), so numerical agreement with HF is not establishable
// here. What IS establishable is the only thing the product cares about: show the model a
// picture whose content is known by construction, and see whether it says what is in it.
//
// A tower with a wrong rotary table, a wrong position interpolation or a transposed patch
// walk still produces finite activations of a sane magnitude. It does not produce the
// right colour.
//
// realmodel: ~19 GB of weights plus the 0.92 GB tower, never run in parallel.

#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/model/backend.hpp"
#include "src/model/chat_template.hpp"
#include "src/model/grammar.hpp"
#include "src/model/image_preprocess.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"

#include "tests/check.hpp"

using namespace lmp::model;

namespace {

std::string qwen_dir() {
    const char* v = std::getenv("LMP_QWEN_DIR");
    return v != nullptr ? std::string(v)
                        : std::string("/Users/dev/.lmstudio/models/lmstudio-community/"
                                      "Qwen3.6-35B-A3B-MLX-4bit");
}

// A synthesized image, so the answer is known by construction rather than by my reading
// of a fixture -- and so the test carries no binary blob. A solid field of one strong
// colour is the most robust signal a vision tower can be asked for: it survives any
// resampling, does not depend on reading text, and a tower that is merely APPROXIMATELY
// right still gets it.
ImageRGB solid(int w, int h, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    ImageRGB img;
    img.width = w;
    img.height = h;
    img.rgb.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3U);
    for (std::size_t p = 0; p < img.rgb.size(); p += 3) {
        img.rgb[p] = r;
        img.rgb[p + 1] = g;
        img.rgb[p + 2] = b;
    }
    return img;
}

std::string lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

// Builds the prompt the checkpoint's own chat template specifies for an image:
// <|vision_start|> <|image_pad|> x N <|vision_end|> then the question.
struct BuiltPrompt {
    std::vector<TokenId> ids;
    std::size_t pad_offset = 0;
    int tokens = 0;
};

// Through ChatTemplate, not by hand. The offset the splice addresses is the thing that
// cannot be checked downstream -- a wrong one overwrites real text and the model reads a
// sentence with a hole in it -- so the template is what must compute it, and this test is
// what proves the template and the splice agree.
BuiltPrompt build(const QwenTokenizer& tok, const ChatTemplate& tmpl,
                  const PreprocessedImage& img, const std::string& question) {
    (void)tok;
    Message user;
    user.role = Role::User;
    user.content = question;
    user.images.push_back({img.token_count()});
    const std::vector<Message> messages = {
        {Role::System, "You are a terse assistant."}, user};

    BuiltPrompt out;
    std::vector<std::size_t> offsets;
    std::vector<ImagePlacement> placements;
    out.ids = tmpl.render_with_offsets(messages, {}, offsets, &placements);
    if (placements.size() != 1) {
        lmp::test::record_failure(__FILE__, __LINE__,
                                  "expected one image placement, got " +
                                      std::to_string(placements.size()));
        return out;
    }
    out.pad_offset = placements[0].pad_offset;
    out.tokens = placements[0].tokens;
    return out;
}

class Sink final : public TokenSink {
  public:
    explicit Sink(TurnGrammar& g) : g_(g) {}
    bool on_token(TokenId id) override {
        last = g_.advance(id);
        return last == Advance::Ok;
    }
    Advance last = Advance::Ok;

  private:
    TurnGrammar& g_;
};

// Runs one image + question and returns the answer channel.
std::string ask(MlxBackend& backend, const QwenTokenizer& tok, const ImageRGB& image,
                const std::string& question, std::uint64_t content_hash) {
    PreprocessConfig pc;
    // A small budget: the colour is legible at any resolution and this keeps the test to
    // a few hundred prompt tokens.
    pc.max_pixels = token_budget_to_max_pixels(64, pc);
    pc.min_pixels = 1;
    PreprocessedImage pre;
    std::string err;
    if (!preprocess_image(image, pc, pre, err)) {
        lmp::test::record_failure(__FILE__, __LINE__, "preprocess: " + err);
        return {};
    }

    ChatTemplate tmpl(tok);
    const BuiltPrompt bp = build(tok, tmpl, pre, question);

    const std::vector<parsephony::ToolSpec> no_tools;
    TurnGrammar grammar(tok, no_tools);
    InferenceTask task;
    task.prompt = bp.ids;
    task.max_new_tokens = 384;
    task.sampling.seed = 11;
    task.mask = &grammar;

    PromptImage pi;
    pi.pad_offset = bp.pad_offset;
    pi.tokens = bp.tokens;
    pi.grid_h = pre.grid_h;
    pi.grid_w = pre.grid_w;
    pi.patch_dim = pre.patch_dim;
    pi.patches = pre.patches;
    pi.content_hash = content_hash;
    task.images.push_back(std::move(pi));

    Sink sink(grammar);
    CancelToken cancel;
    const GenResult r = backend.generate(task, sink, cancel);
    if (!r.error.empty()) {
        lmp::test::record_failure(__FILE__, __LINE__, "generate: " + r.error);
        return {};
    }
    const std::string answer = tok.decode(grammar.text_ids());
    std::fprintf(stderr, "  [vision-e2e] %dx%d -> %d tokens | %s\n", pre.grid_h,
                 pre.grid_w, bp.tokens, answer.c_str());
    return lower(answer);
}

} // namespace

TEST(the_model_reads_the_colour_of_an_image_it_is_shown) {
    QwenTokenizer tok;
    const LoadStatus ts = tok.load(qwen_dir() + "/tokenizer.json", Family::Qwen3);
    REQUIRE(ts.ok);
    REQUIRE(tok.specials().has_vision());

    lmp::platform::SystemClock clock;
    MlxBackend backend(clock);
    MlxBackendConfig cfg;
    cfg.model_dir = qwen_dir();
    cfg.with_vision = true;
    const LoadStatus st = backend.load(cfg);
    REQUIRE(st.ok);

    // Two DIFFERENT images through the same backend, in sequence. That is also the
    // cache-collision case: both are the same run of <|image_pad|> ids at the same
    // offsets, so a ledger comparing ids alone would serve the first one's KV for the
    // second and the model would answer "red" twice.
    const std::string red =
        ask(backend, tok, solid(320, 320, 220, 20, 20),
            "What single colour fills this image? Answer with one word.", 0x9E3779B97F4A7C15ULL);
    CHECK(red.find("red") != std::string::npos);

    const std::string blue =
        ask(backend, tok, solid(320, 320, 20, 40, 220),
            "What single colour fills this image? Answer with one word.", 0xC2B2AE3D27D4EB4FULL);
    CHECK(blue.find("blue") != std::string::npos);
    // The second answer must not be the first one repeated out of a stale cache.
    CHECK(blue.find("red") == std::string::npos);
}
