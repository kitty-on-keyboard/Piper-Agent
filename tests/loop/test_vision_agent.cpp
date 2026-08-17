// THE WHOLE LOOP, WITH EYES. realmodel: ~19 GB of weights plus the 0.92 GB vision tower.
//
// Everything below the agent has been proved separately -- the tower against the real
// checkpoint, the splice end to end (a red image comes back "Red"), each hop of the tool
// chain in the gate. What none of that establishes is the INTEGRATION: that a model told
// there is a picture in its workspace calls `view_image`, that the path it names survives
// ToolResult -> TurnRecord -> ContextStore -> Message -> template placement ->
// PromptImage, and that the pixels arrive spliced over the pads the template reserved.
//
// The failure this catches is the quiet one. Drop the path anywhere along that chain and
// the model is TOLD about an image and never SHOWN one -- so it answers from the tool's
// summary text, fluently, and the run looks like a model ignoring a picture rather than a
// harness losing it. The summary here deliberately does NOT name the colour, so an answer
// that gets the colour right can only have come from the pixels.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <unistd.h>
#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/loop/agent.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
#include "src/tools/registry.hpp"

#include "tests/check.hpp"

namespace {

std::string qwen_dir() {
    const char* v = std::getenv("LMP_QWEN_DIR");
    return v != nullptr ? std::string(v)
                        : std::string("/Users/dev/.lmstudio/models/lmstudio-community/"
                                      "Qwen3.6-35B-A3B-MLX-4bit");
}

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base != nullptr ? base : "/tmp") + "/lmp_vis_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made != nullptr ? std::string(made) : std::string();
}

// A BMP, written by hand. No encoder dependency and no committed binary fixture: the
// format is a 54-byte header and bottom-up BGR rows padded to 4 bytes, ImageIO reads it,
// and the test's image is therefore described by the test rather than by a blob nobody
// opens.
void write_bmp(const std::string& path, int w, int h, std::uint8_t r, std::uint8_t g,
               std::uint8_t b) {
    const int row_bytes = ((w * 3 + 3) / 4) * 4;
    const std::uint32_t pixel_bytes = static_cast<std::uint32_t>(row_bytes) *
                                      static_cast<std::uint32_t>(h);
    const std::uint32_t file_bytes = 54U + pixel_bytes;
    std::vector<std::uint8_t> out(file_bytes, 0);
    const auto put16 = [&out](std::size_t at, std::uint16_t v) {
        out[at] = static_cast<std::uint8_t>(v & 0xFFU);
        out[at + 1] = static_cast<std::uint8_t>((v >> 8U) & 0xFFU);
    };
    const auto put32 = [&out](std::size_t at, std::uint32_t v) {
        for (std::size_t i = 0; i < 4; ++i) {
            out[at + i] = static_cast<std::uint8_t>((v >> (8U * i)) & 0xFFU);
        }
    };
    out[0] = 'B';
    out[1] = 'M';
    put32(2, file_bytes);
    put32(10, 54);
    put32(14, 40);            // DIB header size
    put32(18, static_cast<std::uint32_t>(w));
    put32(22, static_cast<std::uint32_t>(h));
    put16(26, 1);             // planes
    put16(28, 24);            // bits per pixel
    put32(34, pixel_bytes);
    for (int y = 0; y < h; ++y) {
        std::size_t at = 54U + static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(row_bytes);
        for (int x = 0; x < w; ++x) {
            out[at] = b; // BMP is BGR
            out[at + 1] = g;
            out[at + 2] = r;
            at += 3;
        }
    }
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return;
    }
    (void)std::fwrite(out.data(), 1, out.size(), f);
    (void)std::fclose(f);
}

lmp::tools::WorkspaceContext workspace(const std::string& root) {
    lmp::tools::WorkspaceContext ws;
    ws.root = root;
    ws.max_read_bytes = 1U << 20;
    ws.max_model_read_bytes = 16384;
    ws.max_result_bytes = 8192;
    ws.max_observation_bytes = lmp::tools::kObservationBudgetBytes;
    ws.spool_dir = root + "/.spool";
    ws.shell_wall_clock_seconds = 20;
    ws.max_image_tokens = 64; // small: the colour is legible at any resolution
    // This test loads the tower (cfg.with_vision below), so the registry must offer
    // `view_image` and resolve_images must actually resolve. The two flags describe the
    // same fact from either side of the seam and are wired together in production by
    // surface::ensure_registry; a test that set only one would be testing a state the
    // product cannot reach.
    ws.model_can_see = true;
    return ws;
}

std::string lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

} // namespace

TEST(the_agent_looks_at_a_file_and_reports_what_is_in_it) {
    using namespace lmp;

    model::QwenTokenizer tok;
    const model::LoadStatus ts = tok.load(qwen_dir() + "/tokenizer.json", model::Family::Qwen3);
    REQUIRE(ts.ok);
    REQUIRE(tok.specials().has_vision());

    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    // A solid magenta field. Chosen because no part of the prompt says "magenta": the
    // filename does not, the mission does not, and view_image's summary reports only the
    // dimensions and the token cost. The only route from pixels to the word is the tower.
    write_bmp(root + "/mystery.bmp", 256, 256, 200, 30, 190);

    platform::SystemClock clock;
    model::MlxBackend backend(clock);
    model::MlxBackendConfig cfg;
    cfg.model_dir = qwen_dir();
    cfg.with_vision = true;
    const model::LoadStatus st = backend.load(cfg);
    REQUIRE(st.ok);

    tools::Registry registry(workspace(root));
    context::ContextStore ctx(
        "There is an image at mystery.bmp. Look at it with view_image, then tell me in "
        "one short sentence what colour it is.");
    platform::EventLogWriter log;
    loop::AgentConfig config;
    config.auto_syntax_check = false;
    config.max_new_tokens = 1024;
    config.budget.max_iterations = 6;
    loop::Agent agent(tok, backend, registry, ctx, log, clock, config);

    const model::CancelToken cancel;
    const loop::RunReport report = agent.run(cancel);

    // It reached for the tool rather than guessing from the filename.
    bool called_view_image = false;
    // EVERYTHING THE RUN SAID, not just its last line. The colour may be named in the turn
    // that looked at the picture, in a later one, or in the summary `finish` carries --
    // and which of those it is says nothing about whether the model saw the image.
    std::string said;
    for (const context::TurnRecord& t : ctx.recent()) {
        if (t.tool_name == "view_image") {
            called_view_image = true;
            // ...and the path travelled with the result, which is the hop that fails
            // silently.
            CHECK_EQ(t.observed_images.size(), std::size_t{1});
        }
        said += t.assistant_text;
        said += "\n";
        said += t.tool_call_text;
        said += "\n";
    }
    std::fprintf(stderr,
                 "  [vision-agent] turns=%d reason=%s\n--- transcript ---\n%s---\n",
                 report.iterations, report.termination_reason.c_str(), said.c_str());
    CHECK(called_view_image);

    // THE ASSERTION. Nothing in the prompt names this colour, so getting it right means
    // the pixels reached the model through the splice.
    const std::string lowered = lower(said);
    const bool named_it = lowered.find("magenta") != std::string::npos ||
                          lowered.find("pink") != std::string::npos ||
                          lowered.find("purple") != std::string::npos ||
                          lowered.find("violet") != std::string::npos;
    if (!named_it) {
        lmp::test::record_failure(
            __FILE__, __LINE__,
            "the run never named the image's colour anywhere in its transcript");
    }
    ++lmp::test::reg().checks;
}
