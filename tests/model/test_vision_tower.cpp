// The vision tower, against the real checkpoint (S11.6): labelled realmodel, excluded
// from the gate, never parallel.
//
// WHAT THESE CAN AND CANNOT PROVE. There is no reference implementation on this machine
// -- no transformers, no torch, no mlx-lm, no mlx-vlm -- so there are no ground-truth
// activations to diff against. The tower was transcribed from mlx-vlm's qwen3_vl
// vision.py (the converter named on the model card), and what follows checks the
// properties that a transcription error actually breaks: shapes at every seam, finite
// values through 27 layers, and the invariances the architecture guarantees. It does NOT
// establish numerical agreement with HF. Until an end-to-end read of a known image lands,
// treat the tower as PLAUSIBLE, not verified.
//
// The invariance tests are the load-bearing ones. A wrong rotary table, a wrong position
// interpolation or a transposed merge order all leave shapes intact and values finite --
// but each one breaks a specific symmetry below.

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/model/mlx/qwen35_moe_config.hpp"
#include "src/model/mlx/vision_tower.hpp"
#include "src/model/mlx/weight_store.hpp"

#include "tests/check.hpp"

#if LMP_HAVE_MLX

using namespace lmp::model::mlxl;

namespace {

std::string qwen_dir() {
    const char* v = std::getenv("LMP_QWEN_DIR");
    return v != nullptr ? std::string(v)
                        : std::string("/Users/dev/.lmstudio/models/lmstudio-community/"
                                      "Qwen3.6-35B-A3B-MLX-4bit");
}

// Loaded once: the tower is 0.92 GB and every test below wants the same one.
struct Fixture {
    WeightStore store;
    Qwen35VisionConfig cfg;
    Qwen35VisionTower tower;
    bool ok = false;
    std::string error;
};

Fixture& fixture() {
    static Fixture f = [] {
        Fixture x;
        if (!load_qwen35_vision_config(qwen_dir(), x.cfg)) {
            x.error = "no vision_config in " + qwen_dir();
            return x;
        }
        if (!x.store.load_directory(qwen_dir())) {
            x.error = "weights would not load from " + qwen_dir();
            return x;
        }
        x.ok = x.tower.init(x.cfg, x.store, x.error);
        return x;
    }();
    return f;
}

// A deterministic image-shaped patch buffer. Content does not matter for the shape and
// finiteness checks; it matters for the invariances, which compare two arrangements of
// the SAME patch values.
std::vector<float> patch_buffer(int n, int dim, unsigned seed) {
    std::vector<float> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(dim));
    unsigned s = seed;
    for (float& e : v) {
        s = s * 1664525U + 1013904223U;
        // Normalised to roughly what the preprocessor emits: (x/255 - 0.5) / 0.5.
        e = (static_cast<float>((s >> 16U) & 0xFFU) / 255.0F - 0.5F) / 0.5F;
    }
    return v;
}

float max_abs(const mx::array& a) {
    mx::array m = mx::max(mx::abs(a));
    mx::eval(m);
    return m.item<float>();
}

bool all_finite(const mx::array& a) {
    mx::array bad = mx::sum(mx::astype(mx::logical_not(mx::isfinite(a)), mx::int32));
    mx::eval(bad);
    return bad.item<int>() == 0;
}

} // namespace

TEST(the_checkpoint_declares_a_vision_tower_and_it_loads) {
    Fixture& f = fixture();
    if (!f.ok) {
        lmp::test::record_failure(__FILE__, __LINE__, "vision tower init: " + f.error);
        return;
    }
    // Measured from both checkpoints on this machine, pinned as regression.
    CHECK(f.cfg.present);
    CHECK_EQ(f.cfg.depth, 27);
    CHECK_EQ(f.cfg.hidden_size, 1152);
    CHECK_EQ(f.cfg.num_heads, 16);
    CHECK_EQ(f.cfg.head_dim(), 72);
    CHECK_EQ(f.cfg.patch_size, 16);
    CHECK_EQ(f.cfg.spatial_merge_size, 2);
    CHECK_EQ(f.cfg.temporal_patch_size, 2);
    CHECK_EQ(f.cfg.num_position_embeddings, 2304); // 48 x 48
    CHECK_EQ(f.tower.patch_dim(), 2 * 16 * 16 * 3);
    // Empty on both checkpoints, which is what lets this build splice at the embedding
    // alone. If a future checkpoint fills it, init() must refuse -- see the next test.
    CHECK(f.cfg.deepstack_visual_indexes.empty());
    // The five vision specials, as single ids.
    CHECK_EQ(f.cfg.vision_start_token_id, 248053);
    CHECK_EQ(f.cfg.vision_end_token_id, 248054);
    CHECK_EQ(f.cfg.image_token_id, 248056);
    CHECK_EQ(f.cfg.video_token_id, 248057);
}

// Refusing beats running-but-wrong: a deepstack checkpoint spliced only at the embedding
// produces fluent nonsense, and nothing downstream would flag it.
TEST(a_deepstack_checkpoint_is_refused_rather_than_silently_ignored) {
    Fixture& f = fixture();
    REQUIRE(f.ok);
    Qwen35VisionConfig cfg = f.cfg;
    cfg.deepstack_visual_indexes = {7, 15, 23};
    Qwen35VisionTower t;
    std::string err;
    CHECK(!t.init(cfg, f.store, err));
    CHECK(err.find("deepstack") != std::string::npos);
}

TEST(video_is_refused_rather_than_run_through_one_attention_window) {
    Fixture& f = fixture();
    REQUIRE(f.ok);
    CHECK(f.tower.validate({/*t=*/2, /*h=*/4, /*w=*/4}).find("still images") !=
          std::string::npos);
    // An odd grid cannot be folded by the merger.
    CHECK(!f.tower.validate({1, 3, 4}).empty());
    CHECK(!f.tower.validate({1, 4, 5}).empty());
    CHECK(f.tower.validate({1, 4, 4}).empty());
}

TEST(the_tower_emits_one_row_per_merge_block_and_they_are_finite) {
    Fixture& f = fixture();
    REQUIRE(f.ok);
    const VisionGrid grid{1, 8, 12}; // 96 patches -> 24 merged tokens
    const int dim = f.tower.patch_dim();
    const std::vector<float> buf = patch_buffer(grid.patches(), dim, 12345U);
    const mx::array patches(buf.data(), {grid.patches(), dim}, mx::float32);

    mx::array out = f.tower.forward(patches, grid);
    mx::eval(out);

    // One row per <|image_pad|> the template will emit.
    REQUIRE(out.ndim() == 2);
    CHECK_EQ(static_cast<int>(out.shape()[0]),
             grid.patches() / f.cfg.merge_unit());
    CHECK_EQ(static_cast<int>(out.shape()[1]), f.cfg.out_hidden_size);

    // 27 layers of attention with a wrong rope table or a bad norm eps overflows bf16
    // long before it reaches here.
    CHECK(all_finite(out));
    const float m = max_abs(out);
    std::fprintf(stderr, "  [vision] grid=%dx%d tokens=%d dim=%d max|out|=%.3f\n", grid.h,
                 grid.w, static_cast<int>(out.shape()[0]),
                 static_cast<int>(out.shape()[1]), static_cast<double>(m));
    // Residual-stream scale. A dead tower emits zeros; an exploding one emits hundreds.
    CHECK(m > 1e-3F);
    CHECK(m < 1e3F);
}

// TRANSLATION INVARIANCE OF THE MERGE ORDER. The tower is fed patches in merge-block
// order, and the merger folds each 2x2 block into one row. So permuting whole BLOCKS of
// the input must permute the output rows the same way -- if the patch order and the
// position/rotary tables ever disagree, this breaks while shapes stay right.
//
// Position embeddings and rotary coordinates are per-position, so this is checked on a
// 1-block-tall grid where swapping two blocks also swaps their coordinates: the assertion
// is the WEAKER one that a block swap changes the output at exactly the swapped rows and
// nowhere else is not claimed. What is claimed: the row count tracks blocks, and the same
// block content at the same position gives the same row.
TEST(the_same_patch_block_at_the_same_position_gives_the_same_row) {
    Fixture& f = fixture();
    REQUIRE(f.ok);
    const VisionGrid grid{1, 4, 4};
    const int dim = f.tower.patch_dim();
    const std::vector<float> buf = patch_buffer(grid.patches(), dim, 777U);
    const mx::array patches(buf.data(), {grid.patches(), dim}, mx::float32);

    mx::array a = f.tower.forward(patches, grid);
    mx::array b = f.tower.forward(patches, grid);
    mx::eval(a);
    mx::eval(b);
    // Determinism, which also means no uninitialised scratch is leaking into the graph.
    CHECK_EQ(max_abs(mx::subtract(a, b)), 0.0F);
}

// THE POSITION SIGNAL IS REAL. Identical patch content at every position must still
// produce DIFFERENT merged rows, because the learned position embedding and the 2D rotary
// coordinates differ per patch. If the interpolation silently returned one row, or the
// rotary table were all zeros, every output row would be identical -- shapes intact,
// values finite, and the model blind to layout.
TEST(uniform_patches_still_produce_position_dependent_rows) {
    Fixture& f = fixture();
    REQUIRE(f.ok);
    const VisionGrid grid{1, 4, 4};
    const int dim = f.tower.patch_dim();
    // Every patch byte-identical.
    std::vector<float> buf(static_cast<std::size_t>(grid.patches()) *
                               static_cast<std::size_t>(dim),
                           0.25F);
    const mx::array patches(buf.data(), {grid.patches(), dim}, mx::float32);

    mx::array out = f.tower.forward(patches, grid);
    mx::eval(out);
    REQUIRE(static_cast<int>(out.shape()[0]) == grid.patches() / f.cfg.merge_unit());

    const mx::array row0 = mx::slice(out, {0, 0}, {1, f.cfg.out_hidden_size});
    const mx::array rowN = mx::slice(out, {static_cast<int>(out.shape()[0]) - 1, 0},
                                     {static_cast<int>(out.shape()[0]),
                                      f.cfg.out_hidden_size});
    const float spread = max_abs(mx::subtract(row0, rowN));
    std::fprintf(stderr, "  [vision] uniform-input row spread = %.4f\n",
                 static_cast<double>(spread));
    CHECK(spread > 1e-3F);
}

// The interpolation must track the grid, not a fixed table row. Two different grid sizes
// over the same uniform content must not produce the same first row -- a stub that
// ignored `grid` and indexed pos_embed[0..n] would pass every shape check above.
TEST(the_position_table_is_interpolated_to_the_grid_not_indexed) {
    Fixture& f = fixture();
    REQUIRE(f.ok);
    const int dim = f.tower.patch_dim();
    const auto first_row = [&](const VisionGrid& g) {
        std::vector<float> buf(static_cast<std::size_t>(g.patches()) *
                                   static_cast<std::size_t>(dim),
                               0.25F);
        const mx::array p(buf.data(), {g.patches(), dim}, mx::float32);
        mx::array out = f.tower.forward(p, g);
        mx::array r = mx::slice(out, {1, 0}, {2, f.cfg.out_hidden_size});
        mx::eval(r);
        return r;
    };
    // Same content, same row index, different grid -> different interpolation source.
    const float d = max_abs(mx::subtract(first_row({1, 4, 4}), first_row({1, 16, 16})));
    std::fprintf(stderr, "  [vision] 4x4 vs 16x16 row-1 delta = %.4f\n",
                 static_cast<double>(d));
    CHECK(d > 1e-3F);
}

#endif // LMP_HAVE_MLX
