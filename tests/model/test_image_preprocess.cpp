// Image -> patch tensor. GATE: no checkpoint, no GPU, no MLX, no ImageIO.
//
// This is the half of the vision path that can be pinned exactly, and the half where a
// mistake is invisible downstream. The tower cannot check the patch ORDER it is handed --
// raster order and merge-block order have identical shapes and both produce finite
// activations through all 27 layers. The only place that contract can be asserted is
// here, against arithmetic.

#include <cmath>
#include <string>
#include <vector>

#include "src/model/image_decode.hpp"
#include "src/model/image_preprocess.hpp"

#include "tests/check.hpp"

using namespace lmp::model;

namespace {

// A gradient, so every pixel is distinguishable and a transposed axis shows up as a
// wrong value rather than a wrong shape.
ImageRGB gradient(int w, int h) {
    ImageRGB img;
    img.width = w;
    img.height = h;
    img.rgb.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3U);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t p =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)) *
                3U;
            img.rgb[p] = static_cast<std::uint8_t>(x % 256);
            img.rgb[p + 1] = static_cast<std::uint8_t>(y % 256);
            img.rgb[p + 2] = 128;
        }
    }
    return img;
}

PreprocessConfig cfg_default() { return PreprocessConfig{}; }

} // namespace

TEST(smart_resize_rounds_to_whole_merge_blocks) {
    PreprocessConfig cfg = cfg_default();
    std::string err;
    int h = 0;
    int w = 0;

    // Already a multiple of 32 and inside the pixel band: unchanged.
    REQUIRE(smart_resize(512, 512, cfg, h, w, err));
    CHECK_EQ(h, 512);
    CHECK_EQ(w, 512);

    // Rounded to the nearest block in each axis.
    REQUIRE(smart_resize(500, 300, cfg, h, w, err));
    CHECK_EQ(h % cfg.factor(), 0);
    CHECK_EQ(w % cfg.factor(), 0);

    // Never below one block, however small the input.
    REQUIRE(smart_resize(3, 5, cfg, h, w, err));
    CHECK(h >= cfg.factor());
    CHECK(w >= cfg.factor());
    CHECK_EQ(h % cfg.factor(), 0);
    CHECK_EQ(w % cfg.factor(), 0);
}

// Python's round() is round-half-to-EVEN and the reference uses it. 80/32 is exactly 2.5:
// half-to-even gives 2 blocks (64px), half-away-from-zero gives 3 (96px). That is a
// different grid and a different token count, so the tie-break is load-bearing.
TEST(smart_resize_breaks_exact_halves_the_way_python_does) {
    PreprocessConfig cfg = cfg_default();
    cfg.min_pixels = 1; // let the rounding stand rather than be scaled away
    cfg.max_pixels = 1LL << 40;
    std::string err;
    int h = 0;
    int w = 0;
    REQUIRE(smart_resize(80, 80, cfg, h, w, err));
    CHECK_EQ(h, 64); // 2.5 -> 2, not 3
    CHECK_EQ(w, 64);
    REQUIRE(smart_resize(112, 112, cfg, h, w, err));
    CHECK_EQ(h, 128); // 3.5 -> 4, which is even in the other direction
}

TEST(smart_resize_honours_the_pixel_band) {
    PreprocessConfig cfg = cfg_default();
    std::string err;
    int h = 0;
    int w = 0;

    // Far above the ceiling: scaled down, and the result must be UNDER it (the reference
    // floors, so overshoot is not allowed).
    REQUIRE(smart_resize(8000, 8000, cfg, h, w, err));
    CHECK(static_cast<long long>(h) * w <= cfg.max_pixels);
    CHECK_EQ(h % cfg.factor(), 0);

    // Far below the floor: scaled up.
    REQUIRE(smart_resize(64, 64, cfg, h, w, err));
    CHECK(static_cast<long long>(h) * w >= cfg.min_pixels);
    CHECK_EQ(w % cfg.factor(), 0);

    // Aspect ratios the patch grid cannot represent are refused, not clamped.
    CHECK(!smart_resize(1, 4000, cfg, h, w, err));
    CHECK(err.find("aspect") != std::string::npos);
}

// A token budget is the number callers actually reason about: an image competes with text
// for the same context, and the checkpoint's own 16.7M ceiling is 16384 tokens for one
// picture.
TEST(a_token_budget_bounds_the_image) {
    PreprocessConfig cfg = cfg_default();
    cfg.max_pixels = token_budget_to_max_pixels(256, cfg);
    cfg.min_pixels = 1;
    std::string err;
    int h = 0;
    int w = 0;
    REQUIRE(smart_resize(4000, 3000, cfg, h, w, err));
    const int tokens = (h / cfg.factor()) * (w / cfg.factor());
    CHECK(tokens <= 256);
    CHECK(tokens > 128); // and it uses the budget rather than collapsing to one block
}

TEST(the_grid_and_the_buffer_agree) {
    const PreprocessConfig cfg = cfg_default();
    const ImageRGB img = gradient(640, 480);
    PreprocessedImage out;
    std::string err;
    REQUIRE(preprocess_image(img, cfg, out, err));

    CHECK_EQ(out.grid_t, 1);
    CHECK_EQ(out.patch_dim, 2 * 16 * 16 * 3);
    CHECK_EQ(out.grid_h % cfg.merge_size, 0);
    CHECK_EQ(out.grid_w % cfg.merge_size, 0);
    CHECK_EQ(out.patches.size(), static_cast<std::size_t>(out.patch_count()) *
                                    static_cast<std::size_t>(out.patch_dim));
    // One <|image_pad|> per 2x2 block of patches.
    CHECK_EQ(out.token_count(), out.patch_count() / 4);

    // Normalisation is (x/255 - 0.5)/0.5, so everything lands in [-1, 1].
    float lo = 1e9F;
    float hi = -1e9F;
    for (const float v : out.patches) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    CHECK(lo >= -1.0001F);
    CHECK(hi <= 1.0001F);
}

// THE CONTRACT. Patch k of the output must be the pixels the merge-block walk says it is:
// blocks in raster order, and within each block the 2x2 in raster order. Checked by
// reconstructing the expected first pixel of several patches straight from the source.
TEST(patches_come_out_in_merge_block_order) {
    PreprocessConfig cfg = cfg_default();
    // 128x128 is 8x8 patches = 4x4 merge blocks, and sits inside the pixel band once
    // min_pixels is relaxed -- so no resample runs and the source pixels survive exactly.
    cfg.min_pixels = 1;
    const ImageRGB img = gradient(128, 128);
    PreprocessedImage out;
    std::string err;
    REQUIRE(preprocess_image(img, cfg, out, err));
    REQUIRE(out.grid_h == 8);
    REQUIRE(out.grid_w == 8);

    const int ms = cfg.merge_size;
    const int ps = cfg.patch_size;
    const int mw = out.grid_w / ms;
    std::size_t k = 0;
    for (int bi = 0; bi < out.grid_h / ms; ++bi) {
        for (int bj = 0; bj < mw; ++bj) {
            for (int ii = 0; ii < ms; ++ii) {
                for (int ij = 0; ij < ms; ++ij, ++k) {
                    const int pr = bi * ms + ii;
                    const int pc = bj * ms + ij;
                    // First pixel of this patch, first temporal frame, red channel.
                    const float got =
                        out.patches[k * static_cast<std::size_t>(out.patch_dim)];
                    const auto src_x = static_cast<float>((pc * ps) % 256);
                    const float want = (src_x / 255.0F - 0.5F) / 0.5F;
                    if (std::fabs(got - want) > 1e-5F) {
                        lmp::test::record_failure(
                            __FILE__, __LINE__,
                            "patch " + std::to_string(k) + " (block " +
                                std::to_string(bi) + "," + std::to_string(bj) +
                                " intra " + std::to_string(ii) + "," +
                                std::to_string(ij) + ") got " + std::to_string(got) +
                                " want " + std::to_string(want));
                        return;
                    }
                    // ...and the green channel carries the ROW, which is what catches a
                    // transposed patch walk that the red channel alone would miss.
                    const float got_g =
                        out.patches[k * static_cast<std::size_t>(out.patch_dim) + 1];
                    const auto src_y = static_cast<float>((pr * ps) % 256);
                    const float want_g = (src_y / 255.0F - 0.5F) / 0.5F;
                    CHECK(std::fabs(got_g - want_g) < 1e-5F);
                }
            }
        }
    }
    ++lmp::test::reg().checks;
}

// The temporal axis is a REPEAT inside one patch, not a second grid step: a still image
// occupies grid_t = 1 and each patch carries its frame twice.
TEST(a_still_image_repeats_its_frame_inside_the_patch) {
    PreprocessConfig cfg = cfg_default();
    cfg.min_pixels = 1;
    const ImageRGB img = gradient(64, 64);
    PreprocessedImage out;
    std::string err;
    REQUIRE(preprocess_image(img, cfg, out, err));
    CHECK_EQ(out.grid_t, 1);

    const int frame = cfg.patch_size * cfg.patch_size * 3;
    REQUIRE(out.patch_dim == frame * cfg.temporal_patch_size);
    for (int i = 0; i < frame; ++i) {
        CHECK_EQ(out.patches[static_cast<std::size_t>(i)],
                 out.patches[static_cast<std::size_t>(frame + i)]);
    }
}

TEST(the_resampler_preserves_a_flat_field_and_the_corners) {
    // A constant image must survive any rescale exactly: weights that do not sum to 1
    // show up here as a shifted grey, and nowhere else.
    ImageRGB flat;
    flat.width = 100;
    flat.height = 70;
    flat.rgb.assign(static_cast<std::size_t>(100 * 70 * 3), 200);
    ImageRGB dst;
    resample_bicubic(flat, 32, 32, dst);
    REQUIRE(dst.valid());
    for (const std::uint8_t v : dst.rgb) {
        CHECK_EQ(static_cast<int>(v), 200);
    }

    // Identity size is a no-op to within rounding.
    const ImageRGB g = gradient(64, 64);
    ImageRGB same;
    resample_bicubic(g, 64, 64, same);
    REQUIRE(same.valid());
    int worst = 0;
    for (std::size_t i = 0; i < g.rgb.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<int>(g.rgb[i]) -
                                         static_cast<int>(same.rgb[i])));
    }
    CHECK(worst <= 1);
}

TEST(image_paths_are_recognised_by_extension) {
    CHECK(looks_like_image_path("a/b/shot.png"));
    CHECK(looks_like_image_path("SHOT.PNG"));
    CHECK(looks_like_image_path("photo.heic")); // what a macOS screenshot actually is
    CHECK(!looks_like_image_path("src/main.cpp"));
    CHECK(!looks_like_image_path("noextension"));
}

// A decoder that returns success with no pixels would send a blank grid to the tower and
// get a confident description of nothing, so failure must be explicit on every platform.
TEST(decoding_a_non_image_fails_loudly) {
    ImageRGB out;
    std::string err;
    const std::string not_an_image = "this is not a PNG";
    CHECK(!decode_image_bytes(reinterpret_cast<const std::uint8_t*>(not_an_image.data()),
                              not_an_image.size(), 1 << 20, out, err));
    CHECK(!err.empty());
    CHECK(!decode_image_bytes(nullptr, 0, 1 << 20, out, err));
    CHECK(!err.empty());
    CHECK(!decode_image_file("/nonexistent/nope.png", 1 << 20, out, err));
    CHECK(!err.empty());
}
