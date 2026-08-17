#pragma once
//
// Image -> patch tensor, exactly as Qwen3VLImageProcessor does it.
//
// Pure arithmetic over an RGB8 buffer: no MLX, no ImageIO, no I/O. That is deliberate --
// this is the half of the vision path that can be tested in the GATE, on any machine,
// with no checkpoint and no GPU, and the patch ORDER it produces is a contract the tower
// cannot check for itself.
//
// THE ORDER IS THE WHOLE POINT. Qwen3VLVisionModel builds its learned position
// embeddings and its 2D rotary coordinates in MERGE-BLOCK order -- (block_row,
// block_col, intra_row, intra_col) -- and the merger folds each spatial_merge_size square
// into one output row by reshaping. So patches must arrive already permuted that way.
// Emitting them in ordinary raster order costs nothing at any shape check, produces
// finite values through all 27 layers, and describes the wrong image.
//
// Every constant here is read from the checkpoint's own preprocessor_config.json
// (patch 16, merge 2, temporal 2, mean/std 0.5, min 65536, max 16777216 pixels) rather
// than assumed; the defaults below are what those files say and exist so a caller can
// construct the struct without parsing them twice.
#include <cstdint>
#include <string>
#include <vector>

namespace lmp::model {

// Decoded pixels, 3 bytes per pixel, row-major, no padding between rows.
struct ImageRGB {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;

    [[nodiscard]] bool valid() const noexcept {
        return width > 0 && height > 0 &&
               rgb.size() == static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) * 3U;
    }
};

struct PreprocessConfig {
    int patch_size = 16;
    int merge_size = 2;
    int temporal_patch_size = 2;
    // Bounds on the RESIZED pixel count, which is what decides how many tokens the image
    // costs the context: tokens = (h/patch/merge) * (w/patch/merge). At max_pixels that
    // is 16384 tokens for one image, so the caller almost always wants a tighter ceiling
    // than the checkpoint's -- see `token_budget_to_max_pixels`.
    long long min_pixels = 65536;      // 256 x 256
    long long max_pixels = 16777216;   // 4096 x 4096
    float mean[3] = {0.5F, 0.5F, 0.5F};
    float stdev[3] = {0.5F, 0.5F, 0.5F};

    // patch_size * merge_size: the resized image must be a whole number of merge blocks
    // in both axes, because the merger reshapes rather than pads.
    [[nodiscard]] int factor() const noexcept { return patch_size * merge_size; }
    [[nodiscard]] int patch_dim() const noexcept {
        return temporal_patch_size * patch_size * patch_size * 3;
    }
};

struct PreprocessedImage {
    // Patch counts, not pixels. `t` is 1 for a still image: temporal_patch_size folds the
    // duplicated frame pair inside one patch rather than adding a grid step.
    int grid_t = 1;
    int grid_h = 0;
    int grid_w = 0;
    int patch_dim = 0;
    // [grid_h * grid_w, patch_dim], merge-block ordered, each patch laid out
    // (T, H, W, C) flattened -- which is the order the Conv3d kernel is stored in
    // ([out, kT, kH, kW, in]), so flattening that weight to a matrix fixes this too.
    std::vector<float> patches;

    [[nodiscard]] int patch_count() const noexcept { return grid_t * grid_h * grid_w; }
    // One <|image_pad|> per merged token; the caller emits exactly this many.
    [[nodiscard]] int token_count() const noexcept { return patch_count() / 4; }
};

// The largest `max_pixels` that keeps an image under `tokens` merged tokens. Callers use
// this rather than the checkpoint's 16.7M ceiling: an image is charged to the same
// context budget as text, and a 4096x4096 screenshot at full resolution is 16384 tokens
// of it.
[[nodiscard]] long long token_budget_to_max_pixels(int tokens, const PreprocessConfig& cfg);

// Qwen's `smart_resize`: round each side to a whole number of merge blocks, then scale
// the whole image (preserving aspect) if the result falls outside [min_pixels,
// max_pixels]. Returns false and sets `error` on a degenerate aspect ratio, which is what
// the reference raises on.
[[nodiscard]] bool smart_resize(int height, int width, const PreprocessConfig& cfg,
                                int& out_height, int& out_width, std::string& error);

// Decode -> resize -> rescale/normalize -> patchify, in merge-block order.
[[nodiscard]] bool preprocess_image(const ImageRGB& img, const PreprocessConfig& cfg,
                                    PreprocessedImage& out, std::string& error);

// Bicubic resample with antialiasing, which is what PIL does for a downscale and what the
// reference processor calls (Qwen2VLImageProcessor's default resample is BICUBIC).
// Exposed for the tests: a resampler is easy to get subtly wrong and impossible to see in
// the model's output.
void resample_bicubic(const ImageRGB& src, int dst_w, int dst_h, ImageRGB& dst);

} // namespace lmp::model
