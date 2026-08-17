#include "src/model/image_preprocess.hpp"

#include <algorithm>
#include <cmath>

namespace lmp::model {
namespace {

// PIL's bicubic kernel: Catmull-Rom with a = -0.5, support 2.
float cubic(float x) {
    const float a = -0.5F;
    x = std::fabs(x);
    if (x < 1.0F) {
        return ((a + 2.0F) * x - (a + 3.0F)) * x * x + 1.0F;
    }
    if (x < 2.0F) {
        return (((x - 5.0F) * x + 8.0F) * x - 4.0F) * a;
    }
    return 0.0F;
}

// One axis of a separable resample. `scale` is src/dst: when it exceeds 1 the filter is
// WIDENED by it, which is what antialiasing means here -- a plain point-sampled bicubic
// downscale aliases hard, and a screenshot scaled 4:1 is exactly the case that shows it.
struct AxisWeights {
    int support = 0;                 // taps per output pixel
    std::vector<int> first;          // first source index per output pixel
    std::vector<float> w;            // [dst][support]
};

AxisWeights axis_weights(int src_n, int dst_n) {
    AxisWeights aw;
    const float scale = static_cast<float>(src_n) / static_cast<float>(dst_n);
    const float filter_scale = std::max(1.0F, scale);
    const float support = 2.0F * filter_scale;
    aw.support = static_cast<int>(std::ceil(support)) * 2 + 1;
    aw.first.resize(static_cast<std::size_t>(dst_n));
    aw.w.assign(static_cast<std::size_t>(dst_n) * static_cast<std::size_t>(aw.support),
                0.0F);

    for (int i = 0; i < dst_n; ++i) {
        // Pixel CENTRES, which is why the half-pixel offsets appear on both sides. Using
        // corners instead shifts the whole image by half a source pixel per axis -- a
        // shift small enough to look right and large enough to move every patch boundary.
        const float centre = (static_cast<float>(i) + 0.5F) * scale;
        const int begin = std::max(0, static_cast<int>(std::floor(centre - support)));
        const int end = std::min(src_n - 1, static_cast<int>(std::ceil(centre + support)));
        aw.first[static_cast<std::size_t>(i)] = begin;

        float total = 0.0F;
        const std::size_t base =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(aw.support);
        for (int s = begin; s <= end && (s - begin) < aw.support; ++s) {
            const float t =
                (static_cast<float>(s) + 0.5F - centre) / filter_scale;
            const float weight = cubic(t);
            aw.w[base + static_cast<std::size_t>(s - begin)] = weight;
            total += weight;
        }
        if (total != 0.0F) {
            for (int k = 0; k < aw.support; ++k) {
                aw.w[base + static_cast<std::size_t>(k)] /= total;
            }
        }
    }
    return aw;
}

} // namespace

void resample_bicubic(const ImageRGB& src, int dst_w, int dst_h, ImageRGB& dst) {
    dst.width = dst_w;
    dst.height = dst_h;
    dst.rgb.assign(static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h) * 3U,
                   0);
    if (!src.valid() || dst_w <= 0 || dst_h <= 0) {
        return;
    }

    const AxisWeights ax = axis_weights(src.width, dst_w);
    const AxisWeights ay = axis_weights(src.height, dst_h);

    // Horizontal first into a float scratch, then vertical: separable, so this is
    // O(w*h*support) rather than O(w*h*support^2).
    std::vector<float> mid(static_cast<std::size_t>(dst_w) *
                           static_cast<std::size_t>(src.height) * 3U);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const int begin = ax.first[static_cast<std::size_t>(x)];
            const std::size_t wbase =
                static_cast<std::size_t>(x) * static_cast<std::size_t>(ax.support);
            float acc[3] = {0.0F, 0.0F, 0.0F};
            for (int k = 0; k < ax.support; ++k) {
                const int sx = std::min(begin + k, src.width - 1);
                const float weight = ax.w[wbase + static_cast<std::size_t>(k)];
                const std::size_t sp = (static_cast<std::size_t>(y) *
                                            static_cast<std::size_t>(src.width) +
                                        static_cast<std::size_t>(sx)) *
                                       3U;
                acc[0] += weight * static_cast<float>(src.rgb[sp]);
                acc[1] += weight * static_cast<float>(src.rgb[sp + 1]);
                acc[2] += weight * static_cast<float>(src.rgb[sp + 2]);
            }
            const std::size_t dp = (static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(dst_w) +
                                    static_cast<std::size_t>(x)) *
                                   3U;
            mid[dp] = acc[0];
            mid[dp + 1] = acc[1];
            mid[dp + 2] = acc[2];
        }
    }

    for (int y = 0; y < dst_h; ++y) {
        const int begin = ay.first[static_cast<std::size_t>(y)];
        const std::size_t wbase =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(ay.support);
        for (int x = 0; x < dst_w; ++x) {
            float acc[3] = {0.0F, 0.0F, 0.0F};
            for (int k = 0; k < ay.support; ++k) {
                const int sy = std::min(begin + k, src.height - 1);
                const float weight = ay.w[wbase + static_cast<std::size_t>(k)];
                const std::size_t sp = (static_cast<std::size_t>(sy) *
                                            static_cast<std::size_t>(dst_w) +
                                        static_cast<std::size_t>(x)) *
                                       3U;
                acc[0] += weight * mid[sp];
                acc[1] += weight * mid[sp + 1];
                acc[2] += weight * mid[sp + 2];
            }
            const std::size_t dp = (static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(dst_w) +
                                    static_cast<std::size_t>(x)) *
                                   3U;
            for (int c = 0; c < 3; ++c) {
                dst.rgb[dp + static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(
                    std::clamp(std::lround(acc[c]), 0L, 255L));
            }
        }
    }
}

long long token_budget_to_max_pixels(int tokens, const PreprocessConfig& cfg) {
    if (tokens <= 0) {
        return cfg.min_pixels;
    }
    // One merged token covers factor x factor pixels.
    const auto f = static_cast<long long>(cfg.factor());
    return static_cast<long long>(tokens) * f * f;
}

bool smart_resize(int height, int width, const PreprocessConfig& cfg, int& out_height,
                  int& out_width, std::string& error) {
    if (height <= 0 || width <= 0) {
        error = "image has a zero dimension";
        return false;
    }
    const int factor = cfg.factor();
    const double hi = std::max(height, width);
    const double lo = std::min(height, width);
    if (hi / lo > 200.0) {
        error = "image aspect ratio " + std::to_string(hi / lo) +
                " exceeds 200:1, which the patch grid cannot represent";
        return false;
    }

    // std::nearbyint, not std::lround: the reference is Python's round(), which is
    // round-half-to-EVEN, and the default FP rounding mode matches it. They differ on
    // exact halves -- a 80px side at factor 32 is 2.5 blocks, which lround makes 3 and
    // Python makes 2 -- and that is a different grid, a different token count, and a
    // different image.
    const auto round_blocks = [factor](int v) {
        return std::max(factor,
                        static_cast<int>(std::nearbyint(static_cast<double>(v) /
                                                        static_cast<double>(factor))) *
                            factor);
    };
    int h_bar = round_blocks(height);
    int w_bar = round_blocks(width);

    const auto area = static_cast<long long>(h_bar) * static_cast<long long>(w_bar);
    const double px = static_cast<double>(height) * static_cast<double>(width);
    if (area > cfg.max_pixels) {
        // Scale DOWN, then floor to a block boundary: overshooting the ceiling is the
        // failure that matters (it is a context-budget blowout), so the rounding goes the
        // safe way.
        const double beta = std::sqrt(px / static_cast<double>(cfg.max_pixels));
        h_bar = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
    } else if (area < cfg.min_pixels) {
        const double beta = std::sqrt(static_cast<double>(cfg.min_pixels) / px);
        h_bar = std::max(factor, static_cast<int>(std::ceil(height * beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<int>(std::ceil(width * beta / factor)) * factor);
    }
    out_height = h_bar;
    out_width = w_bar;
    return true;
}

bool preprocess_image(const ImageRGB& img, const PreprocessConfig& cfg,
                      PreprocessedImage& out, std::string& error) {
    if (!img.valid()) {
        error = "image buffer is empty or its size does not match its dimensions";
        return false;
    }
    int rh = 0;
    int rw = 0;
    if (!smart_resize(img.height, img.width, cfg, rh, rw, error)) {
        return false;
    }

    ImageRGB scaled;
    if (rh == img.height && rw == img.width) {
        scaled = img; // exact-fit images skip the resampler entirely
    } else {
        resample_bicubic(img, rw, rh, scaled);
    }

    const int ps = cfg.patch_size;
    const int ms = cfg.merge_size;
    const int tp = cfg.temporal_patch_size;
    out.grid_t = 1;
    out.grid_h = rh / ps;
    out.grid_w = rw / ps;
    out.patch_dim = cfg.patch_dim();
    out.patches.assign(static_cast<std::size_t>(out.grid_h) *
                           static_cast<std::size_t>(out.grid_w) *
                           static_cast<std::size_t>(out.patch_dim),
                       0.0F);

    const int mh = out.grid_h / ms;
    const int mw = out.grid_w / ms;
    std::size_t w = 0;
    // MERGE-BLOCK ORDER. See the header: this loop nest IS the contract with
    // Qwen35VisionTower's position and rotary tables, and nothing downstream can detect
    // it being wrong.
    for (int bi = 0; bi < mh; ++bi) {
        for (int bj = 0; bj < mw; ++bj) {
            for (int ii = 0; ii < ms; ++ii) {
                for (int ij = 0; ij < ms; ++ij) {
                    const int pr = bi * ms + ii;
                    const int pc = bj * ms + ij;
                    // (T, H, W, C) within the patch, matching the stored Conv3d kernel.
                    // A still image repeats its single frame across the temporal axis,
                    // which is what the reference processor does -- the pair is one
                    // patch, not two grid steps.
                    for (int t = 0; t < tp; ++t) {
                        for (int py = 0; py < ps; ++py) {
                            for (int px = 0; px < ps; ++px) {
                                const std::size_t sp =
                                    (static_cast<std::size_t>(pr * ps + py) *
                                         static_cast<std::size_t>(rw) +
                                     static_cast<std::size_t>(pc * ps + px)) *
                                    3U;
                                for (int c = 0; c < 3; ++c) {
                                    const float v =
                                        static_cast<float>(
                                            scaled.rgb[sp + static_cast<std::size_t>(c)]) /
                                        255.0F;
                                    out.patches[w++] =
                                        (v - cfg.mean[c]) / cfg.stdev[c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

} // namespace lmp::model
