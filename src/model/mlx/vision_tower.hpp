#ifndef LLM_MODELS_QWEN35_VISION_TOWER_HPP
#define LLM_MODELS_QWEN35_VISION_TOWER_HPP

#if LMP_HAVE_MLX

// The vision tower both checkpoints on this machine already carry, and which the loader
// used to delete on the way in.
//
// 333 tensors, 0.92 GB unquantized against 15.13 GB of quantized text -- so eyes cost
// about 6% more resident memory. `sanitize_weights()` dropped every `vision_tower.*` key
// before this existed, which is why "can the model see?" read as no: the weights were on
// disk, the tokenizer had the five vision specials, the checkpoint's own chat template
// emitted `<|vision_start|><|image_pad|><|vision_end|>`, and the harness threw the tower
// away at load.
//
// TRANSCRIBED FROM mlx-vlm's qwen3_vl/vision.py, which is the implementation that
// PRODUCED these weights (the model card names it as the converter). Two details here
// are not recoverable from the tensor shapes and would have been guessed wrong:
//
//   * The attention carries 2D ROTARY EMBEDDINGS over (row, col) patch coordinates.
//     Nothing in the checkpoint says so -- rope is parameter-free -- and a tower built
//     from the shapes alone omits it, runs, and produces confidently wrong features.
//   * The learned `pos_embed` table is a 48x48 grid BILINEARLY INTERPOLATED to the
//     patch grid, not indexed. Wrong interpolation is likewise silent.
//
// So this file is a transcription, not a derivation, and the reference is named at each
// step that is not obvious. A vision tower that is subtly wrong does not crash; it
// describes the wrong image fluently, which is the worst failure available here.
//
// PATCH ORDER IS PART OF THE CONTRACT. Patches arrive already permuted into merge-block
// order -- (block_row, block_col, intra_row, intra_col) -- because the position
// embeddings and the rotary coordinates are both built in that order and the merger
// folds each 2x2 block by reshaping. See image_preprocess.hpp, which is the only
// supported producer.

#include "qwen35_moe_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "weight_store.hpp"

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

// The grid a preprocessed image occupies, in PATCHES (not pixels). `t` is 1 for a still
// image: temporal_patch_size folds the duplicated frame pair inside a single patch
// rather than adding a grid step.
struct VisionGrid {
    int t{1};
    int h{0};
    int w{0};

    [[nodiscard]] int patches() const noexcept { return t * h * w; }
};

class Qwen35VisionTower {
public:
    // `weights` must outlive the tower: it is the text model's own store, because the
    // vision tensors ship in the same safetensors shards and there is nothing to gain
    // from a second copy.
    bool init(const Qwen35VisionConfig& cfg, const WeightStore& weights,
              std::string& error) {
        cfg_ = cfg;
        weights_ = &weights;
        if (!cfg_.present) {
            error = "vision: this checkpoint declares no vision_config";
            return false;
        }
        // DEEPSTACK IS REFUSED, NOT IGNORED. Qwen3-VL can inject visual features at
        // several LLM layers as well as at the embedding; this build splices at the
        // embedding only. Both checkpoints here declare an EMPTY list, so nothing is
        // lost -- but a checkpoint that wants the extra injection sites and silently
        // does not get them would run and be wrong, and no shape check downstream
        // would notice.
        if (!cfg_.deepstack_visual_indexes.empty()) {
            error = "vision: checkpoint requests deepstack injection at " +
                    std::to_string(cfg_.deepstack_visual_indexes.size()) +
                    " layers, which this build does not implement (it splices at the "
                    "embedding only). Refusing rather than running without it.";
            return false;
        }
        // Every tensor the forward pass will reach for, checked BEFORE the first image
        // rather than at the first token: a missing key throws out of WeightStore::get,
        // and doing that mid-generation loses the turn.
        if (!weights.has(kPrefix + "patch_embed.proj.weight") ||
            !weights.has(kPrefix + "pos_embed.weight") ||
            !weights.has(kPrefix + "merger.linear_fc2.weight")) {
            error = "vision: vision_tower weights absent from the checkpoint (the "
                    "config declares a vision tower, so this is a broken export or a "
                    "text-only re-quantization)";
            return false;
        }
        for (int i = 0; i < cfg_.depth; ++i) {
            const std::string p = block_prefix(i);
            if (!weights.has(p + "attn.qkv.weight") || !weights.has(p + "norm1.weight")) {
                error = "vision: block " + std::to_string(i) + " is missing from the "
                        "checkpoint, but vision_config declares depth " +
                        std::to_string(cfg_.depth);
                return false;
            }
        }
        num_grid_per_side_ =
            static_cast<int>(std::lround(std::sqrt(static_cast<double>(
                cfg_.num_position_embeddings))));
        if (num_grid_per_side_ * num_grid_per_side_ != cfg_.num_position_embeddings) {
            error = "vision: num_position_embeddings (" +
                    std::to_string(cfg_.num_position_embeddings) +
                    ") is not a square, so the learned position table is not a grid and "
                    "cannot be interpolated as one";
            return false;
        }
        return true;
    }

    // The dimension a preprocessed patch vector must have: (T, H, W, C) flattened, in
    // that axis order, because that is how the Conv3d kernel is stored -- the weight is
    // [out, kT, kH, kW, in], so flattening it to a matrix fixes the input's layout too.
    [[nodiscard]] int patch_dim() const noexcept {
        return cfg_.temporal_patch_size * cfg_.patch_size * cfg_.patch_size *
               cfg_.in_channels;
    }

    [[nodiscard]] const Qwen35VisionConfig& config() const noexcept { return cfg_; }

    // Every way `grid` can be wrong for this tower, named. Callers check this before
    // building patches, so a bad grid costs nothing; forward() checks it again because a
    // mis-sized reshape inside the graph reports as an MLX shape error with no idea which
    // of the caller's assumptions produced it.
    [[nodiscard]] std::string validate(const VisionGrid& grid) const {
        const int m = cfg_.spatial_merge_size;
        if (grid.h <= 0 || grid.w <= 0 || grid.t <= 0) {
            return "vision: empty patch grid";
        }
        if (grid.h % m != 0 || grid.w % m != 0) {
            return "vision: patch grid " + std::to_string(grid.h) + "x" +
                   std::to_string(grid.w) + " is not a multiple of spatial_merge_size " +
                   std::to_string(m) + " (the preprocessor must round the image up to a "
                   "multiple of patch_size * merge_size)";
        }
        // MULTI-FRAME IS REFUSED. The reference splits attention at frame boundaries so
        // that a patch never attends across time; this build runs one window over the
        // whole sequence, which is correct for a still image and silently wrong for
        // video. Refusing is the difference between "video is not implemented" and
        // "video produces confident nonsense".
        if (grid.t != 1) {
            return "vision: this build handles still images only (grid.t=" +
                   std::to_string(grid.t) +
                   "); video needs per-frame attention windows, which are not implemented";
        }
        return {};
    }

    // `patches` is [num_patches, patch_dim()], merge-block ordered. Returns
    // [num_patches / merge_unit, out_hidden_size] -- one row per `<|image_pad|>` the
    // template must emit.
    //
    // Unevaluated, like the text model's forward: the caller decides when to force.
    [[nodiscard]] mx::array forward(const mx::array& patches, const VisionGrid& grid) const {
        const std::string bad = validate(grid);
        if (!bad.empty()) {
            throw std::runtime_error(bad);
        }
        const int n = grid.patches();
        const int dim = cfg_.hidden_size;

        // --- patch embedding ------------------------------------------------------
        // The Conv3d is a per-patch dot product: kernel == patch extent and stride ==
        // kernel, so no window ever spans two patches. Folding it to a matmul is exact,
        // not an approximation, and it avoids materialising the image as a 5-D volume.
        const mx::array& pw = weights_->get(kPrefix + "patch_embed.proj.weight");
        const mx::array w2d = mx::reshape(pw, {cfg_.hidden_size, patch_dim()});
        mx::array x = mx::matmul(mx::astype(patches, pw.dtype()),
                                 mx::transpose(w2d, {1, 0}));
        x = mx::add(x, weights_->get(kPrefix + "patch_embed.proj.bias"));

        // --- learned position embedding, bilinear over the 48x48 grid --------------
        x = mx::add(x, interpolate_pos_embed(grid, pw.dtype()));

        // --- 2D rotary coordinates -------------------------------------------------
        const mx::array freqs = rotary_freqs(grid); // [n, head_dim/2]
        const mx::array cos_t = rope_factor(mx::cos(freqs), pw.dtype());
        const mx::array sin_t = rope_factor(mx::sin(freqs), pw.dtype());

        // --- blocks ----------------------------------------------------------------
        for (int i = 0; i < cfg_.depth; ++i) {
            const std::string p = block_prefix(i);
            x = mx::add(x, attention(layer_norm(x, p + "norm1"), cos_t, sin_t, n, p));
            x = mx::add(x, mlp(layer_norm(x, p + "norm2"), p + "mlp."));
        }

        // --- merger ----------------------------------------------------------------
        // LayerNorm runs on the UNMERGED rows (its weight is hidden_size wide), and only
        // then does the reshape fold each merge block into one row. Doing it the other
        // way round normalises across four patches at once, which is a different
        // function and one the weight shape would not catch.
        mx::array m = layer_norm(x, kPrefix + "merger.norm");
        m = mx::reshape(m, {n / cfg_.merge_unit(), dim * cfg_.merge_unit()});
        m = gelu_tanh(linear(m, kPrefix + "merger.linear_fc1"));
        return linear(m, kPrefix + "merger.linear_fc2");
    }

private:
    static constexpr const char* kPrefixLit = "vision_tower.";
    const std::string kPrefix{kPrefixLit};
    // HF's vision LayerNorms carry no eps in config.json; the reference constructs them
    // with the transformers default.
    static constexpr float kLayerNormEps = 1e-6F;
    static constexpr float kRopeTheta = 10000.0F;

    [[nodiscard]] std::string block_prefix(int i) const {
        return kPrefix + "blocks." + std::to_string(i) + ".";
    }

    [[nodiscard]] mx::array linear(const mx::array& x, const std::string& key) const {
        mx::array y = mx::matmul(x, mx::transpose(weights_->get(key + ".weight"), {1, 0}));
        const mx::array* b = weights_->try_get(key + ".bias");
        return b != nullptr ? mx::add(y, *b) : y;
    }

    [[nodiscard]] mx::array layer_norm(const mx::array& x, const std::string& key) const {
        return mx::fast::layer_norm(x, weights_->get(key + ".weight"),
                                    weights_->get(key + ".bias"), kLayerNormEps);
    }

    // gelu_pytorch_tanh, which is what vision_config.hidden_act names. Not mx::erf-based
    // gelu: the two differ by ~1e-3 at the tails, and this is the one the weights were
    // trained and converted against.
    [[nodiscard]] static mx::array gelu_tanh(const mx::array& x) {
        const mx::array x3 = mx::multiply(mx::multiply(x, x), x);
        const mx::array inner = mx::multiply(
            mx::array(0.7978845608028654F),
            mx::add(x, mx::multiply(mx::array(0.044715F), x3)));
        return mx::multiply(mx::multiply(mx::array(0.5F), x),
                            mx::add(mx::array(1.0F), mx::tanh(inner)));
    }

    [[nodiscard]] mx::array mlp(const mx::array& x, const std::string& p) const {
        return linear(gelu_tanh(linear(x, p + "linear_fc1")), p + "linear_fc2");
    }

    // cos/sin shaped for broadcast against [n, heads, head_dim]. The frequency vector is
    // half a head wide and is TILED, not repeated: `rotate_half` splits the head in two
    // and the same frequency must land on both halves.
    [[nodiscard]] mx::array rope_factor(const mx::array& f, mx::Dtype dt) const {
        const mx::array t = mx::tile(mx::expand_dims(f, 1), {1, 1, 2}); // [n,1,head_dim]
        return mx::astype(t, dt);
    }

    [[nodiscard]] static mx::array rotate_half(const mx::array& x) {
        const std::vector<mx::array> halves = mx::split(x, 2, -1);
        return mx::concatenate({mx::negative(halves[1]), halves[0]}, -1);
    }

    [[nodiscard]] mx::array attention(const mx::array& x, const mx::array& cos_t,
                                      const mx::array& sin_t, int n,
                                      const std::string& block) const {
        const int heads = cfg_.num_heads;
        const int hd = cfg_.head_dim();

        const mx::array qkv = linear(x, block + "attn.qkv");
        // [n, 3, heads, hd] -> [3, n, heads, hd]: qkv is laid out q|k|v ACROSS the whole
        // row, so the split axis is the second one after reshaping, not the head axis.
        const mx::array t = mx::transpose(mx::reshape(qkv, {n, 3, heads, hd}), {1, 0, 2, 3});
        const std::vector<mx::array> parts = mx::split(t, 3, 0);

        auto rope = [&](const mx::array& p) {
            const mx::array r = mx::reshape(p, {n, heads, hd});
            return mx::add(mx::multiply(r, cos_t), mx::multiply(rotate_half(r), sin_t));
        };
        // [n, heads, hd] -> [1, heads, n, hd]
        const mx::array q = mx::transpose(mx::expand_dims(rope(parts[0]), 0), {0, 2, 1, 3});
        const mx::array k = mx::transpose(mx::expand_dims(rope(parts[1]), 0), {0, 2, 1, 3});
        const mx::array v =
            mx::transpose(mx::expand_dims(mx::reshape(parts[2], {n, heads, hd}), 0),
                          {0, 2, 1, 3});

        // NO MASK. The reference splits the sequence at `cu_seqlens` so that each frame
        // attends only within itself; a still image is one frame, so the whole sequence
        // is one window and every patch may see every other. forward() refuses t != 1
        // rather than pretend this generalises.
        const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
        mx::array a = mx::fast::scaled_dot_product_attention(q, k, v, scale, "");
        a = mx::reshape(mx::transpose(a, {0, 2, 1, 3}), {n, heads * hd});
        return linear(a, block + "attn.proj");
    }

    // Bilinear interpolation of the learned 48x48 table onto the patch grid, then the
    // same merge-block permutation the patches themselves are in.
    //
    // Index and weight arithmetic runs on the HOST: it is O(h*w) integers built once per
    // image, against a tower that is about to do 27 layers of attention over the same
    // grid, and expressing it as MLX graph ops would cost more to schedule than to
    // compute.
    [[nodiscard]] mx::array interpolate_pos_embed(const VisionGrid& grid,
                                                  mx::Dtype dt) const {
        const int side = num_grid_per_side_;
        const int h = grid.h;
        const int w = grid.w;

        const auto axis = [side](int n, std::vector<int>& floor_i, std::vector<int>& ceil_i,
                                 std::vector<float>& frac) {
            floor_i.resize(static_cast<std::size_t>(n));
            ceil_i.resize(static_cast<std::size_t>(n));
            frac.resize(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                // linspace(0, side-1, n): the target grid spans the SOURCE grid
                // inclusive at both ends. n == 1 degenerates to the first row, which is
                // what linspace does with a single sample.
                const double v = n > 1 ? static_cast<double>(i) *
                                             (static_cast<double>(side - 1) /
                                              static_cast<double>(n - 1))
                                       : 0.0;
                const int f = static_cast<int>(v);
                floor_i[static_cast<std::size_t>(i)] = f;
                ceil_i[static_cast<std::size_t>(i)] = std::min(f + 1, side - 1);
                frac[static_cast<std::size_t>(i)] = static_cast<float>(v - f);
            }
        };
        std::vector<int> hf;
        std::vector<int> hc;
        std::vector<float> dh;
        std::vector<int> wf;
        std::vector<int> wc;
        std::vector<float> dw;
        axis(h, hf, hc, dh);
        axis(w, wf, wc, dw);

        const auto count = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
        std::vector<std::int32_t> idx(count * 4);
        std::vector<float> wt(count * 4);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                const auto c = static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
                               static_cast<std::size_t>(j);
                const std::size_t hi = static_cast<std::size_t>(i);
                const std::size_t wj = static_cast<std::size_t>(j);
                idx[c] = hf[hi] * side + wf[wj];
                idx[count + c] = hf[hi] * side + wc[wj];
                idx[2 * count + c] = hc[hi] * side + wf[wj];
                idx[3 * count + c] = hc[hi] * side + wc[wj];
                wt[c] = (1.0F - dh[hi]) * (1.0F - dw[wj]);
                wt[count + c] = (1.0F - dh[hi]) * dw[wj];
                wt[2 * count + c] = dh[hi] * (1.0F - dw[wj]);
                wt[3 * count + c] = dh[hi] * dw[wj];
            }
        }

        const mx::array table = weights_->get(kPrefix + "pos_embed.weight");
        const auto n = static_cast<int>(count);
        const mx::array ids =
            mx::array(idx.data(), {4, n}, mx::int32);
        const mx::array weights =
            mx::astype(mx::array(wt.data(), {4, n, 1}, mx::float32), table.dtype());
        // [4, n, dim] gathered, weighted, summed over the four corners.
        const mx::array gathered = mx::multiply(mx::take(table, ids, 0), weights);
        const std::vector<mx::array> corners = mx::split(gathered, 4, 0);
        mx::array pos = mx::add(mx::add(corners[0], corners[1]),
                                mx::add(corners[2], corners[3]));
        pos = mx::reshape(pos, {n, cfg_.hidden_size});

        pos = merge_block_permute(pos, grid, cfg_.hidden_size);
        if (grid.t > 1) {
            pos = mx::tile(pos, {grid.t, 1});
        }
        return mx::astype(pos, dt);
    }

    // [h*w, C] in row-major patch order -> merge-block order, which is what the merger's
    // reshape folds and what the rotary coordinates below assume.
    [[nodiscard]] mx::array merge_block_permute(const mx::array& v, const VisionGrid& grid,
                                                int c) const {
        const int m = cfg_.spatial_merge_size;
        mx::array r = mx::reshape(v, {grid.h / m, m, grid.w / m, m, c});
        r = mx::transpose(r, {0, 2, 1, 3, 4});
        return mx::reshape(r, {grid.h * grid.w, c});
    }

    // Per-patch (row, col) rotary frequencies: half the head encodes the row, half the
    // column. Built in merge-block order so index i here is the same patch as row i of
    // the embedding above.
    [[nodiscard]] mx::array rotary_freqs(const VisionGrid& grid) const {
        const int m = cfg_.spatial_merge_size;
        const int half = cfg_.head_dim() / 2; // frequency width before the h|w concat
        const int pairs = half / 2;
        const int max_hw = std::max(grid.h, grid.w);

        std::vector<float> table(static_cast<std::size_t>(max_hw) *
                                 static_cast<std::size_t>(pairs));
        for (int p = 0; p < max_hw; ++p) {
            for (int k = 0; k < pairs; ++k) {
                const float inv = 1.0F / std::pow(kRopeTheta,
                                                  static_cast<float>(2 * k) /
                                                      static_cast<float>(half));
                table[static_cast<std::size_t>(p) * static_cast<std::size_t>(pairs) +
                      static_cast<std::size_t>(k)] = static_cast<float>(p) * inv;
            }
        }

        const int mh = grid.h / m;
        const int mw = grid.w / m;
        std::vector<std::int32_t> rows;
        std::vector<std::int32_t> cols;
        rows.reserve(static_cast<std::size_t>(grid.h) * static_cast<std::size_t>(grid.w));
        cols.reserve(rows.capacity());
        for (int bi = 0; bi < mh; ++bi) {
            for (int bj = 0; bj < mw; ++bj) {
                for (int ii = 0; ii < m; ++ii) {
                    for (int ij = 0; ij < m; ++ij) {
                        rows.push_back(bi * m + ii);
                        cols.push_back(bj * m + ij);
                    }
                }
            }
        }

        const auto n = static_cast<int>(rows.size());
        const mx::array tab = mx::array(table.data(), {max_hw, pairs}, mx::float32);
        const mx::array r = mx::take(tab, mx::array(rows.data(), {n}, mx::int32), 0);
        const mx::array c = mx::take(tab, mx::array(cols.data(), {n}, mx::int32), 0);
        mx::array f = mx::concatenate({r, c}, -1); // [n, head_dim/2]
        if (grid.t > 1) {
            f = mx::tile(f, {grid.t, 1});
        }
        return f;
    }

    Qwen35VisionConfig cfg_{};
    const WeightStore* weights_{nullptr};
    int num_grid_per_side_{0};
};

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MODELS_QWEN35_VISION_TOWER_HPP
