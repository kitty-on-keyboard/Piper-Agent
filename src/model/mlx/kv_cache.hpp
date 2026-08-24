#ifndef LLM_MLX_KV_CACHE_HPP
#define LLM_MLX_KV_CACHE_HPP

#if LMP_HAVE_MLX

#include <algorithm>
#include <optional>
#include <tuple>
#include <vector>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

struct KVCache {
    // Buffers are preallocated in kStep-token blocks along the sequence axis and
    // written in place via slice_update; `offset` is the true token count, the
    // padding beyond it is scratch. The naive alternative — concatenating the
    // full history with each new token — copies the ENTIRE cache per layer per
    // decoded token (O(context) GPU traffic per token, multiple GB/token at long
    // contexts), which is what made decode throughput degrade with context.
    static constexpr int kStep = 256;

    std::optional<mx::array> keys;
    std::optional<mx::array> values;
    int offset{0};

    void clear() noexcept {
        keys.reset();
        values.reset();
        offset = 0;
    }

    std::pair<mx::array, mx::array> update_and_fetch(const mx::array& k, const mx::array& v) {
        const int B = static_cast<int>(k.shape()[0]);
        const int n_kv = static_cast<int>(k.shape()[1]);
        const int n_new = static_cast<int>(k.shape()[2]);
        const int kd = static_cast<int>(k.shape()[3]);
        const int vd = static_cast<int>(v.shape()[3]);
        const int prev = offset;

        if (!keys.has_value() || prev + n_new > static_cast<int>(keys->shape()[2])) {
            const int n_steps = (n_new + kStep - 1) / kStep;
            mx::array k_grow = mx::zeros({B, n_kv, n_steps * kStep, kd}, k.dtype());
            mx::array v_grow = mx::zeros({B, n_kv, n_steps * kStep, vd}, v.dtype());
            if (keys.has_value()) {
                if (prev % kStep != 0) {
                    keys = mx::slice(*keys, {0, 0, 0, 0}, {B, n_kv, prev, kd});
                    values = mx::slice(*values, {0, 0, 0, 0}, {B, n_kv, prev, vd});
                }
                keys = mx::concatenate({*keys, k_grow}, 2);
                values = mx::concatenate({*values, v_grow}, 2);
            } else {
                keys = std::move(k_grow);
                values = std::move(v_grow);
            }
        }

        offset += n_new;
        keys = mx::slice_update(*keys, k, {0, 0, prev, 0}, {B, n_kv, offset, kd});
        values = mx::slice_update(*values, v, {0, 0, prev, 0}, {B, n_kv, offset, vd});
        return {mx::slice(*keys, {0, 0, 0, 0}, {B, n_kv, offset, kd}),
                mx::slice(*values, {0, 0, 0, 0}, {B, n_kv, offset, vd})};
    }

    // Rollback, for speculative decoding. `offset` is the ONLY thing that says how much of
    // the over-allocated buffer is real -- update_and_fetch slices to it, and forward_self_attn
    // reads it as rope's position base -- so discarding a tail is an integer assignment, not
    // a reallocation. The stale keys and values past `n` are never read: the next append
    // overwrites them in place before any slice can reach them.
    //
    // Clamped rather than asserted. `n > offset` is a caller bug either way, but growing the
    // cache by moving an index would hand attention rows of uninitialised scratch and produce
    // a plausible wrong answer; refusing to grow produces a slow one instead.
    void truncate_to(int n) noexcept { offset = std::clamp(n, 0, offset); }

    void sync() const {
        if (keys) {
            mx::eval({*keys});
        }
        if (values) {
            mx::eval({*values});
        }
    }
};

struct SsmCache {
    std::optional<mx::array> conv_state;
    std::optional<mx::array> delta_state;
    int offset{0};

    void clear() noexcept {
        conv_state.reset();
        delta_state.reset();
        offset = 0;
    }

    // The linear layers' half of rollback, and the reason it is snapshot/restore rather
    // than KVCache's index assignment: conv_state and delta_state are a running recurrence
    // over the whole sequence, so there is no prefix of them left to keep. What makes that
    // affordable is that neither has a sequence axis -- conv_state is
    // [B, kernel_dim - 1, conv_dim] and delta_state is [B, Hv, Dv, Dk] -- so the cost is
    // independent of both draft length and context length.
    //
    // Cheaper, in fact, than the "two small tensor copies" this was estimated at.
    // mx::array is an immutable refcounted handle and forward_gated_delta REPLACES
    // cache.conv_state / cache.delta_state rather than writing through them, so capturing
    // a snapshot is two refcount bumps; the buffers are kept alive, never duplicated.
    struct Snapshot {
        std::optional<mx::array> conv_state;
        std::optional<mx::array> delta_state;
        int offset{0};
    };

    [[nodiscard]] Snapshot snapshot() const {
        // Forced to buffers on capture. An unevaluated handle would pin every input of the
        // pass that produced it for as long as the checkpoint lives, which for a checkpoint
        // held across a speculative block is the block's whole activation set. A no-op when
        // the caller has already synced, which the decode loop has.
        sync();
        return Snapshot{conv_state, delta_state, offset};
    }

    void restore(const Snapshot& s) {
        conv_state = s.conv_state;
        delta_state = s.delta_state;
        offset = s.offset;
    }

    void sync() const {
        if (conv_state) {
            mx::eval({*conv_state});
        }
        if (delta_state) {
            mx::eval({*delta_state});
        }
    }
};

// A quantized triple: the packed words plus the affine scale/bias per group. Kept as one
// value because every consumer wants all three and a stray mismatched pair silently
// decodes to noise rather than failing.
struct QTensor {
    mx::array w;
    mx::array scales;
    mx::array biases;
};

// The KV cache, held affine-quantized instead of bf16.
//
// THIS IS A MEMORY LEVER, NOT A SPEED ONE. MEASURED AND REJECTED FOR THROUGHPUT
// 2026-08-19: at 32k of context, 8-bit KV decodes at 43.1 tok/s against bf16's 65.1 --
// a 34% LOSS, six paired runs, every pair the same direction. Default is OFF
// (LMP_KV_BITS unset or 0) and it should stay off unless the goal is context length.
//
// The reasoning that predicted a win, and exactly where it broke: at 48k this model reads
// 0.98 GB of KV per token against 1.82 GB of active weights, so halving the KV term
// should move the ROOFLINE by ~19%. It does. What the arithmetic left out is that using a
// quantized cache means giving up `fast::scaled_dot_product_attention` -- a fused kernel
// that never materialises the score row -- for two `quantized_matmul` calls with an
// explicit [1, S] score matrix and a separate softmax pass between them. That is three
// launches and two passes over the cache where the fused path makes one, and it costs far
// more than the bandwidth it saves. Bytes are not the only thing on the critical path,
// and a roofline says nothing about which kernel you had to give up to reach it.
//
// What it IS good for: 8-bit affine over group 64 stores 272 B per head per token where
// bf16 stores 512 (256 packed + 4 scales + 4 biases, at bf16) -- 10.6 KB/token against
// 20 KB across the 10 full-attention layers. That is close to double the context for the
// same KV budget, which is the binding constraint on how long a run can get. Pay the 34%
// only to buy length that is otherwise unreachable.
//
// WHY NOT ALWAYS: this trades a fused SDPA kernel for two quantized_matmuls and an
// explicit score matrix. At L=1 that is a clear win because the score row is [1, S] and
// the cache read dominates. During PREFILL, L is a 2048-token chunk and the explicit
// [L, S] scores are both slower and far larger than the fused kernel's tiles. So the
// live path keeps bf16 + fused SDPA through prefill and converts once, on the way into
// decode -- see adopt_from(). That is also why this stores the same [B, H, S, D] layout
// bf16 does: conversion is a quantize of the live prefix, not a re-layout.
//
// The sequence axis stays axis 2 and quantization runs along the LAST axis (head_dim), so
// a token is still a row and every operation KVCache does by row -- preallocate in
// kStep blocks, slice_update in place, truncate by moving `offset` -- works unchanged.
struct QuantizedKVCache {
    static constexpr int kStep = 256;

    int group_size{64};
    int bits{8};
    std::optional<QTensor> keys;
    std::optional<QTensor> values;
    int offset{0};

    void clear() noexcept {
        keys.reset();
        values.reset();
        offset = 0;
    }

    [[nodiscard]] bool active() const noexcept { return keys.has_value(); }

    // How many words one row of `dim` values packs into.
    [[nodiscard]] int packed_dim(int dim) const noexcept {
        const int el_per_word = 32 / bits;
        return dim / el_per_word;
    }
    [[nodiscard]] int group_dim(int dim) const noexcept { return dim / group_size; }

    static QTensor quantize_rows(const mx::array& x, int gs, int b) {
        const std::vector<mx::array> q = mx::quantize(x, gs, b, "affine");
        // Affine always yields biases; asserting it here rather than carrying an optional
        // through every call site, because a missing bias would decode as a silent offset.
        return QTensor{q.at(0), q.at(1), q.at(2)};
    }

    // Take over a bf16 cache's live prefix. Called once, when decode begins on a context
    // long enough to be worth it; `src` can be dropped by the caller afterwards.
    void adopt_from(const KVCache& src, int gs, int b) {
        group_size = gs;
        bits = b;
        offset = src.offset;
        if (!src.keys.has_value() || !src.values.has_value() || offset <= 0) {
            clear();
            return;
        }
        const int B = static_cast<int>(src.keys->shape()[0]);
        const int H = static_cast<int>(src.keys->shape()[1]);
        const int kd = static_cast<int>(src.keys->shape()[3]);
        const int vd = static_cast<int>(src.values->shape()[3]);
        // Only the live prefix. The padding past `offset` is scratch and quantizing it
        // would both cost time and put uninitialised values behind a real scale.
        keys = quantize_rows(mx::slice(*src.keys, {0, 0, 0, 0}, {B, H, offset, kd}), gs, b);
        values = quantize_rows(mx::slice(*src.values, {0, 0, 0, 0}, {B, H, offset, vd}), gs, b);
        sync();
    }

    // Append `k`,`v` (bf16, [B, H, n_new, D]) and return the live prefix of the cache.
    std::pair<QTensor, QTensor> update_and_fetch(const mx::array& k, const mx::array& v) {
        const int B = static_cast<int>(k.shape()[0]);
        const int H = static_cast<int>(k.shape()[1]);
        const int n_new = static_cast<int>(k.shape()[2]);
        const int kd = static_cast<int>(k.shape()[3]);
        const int vd = static_cast<int>(v.shape()[3]);
        const int prev = offset;

        const QTensor qk = quantize_rows(k, group_size, bits);
        const QTensor qv = quantize_rows(v, group_size, bits);

        if (!keys.has_value() || prev + n_new > static_cast<int>(keys->w.shape()[2])) {
            grow(B, H, kd, vd, prev, n_new, qk, qv);
        }
        offset += n_new;
        splice(*keys, qk, prev, offset);
        splice(*values, qv, prev, offset);
        return {live(*keys, offset), live(*values, offset)};
    }

    // Same contract as KVCache::truncate_to -- `offset` is the only thing that says how
    // much of the buffer is real, so a rollback is an integer assignment.
    void truncate_to(int n) noexcept { offset = std::clamp(n, 0, offset); }

    void sync() const {
        for (const std::optional<QTensor>* t : {&keys, &values}) {
            if (t->has_value()) {
                mx::eval({(*t)->w, (*t)->scales, (*t)->biases});
            }
        }
    }

  private:
    static QTensor live(const QTensor& t, int n) {
        auto cut = [n](const mx::array& a) {
            mx::Shape lo(a.ndim(), 0);
            mx::Shape hi(a.shape().begin(), a.shape().end());
            hi[2] = n;
            return mx::slice(a, lo, hi);
        };
        return QTensor{cut(t.w), cut(t.scales), cut(t.biases)};
    }

    static void splice(QTensor& dst, const QTensor& src, int from, int to) {
        auto put = [from, to](const mx::array& d, const mx::array& s) {
            mx::Shape lo(d.ndim(), 0);
            lo[2] = from;
            mx::Shape hi(d.shape().begin(), d.shape().end());
            hi[2] = to;
            return mx::slice_update(d, s, lo, hi);
        };
        dst.w = put(dst.w, src.w);
        dst.scales = put(dst.scales, src.scales);
        dst.biases = put(dst.biases, src.biases);
    }

    void grow(int B, int H, int kd, int vd, int prev, int n_new, const QTensor& qk,
              const QTensor& qv) {
        const int n_steps = (n_new + kStep - 1) / kStep;
        auto blank = [&](int last, const mx::array& like) {
            return mx::zeros({B, H, n_steps * kStep, last}, like.dtype());
        };
        auto extend = [&](std::optional<QTensor>& slot, const QTensor& shape_of, int dim) {
            QTensor add{blank(packed_dim(dim), shape_of.w), blank(group_dim(dim), shape_of.scales),
                        blank(group_dim(dim), shape_of.biases)};
            if (!slot.has_value()) {
                slot = add;
                return;
            }
            auto trim_cat = [prev](const mx::array& old, const mx::array& more) {
                mx::Shape lo(old.ndim(), 0);
                mx::Shape hi(old.shape().begin(), old.shape().end());
                hi[2] = prev;
                const mx::array kept =
                    (prev % kStep != 0) ? mx::slice(old, lo, hi) : old;
                return mx::concatenate({kept, more}, 2);
            };
            slot->w = trim_cat(slot->w, add.w);
            slot->scales = trim_cat(slot->scales, add.scales);
            slot->biases = trim_cat(slot->biases, add.biases);
        };
        extend(keys, qk, kd);
        extend(values, qv, vd);
    }
};

// Unused by the live model path. Layers use KVCache + truncate_to; paged / rotating
// multi-session KV remains out of scope (product is one model, one session). Kept as a
// reference shape rather than deleted so a future opt-in can wire it without reinventing
// the slice arithmetic. Do not construct this from mlx_backend.
struct RotatingKVCache {
    int max_size{512};
    std::optional<mx::array> keys;
    std::optional<mx::array> values;
    int offset{0};

    explicit RotatingKVCache(int max_sz = 512) : max_size(max_sz) {}

    void clear() noexcept {
        keys.reset();
        values.reset();
        offset = 0;
    }

    std::pair<mx::array, mx::array> update_and_fetch(const mx::array& k, const mx::array& v) {
        if (!keys.has_value()) {
            keys = k;
            values = v;
        } else {
            keys = mx::concatenate({*keys, k}, 2);
            values = mx::concatenate({*values, v}, 2);
        }
        offset += static_cast<int>(k.shape()[2]);
        if (offset > max_size) {
            const int trim = offset - max_size;
            const int B = static_cast<int>(keys->shape()[0]);
            const int n_kv = static_cast<int>(keys->shape()[1]);
            const int kd = static_cast<int>(keys->shape()[3]);
            const int vd = static_cast<int>(values->shape()[3]);
            keys = mx::slice(*keys, {0, 0, trim, 0}, {B, n_kv, offset, kd});
            values = mx::slice(*values, {0, 0, trim, 0}, {B, n_kv, offset, vd});
            offset = max_size;
        }
        return {*keys, *values};
    }

    void sync() const {
        if (keys) {
            mx::eval({*keys});
        }
        if (values) {
            mx::eval({*values});
        }
    }
};

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MLX_KV_CACHE_HPP
