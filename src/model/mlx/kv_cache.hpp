#ifndef LLM_MLX_KV_CACHE_HPP
#define LLM_MLX_KV_CACHE_HPP

#if LMP_HAVE_MLX

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

    void sync() const {
        if (conv_state) {
            mx::eval({*conv_state});
        }
        if (delta_state) {
            mx::eval({*delta_state});
        }
    }
};

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
