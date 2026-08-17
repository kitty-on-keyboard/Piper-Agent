#ifndef LLM_MLX_WEIGHT_STORE_HPP
#define LLM_MLX_WEIGHT_STORE_HPP

#if LMP_HAVE_MLX

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <simdjson.h>

#include "mlx/array.h"
#include "mlx/io.h"
#include "mlx/ops.h"
// mx::eval lives here. This header calls it and used to rely on some other include in
// the translation unit having pulled it in -- which held only because every existing
// includer reached weight_store.hpp through qwen35_moe_model.hpp. Included first, it
// failed to compile, and not with a missing-declaration error: `mx::eval(buf)` parses as
// a DECLARATION of `buf` with type `mx::eval`, so the diagnostic was "no type named
// 'eval'" plus a shadowing warning, several lines from the cause.
#include "mlx/transforms.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

struct QuantWeight {
    mx::array weight;
    mx::array scales;
    std::optional<mx::array> biases;
    int group_size{64};
    int bits{4};
    std::string mode{"affine"};
};

struct QuantSpec {
    int group_size{64};
    int bits{4};
    std::string mode{"affine"};
};

class WeightStore {
public:
    bool load_directory(const std::string& model_dir) {
        namespace fs = std::filesystem;
        weights_.clear();
        quant_specs_.clear();
        default_quant_ = QuantSpec{};
        load_quantization_config(model_dir);
        for (const auto& entry : fs::directory_iterator(model_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().extension() != ".safetensors") {
                continue;
            }
            auto loaded = mx::load_safetensors(entry.path().string());
            for (auto& [key, tensor] : loaded.first) {
                weights_.emplace(std::move(key), std::move(tensor));
            }
        }
        if (weights_.empty()) {
            return false;
        }
        std::vector<mx::array> eval_buf;
        eval_buf.reserve(weights_.size());
        for (auto& [_, v] : weights_) {
            eval_buf.push_back(v);
        }
        mx::eval(eval_buf);
        return true;
    }

    // Load an ADDITIONAL checkpoint into this store under `prefix`, keeping everything
    // already here. The MTP head ships as its own directory but has to share the target's
    // store: it carries no embedding table and no lm_head, and its one decoder layer is
    // addressed by the same forward_self_attn the target uses, which resolves weights by
    // key. Giving it a separate store would mean duplicating that path.
    //
    // Quantization falls through to this store's defaults (4-bit, group 64, affine),
    // which is what the MTP checkpoint's own config declares; lookup_quant still infers
    // per-tensor bits from shape where a tensor disagrees.
    bool load_directory_merged(const std::string& model_dir, const std::string& prefix) {
        namespace fs = std::filesystem;
        std::vector<mx::array> eval_buf;
        for (const auto& entry : fs::directory_iterator(model_dir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".safetensors") {
                continue;
            }
            auto loaded = mx::load_safetensors(entry.path().string());
            for (auto& [key, tensor] : loaded.first) {
                eval_buf.push_back(tensor);
                weights_.emplace(prefix + key, std::move(tensor));
            }
        }
        if (eval_buf.empty()) {
            return false;
        }
        mx::eval(eval_buf);
        return true;
    }

    [[nodiscard]] bool has(const std::string& key) const {
        return weights_.find(key) != weights_.end();
    }

    [[nodiscard]] const mx::array& get(const std::string& key) const {
        const auto it = weights_.find(key);
        if (it == weights_.end()) {
            throw std::runtime_error("Missing weight: " + key);
        }
        return it->second;
    }

    [[nodiscard]] const mx::array* try_get(const std::string& key) const {
        const auto it = weights_.find(key);
        if (it == weights_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    void set(std::string key, mx::array value) { weights_.insert_or_assign(std::move(key), std::move(value)); }

    void erase(const std::string& key) { weights_.erase(key); }

    [[nodiscard]] std::unordered_map<std::string, mx::array>& mutable_weights() { return weights_; }

    [[nodiscard]] const QuantSpec& default_quant() const noexcept { return default_quant_; }

    [[nodiscard]] std::string quant_label() const {
        return std::to_string(default_quant_.bits) + "bit_" + default_quant_.mode + "_g" +
               std::to_string(default_quant_.group_size);
    }

    [[nodiscard]] bool is_quantized(const std::string& base_key) const {
        return has(base_key + ".weight") && has(base_key + ".scales");
    }

    [[nodiscard]] QuantWeight quant(const std::string& base_key) const {
        const QuantSpec spec = lookup_quant(base_key);
        const mx::array& weight = get(base_key + ".weight");
        const mx::array& scales = get(base_key + ".scales");
        QuantWeight qw{
            .weight = weight,
            .scales = scales,
            .group_size = spec.group_size,
            .bits = spec.bits,
            .mode = spec.mode,
        };
        if (has(base_key + ".biases")) {
            qw.biases = get(base_key + ".biases");
        }
        return qw;
    }

    [[nodiscard]] mx::array linear(const mx::array& x, const std::string& base_key) const {
        if (is_quantized(base_key)) {
            const QuantWeight qw = quant(base_key);
            return mx::quantized_matmul(
                x, qw.weight, qw.scales, qw.biases, /*transpose=*/true,
                qw.group_size, qw.bits, qw.mode);
        }
        const mx::array& w = get(base_key + ".weight");
        return mx::matmul(x, mx::transpose(w, {1, 0}));
    }

    // Token embedding lookup — quantized checkpoints store affine-quantized rows
    // and must be gathered then dequantized (not matmul'd against token ids).
    [[nodiscard]] mx::array embed_lookup(const mx::array& ids, const std::string& base_key) const {
        if (is_quantized(base_key)) {
            const QuantWeight qw = quant(base_key);
            const mx::array w_rows = mx::take(qw.weight, ids, 0);
            const mx::array s_rows = mx::take(qw.scales, ids, 0);
            std::optional<mx::array> b_rows;
            if (qw.biases) {
                b_rows = mx::take(*qw.biases, ids, 0);
            }
            return mx::dequantize(w_rows, s_rows, b_rows, qw.group_size, qw.bits, qw.mode);
        }
        return mx::take(get(base_key + ".weight"), ids, 0);
    }

    // Tied lm_head projection from hidden states onto the embedding table.
    [[nodiscard]] mx::array tied_logits(const mx::array& h, const std::string& base_key) const {
        if (is_quantized(base_key)) {
            const QuantWeight qw = quant(base_key);
            return mx::quantized_matmul(
                h, qw.weight, qw.scales, qw.biases, /*transpose=*/true,
                qw.group_size, qw.bits, qw.mode);
        }
        const mx::array& w = get(base_key + ".weight");
        return mx::matmul(h, mx::transpose(w, {1, 0}));
    }

private:
    void load_quantization_config(const std::string& model_dir) {
        const std::string path = model_dir + "/config.json";
        std::ifstream in(path);
        if (!in) {
            return;
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        if (parser.parse(buf.str()).get(root)) {
            return;
        }
        simdjson::dom::element quant;
        if (root["quantization"].get(quant)) {
            if (root["quantization_config"].get(quant)) {
                return;
            }
        }
        int64_t default_bits = default_quant_.bits;
        int64_t default_group = default_quant_.group_size;
        (void)quant["bits"].get_int64().get(default_bits);
        (void)quant["group_size"].get_int64().get(default_group);
        default_quant_.bits = static_cast<int>(default_bits);
        default_quant_.group_size = static_cast<int>(default_group);
        std::string_view mode;
        if (!quant["mode"].get_string().get(mode)) {
            default_quant_.mode = std::string(mode);
        }
        simdjson::dom::object quant_obj;
        if (quant.get_object().get(quant_obj)) {
            return;
        }
        for (simdjson::dom::key_value_pair field : quant_obj) {
            const std::string key(field.key);
            if (key == "bits" || key == "group_size" || key == "mode") {
                continue;
            }
            simdjson::dom::object spec_obj;
            if (field.value.get_object().get(spec_obj)) {
                continue;
            }
            QuantSpec spec = default_quant_;
            int64_t bits = spec.bits;
            int64_t group = spec.group_size;
            (void)spec_obj["bits"].get_int64().get(bits);
            (void)spec_obj["group_size"].get_int64().get(group);
            spec.bits = static_cast<int>(bits);
            spec.group_size = static_cast<int>(group);
            quant_specs_.emplace(key, spec);
        }
    }

    [[nodiscard]] QuantSpec lookup_quant(const std::string& base_key) const {
        const auto it = quant_specs_.find(base_key);
        if (it != quant_specs_.end()) {
            return it->second;
        }
        const mx::array& weight = get(base_key + ".weight");
        const mx::array& scales = get(base_key + ".scales");
        const int packed_cols = static_cast<int>(weight.shape().back());
        const int scale_cols = static_cast<int>(scales.shape().back());
        QuantSpec spec = default_quant_;
        spec.bits = std::max(1, (packed_cols * 32) / (scale_cols * spec.group_size));
        return spec;
    }

    std::unordered_map<std::string, mx::array> weights_;
    std::unordered_map<std::string, QuantSpec> quant_specs_;
    QuantSpec default_quant_{};
};

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MLX_WEIGHT_STORE_HPP
