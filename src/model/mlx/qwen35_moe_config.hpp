#ifndef LLM_MODELS_QWEN35_MOE_CONFIG_HPP
#define LLM_MODELS_QWEN35_MOE_CONFIG_HPP

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <simdjson.h>

namespace lmp::model::mlxl {

// Which FFN the checkpoint carries. The Qwen3.5/3.6 generation ships one hybrid
// linear-attention backbone with two FFN shapes: routed experts (35B-A3B) and a plain
// gated MLP (27B). Everything else -- the layer schedule, output-gated attention,
// QK-norm, the gated-delta block, the weight prefix -- is common to both, so this is the
// only axis the model graph has to branch on.
enum class FfnKind { Unknown, Moe, Dense };

// An allowlist, not a substring test: an unrecognised checkpoint must be REFUSED at load
// rather than silently run through whichever graph happens to be the default. Both the
// nested `text_config.model_type` spellings and the root ones are accepted, because the
// loader prefers the nested value and the two disagree (`qwen3_5` at the root vs
// `qwen3_5_text` nested, and likewise for the MoE).
[[nodiscard]] inline FfnKind ffn_kind_for(std::string_view model_type) noexcept {
    if (model_type == "qwen3_5_moe" || model_type == "qwen3_5_moe_text") {
        return FfnKind::Moe;
    }
    if (model_type == "qwen3_5" || model_type == "qwen3_5_text") {
        return FfnKind::Dense;
    }
    return FfnKind::Unknown;
}

struct Qwen35MoeConfig {
    std::string model_type{"qwen3_5_moe_text"};
    int hidden_size{2048};
    int num_hidden_layers{40};
    int num_attention_heads{16};
    int num_key_value_heads{2};
    int head_dim{256};
    int vocab_size{248320};
    // From text_config.max_position_embeddings (or the root). The hard sequence ceiling
    // the checkpoint was trained/exported with -- prompt + reserved generation must not
    // exceed it. 0 means the field was absent; callers treat that as "unknown".
    int max_position_embeddings{0};
    int intermediate_size{0};
    int moe_intermediate_size{512};
    int shared_expert_intermediate_size{512};
    int num_experts{256};
    int num_experts_per_tok{8};
    int full_attention_interval{4};
    int linear_num_key_heads{16};
    int linear_num_value_heads{32};
    int linear_key_head_dim{128};
    int linear_value_head_dim{128};
    int linear_conv_kernel_dim{4};
    float rms_norm_eps{1e-6f};
    float rope_theta{10000000.0f};
    float partial_rotary_factor{0.25f};
    bool tie_word_embeddings{false};
    bool norm_topk_prob{true};

    [[nodiscard]] bool is_linear_layer(int layer_idx) const noexcept {
        return (layer_idx + 1) % full_attention_interval != 0;
    }

    [[nodiscard]] FfnKind ffn_kind() const noexcept { return ffn_kind_for(model_type); }
};

inline bool load_qwen35_moe_config(const std::string& model_dir, Qwen35MoeConfig& cfg) {
    const std::string path = model_dir + "/config.json";
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    const std::string json = buf.str();

    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(json).get(root)) {
        return false;
    }

    simdjson::dom::element text_cfg = root;
    simdjson::dom::element nested;
    if (!root["text_config"].get(nested)) {
        text_cfg = nested;
    }

    auto get_i = [&](const char* key, int& out) -> bool {
        int64_t v = 0;
        if (text_cfg[key].get_int64().get(v)) {
            return false;
        }
        out = static_cast<int>(v);
        return true;
    };

    auto get_f = [&](const char* key, float& out) -> bool {
        double v = 0;
        if (text_cfg[key].get_double().get(v)) {
            return false;
        }
        out = static_cast<float>(v);
        return true;
    };

    auto get_b = [&](const char* key, bool& out) -> bool {
        bool v = false;
        if (text_cfg[key].get_bool().get(v)) {
            return false;
        }
        out = v;
        return true;
    };

    std::string_view mt;
    if (!text_cfg["model_type"].get_string().get(mt)) {
        cfg.model_type = std::string(mt);
    }

    if (!get_i("hidden_size", cfg.hidden_size) ||
        !get_i("num_hidden_layers", cfg.num_hidden_layers) ||
        !get_i("vocab_size", cfg.vocab_size)) {
        return false;
    }

    (void)get_i("num_attention_heads", cfg.num_attention_heads);
    (void)get_i("num_key_value_heads", cfg.num_key_value_heads);
    (void)get_i("head_dim", cfg.head_dim);
    // Prefer the nested text_config value; some exports also put it on the root.
    if (!get_i("max_position_embeddings", cfg.max_position_embeddings)) {
        int64_t root_max = 0;
        if (!root["max_position_embeddings"].get_int64().get(root_max) && root_max > 0) {
            cfg.max_position_embeddings = static_cast<int>(root_max);
        }
    }
    (void)get_i("intermediate_size", cfg.intermediate_size);
    (void)get_i("moe_intermediate_size", cfg.moe_intermediate_size);
    (void)get_i("shared_expert_intermediate_size", cfg.shared_expert_intermediate_size);
    (void)get_i("num_experts", cfg.num_experts);
    (void)get_i("num_experts_per_tok", cfg.num_experts_per_tok);
    (void)get_i("full_attention_interval", cfg.full_attention_interval);
    (void)get_i("linear_num_key_heads", cfg.linear_num_key_heads);
    (void)get_i("linear_num_value_heads", cfg.linear_num_value_heads);
    (void)get_i("linear_key_head_dim", cfg.linear_key_head_dim);
    (void)get_i("linear_value_head_dim", cfg.linear_value_head_dim);
    (void)get_i("linear_conv_kernel_dim", cfg.linear_conv_kernel_dim);
    (void)get_f("rms_norm_eps", cfg.rms_norm_eps);
    (void)get_b("tie_word_embeddings", cfg.tie_word_embeddings);
    (void)get_b("norm_topk_prob", cfg.norm_topk_prob);

    // Both keys can sit at this level (stock HF exports, transformers < 4.5x) or inside a
    // rope_parameters sub-object (newer exports). Read this level first so a root-level
    // rope_theta is not lost, then let the sub-object override where it exists. Reading
    // only the sub-object silently left the base at the hardcoded default, which is
    // invisible while a checkpoint happens to ship that same value and turns into
    // fluent-but-wrong output that degrades with sequence length once one does not.
    (void)get_f("rope_theta", cfg.rope_theta);
    (void)get_f("partial_rotary_factor", cfg.partial_rotary_factor);

    simdjson::dom::element rope_params;
    if (!text_cfg["rope_parameters"].get(rope_params)) {
        double partial = static_cast<double>(cfg.partial_rotary_factor);
        if (!rope_params["partial_rotary_factor"].get_double().get(partial)) {
            cfg.partial_rotary_factor = static_cast<float>(partial);
        }
        double theta = static_cast<double>(cfg.rope_theta);
        if (!rope_params["rope_theta"].get_double().get(theta)) {
            cfg.rope_theta = static_cast<float>(theta);
        }
    }

    return cfg.hidden_size > 0 && cfg.num_hidden_layers > 0;
}

// The vision tower, which BOTH checkpoints on this machine already carry and which the
// loader used to delete on the way in (sanitize_weights dropped every `vision_tower.*`
// key). 333 tensors, 0.92 GB unquantized, against 15.13 GB of quantized text -- so the
// image path costs about 6% more resident memory and no accuracy.
//
// Read from the ROOT of config.json, not from text_config: this is the multimodal
// wrapper's own block, and the ids below sit beside it rather than inside it.
struct Qwen35VisionConfig {
    // `present` is the load-time question ("does this checkpoint have eyes?"), and it is
    // answered from the config rather than by probing for a tensor -- a checkpoint whose
    // vision block is declared but whose weights are absent must fail loudly at load, not
    // silently run text-only.
    bool present{false};
    int depth{27};
    int hidden_size{1152};
    int num_heads{16};
    int intermediate_size{4304};
    int in_channels{3};
    int patch_size{16};
    int spatial_merge_size{2};
    int temporal_patch_size{2};
    int num_position_embeddings{2304};
    // The LLM's hidden size, which the merger projects onto. Checked against the text
    // config at load: a mismatch means the two halves of the checkpoint disagree, and
    // splicing a wrongly-sized row into the residual stream is not something a shape
    // check downstream would catch cleanly.
    int out_hidden_size{0};

    // Qwen3-VL's deepstack: visual features injected at several LLM layers rather than
    // only at the embedding. EMPTY on both checkpoints here, which is why this build
    // splices at the embedding alone. A non-empty list must refuse at load rather than
    // quietly ignore the extra injection sites -- the model would run and be subtly
    // wrong, which is the worst available outcome.
    std::vector<int> deepstack_visual_indexes;

    // Single ids in the tokenizer (measured: 248053/248054/248056/248057).
    int image_token_id{-1};
    int video_token_id{-1};
    int vision_start_token_id{-1};
    int vision_end_token_id{-1};

    // Patches per merged token: the merger concatenates a spatial_merge_size square.
    [[nodiscard]] int merge_unit() const noexcept {
        return spatial_merge_size * spatial_merge_size;
    }
    [[nodiscard]] int head_dim() const noexcept {
        return num_heads > 0 ? hidden_size / num_heads : 0;
    }
};

inline bool load_qwen35_vision_config(const std::string& model_dir,
                                      Qwen35VisionConfig& cfg) {
    const std::string path = model_dir + "/config.json";
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(buf.str()).get(root)) {
        return false;
    }
    simdjson::dom::element vc;
    if (root["vision_config"].get(vc)) {
        return false; // text-only checkpoint; not an error
    }
    cfg.present = true;

    const auto get_i = [](simdjson::dom::element scope, const char* key, int& out) {
        int64_t v = 0;
        if (!scope[key].get_int64().get(v)) {
            out = static_cast<int>(v);
        }
    };
    get_i(vc, "depth", cfg.depth);
    get_i(vc, "hidden_size", cfg.hidden_size);
    get_i(vc, "num_heads", cfg.num_heads);
    get_i(vc, "intermediate_size", cfg.intermediate_size);
    get_i(vc, "in_channels", cfg.in_channels);
    get_i(vc, "patch_size", cfg.patch_size);
    get_i(vc, "spatial_merge_size", cfg.spatial_merge_size);
    get_i(vc, "temporal_patch_size", cfg.temporal_patch_size);
    get_i(vc, "num_position_embeddings", cfg.num_position_embeddings);
    get_i(vc, "out_hidden_size", cfg.out_hidden_size);

    simdjson::dom::array deep;
    if (!vc["deepstack_visual_indexes"].get_array().get(deep)) {
        for (simdjson::dom::element e : deep) {
            int64_t v = 0;
            if (!e.get_int64().get(v)) {
                cfg.deepstack_visual_indexes.push_back(static_cast<int>(v));
            }
        }
    }

    get_i(root, "image_token_id", cfg.image_token_id);
    get_i(root, "video_token_id", cfg.video_token_id);
    get_i(root, "vision_start_token_id", cfg.vision_start_token_id);
    get_i(root, "vision_end_token_id", cfg.vision_end_token_id);
    return true;
}

// The MTP head's block size, or 0 when the directory is not a usable MTP checkpoint.
//
// A round drafts block_size - 1 tokens, which is why the Qwen3.6-27B MTP card advertises
// "block size 2" while its own config.json says 3. Both describe something true; the
// config is what code must read, since it is the number the reference loop counts against.
//
// model_type is checked, not assumed: pointing this at the TARGET directory would
// otherwise load 16 GB of the wrong tensors under an mtp. prefix and fail much later.
inline int load_mtp_block_size(const std::string& model_dir) {
    const std::string path = model_dir + "/config.json";
    std::ifstream in(path);
    if (!in) {
        return 0;
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    const std::string json = buf.str();

    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(json).get(root)) {
        return 0;
    }
    std::string_view mt;
    if (root["model_type"].get_string().get(mt) || mt != "qwen3_5_mtp") {
        return 0;
    }
    int64_t block = 0;
    if (root["block_size"].get_int64().get(block) || block < 2) {
        return 0;
    }
    return static_cast<int>(block);
}

// The checkpoint's sequence ceiling, or 0 when config.json is missing / unparseable /
// omits the field. Safe to call without loading weights. Prefer
// model::load_max_position_embeddings (model_limits.hpp) from call sites outside this
// translation unit so consumers do not need simdjson on their include path.
inline int load_max_position_embeddings(const std::string& model_dir) {
    Qwen35MoeConfig cfg;
    if (!load_qwen35_moe_config(model_dir, cfg)) {
        return 0;
    }
    return cfg.max_position_embeddings > 0 ? cfg.max_position_embeddings : 0;
}

// Bytes of KV cache this checkpoint needs for ONE token of context. 0 when unreadable.
//
// Only the FULL-ATTENTION layers hold a KV cache that grows with the sequence; the linear
// (gated-delta) layers carry a fixed-size SSM state that does not. Counting all 64 layers
// of a hybrid checkpoint overstates this by 4x, which would be the difference between a
// budget that fits and a budget that is refused for no reason.
//
// K and V, per full-attention layer, per kv head, head_dim wide, at 2 bytes an element:
// the cache is bf16 even when the WEIGHTS are 4-bit, which is the trap in eyeballing this
// from the quantisation in the directory name.
//
// VERIFIED against a real run rather than asserted. Qwen3.8-27B-MLX-4bit: 64 layers,
// full_attention_interval 4 => 16 full-attention layers, 4 kv heads, head_dim 256:
//   2 * 16 * 4 * 256 * 2 = 65,536 B = 64 KB/token
// and the run's own `memory` events give 72 KB/token against a constant 16.29 GB of
// weights (94,527 tok -> 6.81 GB KV; 112,088 tok -> 8.21 GB KV). The 8 KB difference is
// the MTP draft head's own cache, which is why the caller sums this over both checkpoints.
[[nodiscard]] inline std::size_t kv_bytes_per_token(const std::string& model_dir) {
    Qwen35MoeConfig cfg;
    if (!load_qwen35_moe_config(model_dir, cfg)) {
        return 0;
    }
    if (cfg.num_hidden_layers <= 0 || cfg.num_key_value_heads <= 0 || cfg.head_dim <= 0) {
        return 0;
    }
    int full = 0;
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        if (!cfg.is_linear_layer(i)) {
            ++full;
        }
    }
    // A checkpoint with no linear layers at all is dense: every layer holds KV.
    if (full == 0) {
        full = cfg.num_hidden_layers;
    }
    constexpr std::size_t kKvElementBytes = 2; // bf16
    return std::size_t{2} * static_cast<std::size_t>(full) *
           static_cast<std::size_t>(cfg.num_key_value_heads) *
           static_cast<std::size_t>(cfg.head_dim) * kKvElementBytes;
}

} // namespace lmp::model::mlxl

#endif // LLM_MODELS_QWEN35_MOE_CONFIG_HPP
