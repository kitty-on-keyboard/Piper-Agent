// Adapted from v1 src/llm/models/qwen35_moe_model.hpp -- debugged numerics for the
// exact checkpoint this machine runs (Qwen3.6-35B-A3B-MLX-4bit). The forward pass is
// model math, which passes S2.2's asset test; only the seam around it was rebuilt.
#ifndef LLM_MODELS_QWEN35_MOE_MODEL_HPP
#define LLM_MODELS_QWEN35_MOE_MODEL_HPP

#if LMP_HAVE_MLX

#include "qwen35_moe_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "activations.hpp"
#include "gated_delta.hpp"
#include "gated_residual.hpp"
#include "kv_cache.hpp"
#include "quant_attention.hpp"
#include "moe_trace.hpp"
#include "ple.hpp"
#include "qsa.hpp"
#include "switch_glu.hpp"
#include "vision_tower.hpp"
#include "weight_store.hpp"
#include "src/model/qwen4_exp_ops.hpp"

#include "mlx/fast.h"
#include "mlx/memory.h"
#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;
using lmp::model::mlxl::KVCache;
using lmp::model::mlxl::SsmCache;
using lmp::model::mlxl::WeightStore;

class Qwen35MoeModel {
public:
    // `with_vision` keeps the checkpoint's vision tower resident and initialises it.
    // Off by default: most runs never send an image, and the tower is 0.92 GB.
    bool load(const std::string& model_dir, bool with_vision = false) {
        prefix_ = "language_model.model.";
        load_error_.clear();
        keep_vision_ = with_vision;
        if (!load_qwen35_moe_config(model_dir, cfg_)) {
            return false;
        }
        // Refuse an unrecognised architecture BEFORE reading ~16 GB of weights. The
        // alternative is what this path used to do: build the MoE graph regardless, then
        // fail on a missing expert tensor at the first token with the whole checkpoint
        // already resident and wired.
        ffn_ = cfg_.ffn_kind();
        if (ffn_ == FfnKind::Unknown) {
            load_error_ = "unsupported architecture: model_type=\"" + cfg_.model_type +
                          "\" (this build knows qwen3_5 / qwen3_5_text dense, "
                          "qwen3_5_moe / qwen3_5_moe_text, and qwen4_exp / qwen4_exp_text)";
            return false;
        }
        if (cfg_.is_flash()) {
            if (cfg_.hc_count <= 0 || cfg_.hc_lowrank <= 0) {
                load_error_ = "qwen4_exp requires hc_count and hc_lowrank";
                return false;
            }
            if (cfg_.linear_key_head_dim % 32 != 0) {
                load_error_ = "qwen4_exp: linear_key_head_dim must be divisible by 32";
                return false;
            }
            if (cfg_.indexer_compress_ratio > 0 && cfg_.indexer_budget > 0 &&
                cfg_.indexer_budget % cfg_.indexer_compress_ratio != 0) {
                load_error_ = "qwen4_exp: indexer_budget must be divisible by "
                              "indexer_compress_ratio";
                return false;
            }
        }
        if (!weights_.load_directory(model_dir)) {
            return false;
        }
        if (cfg_.is_flash()) {
            constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
            std::fprintf(stderr,
                         "qwen4_exp weights: resident %.2f GiB, PLE disk %s\n",
                         static_cast<double>(weights_.eager_nbytes()) / kGiB,
                         weights_.has_ple_disk() ? "pread" : "MISSING");
        }
        sanitize_weights();
        if (cfg_.is_flash()) {
            infer_flash_attn_output_gate();
            load_ple_tables();
        }
        if (ffn_ == FfnKind::Moe && fuse_gate_up_enabled()) {
            fuse_expert_gate_up();
        }
        if (ffn_ == FfnKind::Dense && fuse_gate_up_enabled()) {
            fuse_dense_shared_x();
        }
        reset_cache();
        cfg_.vocab_size = cfg_.vocab_size > 0 ? cfg_.vocab_size : 248320;
        if (keep_vision_) {
            if (!load_qwen35_vision_config(model_dir, vision_cfg_)) {
                load_error_ = "vision requested, but this checkpoint declares no "
                              "vision_config (it is a text-only export)";
                return false;
            }
            // The merger projects onto the LLM's residual stream. A disagreement here
            // means the two halves of the checkpoint were exported from different
            // models, and the splice would put a wrongly-sized row into the stream.
            if (vision_cfg_.out_hidden_size != cfg_.hidden_size) {
                load_error_ = "vision: merger emits " +
                              std::to_string(vision_cfg_.out_hidden_size) +
                              " but the text model's hidden size is " +
                              std::to_string(cfg_.hidden_size);
                return false;
            }
            if (!vision_.init(vision_cfg_, weights_, load_error_)) {
                return false;
            }
            vision_loaded_ = true;
        }
        return true;
    }

    [[nodiscard]] bool vision_loaded() const noexcept { return vision_loaded_; }
    [[nodiscard]] const Qwen35VisionTower& vision() const noexcept { return vision_; }
    [[nodiscard]] const Qwen35VisionConfig& vision_config() const noexcept {
        return vision_cfg_;
    }

    void reset_cache() {
        kv_caches_.assign(static_cast<std::size_t>(cfg_.num_hidden_layers), KVCache{});
        qkv_caches_.assign(static_cast<std::size_t>(cfg_.num_hidden_layers), QuantizedKVCache{});
        ssm_caches_.assign(static_cast<std::size_t>(cfg_.num_hidden_layers), SsmCache{});
        indexer_caches_.assign(static_cast<std::size_t>(cfg_.num_hidden_layers), IndexerCache{});
        ple_cache_.clear();
    }

    [[nodiscard]] int cache_seq_len() const noexcept {
        // kv_caches_.front() is layer 0, which is a linear (gated-delta/SSM) layer
        // under the full_attention_interval hybrid schedule -- forward_linear_layer
        // never touches kv_caches_[layer], only ssm_caches_[layer].
        // Reading front() here always reported 0 regardless of how many tokens
        // were actually prefilled, which made the caller's "empty cache but
        // baked_turns != 0" invariant check fire every iteration once baked_turns
        // was nonzero and force-reset session state on top of a KV cache that was
        // never actually cleared -- silently re-prefilling the whole conversation
        // on every turn (see inference_engine.cpp's cache_seq_len()==0 guard).
        // Read the offset from the first real full-attention layer instead.
        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            if (!cfg_.is_linear_layer(layer) &&
                static_cast<std::size_t>(layer) < kv_caches_.size()) {
                return kv_caches_[static_cast<std::size_t>(layer)].offset;
            }
        }
        return 0;
    }

    // Returns the graph, unevaluated. The caller decides what to force and when, which
    // matters twice: it lets the decode loop fold the host-ready float32 conversion into
    // the same evaluation instead of paying a second GPU round-trip for it, and it lets
    // prefill chunks whose logits nobody reads skip the lm_head projection entirely.
    // It also lets `lmp_diag step` time graph construction apart from execution.
    mx::array forward_logits(const mx::array& input_ids) {
        const int seq_len = static_cast<int>(input_ids.shape()[1]);
        mx::array h = forward_hidden(input_ids, seq_len);
        // Contract: only the final position's
        // logits are ever consumed downstream. No-op when seq_len == 1 (decode).
        const int hidden = static_cast<int>(h.shape()[2]);
        mx::array h_last = mx::slice(h, {0, seq_len - 1, 0}, {1, seq_len, hidden});
        return logits_from_hidden(h_last);
    }

    // The opt-in path out of that contract: verifying a k-token speculative draft needs the
    // model's distribution at EVERY drafted position, not just the last one. Returns
    // [1, seq_len, vocab] -- one row per input position -- unevaluated, on the same terms as
    // forward_logits.
    //
    // A second entry point rather than a flag on the first, deliberately. The decode path is
    // tuned and its graph is load-bearing: folding the float32
    // cast into this graph measured WORSE (84.8 -> 83.9 tok/s) despite removing a GPU
    // round-trip, because it extended the step's critical path. So forward_logits still
    // builds the graph it built before this function existed. What the two share is the
    // layer stack, which they need verbatim and identically; the only divergence is the
    // final-position slice that one does and the other skips.
    //
    // Not free at the vocabulary this model has: the lm_head projection runs on seq_len rows
    // instead of one, and each row is 248,320 wide. That is the right trade only when the
    // extra rows are actually verified against -- never call this from the single-token
    // decode path, where it would pay for a slice it then throws away.
    mx::array forward_logits_all(const mx::array& input_ids) {
        const int seq_len = static_cast<int>(input_ids.shape()[1]);
        return logits_from_hidden(forward_hidden(input_ids, seq_len));
    }

    // A restore point for speculative decoding: everything needed to put the caches back the
    // way they were before a block of drafted tokens was forwarded.
    //
    // Qwen 3.6 is a HYBRID and the two kinds of layer roll back differently.
    // Full-attention layers keep a per-token history, so any earlier position is reachable
    // by moving an index. Linear (gated-delta) layers keep a recurrence with no per-token
    // history, so only positions that were snapshotted in advance are reachable at all.
    // That asymmetry is why this is checkpoint/restore rather than the `rollback_to(int n)`
    // the plan sketched: a rollback to an arbitrary n is not implementable for 30 of this
    // model's 40 layers, and an API promising it would be a lie whose only symptom is drift
    // in the output.
    struct CacheCheckpoint {
        int seq_len{0};
        // Indexed by layer, including the full-attention layers, whose SsmCache is empty.
        // Keeping the vector layer-aligned rather than packing the linear layers means
        // restore() cannot silently mis-pair a snapshot with a layer if the hybrid schedule
        // ever changes.
        std::vector<SsmCache::Snapshot> ssm;
        PleCache ple;
    };

    [[nodiscard]] CacheCheckpoint checkpoint() const {
        // ONE eval of every live SSM/PLE buffer, not one eval per layer. snapshot()
        // would mx::eval conv_state and delta_state separately for each of 30 linear
        // layers -- 60 Metal round-trips for handles the previous forward already
        // computed. SpecPhases put that on the dense MTP hot path at ~5 ms/block.
        std::vector<mx::array> live;
        live.reserve(ssm_caches_.size() * 2 + 1);
        for (const auto& c : ssm_caches_) {
            if (c.conv_state) {
                live.push_back(*c.conv_state);
            }
            if (c.delta_state) {
                live.push_back(*c.delta_state);
            }
        }
        if (ple_cache_.conv_state) {
            live.push_back(*ple_cache_.conv_state);
        }
        if (!live.empty()) {
            mx::eval(live);
        }
        CacheCheckpoint cp;
        cp.seq_len = cache_seq_len();
        cp.ssm.reserve(ssm_caches_.size());
        for (const auto& c : ssm_caches_) {
            cp.ssm.push_back({c.conv_state, c.delta_state, c.offset});
        }
        cp.ple = ple_cache_;
        return cp;
    }

    void restore(const CacheCheckpoint& cp) {
        // truncate_to clamps upward-to-current, so the linear layers' unused KVCaches --
        // which sit at offset 0 forever -- are left alone rather than being grown to
        // cp.seq_len tokens of scratch.
        for (auto& c : kv_caches_) {
            c.truncate_to(cp.seq_len);
        }
        // Both halves roll back, and by the same integer assignment. A quantized cache
        // left at the old length would keep answering with rows the bf16 offset says were
        // discarded -- stale context that reads as fluent, which is the failure kv_cache.hpp
        // exists to prevent.
        for (auto& c : qkv_caches_) {
            c.truncate_to(cp.seq_len);
        }
        const std::size_t n = std::min(ssm_caches_.size(), cp.ssm.size());
        for (std::size_t i = 0; i < n; ++i) {
            ssm_caches_[i].restore(cp.ssm[i]);
        }
        for (auto& c : indexer_caches_) {
            c.truncate_to(cp.seq_len);
        }
        ple_cache_ = cp.ple;
    }

    void eval_caches() {
        for (auto& c : kv_caches_) {
            c.sync();
        }
        for (auto& c : qkv_caches_) {
            c.sync();
        }
        for (auto& c : ssm_caches_) {
            c.sync();
        }
        for (auto& c : indexer_caches_) {
            if (c.keys) {
                mx::eval(*c.keys);
            }
        }
        if (ple_cache_.conv_state) {
            mx::eval(*ple_cache_.conv_state);
        }
    }

    [[nodiscard]] const Qwen35MoeConfig& qwen_config() const noexcept { return cfg_; }

    // Ablation, for attribution only. Timing a block in isolation does not measure what
    // it costs inside a real step: `layers 1` charges the MoE 0.673 ms/layer, and the
    // same chained microbenchmark applied to mlx-lm's own block charges 0.304 -- yet
    // mlx-lm's entire 40-layer step is 9.66 ms, so both numbers are dominated by the
    // measurement, not the work. Deleting a block from a full generation run and reading
    // the end-to-end rate has no such artifact. Set LMP_ABLATE=routed|mlp|delta.
    // Output is garbage when this is on; only tok/s means anything.
    enum class Ablate : std::uint8_t { none, routed, mlp, delta, deltakernel };
    static Ablate ablation() {
        // Read once. A getenv per layer would itself be a per-step cost.
        static const Ablate mode = [] {
            const char* v = std::getenv("LMP_ABLATE");
            if (v == nullptr) return Ablate::none;
            const std::string s(v);
            if (s == "routed") return Ablate::routed;
            if (s == "mlp") return Ablate::mlp;
            if (s == "delta") return Ablate::delta;
            if (s == "deltakernel") return Ablate::deltakernel;
            return Ablate::none;
        }();
        return mode;
    }

    // The three blocks a forward pass is made of, public so tests/model/diag_main.cpp
    // can time them individually against the real weights. Attribution has to run the
    // SAME code the model runs; a copy of these bodies in the driver would drift, and
    // the profile would then describe a forward pass nobody executes (S19.3).
    mx::array forward_moe(int layer, const mx::array& x) const {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".mlp.";
        if (ablation() == Ablate::mlp) {
            return mx::zeros_like(x);
        }
        const mx::array gate_logits = weights_.linear(x, p + "gate");
        auto [inds, scores] = lmp::model::mlxl::moe_topk(gate_logits, cfg_.num_experts_per_tok, cfg_.norm_topk_prob);
        // Diagnostic capture (S19.3). Off unless LMP_MOE_TRACE is set; see moe_trace.hpp
        // for why it is a runtime branch rather than a compile-time one, and why a traced
        // run's throughput figures must be discarded.
        if (MoeTrace::instance().enabled() && x.shape()[1] == 1) {
            mx::array ids = mx::astype(mx::reshape(inds, {-1}), mx::int32);
            mx::eval(ids);
            MoeTrace::instance().record(layer, ids.data<int>(),
                                        static_cast<std::size_t>(ids.size()));
        }
        mx::array y = mx::zeros_like(x);
        if (ablation() != Ablate::routed) {
            y = lmp::model::mlxl::switch_glu(
                x, weights_, p + "switch_mlp.gate_proj", p + "switch_mlp.up_proj", p + "switch_mlp.down_proj", inds,
                p + "switch_mlp.gate_up_proj");
            y = mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);
        }

        const mx::array shared = weights_.linear(
            lmp::model::mlxl::swiglu(
                weights_.linear(x, p + "shared_expert.gate_proj"),
                weights_.linear(x, p + "shared_expert.up_proj")),
            p + "shared_expert.down_proj");
        const mx::array shared_gate = mx::sigmoid(weights_.linear(x, p + "shared_expert_gate"));
        return mx::add(y, mx::multiply(shared_gate, shared));
    }

    // The dense generation's FFN: one gated MLP where the MoE has 256 routed experts plus
    // a shared one. Same SwiGLU and same quantized `linear` the shared expert already
    // uses, so nothing new reaches the weight store or the kernels.
    mx::array forward_dense_mlp(int layer, const mx::array& x) const {
        if (ablation() == Ablate::mlp) {
            return mx::zeros_like(x);
        }
        return dense_mlp_at(prefix_ + "layers." + std::to_string(layer) + ".mlp.", x);
    }

    // Prefix-addressed so the MTP head's identically-shaped MLP reuses it rather than
    // carrying a second copy of the same three matmuls.
    mx::array dense_mlp_at(const std::string& p, const mx::array& x) const {
        const std::string fused = p + "gate_up_proj";
        if (weights_.is_quantized(fused)) {
            const mx::array gu = weights_.linear(x, fused);
            const std::vector<mx::array> half = mx::split(gu, 2, -1);
            return weights_.linear(swiglu(half[0], half[1]), p + "down_proj");
        }
        return weights_.linear(weights_.swiglu_gate_up(x, p + "gate_proj", p + "up_proj"),
                               p + "down_proj");
    }

    // post_attention_layernorm → MLP. LMP_FUSE_RMS folds the norm into gate/up QMV on
    // the dense path only. MoE still norms once then routes.
    mx::array ffn_from_residual(int layer, const std::string& layer_p,
                                const mx::array& residual) const {
        const std::string ln = layer_p + "post_attention_layernorm.weight";
        if (q4_fuse_rms_on() && ffn_ == FfnKind::Dense) {
            const std::string mp = layer_p + "mlp.";
            if (weights_.is_quantized(mp + "gate_up_proj") || q4_fuse_glu_on()) {
                return dense_mlp_at(mp, rms_norm(residual, ln));
            }
            const mx::array& gamma = weights_.get(ln);
            const mx::array gate =
                weights_.rms_linear(residual, gamma, cfg_.rms_norm_eps, mp + "gate_proj");
            const mx::array up =
                weights_.rms_linear(residual, gamma, cfg_.rms_norm_eps, mp + "up_proj");
            return weights_.linear(swiglu(gate, up), mp + "down_proj");
        }
        return forward_ffn(layer, rms_norm(residual, ln));
    }

    mx::array dense_mlp_from_residual(const std::string& layer_p,
                                      const mx::array& residual) const {
        const std::string ln = layer_p + "post_attention_layernorm.weight";
        const std::string mp = layer_p + "mlp.";
        if (q4_fuse_rms_on()) {
            if (weights_.is_quantized(mp + "gate_up_proj") || q4_fuse_glu_on()) {
                return dense_mlp_at(mp, rms_norm(residual, ln));
            }
            const mx::array& gamma = weights_.get(ln);
            const mx::array gate =
                weights_.rms_linear(residual, gamma, cfg_.rms_norm_eps, mp + "gate_proj");
            const mx::array up =
                weights_.rms_linear(residual, gamma, cfg_.rms_norm_eps, mp + "up_proj");
            return weights_.linear(swiglu(gate, up), mp + "down_proj");
        }
        return dense_mlp_at(mp, rms_norm(residual, ln));
    }

    // The single axis the two checkpoints differ on. Classified once at load(), which
    // refuses anything it does not recognise, so this cannot silently pick a graph.
    mx::array forward_ffn(int layer, const mx::array& x) const {
        return ffn_ == FfnKind::Dense ? forward_dense_mlp(layer, x) : forward_moe(layer, x);
    }

    mx::array forward_linear_layer(int layer, const mx::array& x, int seq_len) {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".";
        mx::array h = rms_norm(x, p + "input_layernorm.weight");
        if (ablation() == Ablate::delta) {
            h = x;
        } else {
            mx::array attn_out = forward_gated_delta(
                p + "linear_attn.", h, ssm_caches_[static_cast<std::size_t>(layer)], seq_len);
            h = mx::add(x, attn_out);
        }
        return mx::add(h, ffn_from_residual(layer, p, h));
    }

    mx::array forward_full_attn_layer(int layer, const mx::array& x, int seq_len) {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".";
        mx::array h = x;
        const mx::array* fuse_ln = nullptr;
        if (q4_fuse_rms_on()) {
            fuse_ln = &weights_.get(p + "input_layernorm.weight");
        } else {
            h = rms_norm(x, p + "input_layernorm.weight");
        }
        mx::array attn_out =
            forward_self_attn(p + "self_attn.", h, kv_caches_[static_cast<std::size_t>(layer)],
                              qkv_caches_[static_cast<std::size_t>(layer)], seq_len, fuse_ln);
        h = mx::add(x, attn_out);
        return mx::add(h, ffn_from_residual(layer, p, h));
    }

    mx::array forward_flash_layer(int layer, const mx::array& x, int seq_len,
                                  const mx::array& input_ids) {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".";
        mx::array h = x;
        if (cfg_.is_ple_layer(layer) && !ple_mults_.empty()) {
            static const bool skip_ple = [] {
                const char* v = std::getenv("LMP_SKIP_PLE");
                return v != nullptr && std::string_view(v) == "1";
            }();
            if (!skip_ple) {
                h = mx::add(h, ple_forward(h, input_ids, weights_, p + "ple.", ple_cache_, cfg_,
                                           ple_mults_.data(), ple_sizes_.data(),
                                           ple_offs_.data()));
            }
        }
        const int hc = cfg_.hc_count;
        const int d = cfg_.hidden_size;
        const int rank = cfg_.hc_lowrank;
        auto attn_gr = gated_residual_read(h, weights_, p + "attn_hyper_connection.", hc, d,
                                           rank, cfg_.rms_norm_eps, true);
        mx::array attn_y = attn_gr.mixed;
        if (ablation() != Ablate::delta) {
            if (cfg_.is_linear_layer(layer)) {
                attn_y = forward_gated_delta(p + "linear_attn.", attn_gr.mixed,
                                             ssm_caches_[static_cast<std::size_t>(layer)],
                                             seq_len);
            } else {
                attn_y = forward_self_attn(
                    p + "self_attn.", attn_gr.mixed, kv_caches_[static_cast<std::size_t>(layer)],
                    qkv_caches_[static_cast<std::size_t>(layer)], seq_len, nullptr,
                    &indexer_caches_[static_cast<std::size_t>(layer)]);
            }
        }
        h = gated_residual_write(attn_gr, attn_y);
        auto mlp_gr = gated_residual_read(h, weights_, p + "mlp_hyper_connection.", hc, d, rank,
                                          cfg_.rms_norm_eps, true);
        return gated_residual_write(mlp_gr, forward_ffn(layer, mlp_gr.mixed));
    }

    // Set when load() refuses; empty otherwise. MlxBackend surfaces it verbatim so the
    // operator is told WHICH of the load preconditions failed.
    [[nodiscard]] const std::string& load_error() const noexcept { return load_error_; }

    // ---- the MTP head ----------------------------------------------------------------
    //
    // A separate checkpoint, merged into THIS store under an "mtp." prefix. It carries one
    // decoder layer, an fc over concat(embedding, hidden), two pre-fc norms and a final
    // norm -- and no embedding table and no lm_head, because it borrows the target's.
    //
    // Loaded after sanitize_weights() on purpose: that pass DROPS keys containing "mtp.",
    // which is correct for a target checkpoint that ships MTP tensors inline and would
    // otherwise delete exactly what we just merged.
    bool load_mtp(const std::string& mtp_dir) {
        mtp_loaded_ = false;
        mtp_block_size_ = load_mtp_block_size(mtp_dir);
        if (mtp_block_size_ < 2) {
            return false;
        }
        if (!weights_.load_directory_merged(mtp_dir, "mtp.")) {
            mtp_block_size_ = 0;
            return false;
        }
        // Refuse rather than discover it mid-generation: the head is addressed by key, so
        // a checkpoint laid out differently fails at the first draft, not at load.
        //
        // TWO KINDS OF KEY, AND THEY ARE NOT INTERCHANGEABLE. A norm is a single tensor
        // named in full. A linear is addressed by its BASE, and `linear()` resolves that
        // base to `.weight` (plus `.scales` / `.biases` when the tensor is quantized) --
        // there is no tensor at the bare base itself.
        //
        // Testing a base with has() therefore asks for a key no checkpoint on disk
        // contains. A real 4-bit head ships fc.weight, fc.scales and fc.biases and no
        // bare `fc`, so this refused EVERY quantized head it was ever handed --
        // Qwen3.8-27B-MTP-4bit among them -- and reported it as "not a usable MTP draft
        // head", which reads as a bad checkpoint rather than a bad check. It passed in
        // testing because the fixtures build unquantized heads, where `linear()` also
        // wants `.weight` and the base is still not a key. The check was never right; it
        // was only never exercised against a file.
        for (const char* key : {"mtp.norm.weight", "mtp.pre_fc_norm_hidden.weight",
                                "mtp.pre_fc_norm_embedding.weight"}) {
            if (!weights_.has(key)) {
                mtp_block_size_ = 0;
                return false;
            }
        }
        for (const char* base : {"mtp.fc", "mtp.layers.0.self_attn.q_proj"}) {
            if (!weights_.has(std::string(base) + ".weight")) {
                mtp_block_size_ = 0;
                return false;
            }
        }
        mtp_cache_ = KVCache{};
        mtp_qcache_ = QuantizedKVCache{};
        mtp_loaded_ = true;
        if (fuse_gate_up_enabled()) {
            fuse_quant_out_rows({"mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj"},
                                "mtp.layers.0.mlp.gate_up_proj");
            mx::clear_cache();
        }
        return true;
    }

    [[nodiscard]] bool is_dense() const noexcept { return ffn_ == FfnKind::Dense; }
    [[nodiscard]] bool is_flash() const noexcept { return cfg_.is_flash(); }
    [[nodiscard]] bool has_mtp() const noexcept { return mtp_loaded_; }
    [[nodiscard]] int mtp_block_size() const noexcept { return mtp_block_size_; }

    // One MTP position: (token, target hidden) -> next hidden, appending to the head's own
    // KV cache. `hidden` is [1, n, hidden]; `token_ids` is [1, n].
    //
    // The concatenation order is EMBEDDING FIRST. It matches the reference and it is not
    // checkable by shape -- both halves are `hidden` wide, so getting it backwards
    // produces a running model whose drafts are noise.
    mx::array mtp_forward(const mx::array& token_ids, const mx::array& hidden) {
        const int seq_len = static_cast<int>(token_ids.shape()[1]);
        const mx::array e = embed_tokens(token_ids);
        mx::array h = mx::concatenate({rms_norm(e, "mtp.pre_fc_norm_embedding.weight"),
                                       rms_norm(hidden, "mtp.pre_fc_norm_hidden.weight")},
                                      -1);
        h = weights_.linear(h, "mtp.fc");

        const std::string p = "mtp.layers.0.";
        mx::array attn_in = h;
        const mx::array* fuse_ln = nullptr;
        if (q4_fuse_rms_on()) {
            fuse_ln = &weights_.get(p + "input_layernorm.weight");
        } else {
            attn_in = rms_norm(h, p + "input_layernorm.weight");
        }
        h = mx::add(h, forward_self_attn(p + "self_attn.", attn_in, mtp_cache_, mtp_qcache_,
                                         seq_len, fuse_ln));
        h = mx::add(h, dense_mlp_from_residual(p, h));
        return rms_norm(h, "mtp.norm.weight");
    }

    // The head's cache is its own; the target's rollback never touches it, so partial
    // acceptance has to trim it explicitly (MtpProposer owns that decision).
    void mtp_trim(int n) {
        if (n > 0) {
            const int keep = mtp_cache_.offset > n ? mtp_cache_.offset - n : 0;
            mtp_cache_.truncate_to(keep);
            mtp_qcache_.truncate_to(keep);
        }
    }
    void mtp_reset() {
        mtp_cache_ = KVCache{};
        mtp_qcache_ = QuantizedKVCache{};
    }

    // The final-normed hidden from the most recent forward, which is what the head
    // consumes and what the LM head consumes. Captured only while an MTP head is loaded:
    // holding it otherwise would keep a [1, seq, 5120] array alive for nothing.
    [[nodiscard]] const std::optional<mx::array>& last_hidden() const noexcept {
        return last_hidden_;
    }

    [[nodiscard]] mx::array logits_from_hidden_public(const mx::array& h) const {
        return logits_from_hidden(h);
    }

    // --- image embedding splice ------------------------------------------------
    //
    // Visual rows REPLACE the embeddings of the `<|image_pad|>` tokens they cover. That
    // is the WHOLE of the multimodal seam for this checkpoint: `deepstack_visual_indexes`
    // is empty, so nothing is injected at any later layer, and everything downstream --
    // 40 layers, the KV caches, the gated-delta recurrences, the sampler -- runs exactly
    // as it does for text.
    //
    // Set per forward by the backend, because prefill is chunked: an image runs to
    // hundreds of tokens and does not respect the 512-token chunk edge, so each chunk
    // receives the sub-range of rows that lands inside it.
    struct EmbedSplice {
        int offset = 0; // position within THIS chunk
        mx::array rows; // [count, hidden]
    };

    void set_embed_splices(std::vector<EmbedSplice> splices) {
        splices_ = std::move(splices);
    }
    void clear_embed_splices() { splices_.clear(); }

private:
    std::string prefix_;
    Qwen35MoeConfig cfg_{};
    FfnKind ffn_{FfnKind::Moe};
    std::string load_error_;
    bool mtp_loaded_ = false;
    int mtp_block_size_ = 0;
    bool keep_vision_ = false;
    bool vision_loaded_ = false;
    std::vector<EmbedSplice> splices_;
    Qwen35VisionConfig vision_cfg_{};
    Qwen35VisionTower vision_{};
    KVCache mtp_cache_{};
    // The draft head keeps its own pair for the same reason it keeps its own KVCache:
    // the target's rollback must never reach it.
    QuantizedKVCache mtp_qcache_{};
    std::optional<mx::array> last_hidden_;
    WeightStore weights_;
    std::unordered_map<std::string, std::vector<int>> fused_out_cuts_;
    std::vector<KVCache> kv_caches_;
    // Parallel to kv_caches_ and empty until a context gets long enough to be worth
    // converting. `kv_caches_[l].offset` stays the single source of truth for the token
    // count either way -- rope reads it, cache_seq_len() reads it, and restore() rolls it
    // back -- so the quantized cache mirrors that number rather than owning a second one.
    std::vector<QuantizedKVCache> qkv_caches_;
    std::vector<SsmCache> ssm_caches_;
    std::vector<IndexerCache> indexer_caches_;
    PleCache ple_cache_{};
    std::vector<std::uint64_t> ple_mults_;
    std::vector<std::int64_t> ple_sizes_;
    std::vector<std::int64_t> ple_offs_;

    // LMP_FUSE_GATE_UP=0 is the control arm. Default on; read once, because a getenv
    // per layer would itself be a load-time cost and a getenv per token would be worse.
    static bool fuse_gate_up_enabled() {
        static const bool on = [] {
            const char* v = std::getenv("LMP_FUSE_GATE_UP");
            return v == nullptr || std::string_view(v) != "0";
        }();
        return on;
    }

    // Stack each MoE layer's gate_proj and up_proj into one quantized weight so decode
    // issues one gather_qmm per layer instead of two. switch_glu.hpp explains why this is
    // numerically a no-op and why the launch count is what matters.
    //
    // Done at LOAD, not per step: concatenating 281 MB per layer on every token would
    // cost far more traffic than the dispatch it saves. The halves are erased as we go,
    // so the resident total is unchanged and the transient peak is one layer's copy.
    //
    // The eval() before the erase is load-bearing. mx::concatenate is lazy, so dropping
    // the map's references first would leave the fused array holding its inputs alive
    // through the graph and free nothing -- 11 GB of experts would be resident twice.
    void fuse_expert_gate_up() {
        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            const std::string p =
                prefix_ + "layers." + std::to_string(layer) + ".mlp.switch_mlp.";
            const std::string g = p + "gate_proj";
            const std::string u = p + "up_proj";
            const std::string f = p + "gate_up_proj";
            if (!weights_.is_quantized(g) || !weights_.is_quantized(u)) {
                continue; // dense layer, or an unquantized export: nothing to fuse
            }
            const QuantWeight qg = weights_.quant(g);
            const QuantWeight qu = weights_.quant(u);
            // Fusing across two different quantization schemes would silently reinterpret
            // one half's scales under the other's grouping. Refuse rather than guess --
            // this checkpoint overrides bits per key for the ROUTER, so unequal specs
            // inside one layer are a real shape a future export could take.
            if (qg.group_size != qu.group_size || qg.bits != qu.bits ||
                qg.mode != qu.mode || qg.biases.has_value() != qu.biases.has_value()) {
                continue;
            }
            if (qg.weight.ndim() != 3 || qu.weight.ndim() != 3 ||
                qg.weight.shape()[0] != qu.weight.shape()[0] ||
                qg.weight.shape()[1] != qu.weight.shape()[1] ||
                qg.weight.shape()[2] != qu.weight.shape()[2]) {
                continue; // unequal halves would make split(2) the wrong cut
            }
            // Axis 1 is the output-row axis: [experts, out, in]. Grouping lives on the
            // last (input) axis, so stacking rows leaves every group boundary intact.
            mx::array w = mx::concatenate({qg.weight, qu.weight}, 1);
            mx::array sc = mx::concatenate({qg.scales, qu.scales}, 1);
            std::vector<mx::array> force{w, sc};
            std::optional<mx::array> bi;
            if (qg.biases && qu.biases) {
                bi = mx::concatenate({*qg.biases, *qu.biases}, 1);
                force.push_back(*bi);
            }
            mx::eval(force);
            weights_.set(f + ".weight", w);
            weights_.set(f + ".scales", sc);
            if (bi) {
                weights_.set(f + ".biases", *bi);
            }
            for (const std::string& half : {g, u}) {
                weights_.erase(half + ".weight");
                weights_.erase(half + ".scales");
                weights_.erase(half + ".biases");
            }
        }
        // Hand the halves back to the OS rather than leaving them in MLX's allocator
        // cache. Measured: without this the process finishes load holding 12.79 GB of
        // cache against 1.44 GB unfused, for the same 18.24 GB active and 19.00 GB peak.
        // It is reclaimable and so it is not a leak, but on a 48 GB host that also runs
        // two editors it is 11 GB of apparent headroom that is not headroom -- the exact
        // accounting error that made 38 GB look survivable (see model_limits.cpp).
        mx::clear_cache();
    }

    // Dense 2D affine-4: concat packed rows along the output axis. Same contract as
    // fuse_expert_gate_up (eval before erase) but axis 0, because these weights are
    // [out, in/8] rather than [experts, out, in].
    bool fuse_quant_out_rows(const std::vector<std::string>& bases, const std::string& out_key) {
        if (bases.size() < 2) {
            return false;
        }
        std::vector<QuantWeight> qs;
        qs.reserve(bases.size());
        for (const std::string& b : bases) {
            if (!weights_.is_quantized(b)) {
                return false;
            }
            qs.push_back(weights_.quant(b));
        }
        const QuantWeight& q0 = qs[0];
        if (q0.weight.ndim() != 2) {
            return false;
        }
        std::vector<int> cuts;
        cuts.reserve(qs.size());
        std::vector<mx::array> ws;
        std::vector<mx::array> scs;
        std::vector<mx::array> bis;
        for (const QuantWeight& q : qs) {
            if (q.group_size != q0.group_size || q.bits != q0.bits || q.mode != q0.mode ||
                q.biases.has_value() != q0.biases.has_value() || q.weight.ndim() != 2 ||
                q.weight.shape()[1] != q0.weight.shape()[1]) {
                return false;
            }
            cuts.push_back(static_cast<int>(q.scales.shape()[0]));
            ws.push_back(q.weight);
            scs.push_back(q.scales);
            if (q.biases) {
                bis.push_back(*q.biases);
            }
        }
        mx::array w = mx::concatenate(ws, 0);
        mx::array sc = mx::concatenate(scs, 0);
        std::vector<mx::array> force{w, sc};
        std::optional<mx::array> bi;
        if (!bis.empty()) {
            bi = mx::concatenate(bis, 0);
            force.push_back(*bi);
        }
        mx::eval(force);
        weights_.set(out_key + ".weight", w);
        weights_.set(out_key + ".scales", sc);
        if (bi) {
            weights_.set(out_key + ".biases", *bi);
        }
        for (const std::string& b : bases) {
            weights_.erase(b + ".weight");
            weights_.erase(b + ".scales");
            weights_.erase(b + ".biases");
        }
        fused_out_cuts_[out_key] = std::move(cuts);
        return true;
    }

    void fuse_dense_shared_x() {
        int n_glu = 0;
        int n_qkv = 0;
        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            const std::string lp = prefix_ + "layers." + std::to_string(layer) + ".";
            if (fuse_quant_out_rows({lp + "mlp.gate_proj", lp + "mlp.up_proj"},
                                    lp + "mlp.gate_up_proj")) {
                ++n_glu;
            }
            if (!cfg_.is_linear_layer(layer)) {
                const std::string ap = lp + "self_attn.";
                if (weights_.is_quantized(ap + "q_proj") &&
                    weights_.is_quantized(ap + "k_proj") &&
                    weights_.is_quantized(ap + "v_proj")) {
                    const int nq = static_cast<int>(weights_.quant(ap + "q_proj").scales.shape()[0]);
                    const int nk = static_cast<int>(weights_.quant(ap + "k_proj").scales.shape()[0]);
                    const int nv = static_cast<int>(weights_.quant(ap + "v_proj").scales.shape()[0]);
                    // attn_output_gate stores q as [q, gate] (N twice k/v). Do not concat.
                    if (nq == nk && nk == nv &&
                        fuse_quant_out_rows({ap + "q_proj", ap + "k_proj", ap + "v_proj"},
                                            ap + "qkv_proj")) {
                        ++n_qkv;
                    }
                }
            }
        }
        std::fprintf(stderr, "dense concat: gate_up=%d qkv=%d / %d layers\n", n_glu, n_qkv,
                     cfg_.num_hidden_layers);
        mx::clear_cache();
    }

    void sanitize_weights() {
        auto& w = weights_.mutable_weights();

        // Mirrors mlx-lm's Qwen3.5-MoE sanitize(): the RMSNorm "+1" shift and the
        // conv1d axis fixup are only needed for raw (unconverted) HF checkpoints.
        // LM Studio's pre-converted MLX checkpoints already have conv1d weights in
        // MLX layout (last dim == 1) and never ship MTP weights, so applying the
        // shift unconditionally would corrupt every norm weight in the model.
        bool has_mtp_weights = false;
        bool has_unsanitized_conv1d = false;
        bool raw_hf = false;
        for (const auto& [key, val] : w) {
            if (key.find("mtp.") != std::string::npos) {
                has_mtp_weights = true;
            }
            if (key.find("model.language_model.") != std::string::npos ||
                key.find("experts.gate_up_proj") != std::string::npos) {
                raw_hf = true;
            }
            if (key.find("conv1d.weight") != std::string::npos && val.ndim() == 3 &&
                val.shape()[2] != 1) {
                has_unsanitized_conv1d = true;
            }
        }
        // Flash-Next's MLX conversion (sh0wie) leaves HF-style zero-centered RMSNorm
        // gammas. hc_norm mean is ~0 and goes negative; mlx-lm's Qwen3.5 exports already
        // bake (1+w). PLE conv1d is still PyTorch layout ([C,1,K]), which also trips the
        // conv fix, but do not rely on that side-effect: a conversion that ships MLX conv
        // and raw norms would otherwise skip the shift and zero the residual.
        const bool should_shift_norm_weights =
            has_mtp_weights || has_unsanitized_conv1d || raw_hf || cfg_.is_flash();
        if (cfg_.is_flash()) {
            std::fprintf(stderr,
                         "qwen4_exp sanitize: shift_norms=%s conv1d_unaligned=%s raw_hf=%s\n",
                         should_shift_norm_weights ? "yes" : "no",
                         has_unsanitized_conv1d ? "yes" : "no",
                         raw_hf ? "yes" : "no");
        }

        std::vector<std::pair<std::string, mx::array>> updated;
        std::vector<std::string> to_erase;
        int n_norm_shifted = 0;

        for (auto& [key, val] : w) {
            // `vision_tower.*` used to be dropped here unconditionally, which is the
            // whole reason this product could not see: the weights ship in the same
            // shards, and load threw them away before anything could ask. They are kept
            // when the caller asked for vision and dropped otherwise, so a text-only run
            // still pays neither the 0.92 GB nor the load time.
            const bool is_vision = key.find("vision_tower") != std::string::npos;
            if (key.find("mtp.") != std::string::npos || (is_vision && !keep_vision_)) {
                to_erase.push_back(key);
                continue;
            }
            if (is_vision) {
                // The vision tower is unquantized bf16 and shares none of the text
                // stack's fixups -- no conv1d axis move, no RMSNorm "+1" shift (its
                // norms are LayerNorm, with their own bias).
                continue;
            }
            if (key.find("conv1d.weight") != std::string::npos && val.ndim() == 3 &&
                val.shape()[2] != 1) {
                updated.emplace_back(key, mx::moveaxis(val, 2, 1));
                to_erase.push_back(key);
            }
            if (should_shift_norm_weights &&
                (key.ends_with(".input_layernorm.weight") ||
                 key.ends_with(".post_attention_layernorm.weight") ||
                 key.ends_with("model.norm.weight") ||
                 key.ends_with(".q_norm.weight") ||
                 key.ends_with(".k_norm.weight") ||
                 key.ends_with(".hc_norm.weight") ||
                 key.ends_with(".q_layernorm.weight") ||
                 key.ends_with(".k_layernorm.weight") ||
                 key.ends_with(".norm_key.weight") ||
                 key.ends_with(".norm_query.weight") ||
                 key.ends_with(".norm_conv.weight"))) {
                if (val.ndim() == 1) {
                    // astype, not a bare mx::array(1.0f). mlx-lm writes `v + 1.0`, and a
                    // Python float is weakly typed, so the weight stays bf16; the C++
                    // scalar is a strongly typed float32 array and would promote it.
                    // These are norm weights -- they multiply into rms_norm on every
                    // step, including model.norm before lm_head -- so a float32 one puts
                    // float32 back into the residual stream, which is the same fault that
                    // cost 3x in forward_gated_delta. No checkpoint here trips
                    // should_shift_norm_weights (it needs mtp weights or an unsanitized
                    // conv1d), so this is latent rather than measured: it costs nothing
                    // today and is wrong for the first checkpoint that does trip it.
                    updated.emplace_back(key, mx::add(val, mx::astype(mx::array(1.0f),
                                                                      val.dtype())));
                    to_erase.push_back(key);
                    ++n_norm_shifted;
                }
            }
        }
        for (const auto& k : to_erase) {
            w.erase(k);
        }
        for (auto& [k, v] : updated) {
            w.insert_or_assign(std::move(k), std::move(v));
        }
        if (cfg_.is_flash()) {
            std::fprintf(stderr, "qwen4_exp sanitize: norms_shifted=%d conv1d_moved=%d\n",
                         n_norm_shifted, has_unsanitized_conv1d ? 1 : 0);
        }

    }

    static std::vector<std::int64_t> copy_i64(const mx::array& a) {
        mx::array c = mx::astype(mx::reshape(a, {-1}), mx::int64);
        mx::eval(c);
        const int n = static_cast<int>(c.size());
        std::vector<std::int64_t> out(static_cast<std::size_t>(n));
        const std::int64_t* p = c.data<int64_t>();
        std::copy(p, p + n, out.begin());
        return out;
    }

    // sh0wie's conversion stores attn_output_gate: null, but q_proj is still the fused
    // [q, gate] layout (out = 2 * n_heads * head_dim). Trust the tensor, not the JSON.
    void infer_flash_attn_output_gate() {
        const int expected = cfg_.num_attention_heads * cfg_.head_dim;
        if (expected <= 0) {
            return;
        }
        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            if (cfg_.is_linear_layer(layer)) {
                continue;
            }
            const std::string q =
                prefix_ + "layers." + std::to_string(layer) + ".self_attn.q_proj.weight";
            if (!weights_.has(q)) {
                return;
            }
            const int out = static_cast<int>(weights_.get(q).shape()[0]);
            if (out == expected * 2) {
                cfg_.attn_output_gate = true;
            } else if (out == expected) {
                cfg_.attn_output_gate = false;
            }
            return;
        }
    }

    void load_ple_tables() {
        const int ngram = cfg_.ngram_size > 0 ? cfg_.ngram_size : 3;
        const int hpn = cfg_.heads_per_ngram > 0 ? cfg_.heads_per_ngram : 8;
        const int n_heads = (ngram - 1) * hpn;
        int ple_idx = 0;
        std::string pp;
        for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
            if (!cfg_.is_ple_layer(i)) {
                continue;
            }
            auto it = std::find(cfg_.ple_layer_ids.begin(), cfg_.ple_layer_ids.end(), i + 1);
            if (it != cfg_.ple_layer_ids.end()) {
                ple_idx = static_cast<int>(std::distance(cfg_.ple_layer_ids.begin(), it));
            }
            pp = prefix_ + "layers." + std::to_string(i) + ".ple.ple_embedding.";
            break;
        }
        if (pp.empty()) {
            return;
        }
        if (weights_.has(pp + "layer_multipliers")) {
            const auto v = copy_i64(weights_.get(pp + "layer_multipliers"));
            ple_mults_.assign(v.begin(), v.end());
        } else {
            ple_mults_ = lmp::model::qwen4::default_layer_multipliers(
                lmp::model::qwen4::kDefaultHashSeed, ngram, cfg_.vocab_size, ple_idx);
        }
        if (weights_.has(pp + "ngram_heads_vocab_sizes") &&
            weights_.has(pp + "ngram_heads_offsets")) {
            ple_sizes_ = copy_i64(weights_.get(pp + "ngram_heads_vocab_sizes"));
            ple_offs_ = copy_i64(weights_.get(pp + "ngram_heads_offsets"));
        } else {
            ple_sizes_.assign(static_cast<std::size_t>(n_heads), 0);
            ple_offs_.assign(static_cast<std::size_t>(n_heads), 0);
            std::int64_t total = 0;
            const int base =
                cfg_.ngram_vocab_size_base > 0 ? cfg_.ngram_vocab_size_base : 20000000;
            for (int h = 0; h < n_heads; ++h) {
                const int g = ple_idx * n_heads + h;
                const int s = lmp::model::qwen4::nth_prime_after(base - 1, g + 1);
                ple_sizes_[static_cast<std::size_t>(h)] = s;
                ple_offs_[static_cast<std::size_t>(h)] = total;
                total += s;
            }
        }
    }

    // The layer stack, shared verbatim by forward_logits and forward_logits_all so the two
    // cannot drift apart. Returns the final-normed hidden state, [1, seq_len, hidden].
    mx::array forward_hidden(const mx::array& input_ids, int seq_len) {
        mx::array h = apply_embed_splices(embed_tokens(input_ids), seq_len);
        if (MoeTrace::instance().enabled() && seq_len == 1) {
            mx::array t = mx::astype(mx::reshape(input_ids, {-1}), mx::int32);
            mx::eval(t);
            MoeTrace::instance().set_token(t.data<int>()[0]);
        }

        if (cfg_.is_flash()) {
            h = tile_hyper(h, cfg_.hc_count);
            for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
                h = forward_flash_layer(layer, h, seq_len, input_ids);
            }
            auto mix = gated_residual_read(h, weights_, prefix_ + "hyper_connection_mixer.",
                                           cfg_.hc_count, cfg_.hidden_size, cfg_.hc_lowrank,
                                           cfg_.rms_norm_eps, /*use_combine=*/false);
            mx::array out = mix.mixed;
            if (mtp_loaded_) {
                last_hidden_ = out;
            }
            return out;
        }

        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            if (cfg_.is_linear_layer(layer)) {
                h = forward_linear_layer(layer, h, seq_len);
            } else {
                h = forward_full_attn_layer(layer, h, seq_len);
            }
        }

        mx::array out = rms_norm(h, prefix_ + "norm.weight");
        if (mtp_loaded_) {
            last_hidden_ = out;
        }
        return out;
    }

    [[nodiscard]] mx::array embed_tokens(const mx::array& ids) const {
        return weights_.embed_lookup(ids, prefix_ + "embed_tokens");
    }


    // Rebuilt by concatenation rather than scatter: an image is a CONTIGUOUS run of pads,
    // so a chunk carrying k images is at most 2k+1 pieces, and concatenate keeps the
    // whole thing one graph node the compiler can fuse. A scatter over row indices would
    // express the same edit as a gather-modify-write over the full [1, L, hidden] block.
    [[nodiscard]] mx::array apply_embed_splices(mx::array h, int seq_len) const {
        if (splices_.empty()) {
            return h;
        }
        const int hidden = static_cast<int>(h.shape()[2]);
        std::vector<mx::array> pieces;
        int cursor = 0;
        for (const EmbedSplice& s : splices_) {
            const int count = static_cast<int>(s.rows.shape()[0]);
            if (s.offset < cursor || s.offset + count > seq_len) {
                // The backend computed these from the same chunk bounds it passed in, so
                // this is a programming error rather than bad input -- and a silently
                // dropped splice is an image the model never sees while everything
                // downstream reports success.
                throw std::runtime_error(
                    "embed splice [" + std::to_string(s.offset) + "," +
                    std::to_string(s.offset + count) + ") does not fit a chunk of " +
                    std::to_string(seq_len) + " starting at " + std::to_string(cursor));
            }
            if (s.offset > cursor) {
                pieces.push_back(mx::slice(h, {0, cursor, 0}, {1, s.offset, hidden}));
            }
            pieces.push_back(
                mx::astype(mx::reshape(s.rows, {1, count, hidden}), h.dtype()));
            cursor = s.offset + count;
        }
        if (cursor < seq_len) {
            pieces.push_back(mx::slice(h, {0, cursor, 0}, {1, seq_len, hidden}));
        }
        return mx::concatenate(pieces, 1);
    }

    [[nodiscard]] mx::array rms_norm(const mx::array& x, const std::string& wkey) const {
        return mx::fast::rms_norm(x, weights_.get(wkey), cfg_.rms_norm_eps);
    }

    [[nodiscard]] mx::array logits_from_hidden(const mx::array& h) const {
        if (cfg_.tie_word_embeddings) {
            return weights_.tied_logits(h, prefix_ + "embed_tokens");
        }
        return weights_.linear(h, "language_model.lm_head");
    }

    mx::array forward_gated_delta(const std::string& p, const mx::array& inputs, SsmCache& cache, int /*seq_len*/) {
        const int B = static_cast<int>(inputs.shape()[0]);
        const int S = static_cast<int>(inputs.shape()[1]);
        const int key_dim = cfg_.linear_num_key_heads * cfg_.linear_key_head_dim;
        const int value_dim = cfg_.linear_num_value_heads * cfg_.linear_value_head_dim;
        const int conv_dim = key_dim * 2 + value_dim;

        mx::array qkv = weights_.linear(inputs, p + "in_proj_qkv");
        mx::array z = mx::array(0.0f, inputs.dtype());
        mx::array b = mx::array(0.0f, inputs.dtype());
        mx::array a = mx::array(0.0f, inputs.dtype());
        const auto zba_it = fused_out_cuts_.find(p + "in_proj_zba");
        if (zba_it != fused_out_cuts_.end()) {
            const mx::array zba = weights_.linear(inputs, p + "in_proj_zba");
            const int nz = zba_it->second[0];
            const int nb = zba_it->second[1];
            const std::vector<mx::array> parts =
                mx::split(zba, mx::Shape{nz, nz + nb}, -1);
            z = parts[0];
            b = parts[1];
            a = parts[2];
        } else {
            z = weights_.linear(inputs, p + "in_proj_z");
            b = weights_.linear(inputs, p + "in_proj_b");
            a = weights_.linear(inputs, p + "in_proj_a");
        }

        mx::array conv_state = cache.conv_state.value_or(
            mx::zeros({B, cfg_.linear_conv_kernel_dim - 1, conv_dim}, inputs.dtype()));
        mx::array conv_input = mx::concatenate({conv_state, qkv}, 1);
        // mx::contiguous, as mlx-lm does: a bare slice is a strided view that both keeps
        // the whole parent conv_input buffer alive across the step boundary and makes
        // next step's concatenate read from a strided source.
        cache.conv_state = mx::contiguous(
            mx::slice(conv_input, {0, conv_input.shape()[1] - (cfg_.linear_conv_kernel_dim - 1), 0},
                      {B, conv_input.shape()[1], conv_dim}));

        mx::array conv_w = weights_.get(p + "conv1d.weight");
        mx::array conv_out = lmp::model::mlxl::silu(
            mx::conv1d(conv_input, conv_w, /*stride=*/1, /*padding=*/0, /*dilation=*/1, conv_dim));

        // One split, not three slices. mlx-lm splits here and `lmp_diag graph` counted
        // the difference: 110 Slice nodes against the reference's 40 Split, for the same
        // partition of the same buffer.
        const std::vector<mx::array> qkv_parts =
            mx::split(conv_out, mx::Shape{key_dim, 2 * key_dim}, 2);

        mx::array q = mx::reshape(qkv_parts[0], {B, S, cfg_.linear_num_key_heads, cfg_.linear_key_head_dim});
        mx::array k = mx::reshape(qkv_parts[1], {B, S, cfg_.linear_num_key_heads, cfg_.linear_key_head_dim});
        mx::array v = mx::reshape(qkv_parts[2], {B, S, cfg_.linear_num_value_heads, cfg_.linear_value_head_dim});
        z = mx::reshape(z, {B, S, cfg_.linear_num_value_heads, cfg_.linear_value_head_dim});

        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.linear_key_head_dim));
        // In the dtype of q/k, not float32. mlx-lm writes `(inv_scale**2) * rms_norm(q)`
        // with a Python float, which is weakly typed and leaves q in bf16; the obvious
        // C++ transcription, mx::array(inv_scale), is a strongly typed float32 scalar and
        // promotes instead. That promotion does not stay local -- q and k go float32 into
        // the delta kernel, y comes back float32, the gated norm returns float32, out_proj
        // emits float32, and `h + attn_out` makes the residual stream float32 from layer 1
        // to the end of the model. Every quantized matmul downstream then runs its float32
        // activation path against bf16 weights.
        //
        // Built in the target dtype rather than cast into it: mx::array(v, dtype) is the
        // same scalar as astype(mx::array(v), dtype) but is a constant, where the cast was
        // an AsType node in the graph. Sixty of them a step, per `lmp_diag graph`.
        const mx::array inv2 = mx::array(inv_scale * inv_scale, qkv.dtype());
        const mx::array inv1 = mx::array(inv_scale, qkv.dtype());
        // std::nullopt, not ones(): mlx-lm passes no weight here. A literal ones vector is
        // arithmetically the same, but it allocates and evaluates a fresh array on every
        // one of these calls (twice per layer, 30 linear layers per token) and takes
        // rms_norm's weighted path instead of its unweighted one.
        if (cfg_.is_flash()) {
            // llama.cpp / the Flash reference L2-normalise q and k and stop there.
            // The Qwen3.5 path below is RMS * 1/d on q (equivalent to L2/sqrt(d) on
            // unit-scale inputs). Applying that extra 1/sqrt(d) here shrinks every
            // GDN query by ~11x and is not how this checkpoint was trained.
            q = exact_l2_norm(q);
            k = exact_l2_norm(k);
        } else {
            q = mx::multiply(inv2, mx::fast::rms_norm(q, std::nullopt, 1e-6f));
            k = mx::multiply(inv1, mx::fast::rms_norm(k, std::nullopt, 1e-6f));
        }

        mx::array a_log = weights_.get(p + "A_log");
        mx::array dt_bias = weights_.get(p + "dt_bias");
        // deltakernel keeps the projections, the conv and the norms and removes only the
        // fused recurrence, to split "the custom Metal kernel" from "everything else in
        // this block". Removing the whole block (LMP_ABLATE=delta) cannot tell them apart.
        // inputs.dtype(), not float32. A float32 stand-in here re-promotes the residual
        // stream exactly the way the real bug did and costs 11.8 -> 32.8 ms/token, which
        // makes the ablation measure the promotion instead of the kernel. Kept as a
        // comment rather than a footnote because it is a live trap: any substitute value
        // spliced into this block has to carry the block's own dtype.
        mx::array out = mx::zeros({B, S, cfg_.linear_num_value_heads, cfg_.linear_value_head_dim},
                                  inputs.dtype());
        if (ablation() != Ablate::deltakernel) {
            auto [o, state] =
                lmp::model::mlxl::gated_delta_update(q, k, v, a, b, a_log, dt_bias, cache.delta_state);
            out = o;
            cache.delta_state = state;
        }
        cache.offset += S;

        mx::array norm_w = weights_.get(p + "norm.weight");
        // Flash: sigmoid, not silu. 36 of 48 layers are GDN; silu(z)=z*sigmoid(z)
        // compounded across them is what collapsed skip-PLE decode to `!`. Do not
        // key this off cfg_.output_gate_type — that string defaults to "sigmoid"
        // on 27B/A3B, whose mlx-lm path is silu.
        const auto gact = cfg_.is_flash() ? lmp::model::mlxl::GatedNormAct::Sigmoid
                                          : lmp::model::mlxl::GatedNormAct::Silu;
        mx::array gated =
            lmp::model::mlxl::precise_rms_norm_gated(out, z, norm_w, cfg_.rms_norm_eps, gact);
        mx::array flat = mx::reshape(gated, {B, S, value_dim});
        return weights_.linear(flat, p + "out_proj");
    }

    // 0 disables it entirely (the reference path). 8 is the only width worth having:
    // 4-bit KV is a visible quality loss on a model that spends its context on code, and
    // the bandwidth it saves over 8-bit is small once the per-group scales are counted.
    static int kv_quant_bits() {
        static const int b = [] {
            const char* v = std::getenv("LMP_KV_BITS");
            return v == nullptr ? 0 : std::atoi(v);
        }();
        return b;
    }

    // Below this many tokens the fused bf16 SDPA wins and the KV term is not what decode
    // is spending its time on. Above it the cache read dominates. The default is where
    // the roofline arithmetic says the two cross on this machine, not a tuned constant --
    // re-derive it rather than nudging it if the model or the host changes.
    static int kv_quant_after() {
        static const int n = [] {
            const char* v = std::getenv("LMP_KV_QUANT_AFTER");
            return v == nullptr ? 8192 : std::atoi(v);
        }();
        return n;
    }

    mx::array forward_self_attn(const std::string& p, const mx::array& h, KVCache& cache,
                                QuantizedKVCache& qcache, int /*seq_len*/,
                                const mx::array* fuse_ln_gamma = nullptr,
                                IndexerCache* indexer = nullptr) {
        namespace ff = mx::fast;
        const int B = static_cast<int>(h.shape()[0]);
        const int L = static_cast<int>(h.shape()[1]);
        const int n_heads = cfg_.num_attention_heads;
        const int n_kv = cfg_.num_key_value_heads;
        const int head_dim = cfg_.head_dim;
        const int rotary_dim = static_cast<int>(head_dim * cfg_.partial_rotary_factor);

        mx::array q_out = mx::array(0.0f, h.dtype());
        mx::array k = mx::array(0.0f, h.dtype());
        mx::array v = mx::array(0.0f, h.dtype());
        const auto qkv_it = fused_out_cuts_.find(p + "qkv_proj");
        if (qkv_it != fused_out_cuts_.end() && fuse_ln_gamma == nullptr) {
            const mx::array qkv = weights_.linear(h, p + "qkv_proj");
            const int nq = qkv_it->second[0];
            const int nk = qkv_it->second[1];
            const std::vector<mx::array> parts =
                mx::split(qkv, mx::Shape{nq, nq + nk}, -1);
            q_out = parts[0];
            k = parts[1];
            v = parts[2];
        } else if (fuse_ln_gamma != nullptr && q4_fuse_rms_on()) {
            q_out = weights_.rms_linear(h, *fuse_ln_gamma, cfg_.rms_norm_eps, p + "q_proj");
            k = weights_.rms_linear(h, *fuse_ln_gamma, cfg_.rms_norm_eps, p + "k_proj");
            v = weights_.rms_linear(h, *fuse_ln_gamma, cfg_.rms_norm_eps, p + "v_proj");
        } else {
            q_out = weights_.linear(h, p + "q_proj");
            k = weights_.linear(h, p + "k_proj");
            v = weights_.linear(h, p + "v_proj");
        }

        mx::array queries = q_out;
        mx::array gate = mx::array(0.0f, h.dtype());
        const bool fused_gate = cfg_.attn_output_gate;
        if (fused_gate) {
            // q_proj emits [q_h0, g_h0, q_h1, g_h1, ...] per head — not [all_q, all_g].
            const mx::array q_heads = mx::reshape(q_out, {B, L, n_heads, head_dim * 2});
            const std::vector<mx::array> qg = mx::split(q_heads, 2, -1);
            queries = mx::reshape(qg[0], {B, L, n_heads * head_dim});
            gate = mx::reshape(qg[1], {B, L, n_heads * head_dim});
        }

        queries = mx::transpose(
            mx::reshape(ff::rms_norm(
                            mx::reshape(queries, {B, L, n_heads, head_dim}),
                            weights_.get(p + "q_norm.weight"), cfg_.rms_norm_eps),
                        {B, L, n_heads, head_dim}),
            {0, 2, 1, 3});
        k = mx::transpose(
            mx::reshape(ff::rms_norm(
                            mx::reshape(k, {B, L, n_kv, head_dim}),
                            weights_.get(p + "k_norm.weight"), cfg_.rms_norm_eps),
                        {B, L, n_kv, head_dim}),
            {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {B, L, n_kv, head_dim}), {0, 2, 1, 3});

        const int offset = cache.offset;
        queries = ff::rope(queries, rotary_dim, false, cfg_.rope_theta, 1.0f, offset);
        k = ff::rope(k, rotary_dim, false, cfg_.rope_theta, 1.0f, offset);

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::optional<mx::array> qsa_mask;
        if (indexer != nullptr && cfg_.indexer_head_dim > 0 && cfg_.indexer_budget > 0) {
            const int idx_rotary = std::min(rotary_dim, cfg_.indexer_head_dim);
            qsa_mask = qsa_indexer_mask(
                h, weights_, p + "indexer.", *indexer, offset, cfg_.indexer_n_heads,
                cfg_.indexer_kv_heads > 0 ? cfg_.indexer_kv_heads : 1, cfg_.indexer_head_dim,
                cfg_.indexer_budget,
                cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4, idx_rotary,
                cfg_.rope_theta, cfg_.rms_norm_eps);
        }

        // The conversion happens ONCE, on the first decode step of a long context, and
        // never during prefill. L == 1 is exactly the "prefill is behind us" signal: the
        // explicit [L, S] score matrix the quantized path builds is a loss against the
        // fused kernel's tiling at L = 2048 and a win at L = 1, so converting any earlier
        // would pay the cost in the phase that cannot use it.
        if (kv_quant_bits() > 0 && !qcache.active() && L == 1 &&
            cache.offset >= kv_quant_after()) {
            qcache.adopt_from(cache, /*gs=*/64, kv_quant_bits());
            // Drop the bf16 copy. Keeping it would leave the cache costing 20 KB/token
            // AND 10.6 KB/token on a host where the KV budget is what decides how long a
            // run can get -- the bandwidth win would be real and the memory win lost.
            // `offset` deliberately survives: it is still the token count everything reads.
            cache.keys.reset();
            cache.values.reset();
        }

        mx::array attn = mx::zeros({1}, h.dtype());
        if (qcache.active()) {
            auto [qk, qv] = qcache.update_and_fetch(k, v);
            cache.offset = qcache.offset;
            attn = quantized_sdpa(queries, qk, qv, scale, /*causal=*/L > 1,
                                  qcache.group_size, qcache.bits);
        } else {
            auto [k_cat, v_cat] = cache.update_and_fetch(k, v);
            // A single query attends to the whole cache, so decode wants no mask at all.
            // Say that with mask_mode rather than an empty mask argument: the type of that
            // argument changed between MLX 0.29 and 0.31 (vector<array> -> optional<array>),
            // and naming it here is what made the build version-specific.
            //
            // MLX 0.32 refuses mask_mode="causal" together with an array mask. QSA's keep
            // mask is already AND-causal, so pass it as mask_mode="array".
            if (qsa_mask.has_value()) {
                attn = ff::scaled_dot_product_attention(queries, k_cat, v_cat, scale, "array",
                                                        qsa_mask);
            } else {
                attn = ff::scaled_dot_product_attention(
                    queries, k_cat, v_cat, scale, (L > 1) ? "causal" : "");
            }
        }
        attn = mx::reshape(mx::transpose(attn, {0, 2, 1, 3}), {B, L, n_heads * head_dim});
        if (fused_gate) {
            attn = mx::multiply(attn, mx::sigmoid(gate));
        }
        return weights_.linear(attn, p + "o_proj");
    }
};

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MODELS_QWEN35_MOE_MODEL_HPP
