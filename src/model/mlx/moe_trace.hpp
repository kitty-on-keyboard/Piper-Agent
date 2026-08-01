#pragma once
//
// Expert-routing trace capture (diagnostic only, OFF unless LMP_MOE_TRACE is set).
//
// Which 8 of 256 experts each token selects, per layer, is the one thing about this model
// nobody has measured. `lmp_diag moestream` prices the routed-expert stream in aggregate --
// 566,231,040 bytes a token -- but says nothing about WHICH experts, and therefore nothing
// about whether two consecutive tokens re-read the same weights or disjoint ones. That
// distinction decides whether speculative decoding can work on this model at all: verifying
// k drafted tokens costs one pass over the UNION of their experts, so perfect reuse makes it
// nearly free and perfect scattering makes it k times the traffic.
//
// Enabled by an env var and gated behind a branch on a `static const bool` read once, so a
// normal run pays one predictable, always-false test per MoE layer. It is not compiled out
// because the whole point is to be able to take this measurement on the real checkpoint
// without a special build.
//
// CAPTURE FORCES A SYNC. Reading the routing indices means evaluating them mid-graph, which
// serialises a step that is otherwise pipelined. A traced run is much slower than an
// untraced one and its tok/s figures are meaningless -- the ROUTING is what is being
// collected, and routing is not affected by when it is read.
//
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace lmp::model::mlxl {

class MoeTrace {
  public:
    static MoeTrace& instance() {
        static MoeTrace t;
        return t;
    }

    [[nodiscard]] bool enabled() const noexcept { return file_ != nullptr; }

    // The token being processed this step. Recorded alongside the expert set because the
    // question that decides expert-aware drafting is how much of the routing is determined
    // by the TOKEN versus by the context -- and that cannot be asked of a trace that only
    // says which experts fired, not what fired them.
    void set_token(int token) noexcept { token_ = token; }

    // Call once per MoE layer per token, with that token's chosen expert ids.
    void record(int layer, const int* experts, std::size_t count) {
        if (file_ == nullptr) {
            return;
        }
        // The step advances when layer 0 comes round again: forward_moe is called once per
        // layer per token, in order, so layer 0 is the token boundary.
        if (layer <= first_layer_seen_) {
            if (seen_any_) {
                ++step_;
            }
            first_layer_seen_ = layer;
            seen_any_ = true;
        }
        line_.clear();
        line_ += "{\"step\":";
        line_ += std::to_string(step_);
        line_ += ",\"token\":";
        line_ += std::to_string(token_);
        line_ += ",\"layer\":";
        line_ += std::to_string(layer);
        line_ += ",\"experts\":[";
        for (std::size_t i = 0; i < count; ++i) {
            if (i != 0) {
                line_ += ',';
            }
            line_ += std::to_string(experts[i]);
        }
        line_ += "]}\n";
        std::fwrite(line_.data(), 1, line_.size(), file_);
    }

  private:
    MoeTrace() {
        const char* path = std::getenv("LMP_MOE_TRACE");
        if (path != nullptr && *path != '\0') {
            file_ = std::fopen(path, "wb");
        }
    }
    ~MoeTrace() {
        if (file_ != nullptr) {
            std::fclose(file_);
        }
    }

    std::FILE* file_ = nullptr;
    std::string line_;
    long step_ = 0;
    int token_ = -1;
    int first_layer_seen_ = 1 << 30;
    bool seen_any_ = false;
};

} // namespace lmp::model::mlxl
