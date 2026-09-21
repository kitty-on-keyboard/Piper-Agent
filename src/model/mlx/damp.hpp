#ifndef LLM_MLX_DAMP_HPP
#define LLM_MLX_DAMP_HPP

// G2 DAMP (delta-state asymmetric mixed precision): protect Top-K_hi key channels
// per head (ranked offline by error-energy), INT8-affine the rest. Seam is
// SsmCache / forward_gated_delta — NOT QuantizedKVCache. See docs/G2_DAMP.md.
//
// Kill switch (default OFF — product path bit-identical when unset):
//   LMP_DAMP=1           enable pack/dequant around gated-delta state
//   LMP_DAMP_K_HI=16     protected channels per head (default 16)
//   LMP_DAMP_MASK=/path  mask artifact (required when enabled on the live path)
//
// This header is CPU-only so gate CI (LMP_WITH_MLX=OFF) can lock mask load +
// quant/dequant without Metal. MLX glue lives at the SsmCache call site.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lmp::model::mlxl {

// --- env -------------------------------------------------------------------

[[nodiscard]] inline bool parse_lmp_damp_flag(const char* v) noexcept {
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

[[nodiscard]] inline int parse_lmp_damp_k_hi(const char* v, int fallback = 16) noexcept {
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || (end != nullptr && *end != '\0') || n < 1 || n > 1024L) {
        return fallback;
    }
    return static_cast<int>(n);
}

// Cached getenv. Tests that need to inject values call parse_* / resolve_* directly.
[[nodiscard]] inline bool damp_enabled() {
    static const bool on = parse_lmp_damp_flag(std::getenv("LMP_DAMP"));
    return on;
}

[[nodiscard]] inline int damp_k_hi() {
    static const int k = parse_lmp_damp_k_hi(std::getenv("LMP_DAMP_K_HI"), 16);
    return k;
}

[[nodiscard]] inline std::optional<std::string> damp_mask_path_from_env() {
    static const std::optional<std::string> path = []() -> std::optional<std::string> {
        const char* v = std::getenv("LMP_DAMP_MASK");
        if (v == nullptr || v[0] == '\0') {
            return std::nullopt;
        }
        return std::string(v);
    }();
    return path;
}

// --- mask ------------------------------------------------------------------

// Per-layer, per-head protected channel indices into Dk (key-channel axis of
// delta_state [B, Hv, Dv, Dk]). Indices are unique, sorted ascending, length k_hi.
struct DampMask {
    std::string format; // must be "g2_damp_mask_v1"
    std::string source; // e.g. "synthetic_fixture" or "research_cal" — never invent rankings
    int k_hi{0};
    int dk{0};
    int num_layers{0};
    int num_heads{0}; // Hv
    // protected[layer][head] -> k_hi indices in [0, dk)
    std::vector<std::vector<std::vector<int>>> protected_idx;

    [[nodiscard]] bool valid() const noexcept {
        if (format != "g2_damp_mask_v1" || k_hi < 1 || dk < 1 || num_layers < 1 ||
            num_heads < 1 || k_hi > dk) {
            return false;
        }
        if (static_cast<int>(protected_idx.size()) != num_layers) {
            return false;
        }
        for (const auto& layer : protected_idx) {
            if (static_cast<int>(layer.size()) != num_heads) {
                return false;
            }
            for (const auto& head : layer) {
                if (static_cast<int>(head.size()) != k_hi) {
                    return false;
                }
                for (int d : head) {
                    if (d < 0 || d >= dk) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    [[nodiscard]] const std::vector<int>& indices(int layer, int head) const {
        return protected_idx[static_cast<std::size_t>(layer)][static_cast<std::size_t>(head)];
    }
};

namespace damp_detail {

// Tiny fixed-schema JSON pull for g2_damp_mask_v1. Not a general JSON parser.
struct Cursor {
    const std::string& s;
    std::size_t i{0};

    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) {
            ++i;
        }
    }
    [[nodiscard]] bool match(char c) {
        skip_ws();
        if (i < s.size() && s[i] == c) {
            ++i;
            return true;
        }
        return false;
    }
    [[nodiscard]] bool expect(char c) { return match(c); }

    [[nodiscard]] std::optional<std::string> string() {
        skip_ws();
        if (i >= s.size() || s[i] != '"') {
            return std::nullopt;
        }
        ++i;
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                out.push_back(s[i + 1]);
                i += 2;
                continue;
            }
            out.push_back(s[i++]);
        }
        if (i >= s.size() || s[i] != '"') {
            return std::nullopt;
        }
        ++i;
        return out;
    }

    [[nodiscard]] std::optional<long> integer() {
        skip_ws();
        if (i >= s.size()) {
            return std::nullopt;
        }
        char* end = nullptr;
        const long v = std::strtol(s.c_str() + i, &end, 10);
        if (end == s.c_str() + i) {
            return std::nullopt;
        }
        i = static_cast<std::size_t>(end - s.c_str());
        return v;
    }

    [[nodiscard]] bool skip_value() {
        skip_ws();
        if (i >= s.size()) {
            return false;
        }
        if (s[i] == '"') {
            return string().has_value();
        }
        if (s[i] == '{') {
            ++i;
            skip_ws();
            if (match('}')) {
                return true;
            }
            for (;;) {
                if (!string()) {
                    return false;
                }
                if (!expect(':')) {
                    return false;
                }
                if (!skip_value()) {
                    return false;
                }
                skip_ws();
                if (match('}')) {
                    return true;
                }
                if (!expect(',')) {
                    return false;
                }
            }
        }
        if (s[i] == '[') {
            ++i;
            skip_ws();
            if (match(']')) {
                return true;
            }
            for (;;) {
                if (!skip_value()) {
                    return false;
                }
                skip_ws();
                if (match(']')) {
                    return true;
                }
                if (!expect(',')) {
                    return false;
                }
            }
        }
        // number / true / false / null
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
               s[i] != ' ' && s[i] != '\n' && s[i] != '\r' && s[i] != '\t') {
            ++i;
        }
        return true;
    }
};

[[nodiscard]] inline bool parse_int_array(Cursor& c, std::vector<int>& out) {
    out.clear();
    if (!c.expect('[')) {
        return false;
    }
    c.skip_ws();
    if (c.match(']')) {
        return true;
    }
    for (;;) {
        const auto v = c.integer();
        if (!v) {
            return false;
        }
        out.push_back(static_cast<int>(*v));
        c.skip_ws();
        if (c.match(']')) {
            return true;
        }
        if (!c.expect(',')) {
            return false;
        }
    }
}

[[nodiscard]] inline bool normalize_head(std::vector<int>& idx, int k_hi, int dk) {
    if (static_cast<int>(idx.size()) != k_hi) {
        return false;
    }
    std::vector<int> sorted = idx;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i] < 0 || sorted[i] >= dk) {
            return false;
        }
        if (i > 0 && sorted[i] == sorted[i - 1]) {
            return false;
        }
    }
    idx = std::move(sorted);
    return true;
}

} // namespace damp_detail

// Parse a g2_damp_mask_v1 JSON document from memory.
[[nodiscard]] inline bool parse_damp_mask_json(const std::string& json, DampMask& out,
                                               std::string* err = nullptr) {
    out = DampMask{};
    damp_detail::Cursor c{json};
    if (!c.expect('{')) {
        if (err) {
            *err = "expected object";
        }
        return false;
    }
    std::vector<std::pair<int, std::vector<std::vector<int>>>> layer_rows;
    c.skip_ws();
    if (c.match('}')) {
        if (err) {
            *err = "empty object";
        }
        return false;
    }
    for (;;) {
        const auto key = c.string();
        if (!key || !c.expect(':')) {
            if (err) {
                *err = "bad key";
            }
            return false;
        }
        if (*key == "format") {
            auto v = c.string();
            if (!v) {
                return false;
            }
            out.format = *v;
        } else if (*key == "source") {
            auto v = c.string();
            if (!v) {
                return false;
            }
            out.source = *v;
        } else if (*key == "k_hi") {
            auto v = c.integer();
            if (!v) {
                return false;
            }
            out.k_hi = static_cast<int>(*v);
        } else if (*key == "dk") {
            auto v = c.integer();
            if (!v) {
                return false;
            }
            out.dk = static_cast<int>(*v);
        } else if (*key == "num_layers") {
            auto v = c.integer();
            if (!v) {
                return false;
            }
            out.num_layers = static_cast<int>(*v);
        } else if (*key == "num_heads") {
            auto v = c.integer();
            if (!v) {
                return false;
            }
            out.num_heads = static_cast<int>(*v);
        } else if (*key == "layers") {
            if (!c.expect('[')) {
                return false;
            }
            c.skip_ws();
            if (!c.match(']')) {
                for (;;) {
                    if (!c.expect('{')) {
                        return false;
                    }
                    int layer_id = -1;
                    std::vector<std::vector<int>> heads;
                    c.skip_ws();
                    if (!c.match('}')) {
                        for (;;) {
                            const auto lk = c.string();
                            if (!lk || !c.expect(':')) {
                                return false;
                            }
                            if (*lk == "layer") {
                                auto v = c.integer();
                                if (!v) {
                                    return false;
                                }
                                layer_id = static_cast<int>(*v);
                            } else if (*lk == "heads") {
                                if (!c.expect('[')) {
                                    return false;
                                }
                                c.skip_ws();
                                if (!c.match(']')) {
                                    for (;;) {
                                        std::vector<int> one;
                                        if (!damp_detail::parse_int_array(c, one)) {
                                            return false;
                                        }
                                        heads.push_back(std::move(one));
                                        c.skip_ws();
                                        if (c.match(']')) {
                                            break;
                                        }
                                        if (!c.expect(',')) {
                                            return false;
                                        }
                                    }
                                }
                            } else if (!c.skip_value()) {
                                return false;
                            }
                            c.skip_ws();
                            if (c.match('}')) {
                                break;
                            }
                            if (!c.expect(',')) {
                                return false;
                            }
                        }
                    }
                    if (layer_id < 0) {
                        if (err) {
                            *err = "layer entry missing layer id";
                        }
                        return false;
                    }
                    layer_rows.emplace_back(layer_id, std::move(heads));
                    c.skip_ws();
                    if (c.match(']')) {
                        break;
                    }
                    if (!c.expect(',')) {
                        return false;
                    }
                }
            }
        } else if (!c.skip_value()) {
            return false;
        }
        c.skip_ws();
        if (c.match('}')) {
            break;
        }
        if (!c.expect(',')) {
            if (err) {
                *err = "expected comma or end";
            }
            return false;
        }
    }

    if (out.num_layers < 1 || out.num_heads < 1) {
        if (err) {
            *err = "num_layers/num_heads required";
        }
        return false;
    }
    out.protected_idx.assign(static_cast<std::size_t>(out.num_layers),
                             std::vector<std::vector<int>>(static_cast<std::size_t>(out.num_heads)));
    for (auto& row : layer_rows) {
        if (row.first < 0 || row.first >= out.num_layers) {
            if (err) {
                *err = "layer id out of range";
            }
            return false;
        }
        if (static_cast<int>(row.second.size()) != out.num_heads) {
            if (err) {
                *err = "head count mismatch";
            }
            return false;
        }
        for (int h = 0; h < out.num_heads; ++h) {
            if (!damp_detail::normalize_head(row.second[static_cast<std::size_t>(h)], out.k_hi,
                                             out.dk)) {
                if (err) {
                    *err = "bad protected indices";
                }
                return false;
            }
        }
        out.protected_idx[static_cast<std::size_t>(row.first)] = std::move(row.second);
    }
    // Every layer must have been provided (no silent empty rankings).
    for (int L = 0; L < out.num_layers; ++L) {
        for (int h = 0; h < out.num_heads; ++h) {
            if (static_cast<int>(out.protected_idx[static_cast<std::size_t>(L)]
                                                     [static_cast<std::size_t>(h)]
                                                         .size()) != out.k_hi) {
                if (err) {
                    *err = "incomplete layer coverage";
                }
                return false;
            }
        }
    }
    if (!out.valid()) {
        if (err) {
            *err = "mask failed validation";
        }
        return false;
    }
    return true;
}

[[nodiscard]] inline bool load_damp_mask_file(const std::string& path, DampMask& out,
                                              std::string* err = nullptr) {
    std::ifstream in(path);
    if (!in) {
        if (err) {
            *err = "cannot open " + path;
        }
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_damp_mask_json(ss.str(), out, err);
}

// Synthetic mask for tests only — sequential channel indices, NOT measured energy ranks.
[[nodiscard]] inline DampMask make_synthetic_damp_mask(int num_layers, int num_heads, int dk,
                                                       int k_hi) {
    DampMask m;
    m.format = "g2_damp_mask_v1";
    m.source = "synthetic_fixture";
    m.k_hi = k_hi;
    m.dk = dk;
    m.num_layers = num_layers;
    m.num_heads = num_heads;
    m.protected_idx.resize(static_cast<std::size_t>(num_layers));
    for (int L = 0; L < num_layers; ++L) {
        m.protected_idx[static_cast<std::size_t>(L)].resize(static_cast<std::size_t>(num_heads));
        for (int h = 0; h < num_heads; ++h) {
            auto& idx = m.protected_idx[static_cast<std::size_t>(L)][static_cast<std::size_t>(h)];
            idx.resize(static_cast<std::size_t>(k_hi));
            // Offset by head so tiers differ; still synthetic, not Research cal.
            for (int i = 0; i < k_hi; ++i) {
                idx[static_cast<std::size_t>(i)] = (h * k_hi + i) % dk;
            }
            std::sort(idx.begin(), idx.end());
        }
    }
    return m;
}

// --- pack / unpack (CPU, plain INT8 compressed tier) -----------------------

// Mixed-precision packing of one delta_state buffer [B, Hv, Dv, Dk] in row-major.
// Protected channels stored float32; remaining channels INT8 affine with one scale
// per (b, h, dv) over the compressed axis only.
struct DampPackedDelta {
    int B{0};
    int Hv{0};
    int Dv{0};
    int Dk{0};
    int k_hi{0};
    int layer{-1};
    std::vector<float> hi;    // [B, Hv, Dv, k_hi]
    std::vector<std::int8_t> lo; // [B, Hv, Dv, Dk - k_hi]
    std::vector<float> scale; // [B, Hv, Dv] — absmax/127 over compressed channels

    [[nodiscard]] bool empty() const noexcept { return hi.empty() && lo.empty(); }

    [[nodiscard]] std::size_t packed_bytes() const noexcept {
        return hi.size() * sizeof(float) + lo.size() * sizeof(std::int8_t) +
               scale.size() * sizeof(float);
    }

    [[nodiscard]] std::size_t fp32_bytes() const noexcept {
        return static_cast<std::size_t>(B) * static_cast<std::size_t>(Hv) *
               static_cast<std::size_t>(Dv) * static_cast<std::size_t>(Dk) * sizeof(float);
    }
};

[[nodiscard]] inline bool damp_quantize_cpu(const float* state, int B, int Hv, int Dv, int Dk,
                                            const DampMask& mask, int layer, DampPackedDelta& out,
                                            std::string* err = nullptr) {
    out = DampPackedDelta{};
    if (state == nullptr || B < 1 || Hv < 1 || Dv < 1 || Dk < 1) {
        if (err) {
            *err = "bad dims";
        }
        return false;
    }
    if (!mask.valid() || layer < 0 || layer >= mask.num_layers || Hv != mask.num_heads ||
        Dk != mask.dk) {
        if (err) {
            *err = "mask/state shape mismatch";
        }
        return false;
    }
    const int k_hi = mask.k_hi;
    const int k_lo = Dk - k_hi;
    out.B = B;
    out.Hv = Hv;
    out.Dv = Dv;
    out.Dk = Dk;
    out.k_hi = k_hi;
    out.layer = layer;
    out.hi.assign(static_cast<std::size_t>(B) * Hv * Dv * k_hi, 0.f);
    out.lo.assign(static_cast<std::size_t>(B) * Hv * Dv * k_lo, 0);
    out.scale.assign(static_cast<std::size_t>(B) * Hv * Dv, 0.f);

    // Per-head boolean: is channel d protected?
    std::vector<std::vector<char>> is_hi(static_cast<std::size_t>(Hv),
                                         std::vector<char>(static_cast<std::size_t>(Dk), 0));
    std::vector<std::vector<int>> hi_order(static_cast<std::size_t>(Hv));
    std::vector<std::vector<int>> lo_order(static_cast<std::size_t>(Hv));
    for (int h = 0; h < Hv; ++h) {
        const auto& prot = mask.indices(layer, h);
        hi_order[static_cast<std::size_t>(h)] = prot;
        for (int d : prot) {
            is_hi[static_cast<std::size_t>(h)][static_cast<std::size_t>(d)] = 1;
        }
        lo_order[static_cast<std::size_t>(h)].reserve(static_cast<std::size_t>(k_lo));
        for (int d = 0; d < Dk; ++d) {
            if (!is_hi[static_cast<std::size_t>(h)][static_cast<std::size_t>(d)]) {
                lo_order[static_cast<std::size_t>(h)].push_back(d);
            }
        }
        if (static_cast<int>(lo_order[static_cast<std::size_t>(h)].size()) != k_lo) {
            if (err) {
                *err = "lo_order size";
            }
            return false;
        }
    }

    auto at = [&](int b, int h, int dv, int d) -> const float& {
        const std::size_t i =
            ((((static_cast<std::size_t>(b) * Hv + h) * Dv + dv) * Dk) + d);
        return state[i];
    };

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Hv; ++h) {
            for (int dv = 0; dv < Dv; ++dv) {
                const std::size_t base_hi =
                    ((((static_cast<std::size_t>(b) * Hv + h) * Dv + dv) * k_hi));
                const std::size_t base_lo =
                    ((((static_cast<std::size_t>(b) * Hv + h) * Dv + dv) * k_lo));
                const std::size_t scale_i =
                    ((static_cast<std::size_t>(b) * Hv + h) * Dv + dv);

                for (int i = 0; i < k_hi; ++i) {
                    const int d = hi_order[static_cast<std::size_t>(h)][static_cast<std::size_t>(i)];
                    out.hi[base_hi + static_cast<std::size_t>(i)] = at(b, h, dv, d);
                }

                float absmax = 0.f;
                for (int i = 0; i < k_lo; ++i) {
                    const int d = lo_order[static_cast<std::size_t>(h)][static_cast<std::size_t>(i)];
                    absmax = std::max(absmax, std::fabs(at(b, h, dv, d)));
                }
                const float s = absmax > 0.f ? absmax / 127.f : 0.f;
                out.scale[scale_i] = s;
                for (int i = 0; i < k_lo; ++i) {
                    const int d = lo_order[static_cast<std::size_t>(h)][static_cast<std::size_t>(i)];
                    float q = s > 0.f ? at(b, h, dv, d) / s : 0.f;
                    q = std::max(-127.f, std::min(127.f, std::round(q)));
                    out.lo[base_lo + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(q);
                }
            }
        }
    }
    return true;
}

[[nodiscard]] inline bool damp_dequantize_cpu(const DampPackedDelta& packed, const DampMask& mask,
                                              float* out, std::string* err = nullptr) {
    if (out == nullptr || packed.empty()) {
        if (err) {
            *err = "empty packed";
        }
        return false;
    }
    if (!mask.valid() || packed.layer < 0 || packed.layer >= mask.num_layers ||
        packed.Hv != mask.num_heads || packed.Dk != mask.dk || packed.k_hi != mask.k_hi) {
        if (err) {
            *err = "mask/packed mismatch";
        }
        return false;
    }
    const int B = packed.B;
    const int Hv = packed.Hv;
    const int Dv = packed.Dv;
    const int Dk = packed.Dk;
    const int k_hi = packed.k_hi;
    const int k_lo = Dk - k_hi;
    const int layer = packed.layer;

    std::vector<std::vector<char>> is_hi(static_cast<std::size_t>(Hv),
                                         std::vector<char>(static_cast<std::size_t>(Dk), 0));
    std::vector<std::vector<int>> hi_order(static_cast<std::size_t>(Hv));
    std::vector<std::vector<int>> lo_order(static_cast<std::size_t>(Hv));
    for (int h = 0; h < Hv; ++h) {
        const auto& prot = mask.indices(layer, h);
        hi_order[static_cast<std::size_t>(h)] = prot;
        for (int d : prot) {
            is_hi[static_cast<std::size_t>(h)][static_cast<std::size_t>(d)] = 1;
        }
        for (int d = 0; d < Dk; ++d) {
            if (!is_hi[static_cast<std::size_t>(h)][static_cast<std::size_t>(d)]) {
                lo_order[static_cast<std::size_t>(h)].push_back(d);
            }
        }
    }

    const std::size_t n =
        static_cast<std::size_t>(B) * Hv * Dv * Dk;
    std::fill(out, out + n, 0.f);

    auto at = [&](int b, int h, int dv, int d) -> float& {
        const std::size_t i =
            ((((static_cast<std::size_t>(b) * Hv + h) * Dv + dv) * Dk) + d);
        return out[i];
    };

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Hv; ++h) {
            for (int dv = 0; dv < Dv; ++dv) {
                const std::size_t base_hi =
                    ((((static_cast<std::size_t>(b) * Hv + h) * Dv + dv) * k_hi));
                const std::size_t base_lo =
                    ((((static_cast<std::size_t>(b) * Hv + h) * Dv + dv) * k_lo));
                const std::size_t scale_i =
                    ((static_cast<std::size_t>(b) * Hv + h) * Dv + dv);
                const float s = packed.scale[scale_i];

                for (int i = 0; i < k_hi; ++i) {
                    const int d = hi_order[static_cast<std::size_t>(h)][static_cast<std::size_t>(i)];
                    at(b, h, dv, d) = packed.hi[base_hi + static_cast<std::size_t>(i)];
                }
                for (int i = 0; i < k_lo; ++i) {
                    const int d = lo_order[static_cast<std::size_t>(h)][static_cast<std::size_t>(i)];
                    at(b, h, dv, d) =
                        static_cast<float>(packed.lo[base_lo + static_cast<std::size_t>(i)]) * s;
                }
            }
        }
    }
    return true;
}

// Identity helper for "DAMP off" checks: copies FP32 state without touching tiers.
inline void damp_passthrough_cpu(const float* in, float* out, std::size_t n) {
    if (in == nullptr || out == nullptr || n == 0) {
        return;
    }
    if (in != out) {
        std::memcpy(out, in, n * sizeof(float));
    }
}

} // namespace lmp::model::mlxl

#endif // LLM_MLX_DAMP_HPP
