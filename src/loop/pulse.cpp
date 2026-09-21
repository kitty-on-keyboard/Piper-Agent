#include "src/loop/pulse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lmp::loop {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

[[nodiscard]] PulseChoice choice_from_index(std::size_t i) noexcept {
    switch (i) {
        case 0:
            return PulseChoice::ForceTool;
        case 1:
            return PulseChoice::Nudge;
        case 2:
            return PulseChoice::Stall;
        case 3:
            return PulseChoice::Compact;
        default:
            return PulseChoice::Nudge;
    }
}

} // namespace

PulseOptionIds resolve_t1_option_ids(const model::QwenTokenizer& tok) {
    PulseOptionIds out;
    out.ids.reserve(kPulseT1OptionCount);
    const char* const* names = pulse_t1_option_names();
    for (std::size_t i = 0; i < kPulseT1OptionCount; ++i) {
        const std::vector<model::TokenId> ids = tok.encode_content(names[i]);
        if (ids.empty()) {
            out.error = std::string("pulse: empty encode for option ") + names[i];
            out.ids.clear();
            return out;
        }
        const model::TokenId first = ids.front();
        for (std::size_t j = 0; j < out.ids.size(); ++j) {
            if (out.ids[j] == first) {
                out.error = std::string("pulse: first-token collision between ") +
                            names[j] + " and " + names[i];
                out.ids.clear();
                return out;
            }
        }
        out.ids.push_back(first);
    }
    return out;
}

model::TokenMask pulse_option_mask(std::size_t vocab_size,
                                   const std::vector<model::TokenId>& option_ids) {
    model::TokenMask m(vocab_size);
    for (model::TokenId id : option_ids) {
        m.allow(id);
    }
    return m;
}

bool pulse_mask_escape(const model::TokenMask& mask, model::TokenId id) noexcept {
    return !mask.allows(id);
}

PulseMicroResult pulse_decode_from_logits(const std::vector<float>& logits,
                                          const std::vector<model::TokenId>& option_ids,
                                          double latency_ms) {
    PulseMicroResult r;
    r.latency_ms = latency_ms;
    r.kv_reuse = "n/a";
    if (option_ids.empty()) {
        r.error = "pulse: empty option set";
        return r;
    }
    if (option_ids.size() > kPulseT1OptionCount) {
        r.error = "pulse: too many options";
        return r;
    }

    std::vector<float> scores(option_ids.size(), kNegInf);
    bool any = false;
    for (std::size_t i = 0; i < option_ids.size(); ++i) {
        const model::TokenId id = option_ids[i];
        if (id < 0 || static_cast<std::size_t>(id) >= logits.size()) {
            continue;
        }
        scores[i] = logits[static_cast<std::size_t>(id)];
        any = true;
    }
    if (!any) {
        r.error = "pulse: no option id in logits range";
        return r;
    }

    float max_logit = kNegInf;
    for (float s : scores) {
        if (s > max_logit) {
            max_logit = s;
        }
    }
    if (max_logit == kNegInf) {
        r.error = "pulse: all option logits masked";
        return r;
    }

    r.p_vec.assign(option_ids.size(), 0.0F);
    float total = 0.0F;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] == kNegInf) {
            continue;
        }
        const float p = std::exp(scores[i] - max_logit);
        r.p_vec[i] = p;
        total += p;
    }
    if (total <= 0.0F) {
        r.error = "pulse: zero mass over options";
        return r;
    }
    for (float& p : r.p_vec) {
        p /= total;
    }

    std::size_t best = 0;
    for (std::size_t i = 1; i < r.p_vec.size(); ++i) {
        // Strict >: lowest index wins ties (stable, matches sampler greedy tie-break).
        if (r.p_vec[i] > r.p_vec[best]) {
            best = i;
        }
    }
    r.choice = choice_from_index(best);
    r.choice_name = std::string(pulse_choice_name(r.choice));
    r.p = r.p_vec[best];
    r.ok = true;
    return r;
}

PulsePolicy apply_pulse_policy(const PulseMicroResult& result, float p_min) noexcept {
    if (!result.ok || result.p < p_min) {
        return PulsePolicy::Fallback;
    }
    switch (result.choice) {
        case PulseChoice::ForceTool:
            return PulsePolicy::ForceTool;
        case PulseChoice::Nudge:
            return PulsePolicy::Nudge;
        case PulseChoice::Stall:
            return PulsePolicy::Stall;
        case PulseChoice::Compact:
            return PulsePolicy::Compact;
    }
    return PulsePolicy::Fallback;
}

} // namespace lmp::loop
