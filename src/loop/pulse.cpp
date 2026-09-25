#include "src/loop/pulse.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>

namespace lmp::loop {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

[[nodiscard]] bool is_hindsight_key(std::string_view key) noexcept {
    // Post-decision / label leakage fields that must not reach the gate.
    return key == "next" || key == "outcome" || key == "gold" || key == "label" ||
           key == "chosen" || key == "decision" || key == "answer" || key == "y_true" ||
           key == "y_pred";
}

// Drop `key=value` tokens (and key="..." / key='...') when key is hindsight.
void erase_hindsight_assignments(std::string& s) {
    std::size_t i = 0;
    while (i < s.size()) {
        // Find a potential key start: letter/_ after start or non-alnum.
        if (!(std::isalpha(static_cast<unsigned char>(s[i])) || s[i] == '_')) {
            ++i;
            continue;
        }
        const std::size_t key_begin = i;
        while (i < s.size() &&
               (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) {
            ++i;
        }
        const std::string_view key(s.data() + key_begin, i - key_begin);
        if (i >= s.size() || s[i] != '=' || !is_hindsight_key(key)) {
            continue;
        }
        std::size_t val_end = i + 1;
        if (val_end < s.size() && (s[val_end] == '"' || s[val_end] == '\'')) {
            const char q = s[val_end];
            ++val_end;
            while (val_end < s.size() && s[val_end] != q && s[val_end] != '\n') {
                ++val_end;
            }
            if (val_end < s.size() && s[val_end] == q) {
                ++val_end;
            }
        } else {
            while (val_end < s.size() && !std::isspace(static_cast<unsigned char>(s[val_end])) &&
                   s[val_end] != ',' && s[val_end] != ';') {
                ++val_end;
            }
        }
        s.erase(key_begin, val_end - key_begin);
        i = key_begin;
    }
}

} // namespace

std::string format_pulse_features(const PulseFeatures& f) {
    std::ostringstream out;
    out << "consec=" << f.consec << " streak=" << f.streak << " prompt_tok=" << f.prompt_tok
        << " reread_max=" << f.reread_max << '\n'
        << "think=" << f.think << " text=" << f.text << " tool_tok=" << f.tool_tok
        << " why=" << (f.why.empty() ? "none" : f.why);
    return out.str();
}

std::string strip_pulse_hindsight(std::string_view text) {
    std::string out(text);
    erase_hindsight_assignments(out);
    // Also drop lines that are ONLY a hindsight trail after stripping.
    std::string cleaned;
    cleaned.reserve(out.size());
    std::size_t line_start = 0;
    while (line_start <= out.size()) {
        const std::size_t nl = out.find('\n', line_start);
        const std::size_t line_end = nl == std::string::npos ? out.size() : nl;
        std::string_view line(out.data() + line_start, line_end - line_start);
        // Trim spaces for emptiness check.
        std::size_t a = 0;
        std::size_t b = line.size();
        while (a < b && std::isspace(static_cast<unsigned char>(line[a]))) {
            ++a;
        }
        while (b > a && std::isspace(static_cast<unsigned char>(line[b - 1]))) {
            --b;
        }
        if (a < b) {
            if (!cleaned.empty() && cleaned.back() != '\n') {
                cleaned.push_back('\n');
            }
            cleaned.append(line.data() + a, b - a);
        }
        if (nl == std::string::npos) {
            break;
        }
        line_start = nl + 1;
    }
    return cleaned;
}

PulseChoiceOrder shuffle_t1_choices(std::uint64_t seed) noexcept {
    PulseChoiceOrder order = {PulseChoice::ForceTool, PulseChoice::Nudge, PulseChoice::Stall,
                              PulseChoice::Compact};
    std::uint64_t state = seed ^ 0x50554C5345ULL; // "PULSE" ASCII salt
    // Avoid all-zero degenerate: seed 0 still mixes via salt.
    if (state == 0) {
        state = 0xC0FFEEULL;
    }
    for (std::size_t i = kPulseT1OptionCount; i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(splitmix64(state) % i);
        std::swap(order[i - 1], order[j]);
    }
    return order;
}

std::string pulse_t1_forced_prefix(const PulseChoiceOrder& order, const PulseFeatures& features,
                                   std::string_view mission) {
    std::ostringstream out;
    out << "Pick exactly one harness next-step. Reply with only the code letter.\n\n";
    for (std::size_t i = 0; i < kPulseT1OptionCount; ++i) {
        out << kPulseT1Letters[i] << " = " << pulse_choice_def(order[i]) << '\n';
    }
    out << "\nFeatures:\n" << format_pulse_features(features);
    if (!mission.empty()) {
        out << "\nMission: " << mission;
    }
    out << "\n";
    return out.str();
}

PulseOptionIds resolve_t1_letter_ids(const model::QwenTokenizer& tok,
                                     const PulseChoiceOrder& order) {
    PulseOptionIds out;
    out.encoding = "letter";
    out.ids.reserve(kPulseT1OptionCount);
    out.choices.reserve(kPulseT1OptionCount);
    for (std::size_t i = 0; i < kPulseT1OptionCount; ++i) {
        const char letter[2] = {kPulseT1Letters[i], '\0'};
        const std::vector<model::TokenId> ids = tok.encode_content(letter);
        if (ids.empty()) {
            out.error = std::string("pulse: empty encode for letter ") + letter;
            out.ids.clear();
            out.choices.clear();
            return out;
        }
        // Prefer single-token letters; still take first id if BPE splits (rare for A-D).
        const model::TokenId first = ids.front();
        for (std::size_t j = 0; j < out.ids.size(); ++j) {
            if (out.ids[j] == first) {
                out.error = std::string("pulse: letter-token collision between ") +
                            kPulseT1Letters[j] + " and " + kPulseT1Letters[i];
                out.ids.clear();
                out.choices.clear();
                return out;
            }
        }
        out.ids.push_back(first);
        out.choices.push_back(order[i]);
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
                                          const std::vector<PulseChoice>& choices,
                                          double latency_ms) {
    PulseMicroResult r;
    r.latency_ms = latency_ms;
    r.kv_reuse = "n/a";
    r.encoding = "letter";
    if (option_ids.empty() || choices.size() != option_ids.size()) {
        r.error = "pulse: empty or mismatched option/choice sets";
        return r;
    }
    if (option_ids.size() > kPulseT1OptionCount) {
        r.error = "pulse: too many options";
        return r;
    }

    std::ostringstream order_ss;
    for (std::size_t i = 0; i < choices.size(); ++i) {
        if (i > 0) {
            order_ss << ',';
        }
        order_ss << pulse_choice_name(choices[i]);
    }
    r.order = order_ss.str();

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
        if (r.p_vec[i] > r.p_vec[best]) {
            best = i;
        }
    }
    r.choice = choices[best];
    r.choice_name = std::string(pulse_choice_name(r.choice));
    r.letter = best < kPulseT1OptionCount ? kPulseT1Letters[best] : '?';
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
