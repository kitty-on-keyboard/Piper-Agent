// Neutral scoreboard for the SpecVerifier cook-off (Jules round 2, Brief C). Compiled once
// per entrant against that entrant's own include/ and src/, using ONLY the public API the
// brief specified.
//
// ---------------------------------------------------------------------------------------
// READ THIS BEFORE READING A SCORE: the brief contains a contradiction, found by exact
// enumeration while building this harness, and it changes what "correct" can mean here.
//
// Brief C asks for two things that cannot both hold:
//
//   1. A PROCEDURE. On rejecting draft token t, "subtract `q` from the rejected token's
//      mass only, clamp negatives to zero, renormalise, sample."
//   2. A PROPERTY. "The histogram must converge to the target distribution ... If
//      acceptance is implemented naively this test fails and nothing else does."
//
// Textbook speculative sampling draws the replacement from norm((p - q)_+), which needs the
// drafter's WHOLE row. The brief hands the implementer only the scalar q(t), so it
// prescribes the reduction in (1). That reduction is NOT distribution-preserving in
// general. Enumerated exactly over 20,000 random (p,q) pairs, no sampling involved:
//
//     textbook full-row residual : worst TV from target = 5.6e-17   (exact)
//     Brief C scalar reduction   : worst TV from target = 2.3e-01   (not exact)
//
// Worked case, checkable by hand: p=(.5,.3,.2), q=(.8,.1,.1). The full-row residual commits
// (.5,.3,.2). The scalar reduction commits (.5,.28,.22) -- off by 0.02.
//
// So an entrant that follows the procedure fails the property, and an entrant that delivers
// the property did not follow the procedure. Scoring either one against the target
// distribution alone would rank them by which half of the brief they guessed at.
//
// THE MITIGATION, and why this is not fatal for LM_Pipe. The two coincide exactly when the
// drafter is DETERMINISTIC -- q(proposed) = 1 -- which is precisely this project's regime:
// SuffixProposer proposes a concrete continuation from matched history, it is not a
// sampling model. Enumerated the same way:
//
//     q(proposed)=1.00 -> worst TV 8.3e-17      q(proposed)=0.90 -> worst TV 5.5e-02
//     q(proposed)=0.99 -> worst TV 6.6e-03      q(proposed)=0.75 -> worst TV 1.2e-01
//     q(proposed)=0.95 -> worst TV 3.1e-02      q(proposed)=0.50 -> worst TV 1.7e-01
//
// This scoreboard therefore measures BOTH regimes and reports them separately:
//
//   tv_det_*   -- deterministic drafter (q=1). Here the brief is self-consistent, both
//                 readings agree, and any deviation from the target IS the entrant's fault.
//                 THIS IS THE COLUMN THAT DECIDES ADOPTION, because it is the column that
//                 matches how LM_Pipe will actually call this code.
//   tv_soft_*  -- soft drafter (q < 1). Reported for information. An entrant scoring near
//                 zero here implemented the full-row residual (deviating from the brief,
//                 arguably correctly); one scoring near the enumerated bias implemented the
//                 brief as written. Neither is disqualifying; the difference is recorded.
// ---------------------------------------------------------------------------------------
//
// Remaining figures of merit:
//   floor       -- `accepted` is never empty, for any input. The guaranteed-progress
//                  property; without it a decode loop can hang.
//   perfect     -- perfect drafter (draft == argmax, q == p) accepts every token:
//                  accepted.size() == k+1 and accepted_drafts == k.
//   determinism -- same seed and inputs, identical Result, across fresh instances.
//   degenerate  -- empty draft, q == 0, single-token vocabulary, one-hot rows: no crash,
//                  no out-of-range token, floor still held.
//   accept_rate -- mean accepted_drafts under a good drafter. A verifier that rejects
//                  everything is "correct" and useless; this is the efficiency tiebreak.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

#include <spec_verifier.hpp>

using spec::Result;
using spec::SpecVerifier;
using spec::TokenId;

namespace {

constexpr std::size_t kVocab = 12;
constexpr std::size_t kDraft = 4;
constexpr int kRuns = 400000;

using Dist = std::vector<float>;

double tv(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += std::fabs(a[i] - b[i]);
    }
    return 0.5 * s;
}

Dist normalised(std::vector<double> w) {
    double sum = 0.0;
    for (double v : w) {
        sum += v;
    }
    Dist out(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) {
        out[i] = static_cast<float>(w[i] / sum);
    }
    return out;
}

Dist target_row(std::mt19937_64& rng) {
    std::vector<double> w(kVocab);
    for (double& v : w) {
        v = 1.0 + static_cast<double>(rng() % 400);
    }
    return normalised(std::move(w));
}

// The exact first-committed-token distribution the brief's scalar reduction induces, given
// the target row p and the drafter row q. Closed form -- the first token depends only on
// position 0, so no simulation is needed to know what a compliant entrant must produce.
std::vector<double> expected_first_token(const Dist& p, const Dist& q) {
    std::vector<double> out(p.size(), 0.0);
    for (std::size_t t = 0; t < p.size(); ++t) {
        out[t] += std::min(static_cast<double>(q[t]), static_cast<double>(p[t]));
    }
    for (std::size_t t = 0; t < p.size(); ++t) {
        const double rej = std::max(0.0, static_cast<double>(q[t]) - static_cast<double>(p[t]));
        if (rej <= 0.0) {
            continue;
        }
        const double z = 1.0 - static_cast<double>(p[t]);
        if (z <= 0.0) {
            continue;
        }
        for (std::size_t j = 0; j < p.size(); ++j) {
            if (j != t) {
                out[j] += rej * static_cast<double>(p[j]) / z;
            }
        }
    }
    return out;
}

std::size_t sample_from(const Dist& d, std::mt19937_64& rng) {
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double acc = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        acc += static_cast<double>(d[i]);
        if (u < acc) {
            return i;
        }
    }
    return d.size() - 1;
}

std::size_t argmax_of(const Dist& d) {
    return static_cast<std::size_t>(std::max_element(d.begin(), d.end()) - d.begin());
}

struct Histogram {
    std::vector<double> counts;
    std::size_t n = 0;
    std::size_t empty_results = 0;
    std::size_t out_of_range = 0;
    double accepted_drafts_total = 0.0;

    explicit Histogram(std::size_t v) : counts(v, 0.0) {}

    void add(const Result& r) {
        ++n;
        accepted_drafts_total += static_cast<double>(r.accepted_drafts);
        if (r.accepted.empty()) {
            ++empty_results;
            return;
        }
        const auto t = static_cast<std::size_t>(r.accepted[0]);
        if (r.accepted[0] < 0 || t >= counts.size()) {
            ++out_of_range;
            return;
        }
        counts[t] += 1.0;
    }

    [[nodiscard]] std::vector<double> pmf() const {
        std::vector<double> out(counts.size(), 0.0);
        if (n == 0) {
            return out;
        }
        for (std::size_t i = 0; i < counts.size(); ++i) {
            out[i] = counts[i] / static_cast<double>(n);
        }
        return out;
    }
};

// `concentration` == 1.0 gives a deterministic drafter (q = 1 on the proposed token).
// `quality` picks WHICH token it proposes: argmax of p (good), argmin of p (bad), or a
// draw from p (mid).
enum class Quality : std::uint8_t { Good, Bad, Mid };

struct RunResult {
    double tv_from_expected = 0.0;
    double tv_from_target = 0.0;
    std::size_t empty_results = 0;
    std::size_t out_of_range = 0;
    double accept_rate = 0.0;
};

RunResult run_histogram(double concentration, Quality quality, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const Dist p0 = target_row(rng);
    std::vector<Dist> rows;
    rows.push_back(p0);
    for (std::size_t i = 1; i <= kDraft; ++i) {
        rows.push_back(target_row(rng));
    }

    // The drafter's row for position 0, which is what decides the first committed token.
    const std::size_t proposed = quality == Quality::Good
                                     ? argmax_of(p0)
                                     : (quality == Quality::Bad
                                            ? static_cast<std::size_t>(
                                                  std::min_element(p0.begin(), p0.end()) -
                                                  p0.begin())
                                            : 0);
    Dist q0(kVocab, 0.0F);
    if (quality == Quality::Mid) {
        // Mid: a tempered version of p, genuinely soft.
        std::vector<double> w(kVocab);
        for (std::size_t i = 0; i < kVocab; ++i) {
            w[i] = std::pow(static_cast<double>(p0[i]), 0.5);
        }
        q0 = normalised(std::move(w));
    } else {
        const double rest = (1.0 - concentration) / static_cast<double>(kVocab - 1);
        for (std::size_t i = 0; i < kVocab; ++i) {
            q0[i] = static_cast<float>(i == proposed ? concentration : rest);
        }
    }

    SpecVerifier v(seed ^ 0x9E3779B97F4A7C15ULL);
    Histogram h(kVocab);
    std::mt19937_64 draft_rng(seed + 12345);

    std::vector<std::span<const float>> row_spans;
    row_spans.reserve(rows.size());
    for (const Dist& r : rows) {
        row_spans.emplace_back(r.data(), r.size());
    }

    for (int run = 0; run < kRuns; ++run) {
        std::vector<TokenId> draft(kDraft);
        std::vector<float> dprobs(kDraft);
        // Position 0 is the one the histogram measures, so it uses q0 exactly. Later
        // positions are filled from the same drafter shape; they affect acceptance depth
        // but not the first-token law.
        const std::size_t d0 = sample_from(q0, draft_rng);
        draft[0] = static_cast<TokenId>(d0);
        dprobs[0] = q0[d0];
        for (std::size_t i = 1; i < kDraft; ++i) {
            const std::size_t di = sample_from(rows[i], draft_rng);
            draft[i] = static_cast<TokenId>(di);
            dprobs[i] = rows[i][di];
        }
        h.add(v.verify(std::span<const TokenId>(draft), std::span<const float>(dprobs),
                       std::span<const std::span<const float>>(row_spans)));
    }

    std::vector<double> target(kVocab);
    for (std::size_t i = 0; i < kVocab; ++i) {
        target[i] = static_cast<double>(p0[i]);
    }

    RunResult out;
    out.tv_from_expected = tv(h.pmf(), expected_first_token(p0, q0));
    out.tv_from_target = tv(h.pmf(), target);
    out.empty_results = h.empty_results;
    out.out_of_range = h.out_of_range;
    out.accept_rate = h.n ? h.accepted_drafts_total / static_cast<double>(h.n) : 0.0;
    return out;
}

// --- the structural properties ------------------------------------------------------
struct Structural {
    bool perfect_accepts_all = true;
    bool deterministic = true;
    std::size_t floor_violations = 0;
    std::size_t degenerate_failures = 0;
};

Structural check_structural() {
    Structural s;
    std::mt19937_64 rng(555);

    // Perfect drafter: proposes the argmax at every position with q == p.
    {
        std::vector<Dist> rows;
        for (std::size_t i = 0; i <= kDraft; ++i) {
            rows.push_back(target_row(rng));
        }
        std::vector<std::span<const float>> spans;
        for (const Dist& r : rows) {
            spans.emplace_back(r.data(), r.size());
        }
        std::vector<TokenId> draft(kDraft);
        std::vector<float> dprobs(kDraft);
        for (std::size_t i = 0; i < kDraft; ++i) {
            const std::size_t a = argmax_of(rows[i]);
            draft[i] = static_cast<TokenId>(a);
            dprobs[i] = rows[i][a];
        }
        SpecVerifier v(11);
        for (int rep = 0; rep < 500; ++rep) {
            const Result r = v.verify(std::span<const TokenId>(draft),
                                      std::span<const float>(dprobs),
                                      std::span<const std::span<const float>>(spans));
            if (r.accepted.size() != kDraft + 1 || r.accepted_drafts != kDraft ||
                r.ended_on_rejection) {
                s.perfect_accepts_all = false;
                break;
            }
        }
    }

    // Determinism: same seed, same inputs, same answer, from fresh instances.
    {
        std::vector<Dist> rows;
        for (std::size_t i = 0; i <= kDraft; ++i) {
            rows.push_back(target_row(rng));
        }
        std::vector<std::span<const float>> spans;
        for (const Dist& r : rows) {
            spans.emplace_back(r.data(), r.size());
        }
        std::vector<TokenId> draft{1, 2, 3, 4};
        std::vector<float> dprobs{0.1F, 0.2F, 0.3F, 0.4F};
        SpecVerifier a(777);
        SpecVerifier b(777);
        for (int rep = 0; rep < 200; ++rep) {
            const Result x = a.verify(std::span<const TokenId>(draft),
                                      std::span<const float>(dprobs),
                                      std::span<const std::span<const float>>(spans));
            const Result y = b.verify(std::span<const TokenId>(draft),
                                      std::span<const float>(dprobs),
                                      std::span<const std::span<const float>>(spans));
            if (x.accepted != y.accepted || x.accepted_drafts != y.accepted_drafts ||
                x.ended_on_rejection != y.ended_on_rejection) {
                s.deterministic = false;
                break;
            }
        }
    }

    // The floor, under a drafter engineered to be rejected: q = 1 on the token the target
    // likes least.
    {
        SpecVerifier v(99);
        for (int rep = 0; rep < 20000; ++rep) {
            std::vector<Dist> rows;
            for (std::size_t i = 0; i <= kDraft; ++i) {
                rows.push_back(target_row(rng));
            }
            std::vector<std::span<const float>> spans;
            for (const Dist& r : rows) {
                spans.emplace_back(r.data(), r.size());
            }
            std::vector<TokenId> draft(kDraft);
            std::vector<float> dprobs(kDraft, 1.0F);
            for (std::size_t i = 0; i < kDraft; ++i) {
                draft[i] = static_cast<TokenId>(std::min_element(rows[i].begin(), rows[i].end()) -
                                                rows[i].begin());
            }
            const Result r = v.verify(std::span<const TokenId>(draft),
                                      std::span<const float>(dprobs),
                                      std::span<const std::span<const float>>(spans));
            if (r.accepted.empty()) {
                ++s.floor_violations;
            }
        }
    }

    // Degenerate inputs. Each must return a non-empty in-range result and not crash.
    {
        SpecVerifier v(1234);
        auto ok = [&](const Result& r, std::size_t vocab) {
            if (r.accepted.empty()) {
                return false;
            }
            for (TokenId t : r.accepted) {
                if (t < 0 || static_cast<std::size_t>(t) >= vocab) {
                    return false;
                }
            }
            return true;
        };
        // empty draft
        {
            const Dist row = normalised(std::vector<double>{3, 1, 2, 4});
            std::vector<std::span<const float>> spans{std::span<const float>(row.data(), row.size())};
            const Result r = v.verify({}, {}, std::span<const std::span<const float>>(spans));
            if (!ok(r, row.size())) {
                ++s.degenerate_failures;
            }
        }
        // q == 0
        {
            const Dist a = normalised(std::vector<double>{3, 1, 2, 4});
            const Dist b = normalised(std::vector<double>{1, 1, 1, 1});
            std::vector<std::span<const float>> spans{
                std::span<const float>(a.data(), a.size()),
                std::span<const float>(b.data(), b.size())};
            std::vector<TokenId> draft{2};
            std::vector<float> dprobs{0.0F};
            const Result r = v.verify(std::span<const TokenId>(draft),
                                      std::span<const float>(dprobs),
                                      std::span<const std::span<const float>>(spans));
            if (!ok(r, a.size())) {
                ++s.degenerate_failures;
            }
        }
        // single-token vocabulary
        {
            const Dist one{1.0F};
            std::vector<std::span<const float>> spans{
                std::span<const float>(one.data(), 1), std::span<const float>(one.data(), 1)};
            std::vector<TokenId> draft{0};
            std::vector<float> dprobs{1.0F};
            const Result r = v.verify(std::span<const TokenId>(draft),
                                      std::span<const float>(dprobs),
                                      std::span<const std::span<const float>>(spans));
            if (!ok(r, 1)) {
                ++s.degenerate_failures;
            }
        }
        // all mass on one token, drafted token is a different one
        {
            const Dist hot{0.0F, 1.0F, 0.0F, 0.0F};
            std::vector<std::span<const float>> spans{
                std::span<const float>(hot.data(), hot.size()),
                std::span<const float>(hot.data(), hot.size())};
            std::vector<TokenId> draft{3};
            std::vector<float> dprobs{1.0F};
            const Result r = v.verify(std::span<const TokenId>(draft),
                                      std::span<const float>(dprobs),
                                      std::span<const std::span<const float>>(spans));
            if (!ok(r, hot.size())) {
                ++s.degenerate_failures;
            }
        }
    }
    return s;
}

} // namespace

int main() {
    // Deterministic drafter (q=1): the regime LM_Pipe uses, and the only one in which the
    // brief's procedure and the brief's property agree. tv_det_* is measured against the
    // TARGET, because here the two references coincide.
    const RunResult det_good = run_histogram(1.0, Quality::Good, 101);
    const RunResult det_bad = run_histogram(1.0, Quality::Bad, 202);

    // Soft drafter: reported, not scored. tv_soft_brief is distance from what the brief's
    // prescribed reduction must produce; tv_soft_target is distance from the true target.
    const RunResult soft = run_histogram(0.75, Quality::Mid, 303);

    const Structural s = check_structural();

    std::printf("tv_det_good=%.4f tv_det_bad=%.4f | tv_soft_brief=%.4f tv_soft_target=%.4f | "
                "floor_viol=%zu empty=%zu oor=%zu perfect=%d det=%d degen_fail=%zu "
                "accept_rate=%.2f\n",
                det_good.tv_from_target, det_bad.tv_from_target, soft.tv_from_expected,
                soft.tv_from_target, s.floor_violations,
                det_good.empty_results + det_bad.empty_results + soft.empty_results,
                det_good.out_of_range + det_bad.out_of_range + soft.out_of_range,
                s.perfect_accepts_all ? 1 : 0, s.deterministic ? 1 : 0,
                s.degenerate_failures, det_good.accept_rate);
    return 0;
}
