// Neutral scoreboard for the SuffixProposer cook-off. Compiled against each entrant's
// own src/suffix_proposer.cpp, using ONLY the public API the brief specified.
//
// Figures of merit, in the order that decides adoption:
//   accepted/call  -- mean tokens accepted per propose() call on a held-out replay. This
//                     is the actual speedup currency: each accepted token is one free token.
//   waste/call     -- mean tokens proposed but WRONG. On an MoE at batch 1 these are not
//                     free, so a proposer that guesses wildly is worse than one that stays
//                     quiet.
//   random-noise   -- fraction of calls that return a non-empty proposal on a corpus of
//                     uniform random tokens. This is the discriminator: there is nothing to
//                     learn there, so anything above ~0 is confident garbage.
//   p50/p99        -- propose() latency at 100k indexed tokens. Budget was 50us.
//   sound          -- every proposed continuation genuinely occurred after that context.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "suffix_proposer.hpp"

using draft::Config;
using draft::Proposal;
using draft::SuffixProposer;
using draft::TokenId;

namespace {

using Phrases = std::vector<std::vector<TokenId>>;

// The recurring vocabulary an agent reuses: tool scaffolding, file paths, code it just
// read. Built ONCE and shared between the training and held-out corpora -- an agent
// repeating structure it has already emitted is the entire premise of this component, and
// a held-out set drawn from fresh phrases would have nothing to find. (It did, at first,
// and every entrant correctly proposed nothing.)
Phrases make_phrases(std::mt19937_64& rng) {
    Phrases phrases;
    std::uniform_int_distribution<int> tok(1000, 1200);
    for (int p = 0; p < 24; ++p) {
        std::vector<TokenId> phrase;
        const int len = 6 + (p % 11);
        for (int i = 0; i < len; ++i) {
            phrase.push_back(tok(rng));
        }
        phrases.push_back(std::move(phrase));
    }
    return phrases;
}

std::vector<TokenId> repetitive_corpus(const Phrases& phrases, std::mt19937_64& rng,
                                       std::size_t target_len) {
    std::vector<TokenId> out;
    std::uniform_int_distribution<std::size_t> pick(0, phrases.size() - 1);
    std::uniform_int_distribution<int> noise(2000, 2050);
    while (out.size() < target_len) {
        const std::vector<TokenId>& ph = phrases[pick(rng)];
        out.insert(out.end(), ph.begin(), ph.end());
        if ((rng() & 3U) == 0U) {
            out.push_back(noise(rng)); // occasional novel token between phrases
        }
    }
    out.resize(target_len);
    return out;
}

std::vector<TokenId> random_corpus(std::mt19937_64& rng, std::size_t len) {
    std::uniform_int_distribution<int> tok(0, 50000);
    std::vector<TokenId> out(len);
    for (TokenId& t : out) {
        t = tok(rng);
    }
    return out;
}

// Does `cont` actually follow `ctx` somewhere in `corpus`? Brute force on purpose: the
// harness must not share any cleverness with the thing it is judging.
bool is_real_continuation(const std::vector<TokenId>& corpus, std::span<const TokenId> ctx,
                          const std::vector<TokenId>& cont, std::size_t matched_len) {
    if (cont.empty()) {
        return true;
    }
    const std::size_t m = std::min(matched_len, ctx.size());
    if (m == 0) {
        return false;
    }
    const TokenId* tail = ctx.data() + (ctx.size() - m);
    for (std::size_t i = 0; i + m + cont.size() <= corpus.size(); ++i) {
        if (std::equal(tail, tail + m, corpus.begin() + static_cast<long>(i)) &&
            std::equal(cont.begin(), cont.end(),
                       corpus.begin() + static_cast<long>(i + m))) {
            return true;
        }
    }
    return false;
}

struct Score {
    double accepted_per_call = 0.0;
    double waste_per_call = 0.0;
    double propose_rate = 0.0;
    double noise_rate = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
    bool sound = true;
    std::size_t unsound_examples = 0;
    bool memory_ok = true;
    bool deterministic = true;
};

} // namespace

int main() {
    Score s;
    std::mt19937_64 rng(20260801);

    // ---- acceptance on a repetitive corpus, replayed on held-out text ----------
    const Phrases phrases = make_phrases(rng);
    const std::vector<TokenId> train = repetitive_corpus(phrases, rng, 60000);
    const std::vector<TokenId> held = repetitive_corpus(phrases, rng, 4000);

    SuffixProposer p{Config{}};
    p.ingest(train);

    constexpr std::size_t kMaxProposed = 8;
    std::size_t calls = 0, nonempty = 0, accepted = 0, wasted = 0;
    for (std::size_t i = 64; i + kMaxProposed < held.size(); i += 7) {
        std::span<const TokenId> ctx(held.data(), i);
        const Proposal pr = p.propose(ctx, kMaxProposed);
        ++calls;
        if (pr.tokens.empty()) {
            continue;
        }
        ++nonempty;
        // Prefix agreement: speculative decoding stops at the first rejection.
        std::size_t k = 0;
        while (k < pr.tokens.size() && i + k < held.size() && pr.tokens[k] == held[i + k]) {
            ++k;
        }
        accepted += k;
        wasted += pr.tokens.size() - k;
        if (s.unsound_examples < 40) {
            if (!is_real_continuation(train, ctx, pr.tokens, pr.matched_len)) {
                s.sound = false;
                ++s.unsound_examples;
            }
        }
    }
    s.accepted_per_call = calls ? static_cast<double>(accepted) / static_cast<double>(calls) : 0.0;
    s.waste_per_call = calls ? static_cast<double>(wasted) / static_cast<double>(calls) : 0.0;
    s.propose_rate = calls ? static_cast<double>(nonempty) / static_cast<double>(calls) : 0.0;

    // ---- the discriminator: does it shut up on noise? --------------------------
    {
        std::mt19937_64 r2(99);
        const std::vector<TokenId> rnd_train = random_corpus(r2, 60000);
        const std::vector<TokenId> rnd_held = random_corpus(r2, 3000);
        SuffixProposer q{Config{}};
        q.ingest(rnd_train);
        std::size_t c = 0, ne = 0;
        for (std::size_t i = 64; i + kMaxProposed < rnd_held.size(); i += 7) {
            const Proposal pr = q.propose(std::span<const TokenId>(rnd_held.data(), i), kMaxProposed);
            ++c;
            if (!pr.tokens.empty()) {
                ++ne;
            }
        }
        s.noise_rate = c ? static_cast<double>(ne) / static_cast<double>(c) : 0.0;
    }

    // ---- latency at 100k indexed tokens ----------------------------------------
    {
        std::mt19937_64 r3(7);
        const Phrases ph3 = make_phrases(r3);
        const std::vector<TokenId> big = repetitive_corpus(ph3, r3, 100000);
        SuffixProposer q{Config{}};
        q.ingest(big);
        std::vector<double> us;
        us.reserve(4000);
        for (std::size_t i = 200; i < 200 + 4000; ++i) {
            std::span<const TokenId> ctx(big.data(), i);
            const auto t0 = std::chrono::steady_clock::now();
            const Proposal pr = q.propose(ctx, kMaxProposed);
            const auto t1 = std::chrono::steady_clock::now();
            (void)pr;
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(us.begin(), us.end());
        s.p50_us = us[us.size() / 2];
        s.p99_us = us[static_cast<std::size_t>(static_cast<double>(us.size()) * 0.99)];
    }

    // ---- memory bound ----------------------------------------------------------
    {
        Config c;
        c.max_indexed_tokens = 8192;
        SuffixProposer q{c};
        std::mt19937_64 r4(3);
        const Phrases ph4 = make_phrases(r4);
        for (int k = 0; k < 20; ++k) {
            const std::vector<TokenId> chunk = repetitive_corpus(ph4, r4, 4000);
            q.ingest(chunk);
        }
        s.memory_ok = q.indexed_tokens() <= 8192;
        if (!s.memory_ok) {
            std::printf("  (indexed_tokens=%zu over cap 8192)\n", q.indexed_tokens());
        }
    }

    // ---- determinism -----------------------------------------------------------
    {
        SuffixProposer a{Config{}}, b{Config{}};
        a.ingest(train);
        b.ingest(train);
        for (std::size_t i = 100; i < 900; i += 37) {
            std::span<const TokenId> ctx(held.data(), i);
            const Proposal x = a.propose(ctx, kMaxProposed);
            const Proposal y = b.propose(ctx, kMaxProposed);
            if (x.tokens != y.tokens || x.matched_len != y.matched_len) {
                s.deterministic = false;
                break;
            }
        }
    }

    std::printf("accepted/call=%.3f waste/call=%.3f propose_rate=%.3f noise_rate=%.3f "
                "p50=%.2fus p99=%.2fus sound=%d mem_ok=%d det=%d\n",
                s.accepted_per_call, s.waste_per_call, s.propose_rate, s.noise_rate,
                s.p50_us, s.p99_us, s.sound ? 1 : 0, s.memory_ok ? 1 : 0,
                s.deterministic ? 1 : 0);
    return 0;
}
