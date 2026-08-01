// Neutral scoreboard for the PrefixLedger cook-off (Jules round 2, Brief D). Compiled
// once per entrant against that entrant's own include/ and src/, using ONLY the public API
// the brief specified.
//
// Written before any entrant was read, and it grades against a brute-force
// std::vector<TokenId> reference rather than against an opinion. Round 1 is the reason:
// the winning draft proposer was only identifiable because four others did the same thing
// worse, and the one wrong moetrace was only visible because five agreed against it. Five
// implementations and one neutral rule beat five code reads.
//
// Figures of merit, in the order that decides adoption:
//
//   agree        -- disagreements with the reference over a randomised interleaving of
//                   append / truncate_last / truncate_to / plan_reuse. Anything but 0 is
//                   disqualifying; everything below is a tiebreak among the survivors.
//   semantics    -- the five divergence cases the brief names. The trap is the asymmetry:
//                   a strict PREFIX of the ledger is divergent (cached state past it must
//                   be thrown away) while a strict EXTENSION is not. Implementing
//                   `divergent = (reusable != candidate.size())` passes four of the five
//                   and inverts exactly that one.
//   fp_collide   -- distinct token sequences sharing a fingerprint, over a corpus built
//                   from near-misses (transpositions, single-token edits, shared prefixes).
//                   THE DISCRIMINATOR. A fingerprint that is a sum, an xor, or a polynomial
//                   over too small a modulus passes every functional test in the brief and
//                   fails only here -- and it fails as silent stale-context reuse in
//                   production, which is the exact bug KvCacheLedger exists to prevent.
//   fp_path      -- fingerprint agreement across construction paths that end at the same
//                   tokens: appended one at a time, appended in bulk, and appended-then-
//                   truncated. The brief requires equal contents to give equal
//                   fingerprints; a rolling hash that folds in length or history breaks
//                   here and nowhere else.
//   p50/p99      -- plan_reuse latency at 100,000 tokens. Brief's budget: under 20 us.
//   trunc_ratio  -- truncate_last(1) on a 100k ledger divided by the same call on a 1k
//                   ledger. The brief's stated design question is whether the fingerprint
//                   survives truncation cheaply; an implementation that rehashes the whole
//                   ledger is correct and O(size), and this is the only number that says
//                   so. ~1.0 is O(removed); ~100 is O(size).

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <prefix_ledger.hpp>

using kv::PrefixLedger;
using kv::ReuseDecision;
using kv::TokenId;

namespace {

// The reference. Deliberately the dumbest thing that could work: the harness must not
// share any cleverness with the thing it is judging.
struct Reference {
    std::vector<TokenId> ids;

    void append(TokenId id) { ids.push_back(id); }
    void append(std::span<const TokenId> more) { ids.insert(ids.end(), more.begin(), more.end()); }

    void truncate_last(std::size_t n) { ids.resize(n >= ids.size() ? 0 : ids.size() - n); }
    void truncate_to(std::size_t n) {
        if (n < ids.size()) {
            ids.resize(n);
        }
    }
    void clear() { ids.clear(); }

    [[nodiscard]] ReuseDecision plan_reuse(std::span<const TokenId> cand) const {
        std::size_t i = 0;
        while (i < cand.size() && i < ids.size() && cand[i] == ids[i]) {
            ++i;
        }
        // Divergent means the candidate left the ledger BEFORE the ledger ended, so cached
        // state past `reusable` is stale. Equal or extending is not divergent; a strict
        // prefix is.
        return ReuseDecision{i, i < ids.size()};
    }
};

struct Score {
    std::size_t disagreements = 0;
    std::size_t semantics_failed = 0;
    std::size_t fp_collisions = 0;
    std::size_t fp_constructed = 0;
    std::size_t fp_path_failures = 0;
    std::size_t fp_insensitive = 0;
    double p50_us = 0.0;
    double p99_us = 0.0;
    double trunc_ratio = 0.0;
    double bulk_trunc_us = 0.0;
    bool rollback_exact = true;
};

std::vector<TokenId> random_ids(std::mt19937_64& rng, std::size_t n, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    std::vector<TokenId> out(n);
    for (TokenId& t : out) {
        t = static_cast<TokenId>(d(rng));
    }
    return out;
}

// --- 1. agreement with the reference under randomised operation streams ------------
std::size_t check_agreement() {
    std::size_t bad = 0;
    std::mt19937_64 rng(20260801);
    for (int trial = 0; trial < 200; ++trial) {
        PrefixLedger led;
        Reference ref;
        for (int op = 0; op < 300; ++op) {
            const int which = static_cast<int>(rng() % 100);
            if (which < 45) {
                const auto t = static_cast<TokenId>(rng() % 500);
                led.append(t);
                ref.append(t);
            } else if (which < 65) {
                const std::vector<TokenId> chunk = random_ids(rng, rng() % 12, 0, 499);
                led.append(std::span<const TokenId>(chunk));
                ref.append(std::span<const TokenId>(chunk));
            } else if (which < 80) {
                const std::size_t n = rng() % 20;
                led.truncate_last(n);
                ref.truncate_last(n);
            } else if (which < 88) {
                const std::size_t n = rng() % (ref.ids.size() + 3);
                led.truncate_to(n);
                ref.truncate_to(n);
            } else if (which < 90) {
                led.clear();
                ref.clear();
            } else {
                // A candidate that shares a prefix with the ledger and then may diverge.
                std::vector<TokenId> cand(ref.ids.begin(),
                                          ref.ids.begin() + static_cast<std::ptrdiff_t>(
                                                                rng() % (ref.ids.size() + 1)));
                const int tail = static_cast<int>(rng() % 3);
                for (int i = 0; i < tail; ++i) {
                    cand.push_back(static_cast<TokenId>(rng() % 500));
                }
                const ReuseDecision a = led.plan_reuse(std::span<const TokenId>(cand));
                const ReuseDecision b = ref.plan_reuse(std::span<const TokenId>(cand));
                if (a.reusable != b.reusable || a.divergent != b.divergent) {
                    ++bad;
                }
                // plan_reuse must not mutate. The brief says so; it is called on every
                // request, so a mutation here corrupts the cache on a read.
                if (led.size() != ref.ids.size()) {
                    ++bad;
                }
            }
            if (led.size() != ref.ids.size()) {
                ++bad;
                break;
            }
            const std::span<const TokenId> got = led.tokens();
            if (!std::equal(got.begin(), got.end(), ref.ids.begin(), ref.ids.end())) {
                ++bad;
                break;
            }
        }
    }
    return bad;
}

// --- 2. the five divergence cases the brief names ----------------------------------
std::size_t check_semantics() {
    const std::vector<TokenId> base{10, 11, 12, 13, 14};
    struct Case {
        const char* name;
        std::vector<TokenId> candidate;
        std::size_t reusable;
        bool divergent;
    };
    const std::vector<Case> cases{
        {"equal", {10, 11, 12, 13, 14}, 5, false},
        {"strict_prefix", {10, 11, 12}, 3, true},
        {"strict_extension", {10, 11, 12, 13, 14, 15, 16}, 5, false},
        {"differs_at_0", {99, 11, 12, 13, 14}, 0, true},
        {"differs_at_last", {10, 11, 12, 13, 99}, 4, true},
    };
    std::size_t failed = 0;
    for (const Case& c : cases) {
        PrefixLedger led;
        led.append(std::span<const TokenId>(base));
        const ReuseDecision d = led.plan_reuse(std::span<const TokenId>(c.candidate));
        if (d.reusable != c.reusable || d.divergent != c.divergent) {
            std::printf("    semantics FAIL %-18s got(reusable=%zu,divergent=%d) "
                        "want(reusable=%zu,divergent=%d)\n",
                        c.name, d.reusable, d.divergent ? 1 : 0, c.reusable,
                        c.divergent ? 1 : 0);
            ++failed;
        }
    }
    return failed;
}

// --- 3. fingerprint collisions over near-misses (the discriminator) -----------------
std::size_t check_fp_collisions() {
    std::unordered_map<std::uint64_t, std::vector<TokenId>> seen;
    std::mt19937_64 rng(4242);
    std::size_t collisions = 0;

    auto consider = [&](const std::vector<TokenId>& ids) {
        PrefixLedger led;
        led.append(std::span<const TokenId>(ids));
        const std::uint64_t fp = led.fingerprint();
        auto it = seen.find(fp);
        if (it == seen.end()) {
            seen.emplace(fp, ids);
        } else if (it->second != ids) {
            ++collisions;
        }
    };

    // Near-misses, not random sequences: random ones are far apart and a weak hash
    // survives them. These are the shapes a real ledger actually compares -- shared
    // prefixes, one token changed, two tokens swapped.
    for (int seed = 0; seed < 300; ++seed) {
        std::vector<TokenId> base = random_ids(rng, 40, 0, 2000);
        consider(base);
        for (std::size_t i = 0; i < base.size(); ++i) {
            std::vector<TokenId> v = base;
            v[i] = static_cast<TokenId>((v[i] + 1) % 2000);
            consider(v);
        }
        for (std::size_t i = 0; i + 1 < base.size(); i += 7) {
            std::vector<TokenId> v = base;
            std::swap(v[i], v[i + 1]);
            consider(v);
        }
        for (std::size_t cut = 1; cut < base.size(); cut += 3) {
            consider(std::vector<TokenId>(base.begin(),
                                          base.begin() + static_cast<std::ptrdiff_t>(cut)));
        }
    }
    return collisions;
}

// --- 3b. CONSTRUCTED collisions (added after the first scoring round) ----------------
//
// The near-miss corpus above is random, and random near-misses do not find the collision
// that a polynomial hash mod 2^64 actually has. Thue-Morse words do: for a TM word and its
// complement the hash difference is divisible by prod_j (1 - P^(2^j)), whose 2-adic
// valuation grows quadratically, so it exceeds 64 at a SHORT length for any odd multiplier.
//
// This column was missing when e1/e2/e3 were first scored and all three came back
// fp_collide=0. They all collide here, at 1024 tokens. A fingerprint that can be collided by
// a construction this cheap is not a fingerprint the caller should be invited to trust, and
// this is the column that says so.
//
// Returns the shortest Thue-Morse length at which a collision was found, or 0 for none.
std::size_t check_constructed_collision() {
    auto thue_morse_bit = [](unsigned i) {
        int parity = 0;
        while (i != 0U) {
            parity ^= static_cast<int>(i & 1U);
            i >>= 1U;
        }
        return parity;
    };
    for (int k = 1; k <= 16; ++k) {
        const auto n = static_cast<std::size_t>(1) << k;
        std::vector<TokenId> a(n);
        std::vector<TokenId> b(n);
        for (std::size_t i = 0; i < n; ++i) {
            const bool one = thue_morse_bit(static_cast<unsigned>(i)) != 0;
            a[i] = one ? TokenId{1000} : TokenId{2000};
            b[i] = one ? TokenId{2000} : TokenId{1000};
        }
        PrefixLedger la;
        PrefixLedger lb;
        la.append(std::span<const TokenId>(a));
        lb.append(std::span<const TokenId>(b));
        if (la.fingerprint() == lb.fingerprint()) {
            return n;
        }
    }
    return 0;
}

// --- 4. fingerprint depends on contents and ONLY on contents ------------------------
void check_fp_paths(Score& s) {
    std::mt19937_64 rng(31337);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t keep = 20 + (rng() % 200);
        const std::size_t extra = 1 + (rng() % 300);
        const std::vector<TokenId> all = random_ids(rng, keep + extra, 0, 5000);
        const std::vector<TokenId> kept(all.begin(),
                                        all.begin() + static_cast<std::ptrdiff_t>(keep));

        PrefixLedger one_at_a_time;
        for (TokenId t : kept) {
            one_at_a_time.append(t);
        }
        PrefixLedger bulk;
        bulk.append(std::span<const TokenId>(kept));
        PrefixLedger rolled_back;
        rolled_back.append(std::span<const TokenId>(all));
        rolled_back.truncate_last(extra);
        PrefixLedger rolled_to;
        rolled_to.append(std::span<const TokenId>(all));
        rolled_to.truncate_to(keep);

        const std::uint64_t fp = bulk.fingerprint();
        if (one_at_a_time.fingerprint() != fp || rolled_back.fingerprint() != fp ||
            rolled_to.fingerprint() != fp) {
            ++s.fp_path_failures;
        }
        // Rollback equivalence by construction, on the tokens themselves.
        const std::span<const TokenId> a = rolled_back.tokens();
        const std::span<const TokenId> b = rolled_to.tokens();
        if (rolled_back.size() != keep || rolled_to.size() != keep ||
            !std::equal(a.begin(), a.end(), kept.begin(), kept.end()) ||
            !std::equal(b.begin(), b.end(), kept.begin(), kept.end())) {
            s.rollback_exact = false;
        }
    }

    // The other direction: any single-token change anywhere must move the fingerprint.
    const std::vector<TokenId> base = random_ids(rng, 64, 0, 1000);
    PrefixLedger ref;
    ref.append(std::span<const TokenId>(base));
    const std::uint64_t fp0 = ref.fingerprint();
    for (std::size_t i : {std::size_t{0}, base.size() / 2, base.size() - 1}) {
        std::vector<TokenId> v = base;
        v[i] = static_cast<TokenId>(v[i] + 1);
        PrefixLedger led;
        led.append(std::span<const TokenId>(v));
        if (led.fingerprint() == fp0) {
            ++s.fp_insensitive;
        }
    }
}

// --- 5. latency ---------------------------------------------------------------------
void measure_latency(Score& s) {
    std::mt19937_64 rng(7);
    const std::vector<TokenId> big = random_ids(rng, 100000, 0, 50000);
    PrefixLedger led;
    led.append(std::span<const TokenId>(big));

    // A candidate that shares almost the whole ledger: the worst case for a byte-by-byte
    // comparison and the case a real follow-up request actually presents.
    std::vector<TokenId> cand = big;
    cand.back() = 999999;

    std::vector<double> us;
    us.reserve(2000);
    for (int i = 0; i < 2000; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const ReuseDecision d = led.plan_reuse(std::span<const TokenId>(cand));
        const auto t1 = std::chrono::steady_clock::now();
        if (d.reusable == 0 && !d.divergent) {
            std::printf("    (latency loop: suspicious result)\n");
        }
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    s.p50_us = us[us.size() / 2];
    s.p99_us = us[static_cast<std::size_t>(static_cast<double>(us.size()) * 0.99)];
}

// --- 6. does truncation cost scale with what is removed, or with what is kept? -------
double measure_truncate_scaling() {
    auto cost = [](std::size_t n) {
        std::mt19937_64 rng(11);
        const std::vector<TokenId> ids = random_ids(rng, n, 0, 50000);
        PrefixLedger led;
        led.append(std::span<const TokenId>(ids));
        const auto t0 = std::chrono::steady_clock::now();
        constexpr int kReps = 2000;
        for (int i = 0; i < kReps; ++i) {
            led.truncate_last(1);
            led.append(ids[0]);
            // fingerprint() is read here because an implementation may defer the rehash to
            // the next read; charging truncation for work it postponed would be wrong.
            if (led.fingerprint() == 0) {
                std::printf("    (truncate loop: fingerprint 0)\n");
            }
        }
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                   .count() /
               kReps;
    };
    const double small = cost(1000);
    const double large = cost(100000);
    return small > 0.0 ? large / small : 0.0;
}

// --- 7. bulk truncation: does dropping 50,000 tokens cost 50,000 or cost nothing? ----
//
// trunc_ratio above removes ONE token and so cannot tell an O(1) design from an O(removed)
// one -- both are flat. This removes half of a 100,000-token ledger, which is what a real
// rollback to an earlier conversation turn looks like. An un-rolled polynomial hash must
// walk every discarded token; a stored per-position hash reads one slot.
double measure_bulk_truncate() {
    std::mt19937_64 rng(13);
    const std::vector<TokenId> ids = random_ids(rng, 100000, 0, 50000);
    constexpr int kReps = 200;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) {
        PrefixLedger led;
        led.append(std::span<const TokenId>(ids));
        const auto t_start = std::chrono::steady_clock::now();
        led.truncate_to(50000);
        (void)led.fingerprint();
        const auto t_end = std::chrono::steady_clock::now();
        // Only the truncation is charged; building the ledger is setup.
        static double acc = 0.0;
        acc += std::chrono::duration<double, std::micro>(t_end - t_start).count();
        if (i == kReps - 1) {
            const double out = acc / kReps;
            acc = 0.0;
            return out;
        }
    }
    (void)t0;
    return 0.0;
}

} // namespace

int main() {
    Score s;
    s.disagreements = check_agreement();
    s.semantics_failed = check_semantics();
    s.fp_collisions = check_fp_collisions();
    s.fp_constructed = check_constructed_collision();
    check_fp_paths(s);
    measure_latency(s);
    s.trunc_ratio = measure_truncate_scaling();
    s.bulk_trunc_us = measure_bulk_truncate();

    std::printf("agree_fail=%zu semantics_fail=%zu fp_collide=%zu fp_constructed=%zu "
                "fp_path_fail=%zu fp_insensitive=%zu rollback_exact=%d p50=%.2fus "
                "p99=%.2fus trunc_ratio=%.1f bulk_trunc=%.2fus\n",
                s.disagreements, s.semantics_failed, s.fp_collisions, s.fp_constructed,
                s.fp_path_failures, s.fp_insensitive, s.rollback_exact ? 1 : 0, s.p50_us,
                s.p99_us, s.trunc_ratio, s.bulk_trunc_us);
    return 0;
}
