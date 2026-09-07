// GPU-free Flash-Next reference ops: GR, QSA mask, PLE hash. No MLX, no checkpoint.

#include "src/model/qwen4_exp_ops.hpp"

#include <cmath>
#include <vector>

#include "tests/check.hpp"

using namespace lmp::model::qwen4;

TEST(qwen4_exp_allowlist_helpers_live_in_ops_header) {
    // Seed 1234 is the transformers/checkpoint default; seed 0 is the PR bug.
    const auto a = default_layer_multipliers(kDefaultHashSeed, 3, 248320, 0);
    const auto b = default_layer_multipliers(0, 3, 248320, 0);
    REQUIRE(a.size() == 3);
    CHECK_EQ(a[0], 23703573157769ULL);
    CHECK_EQ(a[1], 20109073645365ULL);
    CHECK_EQ(a[2], 8052911324071ULL);
    CHECK(a[0] != b[0]);
    CHECK(a[1] != b[1]);
}

TEST(gated_residual_read_collapses_hc_branches) {
    const int hc = 2, d = 2, rank = 2;
    // Two identical branches of ones, identity-ish mix weights.
    std::vector<float> hyper = {1, 2, 1, 2}; // tile of [1,2]
    std::vector<float> w(4, 1.0f);           // already-folded (1+0)
    std::vector<float> w_down(rank * hc * d, 0.0f);
    // down projects the mean: row 0 attends to all ones-ish
    for (int i = 0; i < hc * d; ++i) {
        w_down[static_cast<std::size_t>(i)] = 1.0f;
    }
    std::vector<float> w_up(hc * d * rank, 0.0f);
    for (int i = 0; i < hc * d; ++i) {
        w_up[static_cast<std::size_t>(i * rank)] = 1.0f; // first rank dim drives all outputs
    }
    std::vector<float> w_inj(hc * hc * d, 0.0f);
    for (int b = 0; b < hc; ++b) {
        w_inj[static_cast<std::size_t>(b * hc * d + b * d)] = 1.0f;
    }
    const auto r = gated_residual_read(hyper.data(), w.data(), w_down.data(), w_up.data(),
                                       w_inj.data(), hc, d, rank, 1e-6f, true);
    REQUIRE(r.mixed.size() == 2);
    REQUIRE(r.inject.size() == 2);
    CHECK(r.mixed[0] > 0.0f);
    CHECK(r.mixed[1] > 0.0f);
    const auto mix_only = gated_residual_read(hyper.data(), w.data(), w_down.data(), w_up.data(),
                                              nullptr, hc, d, rank, 1e-6f, false);
    CHECK(mix_only.inject.empty());
    CHECK_EQ(mix_only.mixed.size(), 2u);
}

TEST(gated_residual_write_is_per_branch_scale) {
    const int hc = 2, d = 2;
    std::vector<float> hyper = {1, 0, 3, 0};
    std::vector<float> y = {10, 0};
    std::vector<float> inj = {0.5f, 2.0f};
    std::vector<float> out(4);
    gated_residual_write(hyper.data(), y.data(), inj.data(), out.data(), hc, d);
    CHECK(std::fabs(out[0] - 6.0f) < 1e-5f); // 1 + 0.5*10
    CHECK(std::fabs(out[2] - 23.0f) < 1e-5f); // 3 + 2*10
}

TEST(l2norm_is_not_rms_norm) {
    float x[2] = {3.0f, 4.0f};
    float out[2];
    l2norm(x, out, 2, 1e-6f);
    // ||x|| = 5, so (0.6, 0.8). RMS would divide by sqrt((9+16)/2)=sqrt(12.5).
    CHECK(std::fabs(out[0] - 0.6f) < 1e-4f);
    CHECK(std::fabs(out[1] - 0.8f) < 1e-4f);
    const float rms = std::sqrt((9.0f + 16.0f) / 2.0f + 1e-6f);
    CHECK(std::fabs(out[0] - 3.0f / rms) > 0.05f);
}

TEST(qsa_short_circuit_matches_budget) {
    CHECK(qsa_short_circuit(2048, 2048));
    CHECK(qsa_short_circuit(2047, 2048));
    CHECK(!qsa_short_circuit(2049, 2048));
}

TEST(qsa_keep_mask_unions_own_tail_and_ands_causal) {
    // kv_len=10, compress=4, n_blocks=2, q_pos=8 (in the trailing partial block).
    // Block 0 (0-3) and block 1 (4-7) are fully observed; own tail is 8.
    const int kv = 10, q = 8, cr = 4, topk = 1;
    float scores[2] = {0.1f, 10.0f}; // pick block 1
    const auto keep = qsa_keep_mask(q, kv, cr, topk, scores);
    REQUIRE(keep.size() == 10u);
    CHECK(!keep[0] && !keep[1] && !keep[2] && !keep[3]);
    CHECK(keep[4] && keep[5] && keep[6] && keep[7]);
    CHECK(keep[8]);  // own query token
    CHECK(!keep[9]); // future
}

TEST(qsa_keep_mask_drops_incomplete_blocks) {
    // q_pos=5: block 0 (end=3) eligible; block 1 (end=7) is NOT (7>5).
    const int kv = 8, q = 5, cr = 4, topk = 2;
    float scores[2] = {1.0f, 100.0f};
    const auto keep = qsa_keep_mask(q, kv, cr, topk, scores);
    CHECK(keep[0] && keep[3]);
    CHECK(!keep[6] && !keep[7]); // block 1 ineligible despite score
    CHECK(keep[4] && keep[5]);   // own trailing partial block + query token
}

TEST(qsa_broken_global_tail_is_detectable) {
    // The bug: one global tail for the whole prefill (last query's tail applied to all).
    // For q_pos=2, own_start=0, causal keys 0,1,2. A global tail from q=8 would keep 8.
    const int kv = 10, cr = 4, topk = 1;
    float scores[2] = {1.0f, 0.0f};
    const auto keep = qsa_keep_mask(2, kv, cr, topk, scores);
    CHECK(!keep[8]);
    CHECK(!keep[9]);
    CHECK(keep[2]);
}

TEST(ple_eos_shift_does_not_cross_eos) {
    const int eos = 9;
    std::vector<int> ids = {1, 2, eos, 4, 5};
    const auto s1 = shift_right_eos(ids, 1, eos);
    CHECK_EQ(s1[0], eos); // nothing before pos 0
    CHECK_EQ(s1[1], 1);
    CHECK_EQ(s1[2], 2);
    CHECK_EQ(s1[3], eos); // reset after EOS; cannot take 2 from previous segment
    CHECK_EQ(s1[4], 4);
}

TEST(ple_ngram_gids_are_eos_aware_and_use_per_head_primes) {
    const int ngram = 3, hpn = 2, eos = 9;
    std::vector<int> sizes_i, offsets_i;
    int total = 0;
    const int n_heads = (ngram - 1) * hpn;
    for (int h = 0; h < n_heads; ++h) {
        const int s = nth_prime_after(20 - 1, h + 1);
        sizes_i.push_back(s);
        offsets_i.push_back(total);
        total += s;
    }
    std::vector<std::int64_t> sizes(sizes_i.begin(), sizes_i.end());
    std::vector<std::int64_t> offsets(offsets_i.begin(), offsets_i.end());
    const auto mults = default_layer_multipliers(kDefaultHashSeed, ngram, 50, 0);
    std::vector<int> ids = {1, 2, 3};
    const auto g = ngram_gids(ids, mults.data(), sizes.data(), offsets.data(), ngram, hpn, eos);
    CHECK_EQ(g.ngram_heads, 4);
    CHECK_EQ(g.gid.size(), 12u);
    // Changing the seed must change at least one gid (the hash-0 bug).
    const auto m0 = default_layer_multipliers(0, ngram, 50, 0);
    const auto g0 = ngram_gids(ids, m0.data(), sizes.data(), offsets.data(), ngram, hpn, eos);
    bool differ = false;
    for (std::size_t i = 0; i < g.gid.size(); ++i) {
        if (g.gid[i] != g0.gid[i]) {
            differ = true;
            break;
        }
    }
    CHECK(differ);
    int shard = 0;
    std::int64_t row = 0;
    shard_row(g.gid[0], 10, &shard, &row);
    CHECK(shard >= 0);
    CHECK(row >= 0);
}

TEST(nth_prime_after_matches_hand_values) {
    CHECK_EQ(nth_prime_after(19, 1), 23);
    CHECK_EQ(nth_prime_after(19, 2), 29);
}

TEST(ple_ngram_gids_chunked_match_single_shot) {
    const int ngram = 3, hpn = 2, eos = 9;
    const int n_heads = (ngram - 1) * hpn;
    std::vector<std::int64_t> sizes, offsets;
    std::int64_t total = 0;
    for (int h = 0; h < n_heads; ++h) {
        const int s = nth_prime_after(20 - 1, h + 1);
        sizes.push_back(s);
        offsets.push_back(total);
        total += s;
    }
    const auto mults = default_layer_multipliers(kDefaultHashSeed, ngram, 50, 0);
    std::vector<int> ids = {1, 2, 3, 4, 5, eos, 7, 8, 9, 10, 11, 12};
    const auto full = ngram_gids(ids, mults.data(), sizes.data(), offsets.data(), ngram, hpn, eos);
    std::vector<int> a(ids.begin(), ids.begin() + 5);
    std::vector<int> all = ids;
    const auto ga = ngram_gids(a, mults.data(), sizes.data(), offsets.data(), ngram, hpn, eos);
    const auto gb = ngram_gids(all, mults.data(), sizes.data(), offsets.data(), ngram, hpn, eos);
    REQUIRE(ga.gid.size() == 5u * static_cast<std::size_t>(n_heads));
    for (std::size_t i = 0; i < ga.gid.size(); ++i) {
        CHECK_EQ(ga.gid[i], gb.gid[i]);
        CHECK_EQ(gb.gid[i], full.gid[i]);
    }
    for (std::size_t i = ga.gid.size(); i < full.gid.size(); ++i) {
        CHECK_EQ(gb.gid[i], full.gid[i]);
    }
}

TEST(qsa_keep_mask_without_causal_and_leaks_future) {
    // Control: if we only expand top-k blocks and skip the causal AND, q_pos=2
    // would keep tokens past the query when a later block wins. The real mask must not.
    const int kv = 10, q = 2, cr = 4, topk = 2;
    float scores[2] = {1.0f, 1.0f};
    const auto keep = qsa_keep_mask(q, kv, cr, topk, scores);
    CHECK(keep[0] && keep[1] && keep[2]);
    CHECK(!keep[3] && !keep[4] && !keep[8] && !keep[9]);
}
