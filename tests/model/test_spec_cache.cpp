// The two model-layer primitives speculative decoding is built on, tested against the
// real checkpoint: all-position logits (forward_logits_all) and cache rollback
// (checkpoint/restore). Labelled realmodel -- excluded from the gate, never parallel.
//
// Both tests are equivalence-by-construction against the path already in production.
// Neither inspects cache internals: a cache is only wrong if the logits it produces are
// wrong, and asserting on `offset` would pass for an implementation that moved the index
// and left attention reading scratch.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tests/check.hpp"

#if LMP_HAVE_MLX

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

#include "src/model/mlx/qwen35_moe_model.hpp"

namespace mx = mlx::core;
namespace mlxl = lmp::model::mlxl;

namespace {

const char* qwen_dir() {
    const char* v = std::getenv("LMP_QWEN_DIR");
    return v != nullptr
               ? v
               : "";
}

// One 19 GB load for the whole file. Two live models on a 48 GB host is the failure mode
// that takes the machine down, and a per-TEST local would hold two at once for as long as
// it takes the first to be destroyed.
mlxl::Qwen35MoeModel& model() {
    static mlxl::Qwen35MoeModel m;
    static const bool ok = m.load(qwen_dir());
    if (!ok) {
        static bool reported = false;
        if (!reported) {
            lmp::test::record_failure(__FILE__, __LINE__,
                                      std::string("model load failed: ") + qwen_dir());
            reported = true;
        }
    }
    return m;
}

// Deterministic filler ids. Real text would be better prose and no better test: what is
// under test is whether two cache paths agree, and any fixed id sequence exercises that.
// Kept clear of the 248,044+ special range so nothing here is a control token.
std::vector<std::int32_t> filler(std::size_t n) {
    std::vector<std::int32_t> ids(n);
    std::uint32_t x = 12345;
    for (std::size_t i = 0; i < n; ++i) {
        x = x * 1664525U + 1013904223U;
        ids[i] = static_cast<std::int32_t>(x % 40000U + 100U);
    }
    return ids;
}

mx::array as_batch(const std::int32_t* p, std::size_t n) {
    return mx::array(p, {1, static_cast<int>(n)}, mx::int32);
}

// [1, S, V] (or [1, 1, V]) -> S rows of V floats on the host.
std::vector<std::vector<float>> rows_to_host(const mx::array& logits) {
    mx::array f = mx::astype(logits, mx::float32);
    mx::eval(f);
    const int seq = static_cast<int>(f.shape()[1]);
    const int vocab = static_cast<int>(f.shape()[2]);
    const float* data = f.data<float>();
    std::vector<std::vector<float>> out;
    out.reserve(static_cast<std::size_t>(seq));
    for (int s = 0; s < seq; ++s) {
        out.emplace_back(data + static_cast<std::ptrdiff_t>(s) * vocab,
                         data + static_cast<std::ptrdiff_t>(s + 1) * vocab);
    }
    return out;
}

std::size_t argmax(const std::vector<float>& row) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < row.size(); ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return best;
}

std::vector<double> softmax(const std::vector<float>& row) {
    double hi = -1e30;
    for (float v : row) {
        hi = std::max(hi, static_cast<double>(v));
    }
    std::vector<double> p(row.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < row.size(); ++i) {
        p[i] = std::exp(static_cast<double>(row[i]) - hi);
        sum += p[i];
    }
    for (double& v : p) {
        v /= sum;
    }
    return p;
}

// Total variation distance, the only distance that means anything to a verifier: it is
// exactly the largest disagreement in probability the two rows can produce for any event.
double tv_distance(const std::vector<float>& a, const std::vector<float>& b) {
    const std::vector<double> pa = softmax(a);
    const std::vector<double> pb = softmax(b);
    double sum = 0.0;
    for (std::size_t i = 0; i < pa.size() && i < pb.size(); ++i) {
        sum += std::fabs(pa[i] - pb[i]);
    }
    return 0.5 * sum;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t i = 0; i < n; ++i) {
        worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return worst;
}

} // namespace

TEST(rollback_is_equivalent_to_never_appending) {
    // The claim: a cache that processed 200 tokens and rolled back to 140 is
    // indistinguishable from one that only ever processed 140. Asserted on the next
    // token's logits, which is the only thing downstream reads.
    //
    // Exact equality, not a tolerance. Both runs forward the SAME single token against
    // caches holding the same 140 positions, so every kernel sees identical shapes and
    // identical values; anything but bit-equality here means the rollback left state
    // behind. (140 and 200 both sit inside one 256-token KVCache block, so neither run's
    // buffer has grown -- see the growth-crossing case below.)
    mlxl::Qwen35MoeModel& m = model();
    REQUIRE(m.qwen_config().num_hidden_layers > 0);

    const std::vector<std::int32_t> ids = filler(200);
    const std::int32_t probe = 4242;

    // Run A: 140, checkpoint, 60 more, roll back, probe.
    m.reset_cache();
    mx::array pre = m.forward_logits(as_batch(ids.data(), 140));
    mx::eval(pre);
    const auto cp = m.checkpoint();
    CHECK_EQ(cp.seq_len, 140);
    mx::array mid = m.forward_logits(as_batch(ids.data() + 140, 60));
    mx::eval(mid);
    CHECK_EQ(m.cache_seq_len(), 200);
    m.restore(cp);
    CHECK_EQ(m.cache_seq_len(), 140);
    const auto rolled = rows_to_host(m.forward_logits(as_batch(&probe, 1)));

    // Run B: 140, probe.
    m.reset_cache();
    mx::array fresh = m.forward_logits(as_batch(ids.data(), 140));
    mx::eval(fresh);
    const auto never = rows_to_host(m.forward_logits(as_batch(&probe, 1)));

    REQUIRE(rolled.size() == 1);
    REQUIRE(never.size() == 1);
    CHECK_EQ(rolled[0].size(), never[0].size());
    CHECK_EQ(max_abs_diff(rolled[0], never[0]), 0.0);
    CHECK_EQ(argmax(rolled[0]), argmax(never[0]));
}

TEST(rollback_across_a_kv_growth_boundary) {
    // Same claim, but the rolled-back run has crossed KVCache::kStep (256) and so holds a
    // 512-token buffer where the fresh run holds 256. The values attention reads are the
    // same slice either way; the buffer they are a view of is not. This was written
    // expecting to need a tolerance for that -- differently strided views could plausibly
    // take a different kernel path -- and measured 0.00e+00, so it asserts equality. If a
    // future MLX makes this drift, relax it to a tolerance and say so here; do not relax it
    // to hide a rollback that started attending over stale keys, which misses by whole
    // tokens rather than by ulps.
    mlxl::Qwen35MoeModel& m = model();
    const std::vector<std::int32_t> ids = filler(300);
    const std::int32_t probe = 4242;

    m.reset_cache();
    mx::array pre = m.forward_logits(as_batch(ids.data(), 200));
    mx::eval(pre);
    const auto cp = m.checkpoint();
    mx::array mid = m.forward_logits(as_batch(ids.data() + 200, 100));
    mx::eval(mid);
    CHECK_EQ(m.cache_seq_len(), 300);
    m.restore(cp);
    CHECK_EQ(m.cache_seq_len(), 200);
    const auto rolled = rows_to_host(m.forward_logits(as_batch(&probe, 1)));

    m.reset_cache();
    mx::array fresh = m.forward_logits(as_batch(ids.data(), 200));
    mx::eval(fresh);
    const auto never = rows_to_host(m.forward_logits(as_batch(&probe, 1)));

    REQUIRE(rolled.size() == 1);
    REQUIRE(never.size() == 1);
    const double worst = max_abs_diff(rolled[0], never[0]);
    std::fprintf(stderr, "  [rollback] max|logit diff| across growth boundary = %.3e\n", worst);
    CHECK_EQ(worst, 0.0);
    CHECK_EQ(argmax(rolled[0]), argmax(never[0]));
}

namespace {

// Forward `k` tokens as one batch and again one at a time from the same cache state, and
// report how far apart the two agree at each position. `prompt_len` sets the cache offset
// the block starts from, which is what separates "the causal mask is misaligned against a
// non-empty cache" from "bf16 accumulates differently in a wider kernel".
struct BatchVsStep {
    double worst_logit = 0.0;
    double worst_tv = 0.0;
    std::size_t argmax_mismatches = 0;
};

BatchVsStep compare_batched_vs_sequential(std::size_t prompt_len, std::size_t k) {
    mlxl::Qwen35MoeModel& m = model();
    const std::vector<std::int32_t> prompt = filler(prompt_len + k);
    const std::int32_t* block = prompt.data() + prompt_len;

    m.reset_cache();
    if (prompt_len > 0) {
        mx::array warm = m.forward_logits(as_batch(prompt.data(), prompt_len));
        mx::eval(warm);
    }
    const auto cp = m.checkpoint();

    const auto batched = rows_to_host(m.forward_logits_all(as_batch(block, k)));
    m.restore(cp);

    BatchVsStep out;
    std::fprintf(stderr, "  [all-pos] offset=%zu k=%zu per-row (max|dlogit|, TV):", prompt_len, k);
    for (std::size_t i = 0; i < k; ++i) {
        const auto one = rows_to_host(m.forward_logits(as_batch(block + i, 1)));
        const double d = max_abs_diff(batched[i], one[0]);
        const double tv = tv_distance(batched[i], one[0]);
        out.worst_logit = std::max(out.worst_logit, d);
        out.worst_tv = std::max(out.worst_tv, tv);
        const std::size_t ab = argmax(batched[i]);
        const std::size_t ao = argmax(one[0]);
        if (ab != ao) {
            ++out.argmax_mismatches;
            std::fprintf(stderr, " MISMATCH i=%zu bat=%zu(%.3f) seq=%zu(%.3f) bat@seq=%.3f", i, ab,
                         static_cast<double>(batched[i][ab]), ao, static_cast<double>(one[0][ao]),
                         static_cast<double>(batched[i][ao]));
        }
        std::fprintf(stderr, " (%.2e,%.2e)", d, tv);
    }
    std::fprintf(stderr, "\n");
    return out;
}

} // namespace


TEST(all_position_logits_match_sequential_decode) {
    // The property speculative verification rests on: forwarding k tokens in ONE pass and
    // reading row i must give the same distribution as having decoded those k tokens one at
    // a time and reading step i. Verification compares the drafter's probability against
    // the target's, so if these rows are not the model's real distributions then the
    // acceptance rule is exact with respect to the wrong thing.
    //
    // Same-shapes check is k=1, not "row 0 of a k=8 batch". A k-token forward runs GDN
    // at T=k, full-attn at L=k, and quantized_matmul at M=k; a one-token step is T=L=M=1.
    // Those are different kernels even at position 0. k=1 is the one case that is the
    // same graph on the same shapes, and it is bit-exact (0.00e+00) on both A3B and 27B.
    // A misaligned causal mask would still show up as TV that grows with k or with
    // position; it does not: k=2 already has the full k=8 first-row error, and later
    // rows are flatter, not worse.
    //
    // MEASURED, 2026-09-05, k=8:
    //   dense 27B: worst logit 0.30 / 0.16, worst TV 0.022 / 0.011 (cold/warm),
    //              1 top-1 flip, two candidates 0.06 apart in bf16.
    //   MoE A3B:   worst logit 2.05 / 1.03, worst TV 0.115 / 0.075,
    //              2+1 top-1 flips, all near-ties (logit gap < 0.2).
    // The 2026-08-01 A3B pin (logit 0.75, TV 0.059, 16/16 top-1) does not hold on
    // MLX 0.32: M=k vs M=1 qmm moved, and MoE routing turns a small logit nudge into
    // a discrete expert-set change that fattens row-0 TV. Top-1 is not a mask
    // detector when two tokens sit inside one bf16 ulp.
    //
    // Speculation remains exact with respect to the rows it is given. Those rows sit
    // ~2% TV (dense) / ~12% TV (MoE, worst row) from sequential decode. That is a
    // property of verifying a batch on a quantized hybrid model, not a rollback bug.
    const BatchVsStep k1 = compare_batched_vs_sequential(0, 1);
    const BatchVsStep cold = compare_batched_vs_sequential(0, 8);
    const BatchVsStep warm = compare_batched_vs_sequential(64, 8);
    std::fprintf(stderr,
                 "  [all-pos] k1 logit=%.3e TV=%.3e | worst logit cold=%.3e warm=%.3e | "
                 "worst TV cold=%.3e warm=%.3e | argmax mismatches cold=%zu warm=%zu\n",
                 k1.worst_logit, k1.worst_tv, cold.worst_logit, warm.worst_logit,
                 cold.worst_tv, warm.worst_tv, cold.argmax_mismatches, warm.argmax_mismatches);

    CHECK_EQ(k1.argmax_mismatches, std::size_t{0});
    CHECK(k1.worst_logit < 1e-5);
    CHECK(cold.argmax_mismatches <= 3);
    CHECK(warm.argmax_mismatches <= 3);
    CHECK(cold.worst_tv < 0.15);
    CHECK(warm.worst_tv < 0.15);
    CHECK(cold.worst_logit < 2.5);
    CHECK(warm.worst_logit < 2.5);
}

TEST(forward_logits_keeps_its_final_position_contract) {
    // forward_logits_all is an EXTENSION of forward_logits: both share forward_hidden;
    // the decode entry slices to the last position before lm_head, the verify entry
    // projects every row. Same k-token hidden, so the last row can differ only by
    // lm_head running at M=k vs M=1. That used to be bit-identical on A3B (2026-08-01);
    // on MLX 0.32 it is one bf16 ulp (0.0625 / 0.0664). Still the same token.
    mlxl::Qwen35MoeModel& m = model();
    const std::vector<std::int32_t> prompt = filler(64);
    const std::vector<std::int32_t> block = filler(8);
    const std::size_t k = block.size();

    m.reset_cache();
    mx::array warm = m.forward_logits(as_batch(prompt.data(), prompt.size()));
    mx::eval(warm);
    const auto cp = m.checkpoint();

    const auto batched = rows_to_host(m.forward_logits_all(as_batch(block.data(), k)));
    REQUIRE(batched.size() == k);
    CHECK_EQ(m.cache_seq_len(), static_cast<int>(prompt.size() + k));

    m.restore(cp);
    const auto last_only = rows_to_host(m.forward_logits(as_batch(block.data(), k)));
    REQUIRE(last_only.size() == 1);
    CHECK_EQ(last_only[0].size(), batched[k - 1].size());
    const double last_diff = max_abs_diff(last_only[0], batched[k - 1]);
    std::fprintf(stderr, "  [last-row] max|logit diff| lm_head M=k vs M=1 = %.3e\n", last_diff);
    CHECK(last_diff < 0.25);
    CHECK(tv_distance(last_only[0], batched[k - 1]) < 0.02);
    CHECK_EQ(argmax(last_only[0]), argmax(batched[k - 1]));
}

#else // !LMP_HAVE_MLX

TEST(spec_cache_requires_mlx) {
    lmp::test::record_failure(__FILE__, __LINE__,
                              "test_spec_cache is a realmodel test and needs an MLX build");
}

#endif // LMP_HAVE_MLX
