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
        if (argmax(batched[i]) != argmax(one[0])) {
            ++out.argmax_mismatches;
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
    // Run at two cache offsets on purpose. At offset 0 the FIRST row is the one case where
    // the batched and sequential paths are the same computation on the same shapes, and it
    // agrees exactly (0.00e+00) -- which is what rules out a misaligned causal mask, the
    // failure this would otherwise look like. Every other row differs, by a flat amount
    // that does not grow with position: that is bf16 taking a different accumulation order
    // in a wider kernel, not error compounding down the block.
    //
    // MEASURED, 2026-08-01, k=8 on the production checkpoint: worst max|logit diff| 0.75
    // (2-6 ulp at these magnitudes), worst total-variation distance 0.059, top-1 agreement
    // at 16 of 16 positions. The tolerances below are those numbers with headroom, not
    // guesses -- an earlier 5e-2 logit bound was a guess and it was wrong by 20x.
    //
    // What this costs speculative decoding, stated plainly because Brief C's whole premise
    // is exactness: the acceptance rule keeps the output distribution exactly equal to
    // sampling from THE ROWS IT IS GIVEN, and those rows sit ~4% TV from the ones
    // one-at-a-time decoding would have produced. So speculation here is distribution-
    // preserving to within batched-forward numerics, not absolutely. That is a property of
    // verifying a batch on a bf16 model and is true of every implementation of this
    // technique; it is not a defect in the rollback or in the verifier.
    const BatchVsStep cold = compare_batched_vs_sequential(0, 8);
    const BatchVsStep warm = compare_batched_vs_sequential(64, 8);
    std::fprintf(stderr,
                 "  [all-pos] worst logit cold=%.3e warm=%.3e | worst TV cold=%.3e warm=%.3e\n",
                 cold.worst_logit, warm.worst_logit, cold.worst_tv, warm.worst_tv);

    // The structural assertions. A mask or rope misalignment moves the top token; numerics
    // do not.
    CHECK_EQ(cold.argmax_mismatches, std::size_t{0});
    CHECK_EQ(warm.argmax_mismatches, std::size_t{0});
    CHECK(cold.worst_tv < 0.10);
    CHECK(warm.worst_tv < 0.10);
    // Logit-space distance is reported for attribution and bounded only loosely: it is a
    // ulp-scale artifact, and a tight bound here would be a flaky assertion about bf16.
    CHECK(cold.worst_logit < 2.0);
    CHECK(warm.worst_logit < 2.0);
}

TEST(forward_logits_keeps_its_final_position_contract) {
    // forward_logits_all must be an EXTENSION of forward_logits, not a replacement: fed the
    // same input, the last row of one is the single row of the other. Bit-identical, which
    // is the strongest available evidence that adding the second entry point left the tuned
    // decode path's graph alone.
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
    CHECK_EQ(max_abs_diff(last_only[0], batched[k - 1]), 0.0);
    CHECK_EQ(argmax(last_only[0]), argmax(batched[k - 1]));
}

#else // !LMP_HAVE_MLX

TEST(spec_cache_requires_mlx) {
    lmp::test::record_failure(__FILE__, __LINE__,
                              "test_spec_cache is a realmodel test and needs an MLX build");
}

#endif // LMP_HAVE_MLX
