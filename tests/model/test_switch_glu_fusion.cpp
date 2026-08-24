// Fusing gate_proj and up_proj into one gather_qmm must not change a single bit.
//
// The claim the fusion rests on is arithmetic, not empirical: each output row of a
// quantized matmul is an independent dot product, and affine grouping runs along the
// INPUT axis, so stacking two weight matrices along their OUTPUT axis cannot move a
// group boundary or mix a scale into the wrong row. If that is true the fused and
// unfused paths agree exactly; if it is false they agree ALMOST everywhere, which is
// the failure this file exists to catch. A drift of one ulp per layer over 40 layers
// does not crash and does not show up in tok/s -- it shows up as slightly different
// text, weeks later, with no way left to bisect it.
//
// Synthetic weights, not the checkpoint: the property is a property of the op, and
// asserting it on four toy experts runs in milliseconds instead of loading 19 GB.
// It still needs Metal, and CI has none, so this is labelled `realmodel` despite
// never opening a checkpoint.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "src/model/mlx/switch_glu.hpp"
#include "src/model/mlx/weight_store.hpp"
#include "tests/check.hpp"

#if LMP_HAVE_MLX

#include "mlx/ops.h"

namespace mx = mlx::core;
using lmp::model::mlxl::WeightStore;

namespace {

constexpr int kExperts = 4;
constexpr int kHidden = 64;  // one full group at group_size 64
// Both matmuls quantize along their LAST axis, so both last axes must be a whole number
// of groups: gate/up contract over kHidden, down contracts over kFfn.
constexpr int kFfn = 128;
constexpr int kTopK = 2;
constexpr int kGroup = 64;
constexpr int kBits = 4;

// Deterministic, and deliberately not mx::random: the same weights on every run mean a
// failure here is reproducible without carrying a seed.
mx::array ramp(const std::vector<int>& shape, float step, float phase) {
    int n = 1;
    for (int d : shape) {
        n *= d;
    }
    std::vector<float> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        // Bounded and sign-changing, so quantization has something to round.
        v[static_cast<std::size_t>(i)] =
            0.5f * std::sin(static_cast<float>(i) * step + phase);
    }
    mx::Shape s(shape.begin(), shape.end());
    return mx::astype(mx::array(v.data(), s, mx::float32), mx::bfloat16);
}

// Quantize `w` and install it under `base` the way a converted checkpoint stores it.
void put_quantized(WeightStore& ws, const std::string& base, const mx::array& w) {
    const std::vector<mx::array> q = mx::quantize(w, kGroup, kBits, "affine");
    ws.set(base + ".weight", q[0]);
    ws.set(base + ".scales", q[1]);
    if (q.size() > 2) {
        ws.set(base + ".biases", q[2]);
    }
}

} // namespace

TEST(fusing_gate_and_up_changes_no_bit_of_the_expert_output) {
    WeightStore ws;

    // [experts, out, in] -- the layout gather_qmm(transpose=true) expects.
    const mx::array gate = ramp({kExperts, kFfn, kHidden}, 0.017f, 0.0f);
    const mx::array up = ramp({kExperts, kFfn, kHidden}, 0.023f, 1.3f);
    const mx::array down = ramp({kExperts, kHidden, kFfn}, 0.031f, 2.7f);

    put_quantized(ws, "e.gate_proj", gate);
    put_quantized(ws, "e.up_proj", up);
    put_quantized(ws, "e.down_proj", down);

    // The stacked entry, built exactly as fuse_expert_gate_up() builds it: gate rows
    // first, then up rows, concatenated on the OUTPUT axis.
    ws.set("e.gate_up_proj.weight",
           mx::concatenate({ws.get("e.gate_proj.weight"), ws.get("e.up_proj.weight")}, 1));
    ws.set("e.gate_up_proj.scales",
           mx::concatenate({ws.get("e.gate_proj.scales"), ws.get("e.up_proj.scales")}, 1));
    CHECK(ws.has("e.gate_proj.biases") == ws.has("e.up_proj.biases"));
    if (ws.has("e.gate_proj.biases")) {
        ws.set("e.gate_up_proj.biases",
               mx::concatenate({ws.get("e.gate_proj.biases"), ws.get("e.up_proj.biases")}, 1));
    }
    CHECK(ws.is_quantized("e.gate_up_proj"));

    const mx::array x = ramp({1, 1, kHidden}, 0.011f, 0.4f);
    const std::vector<int32_t> ids{2, 0};
    const mx::array inds = mx::array(ids.data(), {1, 1, kTopK}, mx::int32);

    const mx::array unfused =
        lmp::model::mlxl::switch_glu(x, ws, "e.gate_proj", "e.up_proj", "e.down_proj", inds);
    const mx::array fused = lmp::model::mlxl::switch_glu(
        x, ws, "e.gate_proj", "e.up_proj", "e.down_proj", inds, "e.gate_up_proj");

    mx::eval({unfused, fused});
    CHECK(unfused.shape() == fused.shape());

    // Exactly equal, not close. `max|a-b| == 0` rather than a tolerance: a tolerance here
    // would pass on precisely the bug described at the top of this file.
    const mx::array diff = mx::max(mx::abs(mx::subtract(unfused, fused)));
    const mx::array worst = mx::astype(diff, mx::float32);
    mx::eval(worst);
    const float max_abs_diff = worst.item<float>();
    std::printf("switch_glu fusion: max|fused - unfused| = %.9g\n",
                static_cast<double>(max_abs_diff));
    CHECK(max_abs_diff == 0.0f);

    // And it must not be trivially equal because both were zero.
    const mx::array mag = mx::astype(mx::max(mx::abs(unfused)), mx::float32);
    mx::eval(mag);
    CHECK(mag.item<float>() > 0.0f);

    // An empty fused key is the two-dispatch path, which is what every caller that was
    // not fused at load still passes.
    const mx::array explicit_unfused = lmp::model::mlxl::switch_glu(
        x, ws, "e.gate_proj", "e.up_proj", "e.down_proj", inds, std::string{});
    mx::eval(explicit_unfused);
    const mx::array d2 = mx::astype(mx::max(mx::abs(mx::subtract(unfused, explicit_unfused))),
                                    mx::float32);
    mx::eval(d2);
    CHECK(d2.item<float>() == 0.0f);

}

#else

TEST(switch_glu_fusion_needs_mlx) {
    std::printf("skipped: built without MLX\n");
}

#endif
