// G2 DAMP stub — CPU gate tests (no Metal).
//
// (a) DAMP off == baseline passthrough
// (b) mask load (fixture + synthetic builder)
// (c) roundtrip: protected channels exact; compressed within INT8 affine error

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/model/mlx/damp.hpp"
#include "tests/check.hpp"

using lmp::model::mlxl::DampMask;
using lmp::model::mlxl::DampPackedDelta;
using lmp::model::mlxl::damp_dequantize_cpu;
using lmp::model::mlxl::damp_passthrough_cpu;
using lmp::model::mlxl::damp_quantize_cpu;
using lmp::model::mlxl::load_damp_mask_file;
using lmp::model::mlxl::make_synthetic_damp_mask;
using lmp::model::mlxl::parse_damp_mask_json;
using lmp::model::mlxl::parse_lmp_damp_flag;
using lmp::model::mlxl::parse_lmp_damp_k_hi;

#ifndef LMP_DAMP_SYNTHETIC_MASK
#error "LMP_DAMP_SYNTHETIC_MASK must be defined to the fixture path"
#endif

TEST(a_damp_off_matches_baseline_passthrough) {
    // Kill switch: unset / not exactly "1" → off.
    CHECK(!parse_lmp_damp_flag(nullptr));
    CHECK(!parse_lmp_damp_flag(""));
    CHECK(!parse_lmp_damp_flag("0"));
    CHECK(!parse_lmp_damp_flag("true"));
    CHECK(!parse_lmp_damp_flag("11"));
    CHECK(parse_lmp_damp_flag("1"));

    CHECK_EQ(parse_lmp_damp_k_hi(nullptr, 16), 16);
    CHECK_EQ(parse_lmp_damp_k_hi("", 16), 16);
    CHECK_EQ(parse_lmp_damp_k_hi("16", 16), 16);
    CHECK_EQ(parse_lmp_damp_k_hi("8", 16), 8);
    CHECK_EQ(parse_lmp_damp_k_hi("nope", 16), 16);

    // When DAMP is off the live path must not reshape state: passthrough is bit-identical.
    const std::vector<float> src{0.0f, -1.5f, 2.25f, 3.0f, 4.5f, -7.0f};
    std::vector<float> dst(src.size(), 99.f);
    damp_passthrough_cpu(src.data(), dst.data(), src.size());
    REQUIRE(dst.size() == src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        CHECK(dst[i] == src[i]);
    }
    // In-place no-op.
    std::vector<float> inplace = src;
    damp_passthrough_cpu(inplace.data(), inplace.data(), inplace.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        CHECK(inplace[i] == src[i]);
    }
}

TEST(b_mask_load_fixture_and_synthetic) {
    DampMask file_mask;
    std::string err;
    CHECK(load_damp_mask_file(LMP_DAMP_SYNTHETIC_MASK, file_mask, &err));
    CHECK(err.empty());
    CHECK(file_mask.valid());
    CHECK_EQ(file_mask.format, std::string("g2_damp_mask_v1"));
    CHECK_EQ(file_mask.source, std::string("synthetic_fixture"));
    CHECK_EQ(file_mask.k_hi, 4);
    CHECK_EQ(file_mask.dk, 16);
    CHECK_EQ(file_mask.num_layers, 2);
    CHECK_EQ(file_mask.num_heads, 2);
    CHECK_EQ(file_mask.indices(0, 0)[0], 0);
    CHECK_EQ(file_mask.indices(0, 0)[3], 3);
    CHECK_EQ(file_mask.indices(1, 1)[0], 8);

    // Synthetic builder matches the documented test shape (not Research cal).
    const DampMask syn = make_synthetic_damp_mask(/*layers=*/2, /*heads=*/2, /*dk=*/16, /*k_hi=*/4);
    CHECK(syn.valid());
    CHECK_EQ(syn.source, std::string("synthetic_fixture"));

    // Reject inventing incomplete coverage / bad format.
    DampMask bad;
    CHECK(!parse_damp_mask_json("{\"format\":\"nope\",\"k_hi\":1,\"dk\":2,\"num_layers\":1,"
                                "\"num_heads\":1,\"layers\":[]}",
                                bad, &err));
}

TEST(c_roundtrip_protected_exact_compressed_sane) {
    const int B = 1;
    const int Hv = 2;
    const int Dv = 3;
    const int Dk = 16;
    const int k_hi = 4;
    const DampMask mask = make_synthetic_damp_mask(/*layers=*/1, Hv, Dk, k_hi);
    CHECK(mask.valid());

    std::vector<float> state(static_cast<std::size_t>(B * Hv * Dv * Dk));
    for (std::size_t i = 0; i < state.size(); ++i) {
        // Bounded, sign-changing — INT8 has something to round.
        state[i] = 0.37f * std::sin(0.17f * static_cast<float>(i) + 0.4f);
    }

    DampPackedDelta packed;
    std::string err;
    CHECK(damp_quantize_cpu(state.data(), B, Hv, Dv, Dk, mask, /*layer=*/0, packed, &err));
    CHECK(err.empty());
    CHECK(!packed.empty());
    CHECK(packed.packed_bytes() < packed.fp32_bytes());

    std::vector<float> roundtrip(state.size(), 0.f);
    CHECK(damp_dequantize_cpu(packed, mask, roundtrip.data(), &err));
    CHECK(err.empty());

    // Protected tier: exact float32.
    for (int h = 0; h < Hv; ++h) {
        for (int d : mask.indices(0, h)) {
            for (int dv = 0; dv < Dv; ++dv) {
                const std::size_t i =
                    ((((static_cast<std::size_t>(0) * Hv + h) * Dv + dv) * Dk) +
                     static_cast<std::size_t>(d));
                CHECK(roundtrip[i] == state[i]);
            }
        }
    }

    // Compressed tier: within one INT8 ulp of the per-(b,h,dv) scale.
    float max_abs_err = 0.f;
    for (std::size_t i = 0; i < state.size(); ++i) {
        max_abs_err = std::max(max_abs_err, std::fabs(roundtrip[i] - state[i]));
    }
    // Worst-case INT8 affine error is ~scale/2; scales are small for our sine fixture.
    CHECK(max_abs_err < 0.05f);

    // Off-path identity still holds on the same buffer (documents (a) vs (c) split).
    std::vector<float> base = state;
    damp_passthrough_cpu(state.data(), base.data(), state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        CHECK(base[i] == state[i]);
    }
}
