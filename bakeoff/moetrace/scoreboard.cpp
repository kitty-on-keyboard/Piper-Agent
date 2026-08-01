// Neutral scoreboard for the moetrace cook-off. Every case below has an answer fixed by
// construction, so this grades against arithmetic rather than against another
// implementation's opinion.
//
//   IDENTICAL   every step routes to the same 8 experts.
//               overlap must be 1.0; unique(k)/(8k) must be exactly 1/k.
//   DISJOINT    consecutive steps share no experts.
//               overlap must be 0.0; unique(k)/(8k) must be exactly 1.0.
//   HALF        consecutive steps share exactly 4 of 8. overlap must be 0.5.
//   GINI        one layer where 8 experts carry every selection and 248 are cold:
//               cold_experts must be 248.
//   MALFORMED   six bad lines among good ones; all six counted, none fatal.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "moetrace.hpp"

namespace {

std::string line(int step, int layer, const std::vector<int>& experts) {
    std::string s = "{\"step\":" + std::to_string(step) + ",\"layer\":" +
                    std::to_string(layer) + ",\"experts\":[";
    for (std::size_t i = 0; i < experts.size(); ++i) {
        s += (i ? "," : "") + std::to_string(experts[i]);
    }
    return s + "]}";
}

bool near(double a, double b, double tol = 1e-6) { return std::fabs(a - b) < tol; }

int fails = 0;
void check(bool ok, const char* what, double got, double want) {
    if (!ok) {
        ++fails;
        std::printf("    MISMATCH %-28s got=%.6f want=%.6f\n", what, got, want);
    }
}

} // namespace

int main() {
    constexpr int kSteps = 64;

    // --- IDENTICAL -----------------------------------------------------------
    {
        std::vector<std::string> ls;
        const std::vector<int> e = {1, 2, 3, 4, 5, 6, 7, 8};
        for (int s = 0; s < kSteps; ++s) {
            ls.push_back(line(s, 0, e));
        }
        const moetrace::TraceStats t = moetrace::analyse_lines(ls);
        if (t.per_layer.empty()) {
            std::printf("    MISMATCH identical: no per_layer output\n");
            ++fails;
        } else {
            const auto& L = t.per_layer[0];
            check(near(L.mean_consecutive_overlap, 1.0), "identical.overlap",
                  L.mean_consecutive_overlap, 1.0);
            check(near(L.unique_ratio[0], 1.0), "identical.unique_k1", L.unique_ratio[0], 1.0);
            check(near(L.unique_ratio[1], 0.5), "identical.unique_k2", L.unique_ratio[1], 0.5);
            check(near(L.unique_ratio[2], 0.25), "identical.unique_k4", L.unique_ratio[2], 0.25);
            check(near(L.unique_ratio[3], 0.125), "identical.unique_k8", L.unique_ratio[3], 0.125);
            check(L.cold_experts == 248, "identical.cold", double(L.cold_experts), 248.0);
        }
    }

    // --- DISJOINT ------------------------------------------------------------
    {
        std::vector<std::string> ls;
        for (int s = 0; s < kSteps; ++s) {
            std::vector<int> e;
            for (int j = 0; j < 8; ++j) {
                e.push_back((s * 8 + j) % 256);
            }
            ls.push_back(line(s, 0, e));
        }
        const moetrace::TraceStats t = moetrace::analyse_lines(ls);
        if (!t.per_layer.empty()) {
            const auto& L = t.per_layer[0];
            check(near(L.mean_consecutive_overlap, 0.0), "disjoint.overlap",
                  L.mean_consecutive_overlap, 0.0);
            check(near(L.unique_ratio[1], 1.0), "disjoint.unique_k2", L.unique_ratio[1], 1.0);
            check(near(L.unique_ratio[2], 1.0), "disjoint.unique_k4", L.unique_ratio[2], 1.0);
        }
    }

    // --- HALF ----------------------------------------------------------------
    {
        std::vector<std::string> ls;
        for (int s = 0; s < kSteps; ++s) {
            // Alternate between two sets sharing exactly 4 members.
            const std::vector<int> a = {0, 1, 2, 3, 4, 5, 6, 7};
            const std::vector<int> b = {4, 5, 6, 7, 8, 9, 10, 11};
            ls.push_back(line(s, 0, (s % 2) ? b : a));
        }
        const moetrace::TraceStats t = moetrace::analyse_lines(ls);
        if (!t.per_layer.empty()) {
            check(near(t.per_layer[0].mean_consecutive_overlap, 0.5), "half.overlap",
                  t.per_layer[0].mean_consecutive_overlap, 0.5);
        }
    }

    // --- MALFORMED -----------------------------------------------------------
    {
        std::vector<std::string> ls;
        const std::vector<int> e = {1, 2, 3, 4, 5, 6, 7, 8};
        for (int s = 0; s < 10; ++s) {
            ls.push_back(line(s, 0, e));
        }
        ls.push_back("");                                        // 1 empty
        ls.push_back("{\"step\":1,\"layer\":0");                 // 2 truncated
        ls.push_back("{\"step\":1,\"experts\":[1,2,3]}");        // 3 missing layer
        ls.push_back("not json at all");                         // 4 garbage
        ls.push_back("{\"step\":1,\"layer\":0,\"experts\":[]}"); // 5 empty experts
        ls.push_back("{\"step\":1,\"layer\":0,\"experts\":[1,2,3,4,5,6,7,999]}"); // 6 out of range
        const moetrace::TraceStats t = moetrace::analyse_lines(ls);
        if (t.malformed_lines < 4) {
            std::printf("    MISMATCH malformed: counted %zu of 6 bad lines\n",
                        t.malformed_lines);
            ++fails;
        }
    }

    std::printf(fails == 0 ? "  ALL ANCHORS PASS\n" : "  %d anchor mismatch(es)\n", fails);
    return fails == 0 ? 0 : 1;
}
