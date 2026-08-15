// Attribution driver (S19.3): every performance claim in this repo is made by running
// one of these subcommands, not by reasoning about the code.
//
//   lmp_diag scan [T ...]   gated-delta recurrence: fused kernel vs the reference
//                           op-per-timestep loop -- max deviation AND wall time
//   lmp_diag mask           what one decode step costs OUTSIDE the forward pass
//   lmp_diag step [n]       the REAL decode forward, split into CPU graph construction
//                           and GPU eval; combine with LMP_ABLATE (see below)
//   lmp_diag moestream [n]  the routed experts under a real step's access pattern
//   lmp_diag bench [runs] [prompt_tokens] [max_new]
//                           the whole stack, N runs, median + spread, next to the
//                           LM Studio numbers derived by scripts/lmstudio_baseline.py
//   lmp_diag graph [prompt] the decode step's graph as dot, unevaluated -- the only
//                           subcommand here that is not a timing. Diff its primitive
//                           histogram against mlx-lm's with scripts/graph_histogram.py
//
// LMP_ABLATE=routed|mlp|delta|deltakernel deletes one block from the forward pass and
// leaves the rest running. Output is garbage; only the rate means anything. Attribution
// by ablation on a real step is the only method here that has not lied: `blocks`, `moe`
// and `layers` are isolation benchmarks and all three produced confidently wrong
// answers (see the warning at the top of diag_moe.cpp).
//
// Ablations are NOT additive, and reading them as if they were is how this driver
// misattributed a 34.9 ms step. Removing the routed experts saved 25.8 ms and removing
// the gated-delta block saved 27.8 ms, inside the same 34.9 ms step -- because both
// blocks were slow for one shared reason (a float32 residual stream), not because
// either cost what its ablation appeared to charge it.
//
// Built on demand: cmake --build --preset dev --target lmp_diag

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "src/model/backend.hpp"
#include "src/model/chat_template.hpp"
#include "src/model/grammar.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/model/sampler.hpp"
#include "src/platform/clock.hpp"
#include "tests/model/diag_common.hpp"

#if LMP_HAVE_MLX
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/graph_utils.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

#include "src/model/mlx/gated_delta.hpp"
#include "src/model/mlx/qwen35_moe_model.hpp"
#include "src/model/mlx/switch_glu.hpp"
#include "src/model/mlx/weight_store.hpp"
#endif

using namespace lmp::model;
using Clock = std::chrono::steady_clock;

namespace {

using lmp::diag::ms;
using lmp::diag::draft_dir;
using lmp::diag::qwen_dir;
#if LMP_HAVE_MLX
using lmp::diag::cmd_blocks;
using lmp::diag::cmd_moe;
using lmp::diag::cmd_moe_stream;
namespace mlxl = lmp::model::mlxl;
namespace mx = mlx::core;
#endif

// Median plus the full spread. A single number is the mistake this driver exists to
// stop: v1 measured 462 s, then 275 s and 258 s, on an identical binary.
struct Ledger {
    std::vector<double> xs;
    void add(double x) { xs.push_back(x); }
    void print(const char* label, const char* unit, double target) {
        if (xs.empty()) {
            std::printf("  %-10s no samples\n", label);
            return;
        }
        std::vector<double> s = xs;
        std::sort(s.begin(), s.end());
        const double med = s.size() % 2 == 1
                               ? s[s.size() / 2]
                               : 0.5 * (s[s.size() / 2 - 1] + s[s.size() / 2]);
        std::printf("  %-10s n=%zu median=%8.1f %s  min=%.1f max=%.1f", label, s.size(), med,
                    unit, s.front(), s.back());
        if (target > 0) {
            std::printf("   [LM Studio %.1f -> %s, %.2fx]", target,
                        med > target ? "PASS" : "FAIL", med / target);
        }
        std::printf("\n");
    }
};

// LM Studio's prefill throughput is a strong function of prompt length -- 824 tok/s at
// 512-1024 uncached tokens, 1529 at 8192-16384 -- so there is no single number to beat.
// Comparing our 547-token prefill against 1347 cost this project a pass: 1347 is the
// median over windows >= 5 s, and those windows have a *median prompt of 9172 tokens*.
//
// Medians per bucket, from scripts/lmstudio_baseline.py's parse over the same logs.
// `trusted` is false where the bucket's median window is 1-2 s: the logs have 1-second
// timestamps, so a sub-second prefill either straddles a tick and reads as a full second
// (understating throughput, badly) or does not and is dropped by the dt > 0 filter. Those
// buckets are a floor, not a measurement, and are printed without a PASS/FAIL verdict.
struct PrefillBar {
    int max_prompt;
    double tok_per_s;
    bool trusted;
};
constexpr PrefillBar kPrefillBars[] = {
    {1024, 824.0, false},   {2048, 1127.0, false},   {4096, 1155.0, false},
    {8192, 1350.0, true},   {16384, 1529.0, true},   {1 << 30, 1334.0, true},
};

const PrefillBar& prefill_bar(int prompt_tokens) {
    for (const PrefillBar& b : kPrefillBars) {
        if (prompt_tokens < b.max_prompt) {
            return b;
        }
    }
    return kPrefillBars[std::size(kPrefillBars) - 1];
}

// --- scan ------------------------------------------------------------------

int cmd_scan(const std::vector<int>& lengths) {
#if !LMP_HAVE_MLX
    std::printf("MLX not compiled in\n");
    return 1;
#else
    namespace mx = mlx::core;
    // Qwen 3.6's linear-attention shapes, from config.json.
    const int B = 1, Hk = 16, Hv = 32, Dk = 128, Dv = 128;
    std::printf("gated-delta scan  B=%d Hk=%d Hv=%d Dk=%d Dv=%d\n", B, Hk, Hv, Dk, Dv);
    std::printf("%6s %14s %14s %9s %12s %12s\n", "T", "ops ms", "kernel ms", "speedup",
                "max|dy|", "max|dstate|");

    for (int T : lengths) {
        mx::random::seed(7);
        // q and k as forward_gated_delta hands them over: rms-normed, then scaled by
        // 1/sqrt(Dk) (twice for q). That scaling is what makes |k|^2 ~ 1, so the
        // transition (I - beta k k^T) is contractive. Feeding raw normals instead
        // makes the recurrence itself divergent -- both implementations then blow up
        // identically in exact arithmetic and exponentially amplify any fp difference,
        // which measures the test's inputs rather than the kernel.
        const float inv = 1.0f / std::sqrt(static_cast<float>(Dk));
        const mx::array ones = mx::ones({Dk}, mx::float32);
        mx::array q = mx::multiply(
            mx::array(inv * inv),
            mx::fast::rms_norm(mx::random::normal({B, T, Hk, Dk}, mx::float32), ones, 1e-6f));
        mx::array k = mx::multiply(
            mx::array(inv),
            mx::fast::rms_norm(mx::random::normal({B, T, Hk, Dk}, mx::float32), ones, 1e-6f));
        mx::array v = mx::random::normal({B, T, Hv, Dv}, mx::float32);
        // g in (0,1): it is exp(-exp(A_log)*softplus(...)), a decay factor.
        mx::array g = mx::sigmoid(mx::random::normal({B, T, Hv}, mx::float32));
        mx::array beta = mx::sigmoid(mx::random::normal({B, T, Hv}, mx::float32));
        // Prefill enters with a zero state; decode enters with the carried one.
        mx::array st0 = mx::zeros({B, Hv, Dv, Dk}, mx::float32);
        mx::eval({q, k, v, g, beta, st0});

        // GQA repeat happens inside each implementation; feed both the same inputs.
        auto t0 = Clock::now();
        auto [y_ops, s_ops] = mlxl::gated_delta_update_ops(q, k, v, g, beta, st0);
        mx::eval({y_ops, s_ops});
        auto t1 = Clock::now();

        // Warm the kernel cache so the first JIT compile is not charged to the timing.
        auto [warm_y, warm_s] = mlxl::gated_delta_update_kernel(q, k, v, g, beta, st0);
        mx::eval({warm_y, warm_s});
        auto t2 = Clock::now();
        auto [y_k, s_k] = mlxl::gated_delta_update_kernel(q, k, v, g, beta, st0);
        mx::eval({y_k, s_k});
        auto t3 = Clock::now();

        // Relative to the reference's own scale: an absolute deviation means nothing
        // without the magnitude it is a deviation from.
        mx::array dy = mx::divide(mx::max(mx::abs(mx::subtract(y_ops, y_k))),
                                  mx::max(mx::abs(y_ops)));
        mx::array ds = mx::divide(mx::max(mx::abs(mx::subtract(s_ops, s_k))),
                                  mx::max(mx::abs(s_ops)));
        mx::eval({dy, ds});
        const double ops_ms = ms(t0, t1);
        const double ker_ms = ms(t2, t3);
        std::printf("%6d %14.1f %14.2f %8.1fx %12.3e %12.3e\n", T, ops_ms, ker_ms,
                    ops_ms / ker_ms, static_cast<double>(dy.item<float>()),
                    static_cast<double>(ds.item<float>()));
    }
    return 0;
#endif
}

// --- step ------------------------------------------------------------------
//
// The real decode forward: prefill a prompt, then time forward_logits() one token at a
// time, exactly as MlxBackend::generate calls it -- one eval per step, nothing else in
// the loop. This is the only per-step number in this driver that is not an isolation
// benchmark, so when it disagrees with `layers` or `moe`, believe this one.
//
// Combine with LMP_ABLATE=routed|mlp|delta for attribution that costs no sampling, no
// grammar and no host copy.

#if LMP_HAVE_MLX
int cmd_step(int steps) {
    mlxl::Qwen35MoeModel model;
    if (!model.load(qwen_dir())) {
        std::printf("model load failed\n");
        return 1;
    }
    std::vector<int32_t> prompt(547, 100);
    mx::array ids = mx::array(prompt.data(), {1, static_cast<int>(prompt.size())}, mx::int32);
    mx::array logits = model.forward_logits(ids);
    mx::eval(logits);

    int32_t tok = 42;
    double build_ms = 0;
    double eval_ms = 0;
    const auto t0 = Clock::now();
    for (int i = 0; i < steps; ++i) {
        mx::array one = mx::array(&tok, {1, 1}, mx::int32);
        const auto tb = Clock::now();
        logits = model.forward_logits(one);
        const auto te = Clock::now();
        mx::eval(logits);
        const auto tf = Clock::now();
        build_ms += ms(tb, te);
        eval_ms += ms(te, tf);
    }
    const double per = ms(t0, Clock::now()) / steps;
    const char* ab = std::getenv("LMP_ABLATE");
    std::printf("  forward_logits T=1   %7.2f ms/token   %6.1f tok/s   [ablate=%s]\n", per,
                1000.0 / per, ab != nullptr ? ab : "none");
    // Graph construction is CPU; eval is the GPU actually running it. If build is the
    // larger half, no amount of kernel work will help -- the GPU is being starved.
    std::printf("      build (cpu) %6.2f ms   eval (gpu) %6.2f ms\n", build_ms / steps,
                eval_ms / steps);

    // Of that build, weight-key resolution -- the std::string concatenation and hash
    // probing this model does per weight access, ~800 a step -- was measured at 0.27 ms.
    // The other ~0.7 ms is MLX op construction itself, which no amount of handle caching
    // removes. That is why there is no weight-handle cache here: it was measured before
    // it was written, and it buys about 2%.
    return 0;
}
#endif

// --- graph -----------------------------------------------------------------
//
// Dump the decode step's graph, unevaluated, as dot. This is the one instrument here
// that is not a timing, and that is the point: every other subcommand answers "how
// long did this take", which on this project has been wrong four times. A primitive
// histogram answers "what work is in the graph", which is a fact about the program
// rather than about the machine it ran on.
//
// It exists to be diffed against the same dump from mlx-lm. Two stacks that claim to
// run the same ops on the same shapes must produce the same histogram; a primitive
// that appears in one and not the other is real work one of them is doing:
//
//   lmp_diag graph > ours.dot
//   scripts/graph_histogram.py --dot ours.dot --json ours.json
//   <lmstudio python> scripts/graph_histogram.py --reference --json ref.json
//   scripts/graph_histogram.py --compare ours.json ref.json

#if LMP_HAVE_MLX
int cmd_graph(int prompt_tokens) {
    mlxl::Qwen35MoeModel model;
    if (!model.load(qwen_dir())) {
        std::fprintf(stderr, "model load failed\n");
        return 1;
    }
    // Prefill first and force it, so the caches hold evaluated state. Without this the
    // dump would contain the whole prefill graph behind the step and the histogram would
    // describe a first-token forward, not a decode step.
    std::vector<int32_t> prompt(static_cast<std::size_t>(prompt_tokens), 100);
    mx::array ids = mx::array(prompt.data(), {1, prompt_tokens}, mx::int32);
    mx::array logits = model.forward_logits(ids);
    mx::eval(logits);
    model.eval_caches();

    int32_t tok = 42;
    mx::array one = mx::array(&tok, {1, 1}, mx::int32);
    mx::array step = model.forward_logits(one);
    mx::export_to_dot(std::cout, {step});
    return 0;
}
#endif

// --- layers ----------------------------------------------------------------
//
// Per-layer cost using the model's OWN block functions, chained the way a real step
// chains them (each layer depends on the previous), so the number includes the
// dependent-op latency that `blocks` deliberately batches away.

#if LMP_HAVE_MLX
int cmd_layers(int T) {
    mlxl::Qwen35MoeModel model;
    if (!model.load(qwen_dir())) {
        std::printf("model load failed\n");
        return 1;
    }
    const int hidden = model.qwen_config().hidden_size;
    // bfloat16, because that is what the checkpoint is. Feeding float16 here promotes
    // inside every quantized matmul and costs ~6x -- measured against mlx-lm's own block
    // on this machine: 0.304 ms/layer in bf16, 0.701 in fp16. This line used to say
    // float16, which is why every number this command has ever printed was an fp16
    // number for a bf16 model.
    mx::array x = mx::zeros({1, T, hidden}, mx::bfloat16);
    mx::eval(x);

    const int reps = T == 1 ? 30 : 4;
    // Two timings of the SAME block, differing only in whether call i+1 consumes call
    // i's output. If chained >> independent, the cost is stall between dependent
    // kernels. If they match, the cost is per-call and the dependency is innocent --
    // and those two findings point at completely different fixes.
    const auto run = [&](const char* label, int layers, int total_layers,
                         const std::function<mx::array(mx::array)>& step) {
        model.reset_cache();
        mx::array warm = step(x);
        mx::eval(warm);

        model.reset_cache();
        auto t0 = Clock::now();
        mx::array h = x;
        for (int i = 0; i < reps; ++i) {
            h = step(h);
        }
        mx::eval(h);
        const double chained = ms(t0, Clock::now()) / reps;

        model.reset_cache();
        auto t1 = Clock::now();
        std::vector<mx::array> outs;
        outs.reserve(static_cast<std::size_t>(reps));
        for (int i = 0; i < reps; ++i) {
            outs.push_back(step(x));
        }
        mx::eval(outs);
        const double indep = ms(t1, Clock::now()) / reps;

        std::printf("  %-24s chained %7.3f  independent %7.3f ms/layer  x%2d = %6.2f ms\n",
                    label, chained, indep, total_layers, chained * total_layers);
        (void)layers;
        return chained * total_layers;
    };

    double total = 0;
    total += run("linear layer (delta+moe)", 1, 30,
                 [&](mx::array h) { return model.forward_linear_layer(0, h, T); });
    total += run("attn layer (attn+moe)", 1, 10,
                 [&](mx::array h) { return model.forward_full_attn_layer(3, h, T); });
    const double moe = run("  of which moe", 1, 40,
                           [&](mx::array h) { return model.forward_moe(0, h); });
    std::printf("  %-24s %7s              %6.2f ms/token  (moe share %.2f ms)\n", "TOTAL 40 layers",
                "", total, moe);
    return 0;
}
#endif

// --- chain -----------------------------------------------------------------
//
// A decode step is ~2,600 MLX ops in a dependency chain. The blocks measurement showed
// the GPU work is only ~8 ms and CPU graph construction is under 1 ms, so the missing
// ~26 ms has to be per-op cost that only shows up when ops cannot overlap. This
// measures exactly that: the same op count, dependent vs independent.

#if LMP_HAVE_MLX
int cmd_chain(int n) {
    const mx::array c = mx::array(1.0001f, mx::float16);
    const mx::array x0 = mx::zeros({1, 1, 2048}, mx::float16);
    mx::eval({c, x0});

    mx::array warm = mx::multiply(x0, c);
    mx::eval(warm);

    auto t0 = Clock::now();
    mx::array chained = x0;
    for (int i = 0; i < n; ++i) {
        chained = mx::multiply(chained, c);
    }
    mx::eval(chained);
    const double dep = ms(t0, Clock::now());

    auto t1 = Clock::now();
    std::vector<mx::array> indep;
    indep.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        indep.push_back(mx::multiply(x0, c));
    }
    mx::eval(indep);
    const double ind = ms(t1, Clock::now());

    // compute_g as it is written today: nine elementwise ops on a [1,1,32] tensor,
    // once per linear layer per token. Compiled, MLX fuses the chain into one kernel.
    const mx::array a = mx::zeros({1, 1, 32}, mx::float32);
    const mx::array bias = mx::zeros({32}, mx::float32);
    const mx::array alog = mx::zeros({32}, mx::float32);
    mx::eval({a, bias, alog});
    const auto cg = [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        return {mlxl::compute_g(in[2], in[0], in[1])};
    };
    auto compiled = mx::compile(cg);
    mx::eval(compiled({a, bias, alog}));
    const int reps = n / 10;

    auto t2 = Clock::now();
    std::vector<mx::array> raw;
    for (int i = 0; i < reps; ++i) {
        raw.push_back(mlxl::compute_g(alog, a, bias));
    }
    mx::eval(raw);
    const double raw_ms = ms(t2, Clock::now());

    auto t3 = Clock::now();
    std::vector<mx::array> comp;
    for (int i = 0; i < reps; ++i) {
        comp.push_back(compiled({a, bias, alog})[0]);
    }
    mx::eval(comp);
    const double comp_ms = ms(t3, Clock::now());

    std::printf("%d tiny ops on [1,1,2048] f16:\n", n);
    std::printf("  dependent chain   : %8.2f ms  = %6.2f us/op\n", dep, dep * 1000.0 / n);
    std::printf("  independent       : %8.2f ms  = %6.2f us/op\n", ind, ind * 1000.0 / n);
    std::printf("compute_g x%d (9 elementwise ops each):\n", reps);
    std::printf("  as written        : %8.2f ms  = %6.2f us/call\n", raw_ms,
                raw_ms * 1000.0 / reps);
    std::printf("  mx::compile'd     : %8.2f ms  = %6.2f us/call   (%.1fx)\n", comp_ms,
                comp_ms * 1000.0 / reps, raw_ms / comp_ms);
    std::printf("\nA decode step is a dependency chain of roughly 2,600 ops, so the\n"
                "dependent number is the one that sets the floor: %.1f ms/token,\n"
                "against LM Studio's 12.7 ms/token (78.5 tok/s).\n",
                dep * 2600.0 / n);
    return 0;
}
#endif

// --- mask ------------------------------------------------------------------

int cmd_mask() {
    QwenTokenizer tok;
    const LoadStatus st = tok.load(std::string(qwen_dir()) + "/tokenizer.json", Family::Qwen3);
    if (!st.ok) {
        std::printf("tok fail: %s\n", st.error.c_str());
        return 1;
    }
    const std::size_t V = tok.vocab_size();
    std::printf("vocab = %zu\n\n", V);

    std::vector<parsephony::ToolSpec> tools;
    parsephony::ToolSpec s;
    s.name = "read_file";
    parsephony::ParamSpec p;
    p.name = "path";
    p.required = true;
    s.params.push_back(p);
    tools.push_back(s);
    TurnGrammar g(tok, tools);

    // 1. the per-id predicate, exactly as the sampler used to call it
    auto t0 = Clock::now();
    std::size_t allowed = 0;
    for (std::size_t i = 0; i < V; ++i) {
        allowed += g.permitted(static_cast<TokenId>(i)) ? 1 : 0;
    }
    auto t1 = Clock::now();
    std::printf("permitted() over full vocab : %8.2f ms/token   (%zu allowed)\n", ms(t0, t1),
                allowed);

    // 2. the bulk mask the sampler actually consults now
    auto t2 = Clock::now();
    const TokenMask& m = g.mask();
    auto t3 = Clock::now();
    std::printf("bulk mask, cold             : %8.3f ms/token   (%zu allowed)\n", ms(t2, t3),
                m.count());
    auto t4 = Clock::now();
    for (int i = 0; i < 100; ++i) {
        const TokenMask& mm = g.mask();
        if (mm.size() == 0) {
            return 1;
        }
    }
    auto t5 = Clock::now();
    std::printf("bulk mask, warm             : %8.3f ms/token\n", ms(t4, t5) / 100.0);

    // 3. the sampler, with and without the mask
    std::vector<float> logits(V);
    for (std::size_t i = 0; i < V; ++i) {
        logits[i] = static_cast<float>(i % 97) * 0.01F;
    }
    Sampler smp{SamplingParams{}};
    auto l2 = logits;
    auto t6 = Clock::now();
    (void)smp.sample(l2, nullptr, {});
    auto t7 = Clock::now();
    std::printf("sampler, no mask            : %8.2f ms/token\n", ms(t6, t7));

    auto l3 = logits;
    auto t8 = Clock::now();
    (void)smp.sample(l3, &m, {});
    auto t9 = Clock::now();
    std::printf("sampler + mask (as shipped) : %8.2f ms/token\n\n", ms(t8, t9));

    const double per_tok = ms(t8, t9);
    std::printf("=> CPU-side ceiling from this alone: %.1f tok/s\n", 1000.0 / per_tok);
    std::printf("   (LM Studio measured 78.5 tok/s median on the same model/machine)\n");
    return 0;
}

// What the mask costs INSIDE a tool call, which is the one number that decides whether
// speculation can be extended there.
//
// Today speculation is gated on `mask_is_block_stable()` -- true only in Think and Text,
// where the legal set is a cached per-phase bitset. Inside a tool call parsephony's mask
// is state-dependent per token, so verifying k drafted positions needs k+1 masks rather
// than one, each computed by advancing a COPY of the grammar over the draft.
//
// That copy is not the problem: ToolCallGuard is 272 bytes, is documented copyable, and
// the mask engine already copies it once per candidate token (mask.hpp:151). The question
// is only what re-deriving the mask costs per position. Against a ~60 ms forward pass a
// few hundred microseconds is free and a few milliseconds is not.
//
// Measured on the real tokenizer and the real guard; loads no weights, so it is safe to
// run beside a live sidecar.
int cmd_toolmask(int k) {
    QwenTokenizer tok;
    const LoadStatus st = tok.load(std::string(qwen_dir()) + "/tokenizer.json", Family::Qwen3);
    if (!st.ok) {
        std::printf("tok fail: %s\n", st.error.c_str());
        return 1;
    }
    std::vector<parsephony::ToolSpec> tools;
    parsephony::ToolSpec s;
    s.name = "read_file";
    parsephony::ParamSpec p;
    p.name = "path";
    p.required = true;
    s.params.push_back(p);
    tools.push_back(s);

    TurnGrammar g(tok, tools);
    // Into the tool-call phase the way a real turn gets there.
    if (g.advance(tok.specials().think_close) == Advance::Rejected ||
        g.advance(tok.specials().tool_call_open) == Advance::Rejected) {
        std::printf("could not open a tool call\n");
        return 1;
    }
    // A realistic body: the guard is past the framing and inside a parameter value, which
    // is where a decode spends nearly all of its in-call tokens.
    const std::vector<TokenId> body = tok.encode_content("<function=read_file>\n<parameter=path>\nsrc/model/");
    for (const TokenId id : body) {
        if (g.advance(id) == Advance::Rejected) {
            break;
        }
    }
    std::printf("  phase inside call = %d (2 == ToolCall), mask allows %zu ids\n",
                static_cast<int>(g.phase()), g.mask().count());

    // One position, warm: what the sampler pays today for every in-call token.
    const std::vector<TokenId> more = tok.encode_content("qwen35_moe_model.hpp");
    const int iters = 200;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        const TokenMask& m = g.mask();
        if (m.size() == 0) {
            return 1;
        }
    }
    const double warm = ms(t0, Clock::now()) / iters;
    std::printf("  mask(), warm (same state)   : %8.4f ms\n", warm);

    // The speculative shape: advance over k fresh tokens, taking a mask at each. Every
    // step lands on a state the cache has not seen, so this is the cold path k times --
    // the honest cost of verifying a k-token draft inside a call.
    auto t1 = Clock::now();
    int rounds = 0;
    for (int r = 0; r < 50; ++r) {
        for (int i = 0; i < k && i < static_cast<int>(more.size()); ++i) {
            if (g.advance(more[static_cast<std::size_t>(i)]) == Advance::Rejected) {
                break;
            }
            const TokenMask& m = g.mask();
            if (m.size() == 0) {
                return 1;
            }
        }
        ++rounds;
    }
    const double per_block = ms(t1, Clock::now()) / rounds;
    std::printf("  %d advance+mask (a block)    : %8.4f ms   (%.4f ms/position)\n", k,
                per_block, per_block / k);
    std::printf("\n  => against a ~60 ms forward pass that is %.2f%% of a block\n",
                100.0 * per_block / 60.0);
    return 0;
}

// The reproduction that was missing when tool-call speculation shipped a regression.
//
// Every test written for that path checked ONE rollback in isolation. What none of them
// did was drive a REAL tokenizer's tool-call grammar through a realistic token stream
// while probing it the way the decoder actually probes -- checkpoint, walk a draft
// forward reading the mask at each position, roll back -- and check that the grammar is
// still bit-identical to one that was never probed at all.
//
// Two grammars, same committed tokens. A is the control and is never probed; B is probed
// at every step. They must agree on phase, on the mask, and on what they parsed. And
// neither may ever produce an EMPTY mask, which is the condition that aborts a run with
// "no legal token".
int cmd_specgrammar(int depth) {
    QwenTokenizer tok;
    const LoadStatus st = tok.load(std::string(qwen_dir()) + "/tokenizer.json", Family::Qwen3);
    if (!st.ok) {
        std::printf("tok fail: %s\n", st.error.c_str());
        return 1;
    }
    std::vector<parsephony::ToolSpec> tools;
    {
        parsephony::ToolSpec s;
        s.name = "read_file";
        parsephony::ParamSpec p;
        p.name = "path";
        p.required = true;
        s.params.push_back(p);
        tools.push_back(s);
    }

    // A real call, of the shape the failing run was emitting when it died.
    const std::string body =
        "<function=read_file>\n<parameter=path>\n"
        "tools/blender_vehicles/render_truck.py\n</parameter>\n</function>\n";
    std::vector<TokenId> stream;
    stream.push_back(tok.specials().think_close);
    stream.push_back(tok.specials().tool_call_open);
    for (const TokenId id : tok.encode_content(body)) {
        stream.push_back(id);
    }
    stream.push_back(tok.specials().tool_call_close);

    TurnGrammar a(tok, tools); // control: never probed
    TurnGrammar b(tok, tools); // probed at every step, exactly as the decoder does

    std::printf("  %zu tokens, probe depth %d\n", stream.size(), depth);
    int problems = 0;
    for (std::size_t i = 0; i < stream.size(); ++i) {
        // The mask BEFORE consuming stream[i] is what the sampler would have used.
        const TokenMask ma = a.mask();
        if (ma.count() == 0) {
            std::printf("  [EMPTY MASK] control grammar at step %zu, phase %d -- this is a "
                        "pre-existing grammar/vocabulary defect, not a rollback bug\n",
                        i, static_cast<int>(a.phase()));
            ++problems;
        }

        // What the decoder does before verifying a block: walk `depth` tokens forward,
        // reading the mask at each, then put the grammar back.
        if (b.can_checkpoint()) {
            b.checkpoint();
            for (int d = 0; d < depth; ++d) {
                // A DRAFTER GUESSES, and mostly guesses wrong. Alternating the real
                // continuation with an arbitrary id is what the decoder actually walks:
                // the probe that gets refused half way through a multi-byte token is the
                // interesting one, because it leaves the automaton mid-literal.
                const std::size_t j = i + static_cast<std::size_t>(d);
                const bool guess = (i + static_cast<std::size_t>(d)) % 2 == 1;
                TokenId cand;
                if (guess) {
                    cand = static_cast<TokenId>((i * 1103515245U + d * 12345U) %
                                                static_cast<std::size_t>(tok.vocab_size()));
                } else if (j < stream.size()) {
                    cand = stream[j];
                } else {
                    break;
                }
                const TokenMask here = b.mask();
                if (!here.allows(cand) || !b.probe_advance(cand)) {
                    break;
                }
            }
            b.rollback();
        }

        const TokenMask mb = b.mask();
        if (mb.words() != ma.words() || a.phase() != b.phase()) {
            std::printf("  [DIVERGED] step %zu: control phase %d mask %zu allowed; probed "
                        "phase %d mask %zu allowed\n",
                        i, static_cast<int>(a.phase()), ma.count(),
                        static_cast<int>(b.phase()), mb.count());
            ++problems;
            if (problems > 3) {
                std::printf("  (stopping after 4)\n");
                return 1;
            }
        }
        const Advance aa = a.advance(stream[i]);
        const Advance ab = b.advance(stream[i]);
        if (aa != ab) {
            std::printf("  [DIVERGED] step %zu: control advance %d, probed advance %d\n", i,
                        static_cast<int>(aa), static_cast<int>(ab));
            ++problems;
        }
        if (aa == Advance::Rejected) {
            std::printf("  [REJECTED] step %zu -- the stream itself is not legal\n", i);
            return 1;
        }
    }

    // DEAD-END SEARCH. The mask is one token deep: it says "this id is legal here", not
    // "this id leaves a continuation". A single-token decoder is protected by the model
    // sampling sensibly; a DRAFTER walks the automaton k tokens forward on a guess, so it
    // can land it somewhere no token in the vocabulary can leave. The position after that
    // has an empty legal set, and an empty row reaching the verifier is what raises
    // "no legal token" and kills the run.
    {
        TurnGrammar g(tok, tools);
        std::size_t found = 0;
        for (std::size_t i = 0; i < stream.size() && found < 5; ++i) {
            if (g.phase() == TurnPhase::ToolCall) {
                const TokenMask here = g.mask();
                // checkpoint/rollback rather than replaying the stream per candidate --
                // the same primitive the decoder uses, which turns a scan that does not
                // finish into one that takes seconds.
                for (std::size_t id = 0; id < tok.vocab_size() && found < 5; ++id) {
                    if (!here.allows(static_cast<TokenId>(id))) {
                        continue;
                    }
                    g.checkpoint();
                    const Advance a2 = g.advance(static_cast<TokenId>(id));
                    const bool dead = a2 != Advance::Rejected &&
                                      g.phase() == TurnPhase::ToolCall &&
                                      g.mask().count() == 0;
                    g.rollback();
                    if (dead) {
                        std::printf("  [DEAD END] step %zu: token %zu (%.24s) is ALLOWED but "
                                    "leaves an EMPTY mask\n",
                                    i, id,
                                    std::string(tok.token_bytes(static_cast<TokenId>(id)))
                                        .c_str());
                        ++found;
                    }
                }
            }
            if (g.advance(stream[i]) == Advance::Rejected) {
                break;
            }
        }
        if (found == 0) {
            std::printf("  no dead-end tokens found on this stream\n");
        } else {
            problems += static_cast<int>(found);
        }
    }

    const bool same_calls = a.has_tool_call() == b.has_tool_call() &&
                            (!a.has_tool_call() ||
                             (a.tool_name() == b.tool_name() &&
                              a.tool_params().size() == b.tool_params().size() &&
                              (a.tool_params().empty() ||
                               a.tool_params()[0].value == b.tool_params()[0].value)));
    if (!same_calls) {
        std::printf("  [DIVERGED] parsed call differs: control '%s' = '%s', probed '%s' = '%s'\n",
                    a.has_tool_call() ? a.tool_name().c_str() : "-",
                    (a.has_tool_call() && !a.tool_params().empty())
                        ? a.tool_params()[0].value.c_str() : "-",
                    b.has_tool_call() ? b.tool_name().c_str() : "-",
                    (b.has_tool_call() && !b.tool_params().empty())
                        ? b.tool_params()[0].value.c_str() : "-");
        ++problems;
    }
    std::printf(problems == 0 ? "  OK: probing left the grammar identical throughout\n"
                              : "  %d problem(s)\n", problems);
    return problems == 0 ? 0 : 1;
}

// The other half of the missing reproduction: the REAL SpeculativeDecoder, driven by a
// REAL TurnGrammar over a REAL tool-call token stream.
//
// `specgrammar` proves the grammar survives being probed. It cannot prove the decoder
// uses the result correctly, because the decoder was only ever tested against a 64-token
// fake vocabulary with no grammar at all. This closes that: if a block inside a tool call
// can produce an empty shaped distribution, it happens here.
int cmd_specrun(int rounds) {
    QwenTokenizer tok;
    const LoadStatus st = tok.load(std::string(qwen_dir()) + "/tokenizer.json", Family::Qwen3);
    if (!st.ok) {
        std::printf("tok fail: %s\n", st.error.c_str());
        return 1;
    }
    std::vector<parsephony::ToolSpec> tools;
    {
        parsephony::ToolSpec s;
        s.name = "read_file";
        parsephony::ParamSpec p;
        p.name = "path";
        p.required = true;
        s.params.push_back(p);
        tools.push_back(s);
    }
    // THE MODEL'S LOGITS ROW IS WIDER THAN THE TOKENIZER'S VOCABULARY -- 248,320 against
    // 248,077 on this checkpoint. The drafter takes an argmax over the MODEL's row, so it
    // can legitimately return an id the tokenizer has never heard of and the mask cannot
    // represent. Sizing this harness to the tokenizer instead of the model is what made
    // the first three attempts at this reproduction pass.
    const std::size_t V = 248320;

    // The turn the failing run was in the middle of: think, then a read_file call.
    std::vector<TokenId> stream;
    for (const TokenId id : tok.encode_content("Let me read the render helper.")) {
        stream.push_back(id);
    }
    stream.push_back(tok.specials().think_close);
    stream.push_back(tok.specials().tool_call_open);
    for (const TokenId id : tok.encode_content(
             "<function=read_file>\n<parameter=path>\n"
             "tools/blender_vehicles/render_truck.py\n</parameter>\n</function>\n")) {
        stream.push_back(id);
    }
    stream.push_back(tok.specials().tool_call_close);
    stream.push_back(tok.specials().im_end);

    // A model that always says "the next token of the stream", plus an MTP head that
    // drafts the same -- so acceptance is high and blocks are wide, which is the
    // condition the regression appeared under.
    class Fwd final : public SpecForward {
      public:
        Fwd(const std::vector<TokenId>& s, std::size_t vocab) : s_(s), v_(vocab) {}
        void forward_all(std::span<const TokenId> t,
                         std::vector<std::vector<float>>& rows) override {
            rows.clear();
            for (std::size_t i = 0; i < t.size(); ++i) {
                ++pos_;
                rows.push_back(row(pos_));
            }
            verified_ = true;
        }
        void forward_last(std::span<const TokenId> t, std::vector<float>& r) override {
            primed_ = false;
            for (std::size_t i = 0; i < t.size(); ++i) {
                ++pos_;
            }
            r = row(pos_);
        }
        void checkpoint() override { mark_ = pos_; }
        void restore() override { pos_ = mark_; }
        [[nodiscard]] bool has_mtp() const override { return true; }
        void last_hidden(std::vector<std::vector<float>>& rows) override {
            rows.assign(1, std::vector<float>{1.0F});
            if (verified_) {
                cur_ = pos_ + 1;
                verified_ = false;
                primed_ = true;
            } else if (!primed_) {
                cur_ = pos_;
            }
        }
        void mtp_step(TokenId, std::span<const float>, std::vector<float>& o) override {
            o = {1.0F};
        }
        void mtp_logits(std::span<const float>, std::vector<float>& r) override {
            // A drafter that is right most of the time and wrong the rest, which is what
            // makes blocks PARTIALLY accepted -- and partial acceptance is what drives
            // restore, a growing deferred prefix, and verification passes that carry a
            // prefix in front of the drafts. A perfect drafter exercises none of it.
            const std::size_t p = cur_++;
            if (wrong_every_ > 0 && (p % wrong_every_) == 0) {
                r.assign(v_, 0.0F);
                // A PLAUSIBLE wrong guess, which is the only kind that matters. An
                // arbitrary id is refused at position 0 and the draft is truncated before
                // it can walk anywhere; a real MTP head proposes tokens that are legal
                // here and wrong later, and those are what carry the automaton into
                // states the well-formed stream never visits.
                std::size_t id = (p % 2 == 0) ? (248077 + (p % 243)) : ((p * 7919U) % v_);
                if (g_ != nullptr) {
                    const TokenMask& m = g_->mask();
                    for (std::size_t t = 0; t < m.size(); ++t) {
                        const auto cand = static_cast<TokenId>((id + t) % m.size());
                        if (m.allows(cand)) {
                            id = static_cast<std::size_t>(cand);
                            break;
                        }
                    }
                }
                r[id] = 50.0F;
                return;
            }
            r = row(p);
        }
        std::size_t wrong_every_ = 0;
        const TurnGrammar* g_ = nullptr;
        [[nodiscard]] std::size_t pos() const { return pos_; }
        [[nodiscard]] std::vector<float> row(std::size_t p) const {
            std::vector<float> r(v_, 0.0F);
            r[static_cast<std::size_t>(p < s_.size() ? s_[p] : s_.back())] = 50.0F;
            return r;
        }

      private:
        const std::vector<TokenId>& s_;
        std::size_t v_;
        std::size_t pos_ = 0;
        std::size_t mark_ = 0;
        std::size_t cur_ = 0;
        bool verified_ = false;
        bool primed_ = false;
    };

    for (int round = 0; round < rounds; ++round) {
        TurnGrammar g(tok, tools);
        Fwd fwd(stream, V);
        SpecConfig cfg;
        cfg.enabled = true;
        cfg.mtp_block_size = 3;
        SamplingParams sp;
        sp.seed = 1234 + static_cast<std::uint64_t>(round);
        sp.temperature = 1.0F;
        sp.repetition_penalty = 1.0F;
        sp.top_p = 1.0F;
        fwd.wrong_every_ = static_cast<std::size_t>(2 + (round % 4));
        fwd.g_ = &g;
        SpeculativeDecoder dec(sp, cfg);
        dec.seed(fwd.row(0));

        std::vector<TokenId> emitted;
        std::vector<TokenId> recent;
        const auto is_special = [&g](TokenId id) { return g.is_block_boundary(id); };

        for (int step = 0; step < 200 && emitted.size() < stream.size(); ++step) {
            const bool may = g.mask_is_block_stable() || g.can_checkpoint();
            SpecStep s = dec.step(&g, recent, std::span<const TokenId>(emitted), may,
                                  is_special, fwd);
            if (s.no_legal_token) {
                std::printf("  [REPRODUCED] round %d step %d: no legal token after %zu "
                            "emitted, grammar phase %d, mask allows %zu\n",
                            round, step, emitted.size(), static_cast<int>(g.phase()),
                            g.mask().count());
                return 1;
            }
            for (const TokenId id : s.committed) {
                const Advance a = g.advance(id);
                if (a == Advance::Rejected) {
                    std::printf("  [REPRODUCED] round %d step %d: the grammar REJECTED a "
                                "committed token (id %d) -- it was sampled from a mask "
                                "that allowed it, so the mask and the automaton disagree\n",
                                round, step, static_cast<int>(id));
                    return 1;
                }
                emitted.push_back(id);
                recent.push_back(id);
                if (recent.size() > 64) {
                    recent.erase(recent.begin());
                }
                if (a == Advance::Accepted) {
                    break;
                }
            }
            dec.observe(std::span<const TokenId>(s.committed));
        }
    }
    std::printf("  OK: %d round(s), no empty distribution and no rejected commit\n", rounds);
    std::printf("  (blocks that had to be abandoned for an ordinary token: see per-round)\n");
    return 0;
}

// --- bench -----------------------------------------------------------------

// What a speculative block's VERIFICATION pass costs, against the single-token step it
// has to beat. This is the whole economics of speculation and it was never measured.
//
// A block spends one forward_logits_all over k positions and commits at most k tokens, so
// speculation only pays when that pass costs less than k single-token steps. The naive
// assumption is that it costs about ONE -- decode is bandwidth-bound on weights, and k
// extra rows flowing through the same weights are nearly free. That assumption holds for
// a pure-attention model. It does NOT obviously hold here: this checkpoint is HYBRID
// (full_attention_interval 4, so 3 of every 4 layers are linear attention), and
// gated_delta_update runs a different kernel at S>1 than the fast recurrence it runs at
// S=1. If that path is superlinear in S, speculation is buying tokens at a loss.
//
// Prints ms, ms-per-position, and the ratio against T=1 -- the last is the number that
// decides whether a draft is worth proposing at all.
int cmd_verify(int max_k, int ctx) {
    mlxl::Qwen35MoeModel model;
    if (!model.load(qwen_dir())) {
        std::printf("model load failed\n");
        return 1;
    }
    std::printf("  verification pass cost by block width (context %d, 10 iters each)\n", ctx);
    double base = 0.0;
    for (int k = 1; k <= max_k; ++k) {
        // A fresh cache per width, so every k is measured at the same context length
        // rather than inheriting the previous k's appended tokens.
        model.reset_cache();
        // CHUNKED, like the real prefill path. A single 28k-token pass builds a
        // [1, heads, 28000, 28000] attention score buffer and asks Metal for 37.6 GB --
        // which is what this measurement did on its first run, and it is a property of
        // the instrument, not of the model.
        const int chunk = 2048;
        std::vector<int32_t> prompt(static_cast<std::size_t>(ctx), 100);
        for (int off = 0; off < ctx; off += chunk) {
            const int n = std::min(chunk, ctx - off);
            mx::array ids = mx::array(prompt.data() + off, {1, n}, mx::int32);
            mx::array warm = model.forward_logits(ids);
            mx::eval(warm);
        }

        std::vector<int32_t> block(static_cast<std::size_t>(k), 42);
        const int iters = 10;
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            mx::array b = mx::array(block.data(), {1, k}, mx::int32);
            mx::array logits = model.forward_logits_all(b);
            mx::eval(logits);
        }
        const double per = ms(t0, Clock::now()) / iters;
        if (k == 1) {
            base = per;
        }
        std::printf("    k=%-2d  %7.2f ms/pass   %6.2f ms/position   %.2fx of k=1"
                    "   break-even accept >= %.0f%%\n",
                    k, per, per / k, per / base,
                    100.0 * ((per / base) - 1.0) / static_cast<double>(k > 1 ? k - 1 : 1));
    }
    return 0;
}

int cmd_bench(int runs, int prompt_tokens, int max_new) {
    QwenTokenizer tok;
    LoadStatus st = tok.load(std::string(qwen_dir()) + "/tokenizer.json", Family::Qwen3);
    if (!st.ok) {
        std::printf("tok fail: %s\n", st.error.c_str());
        return 1;
    }

    lmp::platform::SystemClock clock;
    MlxBackend backend(clock);
    auto t_load0 = Clock::now();
    st = backend.load({qwen_dir(), draft_dir()});
    if (!st.ok) {
        std::printf("model load fail: %s\n", st.error.c_str());
        return 1;
    }
    std::printf("model load: %.1f s%s\n", ms(t_load0, Clock::now()) / 1000.0,
                draft_dir()[0] != '\0' ? "  [MTP draft head loaded]" : "  [plain decode]");

    // A prompt of the requested length, built from real text so the tokenizer does not
    // collapse it into a handful of repeated ids.
    std::string filler;
    int n = 0;
    while (n < prompt_tokens) {
        filler += "The build system compiles each translation unit separately, then the "
                  "linker resolves symbols across them and emits one binary. Line " +
                  std::to_string(n) + ".\n";
        n = static_cast<int>(tok.encode_content(filler).size());
    }

    ChatTemplate tmpl(tok);
    const std::vector<parsephony::ToolSpec> no_tools;

    Ledger prefill;
    Ledger decode;
    Ledger ttft;
    std::size_t measured_n = 0;
    for (int run = 0; run < runs; ++run) {
        TurnGrammar grammar(tok, no_tools);
        class GrammarSink final : public TokenSink {
          public:
            explicit GrammarSink(TurnGrammar& gr) : g_(gr) {}
            bool on_token(TokenId id) override {
                last = g_.advance(id);
                return last == Advance::Ok;
            }
            Advance last = Advance::Ok;

          private:
            TurnGrammar& g_;
        };
        GrammarSink sink(grammar);

        InferenceTask task;
        task.prompt = tmpl.render({{Role::System, "You are a terse assistant."},
                                   {Role::User, filler + "\nSummarise the above in one line."}},
                                  "");
        task.max_new_tokens = max_new;
        task.sampling.seed = 7;
        task.mask = &grammar;
        measured_n = task.prompt.size();

        // Each run pays a full prefill: reuse across runs would measure the ledger,
        // not the forward pass.
        backend.reset_cache();
        CancelToken cancel;
        const GenResult r = backend.generate(task, sink, cancel);
        std::printf("  run %d: prompt=%zu tokens=%d status=%d ttft=%.0fms "
                    "prefill=%.1f decode=%.1f  (forward=%.0fms copy=%.0fms sample=%.0fms)\n",
                    run, task.prompt.size(), r.tokens_generated, static_cast<int>(r.status),
                    r.ttft_ms, r.prefill_tok_per_s, r.decode_tok_per_s, r.forward_ms,
                    r.logits_copy_ms, r.sample_ms);
        prefill.add(r.prefill_tok_per_s);
        decode.add(r.decode_tok_per_s);
        ttft.add(r.ttft_ms);
    }

    // The bar has to be the one for the length we actually measured, not the aggregate.
    const PrefillBar& bar = prefill_bar(static_cast<int>(measured_n));
    std::printf("\nLM_Pipe, prompt=%zu tokens, %d runs:\n", measured_n, runs);
    prefill.print("prefill", "tok/s", bar.trusted ? bar.tok_per_s : 0.0);
    if (!bar.trusted) {
        std::printf("             (LM Studio logs ~%.0f tok/s at this length, but that bucket's "
                    "median\n              window is 1-2 s against a 1-second clock -- a floor, "
                    "not a bar.\n              scripts/mlxlm_reference.py measures the live "
                    "stack at this length.)\n",
                    bar.tok_per_s);
    }
    decode.print("decode", "tok/s", 78.5);
    ttft.print("ttft", "ms", 0.0);
    // mlx-lm reports 20.18 GB peak on this checkpoint. Ours being much higher would mean
    // we are holding intermediates the reference frees, which is the kind of thing that
    // shows up as a uniform slowdown rather than one slow op.
    const double gb = 1024.0 * 1024.0 * 1024.0;
    std::printf("  memory     peak %.2f GB  active %.2f GB  cache %.2f GB\n",
                static_cast<double>(mx::get_peak_memory()) / gb,
                static_cast<double>(mx::get_active_memory()) / gb,
                static_cast<double>(mx::get_cache_memory()) / gb);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "bench";
    if (cmd == "scan") {
        std::vector<int> lengths;
        for (int i = 2; i < argc; ++i) {
            lengths.push_back(std::atoi(argv[i]));
        }
        if (lengths.empty()) {
            lengths = {1, 8, 64, 287, 512};
        }
        return cmd_scan(lengths);
    }
    if (cmd == "mask") {
        return cmd_mask();
    }
#if LMP_HAVE_MLX
    if (cmd == "moe") {
        return cmd_moe(argc > 2 ? std::atoi(argv[2]) : 1);
    }
    if (cmd == "moestream") {
        return cmd_moe_stream(argc > 2 ? std::atoi(argv[2]) : 20);
    }
    if (cmd == "step") {
        return cmd_step(argc > 2 ? std::atoi(argv[2]) : 50);
    }
    if (cmd == "graph") {
        return cmd_graph(argc > 2 ? std::atoi(argv[2]) : 547);
    }
    if (cmd == "layers") {
        return cmd_layers(argc > 2 ? std::atoi(argv[2]) : 1);
    }
    if (cmd == "chain") {
        return cmd_chain(argc > 2 ? std::atoi(argv[2]) : 2000);
    }
    if (cmd == "blocks") {
        return cmd_blocks(argc > 2 ? std::atoi(argv[2]) : 1);
    }
#endif
    if (cmd == "verify") {
        return cmd_verify(argc > 2 ? std::atoi(argv[2]) : 4,
                          argc > 3 ? std::atoi(argv[3]) : 547);
    }
    if (cmd == "specrun") {
        return cmd_specrun(argc > 2 ? std::atoi(argv[2]) : 5);
    }
    if (cmd == "specgrammar") {
        return cmd_specgrammar(argc > 2 ? std::atoi(argv[2]) : 3);
    }
    if (cmd == "toolmask") {
        return cmd_toolmask(argc > 2 ? std::atoi(argv[2]) : 3);
    }
    if (cmd == "bench") {
        const int runs = argc > 2 ? std::atoi(argv[2]) : 3;
        const int prompt_tokens = argc > 3 ? std::atoi(argv[3]) : 512;
        const int max_new = argc > 4 ? std::atoi(argv[4]) : 256;
        return cmd_bench(runs, prompt_tokens, max_new);
    }
    std::printf("usage: lmp_diag [scan|mask|bench] ...\n");
    return 2;
}
