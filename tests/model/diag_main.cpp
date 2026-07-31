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
        logits = model.forward_logits_lazy(one);
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

// --- bench -----------------------------------------------------------------

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
    st = backend.load({qwen_dir(), ""});
    if (!st.ok) {
        std::printf("model load fail: %s\n", st.error.c_str());
        return 1;
    }
    std::printf("model load: %.1f s\n", ms(t_load0, Clock::now()) / 1000.0);

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

    std::printf("\nLM_Pipe, prompt=%d tokens, %d runs:\n", prompt_tokens, runs);
    prefill.print("prefill", "tok/s", 1347.0);
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
    if (cmd == "bench") {
        const int runs = argc > 2 ? std::atoi(argv[2]) : 3;
        const int prompt_tokens = argc > 3 ? std::atoi(argv[3]) : 512;
        const int max_new = argc > 4 ? std::atoi(argv[4]) : 256;
        return cmd_bench(runs, prompt_tokens, max_new);
    }
    std::printf("usage: lmp_diag [scan|mask|bench] ...\n");
    return 2;
}
