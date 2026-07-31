// Micro-benchmark: what does ONE decode step cost outside the model forward pass?
#include <chrono>
#include <cstdio>
#include "src/model/grammar.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/model/sampler.hpp"

using namespace lmp::model;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
    QwenTokenizer tok;
    auto st = tok.load("/Users/dev/.lmstudio/models/lmstudio-community/"
                       "Qwen3.6-35B-A3B-MLX-4bit/tokenizer.json", Family::Qwen3);
    if (!st.ok) { std::printf("tok fail: %s\n", st.error.c_str()); return 1; }
    const std::size_t V = tok.vocab_size();
    std::printf("vocab = %zu\n\n", V);

    std::vector<parsephony::ToolSpec> tools;
    parsephony::ToolSpec s; s.name = "read_file";
    parsephony::ParamSpec p; p.name = "path"; p.required = true;
    s.params.push_back(p); tools.push_back(s);
    TurnGrammar g(tok, tools);

    // 1. the mask, exactly as the sampler calls it
    auto t0 = Clock::now();
    std::size_t allowed = 0;
    for (std::size_t i = 0; i < V; ++i) allowed += g.permitted(static_cast<TokenId>(i)) ? 1 : 0;
    auto t1 = Clock::now();
    std::printf("mask over full vocab      : %8.1f ms/token   (%zu allowed)\n", ms(t0,t1), allowed);

    // 2. the sampler with a null mask (softmax + top-k + top-p sort)
    std::vector<float> logits(V);
    for (std::size_t i = 0; i < V; ++i) logits[i] = static_cast<float>(i % 97) * 0.01F;
    Sampler smp{SamplingParams{}};
    auto l2 = logits;
    auto t2 = Clock::now();
    (void)smp.sample(l2, nullptr, {});
    auto t3 = Clock::now();
    std::printf("sampler, no mask          : %8.1f ms/token\n", ms(t2,t3));

    // 3. both together = what actually runs today
    auto l3 = logits;
    auto t4 = Clock::now();
    (void)smp.sample(l3, [&g](TokenId id){ return g.permitted(id); }, {});
    auto t5 = Clock::now();
    std::printf("sampler + mask (as shipped): %7.1f ms/token\n\n", ms(t4,t5));

    const double per_tok = ms(t4,t5);
    std::printf("=> CPU-side ceiling from this alone: %.1f tok/s\n", 1000.0 / per_tok);
    std::printf("   (LM Studio measured 78.5 tok/s on the same model/machine)\n");
}
