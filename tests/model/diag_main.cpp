#include "src/model/chat_template.hpp"
#include "src/model/grammar.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"
#include <cstdio>
using namespace lmp::model;
int main(){
  QwenTokenizer tok;
  auto st = tok.load("/Users/dev/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit/tokenizer.json", Family::Qwen3);
  if(!st.ok){ std::printf("tok fail %s\n", st.error.c_str()); return 1; }
  ChatTemplate tmpl(tok);
  auto ids = tmpl.render({{Role::System,"You are terse."},{Role::User,"What is 2+2? One short sentence."}}, "");
  std::string p = tok.decode(ids);
  std::printf("PROMPT TAIL: %s|END|\n", p.substr(p.size()>40?p.size()-40:0).c_str());

  lmp::platform::SystemClock clock;
  MlxBackend b(clock);
  auto ls = b.load({"/Users/dev/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",""});
  if(!ls.ok){ std::printf("load fail %s\n", ls.error.c_str()); return 1; }
  std::vector<parsephony::ToolSpec> none;
  TurnGrammar g(tok, none);
  struct S : TokenSink {
    TurnGrammar& g; QwenTokenizer::Stream st; int n=0; Advance last=Advance::Ok;
    S(TurnGrammar& gg, const QwenTokenizer& t):g(gg),st(t){}
    bool on_token(TokenId id) override {
      ++n; last=g.advance(id);
      std::string t = st.push(id);
      std::fwrite(t.data(),1,t.size(),stdout);
      if(last!=Advance::Ok) std::printf("\n[STOP after %d tokens, advance=%d, phase=%d]\n",n,static_cast<int>(last),static_cast<int>(g.phase()));
      return last==Advance::Ok;
    }
  } sink(g, tok);
  InferenceTask t; t.prompt=ids; t.max_new_tokens=2000; t.sampling.seed=7;
  t.mask=[&g](TokenId id){ return g.permitted(id); };
  CancelToken c;
  auto r=b.generate(t,sink,c);
  std::printf("\n[status=%d tokens=%d think=%zu text=%zu decode=%.1f tok/s]\n",
    static_cast<int>(r.status),r.tokens_generated,g.think_ids().size(),g.text_ids().size(),r.decode_tok_per_s);
}
