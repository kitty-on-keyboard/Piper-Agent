// Canonical event trace for one entrant tree, over corpus.txt. Both this and the
// TypeScript port print the SAME format, so `diff` is the equivalence test.
#include <markdown_stream.hpp>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
static const char* kn(md::EventKind k){switch(k){
case md::EventKind::Text:return "text";case md::EventKind::CodeBlockOpen:return "codeOpen";
case md::EventKind::CodeBlockText:return "codeText";case md::EventKind::CodeBlockClose:return "codeClose";
case md::EventKind::InlineCodeOpen:return "inlineOpen";case md::EventKind::InlineCodeClose:return "inlineClose";
case md::EventKind::HeadingOpen:return "headingOpen";case md::EventKind::HeadingClose:return "headingClose";
case md::EventKind::ListItemOpen:return "listOpen";case md::EventKind::ListItemClose:return "listClose";
case md::EventKind::ParagraphBreak:return "paraBreak";} return "?";}
static std::string esc(const std::string& s){std::string o;for(char c:s){
  if(c=='\\')o+="\\\\";else if(c=='\n')o+="\\n";else if(c=='\t')o+="\\t";else o+=c;}return o;}
static std::string unesc(const std::string& s){std::string o;for(size_t i=0;i<s.size();++i){
  if(s[i]=='\\'&&i+1<s.size()){char n=s[++i];o+=(n=='n'?'\n':n=='t'?'\t':'\\');}else o+=s[i];}return o;}
// Adjacent same-kind text runs are merged: how a chunk is CARVED is a free choice, what it
// contains is not. Without this the byte-at-a-time trace differs trivially from the whole one.
static void emit(std::vector<std::pair<std::string,md::Event>>& acc, const std::vector<md::Event>& v){
  for(const auto& e:v){const char* k=kn(e.kind);
    if(!acc.empty()&&acc.back().first==k&&(e.kind==md::EventKind::Text||e.kind==md::EventKind::CodeBlockText))
      acc.back().second.text+=e.text;
    else acc.push_back({k,e});}}
static void show(const char* tag,size_t i,const std::vector<std::pair<std::string,md::Event>>& acc){
  for(const auto& p:acc) std::printf("%s[%zu] %s|%s|%s|%d\n",tag,i,p.first.c_str(),
    esc(p.second.text).c_str(),esc(p.second.info).c_str(),p.second.level);}
int main(){
  std::ifstream f(CORPUS_PATH); std::string line; size_t i=0;
  while(std::getline(f,line)){
    const std::string doc=unesc(line);
    { md::MarkdownStream ms; std::vector<std::pair<std::string,md::Event>> a;
      emit(a,ms.feed(doc)); emit(a,ms.finish()); show("whole",i,a); }
    { md::MarkdownStream ms; std::vector<std::pair<std::string,md::Event>> a;
      for(char c:doc) emit(a,ms.feed(std::string_view(&c,1)));
      emit(a,ms.finish()); show("bytes",i,a); }
    ++i;
  }
  return 0;
}
