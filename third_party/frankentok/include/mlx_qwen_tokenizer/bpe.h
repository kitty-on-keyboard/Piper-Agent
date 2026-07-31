#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include "mlx_qwen_tokenizer/vocab.h"

namespace mlx_qwen_tokenizer {

class BPE {
public:
    BPE(const Vocab& vocab) : vocab_(vocab) {}

    // Symbols live in one flat vector and are unlinked in place by index.
    //
    // The list-of-nodes version allocated a heap node *and* a std::string per character of every
    // word, then rebuilt that string on every merge. Nothing downstream needs the text — the caller
    // wants ids, and text is recoverable from an id — so symbols carry only the id here.
    struct Symbol {
        int32_t id;
        int32_t prev;
        int32_t next;
    };

    // A merge parked in the heap used for long words. `lver`/`rver` pin the identities of the two
    // endpoints as of the push, so an entry invalidated by some later merge is recognised on pop
    // rather than needing to be found and erased when the merge happens.
    struct Candidate {
        int32_t rank;
        int32_t left;
        int32_t right;
        int32_t merged_id;
        int32_t lver;
        int32_t rver;
    };

    // Buffers reused across a run of encode_word calls.
    //
    // Held by the caller rather than as members of BPE: encode_word is const and one BPE is shared
    // by every thread encoding against it, so mutable members would be a silent data race. A
    // scratch local to each encode() call keeps concurrent encodes independent while still turning
    // two mallocs per pre-token into two for the whole call.
    struct Scratch {
        std::vector<Symbol> symbols;
        // Only touched on the long-word path; both stay empty for ordinary text.
        std::vector<int32_t> version;
        std::vector<Candidate> heap;
    };

    // Appends the ids for one pre-tokenized chunk to `out`.
    //
    // Appending rather than returning matters at this call rate: pre-tokens average about four
    // bytes, so a returned vector was one malloc and one free per word, and every id was copied
    // twice — once into that vector and again into the caller's.
    void encode_word(std::string_view word, Scratch& scratch, std::vector<int32_t>& out) const;

    // Convenience overload for single-shot callers; allocates its own scratch and result.
    std::vector<int32_t> encode_word(const std::string& word) const;

private:
    const Vocab& vocab_;
};

} // namespace mlx_qwen_tokenizer
