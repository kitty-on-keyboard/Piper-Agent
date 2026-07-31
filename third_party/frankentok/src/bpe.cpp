#include "mlx_qwen_tokenizer/bpe.h"
#include <algorithm>
#include <climits>

namespace mlx_qwen_tokenizer {

namespace {

// Above this many symbols the rescan's O(k^2) overtakes the heap's bookkeeping.
//
// Measured against this vocabulary: the two strategies are even at 8 symbols, the heap is ~1.6x
// cheaper at 16, and the gap then widens without bound — at 4096 symbols the rescan costs 40ms per
// word against the heap's 0.36ms.
//
// Ordinary text never crosses this line: pre-tokens on the benchmark corpus average 4.1 bytes and
// top out at 13, which is why the rescan is still the default. But the pretokenizer's `\p{L}+`
// alternative is unbounded, so one long unbroken run of letters arrives as a *single* pre-token,
// and leaving that on the rescan made a 32KB word take 4.3 seconds — worse than 2MB of prose.
constexpr size_t kHeapMergeThreshold = 16;

// Min-heap ordering. Ties go to the leftmost pair, matching the rescan: it accepts only a strictly
// lower rank, so among equal ranks the first one encountered — the leftmost — wins. Without this
// tie-break the two strategies would disagree on words containing a repeated pair.
struct CandidateGreater {
    bool operator()(const BPE::Candidate& a, const BPE::Candidate& b) const {
        if (a.rank != b.rank) return a.rank > b.rank;
        return a.left > b.left;
    }
};

// Rescan every adjacent pair after each merge. Cheapest when the word is short enough that the
// whole symbol list sits in a cache line or two.
void merge_by_rescan(const Vocab& vocab, std::vector<BPE::Symbol>& symbols) {
    while (true) {
        int32_t best_rank = INT_MAX;
        int32_t best_idx = -1;
        int32_t best_id = -1;

        for (int32_t it = 0; it != -1; it = symbols[it].next) {
            const int32_t nxt = symbols[it].next;
            if (nxt == -1) break;
            if (symbols[it].id == -1 || symbols[nxt].id == -1) continue;

            int32_t rank = -1;
            int32_t merged_id = -1;
            if (vocab.get_merge(symbols[it].id, symbols[nxt].id, rank, merged_id)) {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = it;
                    best_id = merged_id;
                }
            }
        }

        if (best_idx == -1) break;

        // Apply: absorb the right neighbour into the left, then unlink it.
        const int32_t victim = symbols[best_idx].next;
        symbols[best_idx].id = best_id;
        symbols[best_idx].next = symbols[victim].next;
        if (symbols[victim].next != -1) {
            symbols[symbols[victim].next].prev = best_idx;
        }
    }
}

// Same merge order, reached by popping a heap instead of rescanning. Only the two pairs touching a
// just-merged symbol can change, so each merge pushes at most two entries rather than re-testing
// the whole word.
void merge_by_heap(const Vocab& vocab, BPE::Scratch& scratch) {
    std::vector<BPE::Symbol>& symbols = scratch.symbols;
    std::vector<int32_t>& version = scratch.version;
    std::vector<BPE::Candidate>& heap = scratch.heap;

    version.assign(symbols.size(), 0);
    heap.clear();

    const auto push_pair = [&](int32_t left) {
        if (left == -1) return;
        const int32_t right = symbols[left].next;
        if (right == -1) return;
        if (symbols[left].id == -1 || symbols[right].id == -1) return;

        int32_t rank = -1;
        int32_t merged_id = -1;
        if (!vocab.get_merge(symbols[left].id, symbols[right].id, rank, merged_id)) return;

        heap.push_back({rank, left, right, merged_id, version[left], version[right]});
        std::push_heap(heap.begin(), heap.end(), CandidateGreater{});
    };

    for (int32_t it = 0; it != -1; it = symbols[it].next) push_pair(it);

    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), CandidateGreater{});
        const BPE::Candidate cand = heap.back();
        heap.pop_back();

        // Stale if either endpoint has since been rewritten or consumed, or if the two are no
        // longer neighbours. Symbols are only ever unlinked, never moved, so an index stays valid.
        if (version[cand.left] != cand.lver || version[cand.right] != cand.rver) continue;
        if (symbols[cand.left].next != cand.right) continue;

        symbols[cand.left].id = cand.merged_id;
        symbols[cand.left].next = symbols[cand.right].next;
        if (symbols[cand.right].next != -1) {
            symbols[symbols[cand.right].next].prev = cand.left;
        }
        version[cand.left]++;
        version[cand.right]++;

        push_pair(cand.left);
        push_pair(symbols[cand.left].prev);
    }
}

} // namespace

void BPE::encode_word(std::string_view word, Scratch& scratch, std::vector<int32_t>& out) const {
    if (word.empty()) return;

    // 1. Initial split: one symbol per input byte.
    //
    // The ByteLevel mapping and the vocabulary lookup are already composed into a 256-entry table
    // (see Vocab::byte_token_ids), so this needs neither the intermediate ByteLevel string nor a
    // hash lookup per character. It is exact rather than an approximation: the mapping sends each
    // of the 256 byte values to its own single character, so bytes and symbols correspond one to
    // one regardless of how the input happens to be encoded.
    const std::array<int32_t, 256>& byte_ids = vocab_.byte_token_ids();
    std::vector<Symbol>& symbols = scratch.symbols;
    symbols.clear();
    symbols.reserve(word.size());
    for (size_t i = 0; i < word.size(); ++i) {
        const int32_t idx = static_cast<int32_t>(i);
        symbols.push_back({byte_ids[static_cast<unsigned char>(word[i])], idx - 1, idx + 1});
    }

    if (symbols.empty()) return;
    symbols.back().next = -1;

    if (symbols.size() == 1) {
        if (symbols.front().id != -1) out.push_back(symbols.front().id);
        return;
    }

    // 2. Merge the lowest-ranked adjacent pair until none remain. Both strategies below produce
    // the same ids; they differ only in how the next pair is found.
    if (symbols.size() <= kHeapMergeThreshold) {
        merge_by_rescan(vocab_, symbols);
    } else {
        merge_by_heap(vocab_, scratch);
    }

    for (int32_t it = 0; it != -1; it = symbols[it].next) {
        // Fallback for unmerged/unknown tokens
        if (symbols[it].id != -1) out.push_back(symbols[it].id);
    }
}

std::vector<int32_t> BPE::encode_word(const std::string& word) const {
    Scratch scratch;
    std::vector<int32_t> out;
    encode_word(std::string_view(word), scratch, out);
    return out;
}

} // namespace mlx_qwen_tokenizer
