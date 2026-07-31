#pragma once

// Token-level constrained decoding over any byte automaton.
//
// TokenMaskT answers "which tokens may the model emit next" for a vocabulary of
// arbitrary byte strings. It is templated over the automaton so the same engine
// serves both grammars in this library: the JSON PDA (StreamParser) and the
// Qwen XML tool-call guard (ToolCallGuard).
//
// Naively this is |vocab| simulations per sampling step — at Qwen scale
// (248k tokens) that is milliseconds, far too slow to sit in a sampler. Three
// observations make it cheap:
//
//   1. Automaton configurations recur. Two steps whose state_signature() match
//      accept exactly the same token set, so masks are cached per signature. A
//      whole tool call visits a few dozen distinct configurations.
//
//   2. Most of generation happens *inside strings*, where almost the entire
//      vocabulary is legal. Tokens are pre-classified once at construction:
//      a token whose every byte is string-safe can be accepted with no
//      simulation at all, so string states start from a precomputed base mask
//      and simulate only the small minority of tokens containing quotes,
//      backslashes, or control bytes.
//
//   3. Everywhere else the grammar is narrow (a handful of legal first bytes),
//      so gating candidates on their first byte eliminates nearly all of the
//      vocabulary before any simulation happens.
//
// The automaton P must provide:
//   uint64_t  state_signature() const
//   MaskClass mask_class() const
//   ByteSet   allowed_bytes() const
//   Error     probe_byte(unsigned char)      // advance without user callbacks
//   void      mute()                         // silence callbacks on a copy
// and be copyable.

#include "parsephony/stream.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>

namespace parsephony {

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

struct Vocab {
    std::vector<std::string> tokens;
    std::vector<uint8_t> special;   // parallel to tokens; empty = none special

    size_t size() const noexcept { return tokens.size(); }
    bool is_special(size_t i) const noexcept {
        return i < special.size() && special[i] != 0;
    }

    // Loads the binary produced by scripts/export_vocab.py.
    // Returns false (and leaves the vocab empty) on any malformation.
    bool load(const char* path);
};

// ---------------------------------------------------------------------------
// TokenMaskT
// ---------------------------------------------------------------------------

template <class P>
class TokenMaskT {
public:
    explicit TokenMaskT(const Vocab& v, Options o = {})
        : vocab_(v), opts_(o), words_((v.size() + 63) / 64) {
        classify();
    }

    // Bitmask over the vocabulary; bit i set means token i is legal next.
    // The reference is owned by the cache and stays valid for this object's
    // lifetime.
    const std::vector<uint64_t>& compute(const P& p) {
        uint64_t sig = p.state_signature();
        auto it = cache_.find(sig);
        if (it != cache_.end()) { ++hits_; return it->second; }
        ++misses_;

        std::vector<uint64_t> m;
        switch (p.mask_class()) {
            case MaskClass::FreeText:
                // Raw text: everything non-special is legal except the rare
                // token that runs *through* the section terminator and out
                // into constrained territory.
                m = all_ok_;
                for (uint32_t i : free_sim_) {
                    if (!simulate(p, i)) m[i >> 6] &= ~(1ull << (i & 63));
                }
                break;

            case MaskClass::JsonString:
                // Inside a JSON string: start from the precomputed all-safe
                // base, then simulate only tokens containing bytes that
                // interact with string syntax.
                m = json_safe_;
                for (uint32_t i : json_sim_) {
                    if (simulate(p, i)) m[i >> 6] |= (1ull << (i & 63));
                }
                break;

            case MaskClass::Other: {
                // Only tokens whose first byte the grammar accepts can possibly
                // be legal, and tokens are bucketed by first byte up front — so
                // a narrow state (one legal byte) touches a couple of thousand
                // candidates instead of the whole vocabulary.
                m.assign(words_, 0);
                ByteSet first = p.allowed_bytes();
                for (unsigned b = 0; b < 256; ++b) {
                    if (!first.contains(static_cast<unsigned char>(b))) continue;
                    for (uint32_t k = bucket_off_[b]; k < bucket_off_[b + 1]; ++k) {
                        uint32_t i = bucket_ids_[k];
                        if (simulate(p, i)) m[i >> 6] |= (1ull << (i & 63));
                    }
                }
                break;
            }
        }

        auto [pos, inserted] = cache_.emplace(sig, std::move(m));
        (void)inserted;
        return pos->second;
    }

    static bool test(const std::vector<uint64_t>& mask, size_t i) noexcept {
        return (mask[i >> 6] >> (i & 63)) & 1;
    }

    static size_t count(const std::vector<uint64_t>& mask) noexcept {
        size_t c = 0;
        for (uint64_t w : mask) c += size_t(__builtin_popcountll(w));
        return c;
    }

    size_t cache_size() const noexcept { return cache_.size(); }
    size_t hits() const noexcept { return hits_; }
    size_t misses() const noexcept { return misses_; }
    size_t json_sim_count() const noexcept { return json_sim_.size(); }
    size_t free_sim_count() const noexcept { return free_sim_.size(); }

private:
    static constexpr uint16_t kNoByte = 0xFFFF;

    bool simulate(const P& base, uint32_t i) const {
        P probe = base;
        probe.mute();
        for (unsigned char c : vocab_.tokens[i]) {
            if (probe.probe_byte(c) != Error::Ok) return false;
        }
        return true;
    }

    void classify() {
        const size_t n = vocab_.size();
        all_ok_.assign(words_, 0);
        json_safe_.assign(words_, 0);
        first_.assign(n, kNoByte);

        static constexpr char kTerm[] = "\n</parameter>\n";

        // First-byte buckets (CSR layout) for the Other path.
        uint32_t counts[257] = {};
        for (size_t i = 0; i < n; ++i) {
            const std::string& t = vocab_.tokens[i];
            if (!t.empty() && !vocab_.is_special(i))
                ++counts[static_cast<unsigned char>(t[0]) + 1];
        }
        bucket_off_.assign(257, 0);
        for (int b = 0; b < 256; ++b) bucket_off_[b + 1] = bucket_off_[b] + counts[b + 1];
        bucket_ids_.resize(bucket_off_[256]);
        {
            uint32_t cur[256];
            for (int b = 0; b < 256; ++b) cur[b] = bucket_off_[b];
            for (size_t i = 0; i < n; ++i) {
                const std::string& t = vocab_.tokens[i];
                if (!t.empty() && !vocab_.is_special(i))
                    bucket_ids_[cur[static_cast<unsigned char>(t[0])]++] = uint32_t(i);
            }
        }

        for (size_t i = 0; i < n; ++i) {
            const std::string& t = vocab_.tokens[i];
            if (t.empty() || vocab_.is_special(i)) continue;

            first_[i] = static_cast<unsigned char>(t[0]);
            all_ok_[i >> 6] |= (1ull << (i & 63));

            // JSON-string classification.
            bool pure = true;
            for (unsigned char c : t) {
                if (c == '"' || c == '\\' || c < 0x20) { pure = false; break; }
            }
            if (pure) json_safe_[i >> 6] |= (1ull << (i & 63));
            else      json_sim_.push_back(uint32_t(i));

            // Free-text classification: only a token containing the entire
            // parameter terminator can leave the free-text state mid-token.
            if (t.find(kTerm) != std::string::npos) {
                free_sim_.push_back(uint32_t(i));
            }
        }
    }

    const Vocab& vocab_;
    Options opts_;
    size_t words_;

    std::vector<uint64_t> all_ok_;     // every usable (non-special) token
    std::vector<uint64_t> json_safe_;  // tokens legal anywhere inside a JSON string
    std::vector<uint32_t> json_sim_;   // must be simulated in JsonString states
    std::vector<uint32_t> free_sim_;   // must be simulated in FreeText states
    std::vector<uint16_t> first_;
    std::vector<uint32_t> bucket_off_; // CSR: token ids grouped by first byte
    std::vector<uint32_t> bucket_ids_;

    std::unordered_map<uint64_t, std::vector<uint64_t>> cache_;
    size_t hits_ = 0, misses_ = 0;
};

using TokenMask = TokenMaskT<StreamParser>;

} // namespace parsephony
