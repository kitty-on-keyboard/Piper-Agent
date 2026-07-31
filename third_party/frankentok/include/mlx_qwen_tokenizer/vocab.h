#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "robin_hood.h"

namespace mlx_qwen_tokenizer {

// Open-addressed (id1,id2) -> (rank, merged_id) table for the BPE merge rules.
//
// This replaces a robin_hood map for two reasons. The hot one: get_merge is the innermost call of
// the merge loop, and a flat POD table with linear probing keys on one uint64 with no per-entry
// metadata. The structural one: the slots vector is a single contiguous allocation with no heap
// pointers, so the binary vocab cache can write it to disk and read it back verbatim instead of
// re-inserting a quarter-million entries at every load.
class MergeTable {
public:
    struct Slot {
        uint64_t key;        // (id1 << 32) | id2; kEmpty marks a free slot
        int32_t rank;
        int32_t merged_id;
    };
    static constexpr uint64_t kEmpty = ~0ull;   // unreachable: real ids are non-negative int32

    void build(size_t expected) {
        size_t cap = 16;
        // Size for a load factor under 0.5 so probe chains stay short.
        while (cap < expected * 2) cap <<= 1;
        slots_.assign(cap, Slot{kEmpty, -1, -1});
        mask_ = cap - 1;
        count_ = 0;
    }

    void insert(int32_t id1, int32_t id2, int32_t rank, int32_t merged_id) {
        const uint64_t key = make_key(id1, id2);
        size_t i = hash(key) & mask_;
        while (slots_[i].key != kEmpty) {
            if (slots_[i].key == key) { slots_[i] = {key, rank, merged_id}; return; }
            i = (i + 1) & mask_;
        }
        slots_[i] = {key, rank, merged_id};
        count_++;
    }

    bool find(int32_t id1, int32_t id2, int32_t& out_rank, int32_t& out_merged) const {
        const uint64_t key = make_key(id1, id2);
        size_t i = hash(key) & mask_;
        while (true) {
            const Slot& s = slots_[i];
            if (s.key == key) { out_rank = s.rank; out_merged = s.merged_id; return true; }
            if (s.key == kEmpty) return false;
            i = (i + 1) & mask_;
        }
    }

    size_t count() const { return count_; }

    // Raw access for the binary cache.
    std::vector<Slot>& slots() { return slots_; }
    const std::vector<Slot>& slots() const { return slots_; }
    void set_loaded(size_t count) { mask_ = slots_.size() - 1; count_ = count; }

private:
    static uint64_t make_key(int32_t id1, int32_t id2) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(id1)) << 32) |
               static_cast<uint32_t>(id2);
    }
    // splitmix64 finalizer: cheap and well distributed for sequential-ish id pairs.
    static uint64_t hash(uint64_t x) {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    std::vector<Slot> slots_;
    size_t mask_ = 0;
    size_t count_ = 0;
};

struct LoadOptions {
    // Consult and maintain the binary vocab cache. Parsing 19 MB of JSON costs ~350 ms; reading
    // the same tables back as flat binary costs ~2 ms.
    bool use_cache = true;

    // Directory for cache sidecars. Empty means this library's own per-user cache directory
    // (XDG_CACHE_HOME, else ~/Library/Caches on Apple, else ~/.cache).
    //
    // The directory holding tokenizer.json is *never* written to. It belongs to whoever installed
    // the model — LM Studio re-verifies its model directories, a downloaded model may sit on
    // read-only storage, and a sidecar there would vanish on re-download anyway. Reads come from
    // the model directory; writes go here.
    std::string cache_dir;
};

class Vocab {
public:
    Vocab() = default;

    // Load from a huggingface tokenizer.json path. The cache is consulted first and rebuilt
    // best-effort after a JSON parse; it is invalidated by source size/mtime and verified by
    // checksum. Failure to write it is silent — it is an optimization, never a requirement.
    bool load_from_file(const std::string& path, const LoadOptions& options = {});
    bool load_from_string(const std::string& json_content);

    // Absolute path of the cache sidecar this library would use for `source_path`, honouring
    // `options.cache_dir`. Exposed so a consumer can locate, pre-warm, or clear it.
    static std::string cache_path_for(const std::string& source_path,
                                      const std::string& cache_dir = {});

    // Lookups. Tokens here are in ByteLevel text space (the vocab's native key space).
    int32_t token_to_id(std::string_view token) const;
    std::string id_to_token(int32_t id) const;

    // Borrowed view of a token's ByteLevel text, no copy. Empty for out-of-range ids.
    std::string_view id_to_token_view(int32_t id) const {
        if (id >= 0 && static_cast<size_t>(id) < count_) {
            return {bl_blob_.data() + bl_offsets_[id], bl_offsets_[id + 1] - bl_offsets_[id]};
        }
        return {};
    }

    // Borrowed view of the raw bytes a token decodes to — ByteLevel-unmapped for ordinary
    // tokens, the literal content for added/special tokens. This is the table the decoder and
    // anything else that would otherwise call decode() in a loop should read.
    std::string_view id_to_bytes_view(int32_t id) const {
        if (id >= 0 && static_cast<size_t>(id) < count_) {
            return {raw_blob_.data() + raw_offsets_[id], raw_offsets_[id + 1] - raw_offsets_[id]};
        }
        return {};
    }

    // Fills out[id] with the single byte token `id` decodes to, or '\0' for every token that
    // decodes to zero or several bytes. One pass over tables already in memory — the consumer's
    // alternative was 248k allocating decode calls through an FFI (measured at ~100 ms).
    void single_char_table(std::vector<char>& out) const {
        out.assign(count_, '\0');
        for (size_t id = 0; id < count_; ++id) {
            if (raw_offsets_[id + 1] - raw_offsets_[id] == 1) {
                out[id] = raw_blob_[raw_offsets_[id]];
            }
        }
    }

    // Allocation-free merge lookup, for the BPE inner loop.
    bool get_merge(int32_t id1, int32_t id2, int32_t& out_rank, int32_t& out_id) const {
        return merges_.find(id1, id2, out_rank, out_id);
    }

    // Check if pair can be merged, returns the merged rank if true, else -1
    int32_t get_merge_rank(int32_t token1, int32_t token2) const;

    // Check if pair can be merged, returns string token and rank if true, else false
    bool get_merge_result(int32_t id1, int32_t id2, std::string& out_str, int32_t& out_rank, int32_t& out_id) const;

    bool is_special_id(int32_t id) const {
        return special_ids_.find(id) != special_ids_.end();
    }

    size_t size() const { return count_; }

    const std::vector<std::string>& get_special_tokens() const { return special_tokens_; }
    const robin_hood::unordered_flat_map<std::string, int32_t>& get_special_token_map() const { return special_token_to_id_; }

    // The Split pattern declared in tokenizer.json's pre_tokenizer, or empty if none was found.
    //
    // Hardcoding the pattern instead is how this library shipped a regex that disagreed with the
    // vocabulary it was encoding for: Qwen 3.6's 248k vocab pre-tokenizes letters as
    // [\p{L}\p{M}]+ — combining marks stay attached — while the older pattern split marks off,
    // fragmenting any text with combining accents, Thai, or Devanagari into the wrong pieces.
    // The file that ships the vocab is the authority on how to split for it.
    const std::string& pretokenizer_regex() const { return pretokenizer_regex_; }

    // True if tokenizer.json declares an NFC normalizer.
    bool wants_nfc() const { return wants_nfc_; }

    // Token id of each input byte under the ByteLevel mapping, or -1 if the vocabulary lacks it.
    //
    // BPE's initial split produces exactly one symbol per input byte: byte_to_unicode maps the 256
    // byte values onto 256 distinct single characters, so no two bytes ever combine into one
    // character and none is dropped. Composing that map with token_to_id once at load time turns
    // the per-word ByteLevel string build and the per-character hash lookup into an array index.
    const std::array<int32_t, 256>& byte_token_ids() const { return byte_token_ids_; }

private:
    // Token text lives in two parallel blob+offset tables rather than vector<std::string>: one
    // allocation per table instead of one per token, which is most of the difference between a
    // 300 ms load and a 10 ms one, and cache-friendlier to scan.
    size_t count_ = 0;
    std::string bl_blob_;                  // ByteLevel text, concatenated by id
    std::vector<uint32_t> bl_offsets_;     // count_+1 entries
    std::string raw_blob_;                 // decoded raw bytes, concatenated by id
    std::vector<uint32_t> raw_offsets_;    // count_+1 entries

    // Ids sorted by ByteLevel text, for token_to_id by binary search. A hash map would answer in
    // O(1) but costs tens of ms to build at load for 248k string keys; every runtime caller of
    // token_to_id is setup-frequency, so ~18 comparisons is the right trade.
    std::vector<int32_t> sorted_ids_;

    MergeTable merges_;

    std::vector<std::string> special_tokens_;
    robin_hood::unordered_flat_map<std::string, int32_t> special_token_to_id_;
    // Reverse of `special_token_to_id_`. The decoder asks "is this id special?" once per token,
    // and answering it by id avoids materialising the token string just to hash it.
    robin_hood::unordered_flat_set<int32_t> special_ids_;

    // Built once at load; see byte_token_ids().
    std::array<int32_t, 256> byte_token_ids_{};

    std::string pretokenizer_regex_;
    bool wants_nfc_ = false;

    void parse_tokenizer_json(const std::string& json_content);
    void finalize_tables();   // raw blob, sorted index, byte table — everything derived
    bool load_cache(const std::string& cache_path, uint64_t src_size, int64_t src_mtime);
    void write_cache(const std::string& cache_path, uint64_t src_size, int64_t src_mtime) const;
};

// ByteLevel mapping
std::string bytes_to_unicode(const std::string& bytes);
std::string unicode_to_bytes(const std::string& unicode);
void initialize_bytes_to_unicode_map();

} // namespace mlx_qwen_tokenizer
