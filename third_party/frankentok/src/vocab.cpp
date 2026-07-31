#include "mlx_qwen_tokenizer/vocab.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <iostream>

using json = nlohmann::json;

namespace mlx_qwen_tokenizer {

static std::vector<std::string> byte_to_unicode_vec(256);
static robin_hood::unordered_flat_map<std::string, uint8_t> unicode_to_byte_map;
// ByteLevel characters are all below U+0300; this array inverts the mapping without a hash
// lookup, for the load-time pass that unmaps every vocab token. -1 = not a ByteLevel char.
static std::array<int16_t, 0x300> codepoint_to_byte;
// Guarded by std::once_flag rather than a plain bool: two Vocabs constructed concurrently would
// otherwise both see the flag unset and race on the same globals.
static std::once_flag byte_maps_once;

static void build_byte_maps() {
    std::vector<int> bs;
    std::vector<int> cs;

    // 33 to 126, 161 to 172, 174 to 255
    for (int b = 33; b <= 126; ++b) { bs.push_back(b); }
    for (int b = 161; b <= 172; ++b) { bs.push_back(b); }
    for (int b = 174; b <= 255; ++b) { bs.push_back(b); }

    cs = bs;

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    codepoint_to_byte.fill(-1);
    for (size_t i = 0; i < bs.size(); ++i) {
        int c = cs[i];
        std::string s;
        // UTF-8 encode c
        if (c < 0x80) {
            s.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (c >> 6)));
            s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
        byte_to_unicode_vec[bs[i]] = s;
        unicode_to_byte_map[s] = static_cast<uint8_t>(bs[i]);
        codepoint_to_byte[c] = static_cast<int16_t>(bs[i]);
    }
}

void initialize_bytes_to_unicode_map() {
    std::call_once(byte_maps_once, build_byte_maps);
}

std::string bytes_to_unicode(const std::string& bytes) {
    initialize_bytes_to_unicode_map();
    std::string res;
    for (unsigned char b : bytes) {
        res += byte_to_unicode_vec[b];
    }
    return res;
}

std::string unicode_to_bytes(const std::string& unicode) {
    initialize_bytes_to_unicode_map();
    std::string res;
    size_t i = 0;
    while (i < unicode.size()) {
        size_t len = 1;
        if ((unicode[i] & 0xE0) == 0xC0) len = 2;
        else if ((unicode[i] & 0xF0) == 0xE0) len = 3;
        else if ((unicode[i] & 0xF8) == 0xF0) len = 4;

        std::string char_str = unicode.substr(i, len);
        auto it = unicode_to_byte_map.find(char_str);
        if (it != unicode_to_byte_map.end()) {
            res.push_back(static_cast<char>(it->second));
        } else {
            // Unmapped unicode, just copy bytes? Usually shouldn't happen with valid byte-level mapped text.
            res += char_str;
        }
        i += len;
    }
    return res;
}

// ---------------------------------------------------------------------------------------------
// Binary vocab cache.
//
// Every table below is either a POD vector or a byte blob, so the cache is a header plus a
// sequence of (length, bytes) sections written and read with no per-entry work at all. The JSON
// path costs ~300 ms on the 19 MB production vocab — almost all of it JSON parsing and string
// allocation — and this path replaces it with sequential reads.
//
// Host-only by design (no endianness or layout negotiation): it lives next to the JSON it was
// built from, is invalidated by source size/mtime, and is rebuilt silently whenever invalid.

namespace {

constexpr char kCacheMagic[8] = {'F', 'T', 'O', 'K', 'B', 'I', 'N', '5'};

struct CacheHeader {
    char magic[8];
    uint64_t src_size;
    int64_t src_mtime;
    uint64_t count;          // id space
    uint64_t merge_slots;
    uint64_t merge_count;
    uint64_t special_count;
    uint64_t regex_len;
    uint64_t payload_hash;   // FNV-1a of everything after the header
    uint8_t wants_nfc;
    uint8_t pad[7];
};

// The size/mtime check catches a *stale* cache; this catches a *corrupt* one. Without it, a
// flipped bit in the merge table loads cleanly and mis-tokenizes forever — the one failure mode
// worse than being slow.
//
// Consumed 8 bytes at a time. A byte-at-a-time FNV-1a over the same ~16 MB payload cost ~20 ms,
// which is most of a warm load's budget — the integrity check has to be cheap enough that it
// doesn't undo the reason the cache exists. The tail stays bytewise so the result is defined for
// any length.
uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h ^= w;
        h *= 0x9e3779b97f4a7c15ull;
        h ^= h >> 29;
    }
    for (; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}
constexpr uint64_t kFnvSeed = 0xcbf29ce484222325ull;

template <typename T>
bool read_vec(std::ifstream& f, std::vector<T>& v, size_t n) {
    v.resize(n);
    return bool(f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))));
}

bool read_str(std::ifstream& f, std::string& s, size_t n) {
    s.resize(n);
    return bool(f.read(s.data(), std::streamsize(n)));
}

template <typename T>
void write_vec(std::ofstream& f, const std::vector<T>& v) {
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
}

} // namespace

bool Vocab::load_cache(const std::string& cache_path, uint64_t src_size, int64_t src_mtime) {
    std::ifstream f(cache_path, std::ios::binary);
    if (!f.is_open()) return false;

    CacheHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    if (std::memcmp(h.magic, kCacheMagic, sizeof(kCacheMagic)) != 0) return false;
    if (h.src_size != src_size || h.src_mtime != src_mtime) return false;

    count_ = h.count;
    uint64_t bl_bytes = 0, raw_bytes = 0;
    if (!f.read(reinterpret_cast<char*>(&bl_bytes), sizeof(bl_bytes))) return false;
    if (!f.read(reinterpret_cast<char*>(&raw_bytes), sizeof(raw_bytes))) return false;

    if (!read_vec(f, bl_offsets_, count_ + 1)) return false;
    if (!read_str(f, bl_blob_, bl_bytes)) return false;
    if (!read_vec(f, raw_offsets_, count_ + 1)) return false;
    if (!read_str(f, raw_blob_, raw_bytes)) return false;
    if (!read_vec(f, merges_.slots(), h.merge_slots)) return false;
    merges_.set_loaded(h.merge_count);
    if (!read_vec(f, sorted_ids_, count_)) return false;

    std::vector<int32_t> bytes_tbl;
    if (!read_vec(f, bytes_tbl, 256)) return false;
    std::copy(bytes_tbl.begin(), bytes_tbl.end(), byte_token_ids_.begin());

    special_tokens_.clear();
    special_token_to_id_.clear();
    special_ids_.clear();
    for (uint64_t i = 0; i < h.special_count; ++i) {
        uint32_t id = 0, len = 0;
        if (!f.read(reinterpret_cast<char*>(&id), sizeof(id))) return false;
        if (!f.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
        std::string content;
        if (!read_str(f, content, len)) return false;
        special_tokens_.push_back(content);
        special_token_to_id_[content] = int32_t(id);
        special_ids_.insert(int32_t(id));
    }

    if (!read_str(f, pretokenizer_regex_, h.regex_len)) return false;
    wants_nfc_ = h.wants_nfc != 0;

    // Verify the payload before trusting any of it.
    uint64_t hash = kFnvSeed;
    hash = fnv1a(hash, bl_offsets_.data(), bl_offsets_.size() * sizeof(uint32_t));
    hash = fnv1a(hash, bl_blob_.data(), bl_blob_.size());
    hash = fnv1a(hash, raw_offsets_.data(), raw_offsets_.size() * sizeof(uint32_t));
    hash = fnv1a(hash, raw_blob_.data(), raw_blob_.size());
    hash = fnv1a(hash, merges_.slots().data(), merges_.slots().size() * sizeof(MergeTable::Slot));
    hash = fnv1a(hash, sorted_ids_.data(), sorted_ids_.size() * sizeof(int32_t));
    if (hash != h.payload_hash) return false;

    initialize_bytes_to_unicode_map();
    return true;
}

void Vocab::write_cache(const std::string& cache_path, uint64_t src_size, int64_t src_mtime) const {
    // Write to a temp name then rename, so a crash mid-write can't leave a truncated cache that
    // a later load would half-read.
    std::error_code dir_ec;
    std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path(), dir_ec);
    if (dir_ec) return;

    const std::string tmp = cache_path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) return;

        CacheHeader h{};
        std::memcpy(h.magic, kCacheMagic, sizeof(kCacheMagic));
        h.src_size = src_size;
        h.src_mtime = src_mtime;
        h.count = count_;
        h.merge_slots = merges_.slots().size();
        h.merge_count = merges_.count();
        h.special_count = special_tokens_.size();
        h.regex_len = pretokenizer_regex_.size();
        h.wants_nfc = wants_nfc_ ? 1 : 0;

        uint64_t hash = kFnvSeed;
        hash = fnv1a(hash, bl_offsets_.data(), bl_offsets_.size() * sizeof(uint32_t));
        hash = fnv1a(hash, bl_blob_.data(), bl_blob_.size());
        hash = fnv1a(hash, raw_offsets_.data(), raw_offsets_.size() * sizeof(uint32_t));
        hash = fnv1a(hash, raw_blob_.data(), raw_blob_.size());
        hash = fnv1a(hash, merges_.slots().data(), merges_.slots().size() * sizeof(MergeTable::Slot));
        hash = fnv1a(hash, sorted_ids_.data(), sorted_ids_.size() * sizeof(int32_t));
        h.payload_hash = hash;

        f.write(reinterpret_cast<const char*>(&h), sizeof(h));

        const uint64_t bl_bytes = bl_blob_.size(), raw_bytes = raw_blob_.size();
        f.write(reinterpret_cast<const char*>(&bl_bytes), sizeof(bl_bytes));
        f.write(reinterpret_cast<const char*>(&raw_bytes), sizeof(raw_bytes));
        write_vec(f, bl_offsets_);
        f.write(bl_blob_.data(), std::streamsize(bl_blob_.size()));
        write_vec(f, raw_offsets_);
        f.write(raw_blob_.data(), std::streamsize(raw_blob_.size()));
        write_vec(f, merges_.slots());
        write_vec(f, sorted_ids_);

        std::vector<int32_t> bytes_tbl(byte_token_ids_.begin(), byte_token_ids_.end());
        write_vec(f, bytes_tbl);

        for (size_t i = 0; i < special_tokens_.size(); ++i) {
            const std::string& content = special_tokens_[i];
            const uint32_t id = uint32_t(special_token_to_id_.at(content));
            const uint32_t len = uint32_t(content.size());
            f.write(reinterpret_cast<const char*>(&id), sizeof(id));
            f.write(reinterpret_cast<const char*>(&len), sizeof(len));
            f.write(content.data(), std::streamsize(len));
        }
        f.write(pretokenizer_regex_.data(), std::streamsize(pretokenizer_regex_.size()));
        if (!f.good()) { f.close(); std::remove(tmp.c_str()); return; }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, cache_path, ec);
    if (ec) std::remove(tmp.c_str());
}

std::string Vocab::cache_path_for(const std::string& source_path, const std::string& cache_dir) {
    std::filesystem::path dir;
    if (!cache_dir.empty()) {
        dir = cache_dir;
    } else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        dir = std::filesystem::path(xdg) / "mlx_qwen_tokenizer";
    } else if (const char* home = std::getenv("HOME"); home && *home) {
#ifdef __APPLE__
        dir = std::filesystem::path(home) / "Library" / "Caches" / "mlx_qwen_tokenizer";
#else
        dir = std::filesystem::path(home) / ".cache" / "mlx_qwen_tokenizer";
#endif
    } else {
        std::error_code ec;
        dir = std::filesystem::temp_directory_path(ec) / "mlx_qwen_tokenizer";
    }

    // Name by the source's absolute path, so two models whose tokenizer.json share a basename
    // cannot collide. The stem is kept in the name only to make the directory readable.
    std::error_code ec;
    std::filesystem::path abs = std::filesystem::absolute(source_path, ec);
    const std::string key = ec ? source_path : abs.string();
    const uint64_t h = fnv1a(kFnvSeed, key.data(), key.size());

    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "-%016llx.ftbin", static_cast<unsigned long long>(h));
    std::string stem = std::filesystem::path(source_path).stem().string();
    if (stem.empty()) stem = "tokenizer";
    return (dir / (stem + suffix)).string();
}

bool Vocab::load_from_file(const std::string& path, const LoadOptions& options) {
    uint64_t src_size = 0;
    int64_t src_mtime = 0;
    std::string cache_path;
    if (options.use_cache) {
        std::error_code ec;
        src_size = std::filesystem::file_size(path, ec);
        if (!ec) {
            const auto t = std::filesystem::last_write_time(path, ec);
            if (!ec) {
                src_mtime = int64_t(t.time_since_epoch().count());
                cache_path = cache_path_for(path, options.cache_dir);
                if (load_cache(cache_path, src_size, src_mtime)) return true;
            }
        }
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    if (!load_from_string(buffer.str())) return false;

    if (!cache_path.empty()) write_cache(cache_path, src_size, src_mtime);
    return true;
}

bool Vocab::load_from_string(const std::string& json_content) {
    try {
        parse_tokenizer_json(json_content);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse tokenizer.json: " << e.what() << std::endl;
        return false;
    }
}

void Vocab::parse_tokenizer_json(const std::string& json_content) {
    auto j = json::parse(json_content);
    initialize_bytes_to_unicode_map();

    // 1. Load Vocab. A local map serves the merge resolution below; it is not kept — runtime
    // token_to_id goes through the sorted index instead.
    auto vocab = j.at("model").at("vocab");
    robin_hood::unordered_flat_map<std::string, int32_t> token_to_id;
    token_to_id.reserve(vocab.size());
    int32_t max_id = -1;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int32_t id = it.value();
        token_to_id[it.key()] = id;
        if (id > max_id) max_id = id;
    }

    // 3. Added tokens first (they may extend the id space), then lay out the blobs.
    struct Added { std::string content; int32_t id; };
    std::vector<Added> added_tokens;
    if (j.contains("added_tokens")) {
        for (const auto& added : j.at("added_tokens")) {
            Added a{added.at("content"), added.at("id")};
            if (a.id > max_id) max_id = a.id;
            // Every added token goes through the trie, special-flagged or not: Qwen's ChatML tags
            // must never be reachable by BPE, and an added token that BPE could fragment would
            // encode differently from the reference tokenizer.
            added_tokens.push_back(std::move(a));
        }
    }

    count_ = size_t(max_id) + 1;
    std::vector<std::string_view> by_id(count_);
    for (const auto& [tok, id] : token_to_id) by_id[size_t(id)] = tok;
    for (const auto& a : added_tokens) by_id[size_t(a.id)] = a.content;

    bl_offsets_.assign(count_ + 1, 0);
    size_t total = 0;
    for (size_t id = 0; id < count_; ++id) {
        bl_offsets_[id] = uint32_t(total);
        total += by_id[id].size();
    }
    bl_offsets_[count_] = uint32_t(total);
    bl_blob_.clear();
    bl_blob_.reserve(total);
    for (size_t id = 0; id < count_; ++id) bl_blob_.append(by_id[id]);

    special_tokens_.clear();
    special_token_to_id_.clear();
    special_ids_.clear();
    for (const auto& a : added_tokens) {
        token_to_id[a.content] = a.id;
        special_tokens_.push_back(a.content);
        special_token_to_id_[a.content] = a.id;
        special_ids_.insert(a.id);
    }

    // 2. Load Merges
    if (j.at("model").contains("merges")) {
        auto merges = j.at("model").at("merges");
        merges_.build(merges.size());
        int32_t rank = 0;
        for (const auto& merge : merges) {
            std::string t1, t2;
            if (merge.is_string()) {
                std::string merge_str = merge.get<std::string>();
                size_t space_pos = merge_str.find(' ');
                if (space_pos != std::string::npos) {
                    t1 = merge_str.substr(0, space_pos);
                    t2 = merge_str.substr(space_pos + 1);
                }
            } else if (merge.is_array() && merge.size() == 2) {
                t1 = merge[0].get<std::string>();
                t2 = merge[1].get<std::string>();
            }
            if (!t1.empty() && !t2.empty()) {
                auto id1_it = token_to_id.find(t1);
                auto id2_it = token_to_id.find(t2);
                auto merged_id_it = token_to_id.find(t1 + t2);

                if (id1_it != token_to_id.end() && id2_it != token_to_id.end() && merged_id_it != token_to_id.end()) {
                    merges_.insert(id1_it->second, id2_it->second, rank, merged_id_it->second);
                }
            }
            rank++;
        }
    } else {
        merges_.build(1);
    }

    // 4. Pre-tokenizer Split pattern and normalizer, straight from the file. The vocabulary was
    // trained against these; substituting our own is how encode drifts from the reference.
    pretokenizer_regex_.clear();
    wants_nfc_ = false;
    if (j.contains("pre_tokenizer") && !j.at("pre_tokenizer").is_null()) {
        const auto& pt = j.at("pre_tokenizer");
        const auto take_split = [this](const json& node) {
            if (node.value("type", "") == "Split" && node.contains("pattern") &&
                node.at("pattern").contains("Regex")) {
                pretokenizer_regex_ = node.at("pattern").at("Regex").get<std::string>();
            }
        };
        if (pt.value("type", "") == "Sequence" && pt.contains("pretokenizers")) {
            for (const auto& sub : pt.at("pretokenizers")) take_split(sub);
        } else {
            take_split(pt);
        }
    }
    if (j.contains("normalizer") && !j.at("normalizer").is_null()) {
        const auto& nm = j.at("normalizer");
        wants_nfc_ = nm.value("type", "") == "NFC";
        if (!wants_nfc_ && nm.value("type", "") == "Sequence" && nm.contains("normalizers")) {
            for (const auto& sub : nm.at("normalizers")) {
                if (sub.value("type", "") == "NFC") { wants_nfc_ = true; break; }
            }
        }
    }

    finalize_tables();
}

void Vocab::finalize_tables() {
    // Raw-bytes blob: ByteLevel-unmap every ordinary token; added tokens keep their literal
    // content (they are never ByteLevel-encoded). This is the table decode reads at runtime —
    // built once here so no decode call ever unmaps a character again.
    raw_offsets_.assign(count_ + 1, 0);
    raw_blob_.clear();
    raw_blob_.reserve(bl_blob_.size());
    for (size_t id = 0; id < count_; ++id) {
        raw_offsets_[id] = uint32_t(raw_blob_.size());
        const std::string_view t = id_to_token_view(int32_t(id));
        if (special_ids_.find(int32_t(id)) != special_ids_.end()) {
            raw_blob_.append(t);
            continue;
        }
        size_t i = 0;
        while (i < t.size()) {
            const unsigned char c = static_cast<unsigned char>(t[i]);
            int32_t cp;
            size_t len;
            if (c < 0x80) { cp = c; len = 1; }
            else if ((c & 0xE0) == 0xC0 && i + 1 < t.size()) {
                cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(t[i + 1]) & 0x3F);
                len = 2;
            } else {
                // Not a ByteLevel character; copy through (mirrors unicode_to_bytes' fallback).
                raw_blob_.push_back(t[i]);
                i += 1;
                continue;
            }
            const int16_t b = (cp < 0x300) ? codepoint_to_byte[cp] : int16_t(-1);
            if (b >= 0) raw_blob_.push_back(static_cast<char>(b));
            else raw_blob_.append(t.substr(i, len));
            i += len;
        }
    }
    raw_offsets_[count_] = uint32_t(raw_blob_.size());

    // Sorted index for token_to_id.
    sorted_ids_.resize(count_);
    for (size_t id = 0; id < count_; ++id) sorted_ids_[id] = int32_t(id);
    std::sort(sorted_ids_.begin(), sorted_ids_.end(), [this](int32_t a, int32_t b) {
        return id_to_token_view(a) < id_to_token_view(b);
    });

    // Compose the ByteLevel map with the vocabulary so BPE's initial split needs neither.
    for (int b = 0; b < 256; ++b) {
        byte_token_ids_[b] = token_to_id(byte_to_unicode_vec[b]);
    }
}

int32_t Vocab::token_to_id(std::string_view token) const {
    auto it = std::lower_bound(sorted_ids_.begin(), sorted_ids_.end(), token,
                               [this](int32_t id, std::string_view t) {
                                   return id_to_token_view(id) < t;
                               });
    if (it != sorted_ids_.end() && id_to_token_view(*it) == token) return *it;
    return -1; // Not found
}

std::string Vocab::id_to_token(int32_t id) const {
    return std::string(id_to_token_view(id));
}

int32_t Vocab::get_merge_rank(int32_t token1, int32_t token2) const {
    int32_t rank = -1, merged = -1;
    if (merges_.find(token1, token2, rank, merged)) return rank;
    return -1; // Cannot be merged
}

bool Vocab::get_merge_result(int32_t id1, int32_t id2, std::string& out_str, int32_t& out_rank, int32_t& out_id) const {
    if (merges_.find(id1, id2, out_rank, out_id)) {
        out_str = id_to_token(out_id);
        return true;
    }
    return false;
}

} // namespace mlx_qwen_tokenizer
