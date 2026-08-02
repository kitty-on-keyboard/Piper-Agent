#include "src/pcc/cas.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

#include <CommonCrypto/CommonDigest.h>
#include <zlib.h>

#include "src/pcc/diff.hpp"

namespace lmp::pcc {
namespace {

// Raw deflate. The zlib wrapper's only contribution here would be an adler32 of the
// dictionary, which drives the Z_NEED_DICT handshake -- and we always know the base
// before we start, so the handshake is pure ceremony. Raw also saves six bytes a blob.
constexpr int kRawWindowBits = -15;

class ZStream {
  public:
    explicit ZStream(bool deflating) : deflating_(deflating) {}
    ~ZStream() {
        if (deflating_) {
            deflateEnd(&s_);
        } else {
            inflateEnd(&s_);
        }
    }
    ZStream(const ZStream&) = delete;
    ZStream& operator=(const ZStream&) = delete;

    z_stream& get() noexcept { return s_; }

  private:
    z_stream s_{};
    bool deflating_;
};

const Bytef* as_bytes(std::string_view s) {
    return reinterpret_cast<const Bytef*>(s.data());
}

std::vector<unsigned char> compress(std::string_view data, std::string_view dict) {
    ZStream z(true);
    z_stream& s = z.get();
    if (deflateInit2(&s, Z_BEST_COMPRESSION, Z_DEFLATED, kRawWindowBits, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK) {
        throw SqlError("deflateInit2 failed");
    }
    if (!dict.empty() &&
        deflateSetDictionary(&s, as_bytes(dict), static_cast<uInt>(dict.size())) != Z_OK) {
        throw SqlError("deflateSetDictionary failed");
    }

    std::vector<unsigned char> out(deflateBound(&s, static_cast<uLong>(data.size())) + 16);
    s.next_in = const_cast<Bytef*>(as_bytes(data));
    s.avail_in = static_cast<uInt>(data.size());
    s.next_out = out.data();
    s.avail_out = static_cast<uInt>(out.size());
    if (deflate(&s, Z_FINISH) != Z_STREAM_END) {
        throw SqlError("deflate did not finish in one pass");
    }
    out.resize(s.total_out);
    return out;
}

// `expected` is the exact uncompressed size, read from the row. Sizing the buffer from
// stored metadata rather than growing it means one allocation and no resize loop; a row
// whose size field disagrees with its data fails here instead of silently truncating.
std::string decompress(std::span<const unsigned char> data, std::string_view dict,
                       std::size_t expected) {
    ZStream z(false);
    z_stream& s = z.get();
    if (inflateInit2(&s, kRawWindowBits) != Z_OK) {
        throw SqlError("inflateInit2 failed");
    }
    if (!dict.empty() &&
        inflateSetDictionary(&s, as_bytes(dict), static_cast<uInt>(dict.size())) != Z_OK) {
        throw SqlError("inflateSetDictionary failed");
    }

    std::string out(expected, '\0');
    s.next_in = const_cast<Bytef*>(data.data());
    s.avail_in = static_cast<uInt>(data.size());
    s.next_out = reinterpret_cast<Bytef*>(out.data());
    s.avail_out = static_cast<uInt>(expected);
    const int rc = inflate(&s, Z_FINISH);
    if (rc != Z_STREAM_END || s.total_out != expected) {
        throw SqlError("inflate produced " + std::to_string(s.total_out) + " bytes, the "
                       "row claims " + std::to_string(expected));
    }
    return out;
}

} // namespace

std::string sha256_hex(std::string_view data) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(data.data(), static_cast<CC_LONG>(data.size()), digest.data());

    std::string hex(digest.size() * 2, '\0');
    constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < digest.size(); ++i) {
        hex[i * 2] = kHex[digest[i] >> 4];
        hex[i * 2 + 1] = kHex[digest[i] & 0x0F];
    }
    return hex;
}

void Cas::migrate(Db& db) {
    db.exec(R"sql(
        CREATE TABLE IF NOT EXISTS blob (
            hash   TEXT PRIMARY KEY,
            base   TEXT,              -- NULL when stored whole
            depth  INTEGER NOT NULL,  -- links to the nearest whole copy; 0 when whole
            size   INTEGER NOT NULL,  -- uncompressed bytes
            stored INTEGER NOT NULL,  -- bytes this row actually holds
            data   BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS blob_base ON blob(base);
    )sql");
}

std::string Cas::put(std::string_view content, std::string_view base) {
    const std::string hash = sha256_hex(content);
    if (contains(hash)) {
        return hash; // deduplicated: the same bytes are stored once, however they arrive
    }

    // The base is a hint, and every way it can be unusable ends the same way -- store
    // whole. Resolving that here keeps the decision in one place instead of spreading a
    // chain-depth check across every caller.
    std::string base_content;
    int depth = 0;
    if (!base.empty()) {
        Stmt probe(*db_, "SELECT depth FROM blob WHERE hash = ?");
        probe.bind(1, base);
        const bool known = probe.step();
        const int base_depth = known ? static_cast<int>(probe.column_int(0)) : -1;
        if (known && base_depth < kMaxChainDepth) {
            std::optional<std::string> loaded = get(base);
            if (loaded.has_value()) {
                base_content = std::move(*loaded);
                depth = base_depth + 1;
            }
        }
    }

    const std::vector<unsigned char> packed = compress(content, base_content);
    Stmt ins(*db_, "INSERT INTO blob (hash, base, depth, size, stored, data) "
                   "VALUES (?, ?, ?, ?, ?, ?)");
    ins.bind(1, hash);
    if (depth == 0) {
        ins.bind_null(2);
    } else {
        ins.bind(2, base);
    }
    ins.bind(3, static_cast<std::int64_t>(depth));
    ins.bind(4, static_cast<std::int64_t>(content.size()));
    ins.bind(5, static_cast<std::int64_t>(packed.size()));
    ins.bind_blob(6, packed);
    ins.run();
    return hash;
}

bool Cas::contains(std::string_view hash) const {
    Stmt stmt(*db_, "SELECT 1 FROM blob WHERE hash = ?");
    stmt.bind(1, hash);
    return stmt.step();
}

std::optional<std::string> Cas::load(std::string_view hash, int budget) const {
    if (budget < 0) {
        // Unreachable while put() maintains the depth invariant. It is checked anyway
        // because the alternative failure -- a cycle introduced by a corrupt row -- is
        // unbounded recursion, and a stack overflow is a much worse way to find out.
        throw SqlError("delta chain exceeds the depth bound; blob table is corrupt");
    }
    Stmt stmt(*db_, "SELECT base, size, data FROM blob WHERE hash = ?");
    stmt.bind(1, hash);
    if (!stmt.step()) {
        return std::nullopt;
    }
    const bool whole = stmt.column_is_null(0);
    const std::string base = whole ? std::string() : stmt.column_text(0);
    const auto size = static_cast<std::size_t>(stmt.column_int(1));
    const std::vector<unsigned char> data = stmt.column_blob(2);

    std::string dict;
    if (!whole) {
        std::optional<std::string> base_content = load(base, budget - 1);
        if (!base_content.has_value()) {
            throw SqlError("blob " + std::string(hash) + " is a delta against " + base +
                           ", which is missing");
        }
        dict = std::move(*base_content);
    }
    return decompress(data, dict, size);
}

std::optional<std::string> Cas::get(std::string_view hash) const {
    std::optional<std::string> content = load(hash, kMaxChainDepth);
    if (!content.has_value()) {
        return std::nullopt;
    }
    // Verifying costs one hash of data already in hand and turns a whole class of silent
    // corruption -- a truncated blob, a rebuilt chain, a restored backup that mixed
    // generations -- into a loud failure at the point of use.
    if (sha256_hex(*content) != hash) {
        throw SqlError("blob " + std::string(hash) + " reconstructs to different bytes");
    }
    return content;
}

std::optional<std::string> Cas::diff(std::string_view from, std::string_view to,
                                     int context) const {
    std::optional<std::string> a = get(from);
    std::optional<std::string> b = get(to);
    if (!a.has_value() || !b.has_value()) {
        return std::nullopt;
    }
    return unified_diff(*a, *b, std::string(from.substr(0, 12)),
                        std::string(to.substr(0, 12)), context);
}

BlobStats Cas::stats() const {
    Stmt stmt(*db_, "SELECT COUNT(*), COALESCE(SUM(size), 0), COALESCE(SUM(stored), 0), "
                    "COALESCE(SUM(depth = 0), 0) FROM blob");
    BlobStats s;
    if (stmt.step()) {
        s.blobs = stmt.column_int(0);
        s.logical_bytes = stmt.column_int(1);
        s.stored_bytes = stmt.column_int(2);
        s.whole_blobs = stmt.column_int(3);
        s.delta_blobs = s.blobs - s.whole_blobs;
    }
    return s;
}

} // namespace lmp::pcc
