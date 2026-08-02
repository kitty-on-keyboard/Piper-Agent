#pragma once
//
// Content-addressed artifact store with bounded delta chains (spec S8, PCC).
//
// An agent rewrites the same file twenty times in a run. Storing twenty copies is
// wasteful; storing nineteen patches against the first is worse, because reading the
// twentieth then costs nineteen decompressions and one corrupt link loses everything
// after it. Both mistakes were made by the cook-off this component learned from -- every
// entrant that implemented deltas recursed to reconstruct and none bounded the chain.
//
// WHAT IS STORED
//   Content is keyed by the SHA-256 of the UNCOMPRESSED bytes, so identical content is
//   stored once no matter how it arrives. A revision may be stored as a delta against a
//   base revision, compressed with the base as a zlib preset dictionary.
//
// WHY A DICTIONARY RATHER THAN A TEXT PATCH
//   The cook-off's other half reached for diff-match-patch, which is text-only, lossy on
//   invalid UTF-8, and produces a patch that still has to be compressed. A preset
//   dictionary is binary-clean, needs no second pass, and costs one call. The price is
//   zlib's 32 KiB window: for artifacts larger than that only the TAIL of the base seeds
//   the dictionary and the ratio decays toward plain deflate. That is a real limit and it
//   is measured rather than asserted -- see tests/pcc/test_cas.cpp, which pins the ratio
//   on both sides of the window.
//
// THE CHAIN BOUND
//   A delta may be at most kMaxChainDepth links from a whole copy. At the bound the next
//   revision is stored WHOLE, which re-bases the chain. Reconstruction is therefore
//   O(kMaxChainDepth) inflates, always, and a damaged blob can lose at most that many
//   revisions rather than the file's entire history.
//
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/pcc/sqlite.hpp"

namespace lmp::pcc {

// Eight was chosen against the measurement in tests/pcc/test_cas.cpp, not by taste: it
// keeps worst-case reconstruction under the cost of one whole-blob inflate for the
// artifact sizes an agent actually writes, and it is small enough that a rebased chain
// stays a rounding error on total stored bytes.
inline constexpr int kMaxChainDepth = 8;

struct BlobStats {
    std::int64_t blobs = 0;
    std::int64_t logical_bytes = 0; // total uncompressed size of every distinct blob
    std::int64_t stored_bytes = 0;  // what the database actually holds
    std::int64_t whole_blobs = 0;
    std::int64_t delta_blobs = 0;
};

// SHA-256 of `data`, lowercase hex. The artifact identity used everywhere in PCC.
[[nodiscard]] std::string sha256_hex(std::string_view data);

class Cas {
  public:
    // Does not own the handle; the Store that opens the database outlives every Cas.
    explicit Cas(Db& db) : db_(&db) {}

    // Creates the blob table. Idempotent.
    static void migrate(Db& db);

    // Stores `content` and returns its hash.
    //
    // `base` names a revision this content is a successor of. It is a HINT: when it is
    // absent, unknown, or already at the chain bound, the content is stored whole and
    // the call still succeeds. A caller therefore never has to reason about the chain,
    // which is what stops the bound from leaking into every call site.
    std::string put(std::string_view content, std::string_view base = {});

    // Returns the reconstructed content, or nullopt when the hash is unknown.
    //
    // The reconstruction is verified against the requested hash before it is returned:
    // a delta chain that reconstructs to the wrong bytes reports corruption instead of
    // handing back plausible garbage.
    [[nodiscard]] std::optional<std::string> get(std::string_view hash) const;

    [[nodiscard]] bool contains(std::string_view hash) const;

    // A unified diff between two stored revisions, in the format `git diff` emits, with
    // `context` lines either side of each hunk. Returns nullopt if either hash is
    // unknown. This is the tool-facing shape: an agent asking what changed between two
    // snapshots wants the diff, not two files it has to compare itself.
    [[nodiscard]] std::optional<std::string> diff(std::string_view from,
                                                  std::string_view to,
                                                  int context = 3) const;

    [[nodiscard]] BlobStats stats() const;

  private:
    [[nodiscard]] std::optional<std::string> load(std::string_view hash, int budget) const;

    Db* db_;
};

} // namespace lmp::pcc
