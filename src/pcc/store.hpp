#pragma once
//
// PCC -- the Persistent Context Core (spec S8).
//
// WHAT THIS IS FOR
//   src/context holds the working context: four tiers, compaction on trim. It is very
//   good at deciding what belongs in the prompt and it forgets everything else, because
//   compact_oldest() summarizes a span to one line per turn and erases the turns. The
//   summary even prints the event range it came from -- a provenance pointer with
//   nothing on the other end of it.
//
//   This is the other end of it. Every turn is written here before the working context
//   is allowed to drop it, so a trim stops being destructive: the prompt keeps the
//   summary, and the full text stays one query away. That is the whole idea. An agent's
//   effective context is not the size of its window, it is the size of what it can get
//   back.
//
// BI-TEMPORAL, BECAUSE AN AGENT'S FACTS GO STALE
//   "the build is broken" is true, and then it is not, and both are worth keeping. Every
//   row carries VALID time (when the fact was true of the world) and SYSTEM time (when we
//   recorded it). Superseding a fact closes the old row rather than deleting it, so:
//
//     - retrieval defaults to now and returns the CURRENT truth, which is what stops a
//       stale note from outranking what the agent can observe today;
//     - "what did I believe at 14:30" is a query, not an archaeology exercise, which is
//       how you debug a run that went wrong three hours ago.
//
//   The one thing this rules out is the failure mode the flat memory file has today:
//   .lmp-memory.md is 16 KiB of undated one-liners, and a note written last week that is
//   now false is indistinguishable from one written this morning that is not.
//
// WHAT IT IS NOT
//   Not a vector database. There is no embedding here and no fake one either. Of the
//   twelve cook-off entrants that motivated this component, eleven implemented semantic
//   search and SIX of those did it over an embedding they had invented: character sums,
//   SHA-256 bytes reinterpreted as floats, `[0.1] * 384`, and twice `np.random.seed(
//   hash(text))`. Every one returns arbitrary rows with total confidence, and two of them
//   then built "semantic deduplication" on top, which can only ever match exact
//   duplicates. Counted from the branches, not remembered: docs/BAKEOFF_PCC.md.
//
//   Retrieval here is BM25 over SQLite's FTS5: real, tested, deterministic, and strong on
//   the identifiers, paths and error strings that agent memory is mostly made of. Adding
//   a real embedder is worth doing and is not free -- it needs a model this component
//   deliberately does not link. src/pcc/recall.hpp fuses rank lists rather than scores, so
//   that embedder arrives as a third list without any of this changing.
//
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/pcc/cas.hpp"
#include "src/pcc/sqlite.hpp"

namespace lmp::pcc {

// Microseconds since the Unix epoch, matching src/platform/clock.hpp's resolution.
using TimeUs = std::int64_t;

// A row that has not been superseded is valid until the end of time. Using a sentinel
// rather than NULL keeps every temporal predicate a plain comparison -- a NULL valid_to
// makes `valid_to > ?` silently false, which reads as "no current facts exist" and is a
// bug you find weeks later.
inline constexpr TimeUs kOpenEnded = INT64_MAX;

// Passed as a query time to mean "whatever the clock says at the moment of the call".
inline constexpr TimeUs kNow = -1;

// The kinds are a closed set, and deliberately few. Each one answers a different
// question, and a kind nobody queries differently from another kind is a tag, not a
// kind -- tags belong in the body where BM25 can see them.
namespace kind {
inline constexpr const char* kTurn = "turn";         // one exchange, verbatim
inline constexpr const char* kSpan = "span";         // a compacted range's summary
inline constexpr const char* kFact = "fact";         // a durable claim; supersedable
inline constexpr const char* kArtifact = "artifact"; // a named revision of content
} // namespace kind

// Titles this component itself reasons about. `title` is otherwise free-form -- it is the
// tool name for a turn row -- but the mission is special enough to be recognised by two
// places at once, so the string lives in one.
namespace title {
inline constexpr const char* kMission = "mission";
} // namespace title

struct Item {
    std::int64_t id = 0;
    std::string session;
    std::string kind;
    std::string key;   // supersession identity; empty for append-only kinds
    std::string title; // a short human/model-facing label, indexed with more weight
    std::string body;
    std::string hash; // the CAS hash when kind is artifact; empty otherwise
    TimeUs valid_from = 0;
    TimeUs valid_to = kOpenEnded;
    TimeUs system_time = 0;
    std::uint64_t first_event = 0;
    std::uint64_t last_event = 0;
};

// What a caller hands in to write a row. Separate from Item because the store owns id,
// system_time, and the valid-time bookkeeping, and a caller that can set those can
// corrupt the temporal invariants without meaning to.
struct Record {
    std::string session;
    std::string kind;
    std::string key;
    std::string title;
    std::string body;
    std::string hash;
    std::uint64_t first_event = 0;
    std::uint64_t last_event = 0;
    // When the fact became true of the world. Defaults to the write time, which is right
    // for a turn (it happened when it happened) and wrong for an imported fact that was
    // true earlier -- so it is settable.
    TimeUs valid_from = kNow;
};

struct StoreStats {
    std::int64_t items = 0;
    std::int64_t current_items = 0; // not superseded
    std::int64_t sessions = 0;
    BlobStats blobs;
};

// A point in bi-temporal space. Both default to now, which is the common query and the
// one a caller who has not thought about time should get.
struct AsOf {
    TimeUs valid = kNow;
    TimeUs system = kNow;
};

class Store {
  public:
    // Opens or creates the database at `path`. ":memory:" is a valid path and is what
    // the tests use.
    explicit Store(std::string path);

    // --- writing ------------------------------------------------------------

    // Appends a row. For append-only kinds (turn, span) this is the only writer; for
    // facts, prefer remember(), which maintains the supersession chain.
    std::int64_t append(Record record);

    // Writes a fact under `key`, closing whichever row currently holds that key.
    //
    // Returns the new row's id. Recording the same body that is already current is a
    // no-op that returns the existing id: an agent that re-derives the same conclusion
    // every turn would otherwise fill the timeline with identical revisions, which is
    // how the flat memory file's dedupe rule came about.
    std::int64_t remember(Record record);

    // Closes `key` without replacing it -- "this stopped being true". The history stays
    // queryable; only the present changes. Returns false if the key had nothing current.
    bool forget(std::string_view key, std::string_view session = {});

    // Stores content in the CAS and records an artifact row pointing at it. `base` is
    // the previous revision's hash, used as a delta base when it is one.
    std::int64_t put_artifact(Record record, std::string_view content,
                              std::string_view base = {});

    // --- reading ------------------------------------------------------------

    [[nodiscard]] std::optional<Item> get(std::int64_t id) const;

    // The row holding `key` at the given point in bi-temporal space.
    [[nodiscard]] std::optional<Item> current(std::string_view key, AsOf as_of = {},
                                              std::string_view session = {}) const;

    // Every revision ever written under `key`, oldest first. The audit trail: what was
    // believed, when it was believed, and when it stopped being believed.
    [[nodiscard]] std::vector<Item> history(std::string_view key,
                                            std::string_view session = {}) const;

    // Every current row of a kind, newest first. `limit` of 0 means no limit.
    [[nodiscard]] std::vector<Item> by_kind(std::string_view kind, AsOf as_of = {},
                                            std::string_view session = {},
                                            int limit = 0) const;

    // The turns whose event range overlaps [first, last] -- the query that turns a
    // compacted span's provenance pointer back into the turns it was made from.
    //
    // Restricted to kTurn. Span rows carry the same range as the turns they cover, so
    // without that restriction this returns the summary alongside its own source.
    [[nodiscard]] std::vector<Item> events_between(std::uint64_t first,
                                                   std::uint64_t last,
                                                   std::string_view session = {}) const;

    // BM25-ranked full-text search, best first. `text` is a user query, not an FTS5
    // expression: it is escaped before it reaches MATCH, so a stray quote or AND is
    // searched for rather than executed.
    [[nodiscard]] std::vector<Item> search(std::string_view text, AsOf as_of = {},
                                           std::string_view session = {},
                                           int limit = 20) const;

    // The reconstructed content behind an artifact row.
    [[nodiscard]] std::optional<std::string> artifact_content(const Item& item) const;

    [[nodiscard]] StoreStats stats() const;

    [[nodiscard]] Cas& artifacts() noexcept { return cas_; }
    [[nodiscard]] Db& db() noexcept { return db_; }

    // Injectable so tests get a deterministic timeline. Production leaves it alone and
    // reads the system clock.
    void set_clock(TimeUs (*clock)()) noexcept { clock_ = clock; }
    [[nodiscard]] TimeUs now() const { return clock_(); }

  private:
    [[nodiscard]] TimeUs resolve(TimeUs t) const { return t == kNow ? now() : t; }

    Db db_;
    Cas cas_;
    TimeUs (*clock_)() = nullptr;
};

} // namespace lmp::pcc
