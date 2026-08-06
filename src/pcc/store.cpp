#include "src/pcc/store.hpp"

#include <chrono>
#include <utility>

namespace lmp::pcc {
namespace {

TimeUs system_clock_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

constexpr const char* kSchema = R"sql(
    CREATE TABLE IF NOT EXISTS item (
        id          INTEGER PRIMARY KEY,
        session     TEXT    NOT NULL,
        kind        TEXT    NOT NULL,
        key         TEXT    NOT NULL DEFAULT '',
        title       TEXT    NOT NULL DEFAULT '',
        body        TEXT    NOT NULL,
        hash        TEXT    NOT NULL DEFAULT '',
        valid_from  INTEGER NOT NULL,
        valid_to    INTEGER NOT NULL,
        system_time INTEGER NOT NULL,
        first_event INTEGER NOT NULL DEFAULT 0,
        last_event  INTEGER NOT NULL DEFAULT 0
    );

    -- The supersession lookup: "the row currently holding this key". Partial, because
    -- the closed rows are the overwhelming majority in a long-lived store and none of
    -- them can ever match it.
    CREATE INDEX IF NOT EXISTS item_key_open ON item(session, key)
        WHERE valid_to = 9223372036854775807;
    CREATE INDEX IF NOT EXISTS item_key ON item(key, valid_from);
    CREATE INDEX IF NOT EXISTS item_kind ON item(kind, valid_to, id);
    CREATE INDEX IF NOT EXISTS item_events ON item(first_event, last_event);

    -- External-content FTS5: the index stores terms, `item` stays the single source of
    -- truth for the text. Without content= the body would be held twice, which on a
    -- store whose whole job is to remember everything is not a detail.
    CREATE VIRTUAL TABLE IF NOT EXISTS item_fts USING fts5(
        title, body, content='item', content_rowid='id', tokenize='porter unicode61'
    );

    CREATE TRIGGER IF NOT EXISTS item_ai AFTER INSERT ON item BEGIN
        INSERT INTO item_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
    END;
    CREATE TRIGGER IF NOT EXISTS item_ad AFTER DELETE ON item BEGIN
        INSERT INTO item_fts(item_fts, rowid, title, body)
            VALUES('delete', old.id, old.title, old.body);
    END;
    CREATE TRIGGER IF NOT EXISTS item_au AFTER UPDATE ON item BEGIN
        INSERT INTO item_fts(item_fts, rowid, title, body)
            VALUES('delete', old.id, old.title, old.body);
        INSERT INTO item_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
    END;

    CREATE TABLE IF NOT EXISTS scratchpad (
        session        TEXT PRIMARY KEY,
        goal           TEXT NOT NULL DEFAULT '',
        active_errors  TEXT NOT NULL DEFAULT '',
        target_files   TEXT NOT NULL DEFAULT '',
        updated_at     INTEGER NOT NULL
    );
)sql";

// Table-qualified throughout. search() joins item against item_fts, which carries its
// own `title` and `body`, and an unqualified list is an "ambiguous column name" error
// there -- one that only appears on the one query that matters most.
constexpr const char* kColumns =
    "item.id, item.session, item.kind, item.key, item.title, item.body, item.hash, "
    "item.valid_from, item.valid_to, item.system_time, item.first_event, item.last_event";

Item read_item(const Stmt& stmt) {
    Item item;
    item.id = stmt.column_int(0);
    item.session = stmt.column_text(1);
    item.kind = stmt.column_text(2);
    item.key = stmt.column_text(3);
    item.title = stmt.column_text(4);
    item.body = stmt.column_text(5);
    item.hash = stmt.column_text(6);
    item.valid_from = stmt.column_int(7);
    item.valid_to = stmt.column_int(8);
    item.system_time = stmt.column_int(9);
    item.first_event = static_cast<std::uint64_t>(stmt.column_int(10));
    item.last_event = static_cast<std::uint64_t>(stmt.column_int(11));
    return item;
}

std::vector<Item> read_all(Stmt& stmt) {
    std::vector<Item> items;
    while (stmt.step()) {
        items.push_back(read_item(stmt));
    }
    return items;
}

// FTS5 treats bare input as a query language: unbalanced quotes are a syntax error, and
// NEAR/AND/OR/NOT are operators. A user asking about `NOT FOUND` would get a parse
// failure at best and a silently inverted query at worst. Quoting every token turns the
// whole input into a phrase-per-term conjunction, which is what a search box means.
std::string escape_fts(std::string_view text) {
    std::string out;
    out.reserve(text.size() + (text.size() >> 2));
    bool in_token = false;
    for (const char c : text) {
        const bool word = (static_cast<unsigned char>(c) > 127) || std::isalnum(c) != 0;
        if (!word) {
            if (in_token) {
                out += "\" ";
                in_token = false;
            }
            continue;
        }
        if (!in_token) {
            out += '"';
            in_token = true;
        }
        out += c;
    }
    if (in_token) {
        out += '"';
    }
    return out;
}

} // namespace

Store::Store(std::string path, LinkPolicy links)
    : db_(path, links), cas_(db_), clock_(&system_clock_us) {
    // WAL so a reader (a recall running mid-turn) never blocks the writer that is
    // journalling the turn. NORMAL rather than FULL because a torn final turn on power
    // loss is recoverable -- the working context still has it -- and FULL costs an fsync
    // per turn on the agent's hot path.
    db_.exec("PRAGMA journal_mode=WAL");
    db_.exec("PRAGMA synchronous=NORMAL");
    db_.exec("PRAGMA foreign_keys=ON");
    db_.exec(kSchema);
    Cas::migrate(db_);
}

std::int64_t Store::append(Record record) {
    const TimeUs recorded = now();
    Stmt stmt(db_, "INSERT INTO item (session, kind, key, title, body, hash, valid_from, "
                   "valid_to, system_time, first_event, last_event) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
    stmt.bind(1, record.session);
    stmt.bind(2, record.kind);
    stmt.bind(3, record.key);
    stmt.bind(4, record.title);
    stmt.bind(5, record.body);
    stmt.bind(6, record.hash);
    stmt.bind(7, resolve(record.valid_from));
    stmt.bind(8, kOpenEnded);
    stmt.bind(9, recorded);
    stmt.bind(10, static_cast<std::int64_t>(record.first_event));
    stmt.bind(11, static_cast<std::int64_t>(record.last_event));
    stmt.run();
    return db_.last_insert_rowid();
}

std::int64_t Store::remember(Record record) {
    if (record.kind.empty()) {
        record.kind = kind::kFact;
    }
    Transaction txn(db_);

    // Re-deriving the same conclusion is the common case, not the exception, and every
    // identical revision costs a row, an index entry and an FTS posting forever.
    Stmt open(db_, "SELECT id, body FROM item WHERE session = ? AND key = ? "
                   "AND valid_to = ?");
    open.bind(1, record.session);
    open.bind(2, record.key);
    open.bind(3, kOpenEnded);
    std::int64_t previous = 0;
    if (open.step()) {
        previous = open.column_int(0);
        if (open.column_text(1) == record.body) {
            txn.commit();
            return previous;
        }
    }

    const TimeUs valid_from = resolve(record.valid_from);
    if (previous != 0) {
        // Closed at exactly the instant its successor becomes valid. Half-open intervals
        // ([from, to)) mean the two never both match a point-in-time query, so there is
        // no instant at which a key appears to hold two values.
        Stmt close(db_, "UPDATE item SET valid_to = ? WHERE id = ?");
        close.bind(1, valid_from);
        close.bind(2, previous);
        close.run();
    }
    record.valid_from = valid_from;
    const std::int64_t id = append(std::move(record));
    txn.commit();
    return id;
}

bool Store::forget(std::string_view key, std::string_view session) {
    Stmt stmt(db_, "UPDATE item SET valid_to = ? WHERE session = ? AND key = ? "
                   "AND valid_to = ?");
    stmt.bind(1, now());
    stmt.bind(2, session);
    stmt.bind(3, key);
    stmt.bind(4, kOpenEnded);
    stmt.run();
    return sqlite3_changes(db_.handle()) > 0;
}

std::int64_t Store::put_artifact(Record record, std::string_view content,
                                 std::string_view base) {
    Transaction txn(db_);
    record.hash = cas_.put(content, base);
    if (record.kind.empty()) {
        record.kind = kind::kArtifact;
    }
    // The body is the SEARCHABLE projection of the artifact, not the artifact. Indexing
    // a megabyte of file content per revision would bloat the FTS index by the size of
    // every revision ever written, which is exactly what the CAS exists to avoid.
    if (record.body.empty()) {
        record.body = record.title + "\n" + record.hash;
    }
    // An artifact revision is a fact about a path: revision N supersedes revision N-1,
    // and `current(path)` should name the newest. Routing through remember() rather than
    // append() is what makes that true without a second code path.
    const std::int64_t id = record.key.empty() ? append(std::move(record))
                                               : remember(std::move(record));
    txn.commit();
    return id;
}

std::optional<Item> Store::get(std::int64_t id) const {
    Stmt stmt(db_, std::string("SELECT ") + kColumns + " FROM item WHERE id = ?");
    stmt.bind(1, id);
    if (!stmt.step()) {
        return std::nullopt;
    }
    return read_item(stmt);
}

std::optional<Item> Store::current(std::string_view key, AsOf as_of,
                                   std::string_view session) const {
    // THE bi-temporal predicate. Valid time is half-open; system time is "as we knew it
    // then", so a row recorded after the system instant is invisible -- which is what
    // makes a replay of an old decision see what the agent actually saw.
    Stmt stmt(db_, std::string("SELECT ") + kColumns +
                       " FROM item WHERE key = ? AND (? = '' OR session = ?) "
                       "AND valid_from <= ? AND valid_to > ? AND system_time <= ? "
                       "ORDER BY system_time DESC, id DESC LIMIT 1");
    stmt.bind(1, key);
    stmt.bind(2, session);
    stmt.bind(3, session);
    stmt.bind(4, resolve(as_of.valid));
    stmt.bind(5, resolve(as_of.valid));
    stmt.bind(6, resolve(as_of.system));
    if (!stmt.step()) {
        return std::nullopt;
    }
    return read_item(stmt);
}

std::vector<Item> Store::history(std::string_view key, std::string_view session) const {
    Stmt stmt(db_, std::string("SELECT ") + kColumns +
                       " FROM item WHERE key = ? AND (? = '' OR session = ?) "
                       "ORDER BY valid_from ASC, id ASC");
    stmt.bind(1, key);
    stmt.bind(2, session);
    stmt.bind(3, session);
    return read_all(stmt);
}

std::vector<Item> Store::by_kind(std::string_view kind_name, AsOf as_of,
                                 std::string_view session, int limit) const {
    Stmt stmt(db_, std::string("SELECT ") + kColumns +
                       " FROM item WHERE kind = ? AND (? = '' OR session = ?) "
                       "AND valid_from <= ? AND valid_to > ? AND system_time <= ? "
                       "ORDER BY id DESC LIMIT ?");
    stmt.bind(1, kind_name);
    stmt.bind(2, session);
    stmt.bind(3, session);
    stmt.bind(4, resolve(as_of.valid));
    stmt.bind(5, resolve(as_of.valid));
    stmt.bind(6, resolve(as_of.system));
    stmt.bind(7, static_cast<std::int64_t>(limit > 0 ? limit : -1));
    return read_all(stmt);
}

std::vector<Item> Store::events_between(std::uint64_t first, std::uint64_t last,
                                        std::string_view session) const {
    // Overlap, not containment: a turn straddling the edge of a compacted span is part
    // of that span's story, and asking for containment would silently drop it.
    //
    // TURNS ONLY, and that is now load-bearing rather than incidental. A compacted span
    // is written with the SAME event range as the turns it summarizes -- that is what
    // makes the range in the prompt resolvable -- so without this predicate rehydrating
    // "events 40-91" hands back the summary the model is already looking at, charged
    // against the budget it was trying to spend on the detail underneath it.
    Stmt stmt(db_, std::string("SELECT ") + kColumns +
                       " FROM item WHERE (? = '' OR session = ?) AND kind = 'turn' "
                       "AND last_event >= ? AND first_event <= ? AND first_event > 0 "
                       "ORDER BY first_event ASC, id ASC");
    stmt.bind(1, session);
    stmt.bind(2, session);
    stmt.bind(3, static_cast<std::int64_t>(first));
    stmt.bind(4, static_cast<std::int64_t>(last));
    return read_all(stmt);
}

std::vector<Item> Store::search(std::string_view text, AsOf as_of,
                                std::string_view session, int limit) const {
    const std::string match = escape_fts(text);
    if (match.empty()) {
        return {};
    }
    // bm25() weights the title above the body: a match on what something IS beats a
    // passing mention inside it. Ranks ascend because FTS5 returns bm25 negated, so the
    // most relevant row is the most negative one.
    Stmt stmt(db_, std::string("SELECT ") + kColumns +
                       " FROM item JOIN item_fts ON item_fts.rowid = item.id "
                       "WHERE item_fts MATCH ? AND (? = '' OR session = ?) "
                       "AND valid_from <= ? AND valid_to > ? AND system_time <= ? "
                       "ORDER BY bm25(item_fts, 4.0, 1.0) ASC LIMIT ?");
    stmt.bind(1, match);
    stmt.bind(2, session);
    stmt.bind(3, session);
    stmt.bind(4, resolve(as_of.valid));
    stmt.bind(5, resolve(as_of.valid));
    stmt.bind(6, resolve(as_of.system));
    stmt.bind(7, static_cast<std::int64_t>(limit > 0 ? limit : -1));
    return read_all(stmt);
}

std::optional<std::string> Store::artifact_content(const Item& item) const {
    if (item.hash.empty()) {
        return std::nullopt;
    }
    return cas_.get(item.hash);
}

StoreStats Store::stats() const {
    StoreStats s;
    Stmt stmt(db_, "SELECT COUNT(*), COALESCE(SUM(valid_to = ?), 0), "
                   "COUNT(DISTINCT session) FROM item");
    stmt.bind(1, kOpenEnded);
    if (stmt.step()) {
        s.items = stmt.column_int(0);
        s.current_items = stmt.column_int(1);
        s.sessions = stmt.column_int(2);
    }
    s.blobs = cas_.stats();
    return s;
}

std::vector<Item> Store::error_signature_search(std::string_view error_text,
                                                      int limit) const {
    if (error_text.empty()) {
        return {};
    }
    return search(error_text, {}, {}, limit);
}

void Store::save_scratchpad(Scratchpad scratchpad) {
    TimeUs ts = scratchpad.updated_at != 0 ? scratchpad.updated_at : now();
    Stmt stmt(db_, "INSERT INTO scratchpad(session, goal, active_errors, target_files, updated_at) "
                   "VALUES(?, ?, ?, ?, ?) "
                   "ON CONFLICT(session) DO UPDATE SET "
                   "goal = excluded.goal, active_errors = excluded.active_errors, "
                   "target_files = excluded.target_files, updated_at = excluded.updated_at");
    stmt.bind(1, scratchpad.session);
    stmt.bind(2, scratchpad.goal);
    stmt.bind(3, scratchpad.active_errors);
    stmt.bind(4, scratchpad.target_files);
    stmt.bind(5, ts);
    stmt.run();
}

std::optional<Scratchpad> Store::get_scratchpad(std::string_view session) const {
    Stmt stmt(db_, "SELECT session, goal, active_errors, target_files, updated_at "
                   "FROM scratchpad WHERE session = ?");
    stmt.bind(1, session);
    if (!stmt.step()) {
        return std::nullopt;
    }
    Scratchpad pad;
    pad.session = stmt.column_text(0);
    pad.goal = stmt.column_text(1);
    pad.active_errors = stmt.column_text(2);
    pad.target_files = stmt.column_text(3);
    pad.updated_at = stmt.column_int(4);
    return pad;
}

} // namespace lmp::pcc
