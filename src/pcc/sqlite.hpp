#pragma once
//
// The smallest RAII layer over SQLite that makes the rest of src/pcc readable.
//
// Not a query builder and not an ORM. The queries in this component are hand-written
// because their shape IS the design -- the bi-temporal predicate in store.cpp is the
// component's whole contract, and hiding it behind a fluent API would make the one
// thing worth reviewing the one thing you cannot see.
//
// What this does buy: sqlite3_finalize on every path. The cook-off entrants this
// component learned from leaked a statement on every early return, and in C++ that is
// not a style problem -- an unfinalized statement holds a read transaction open and the
// next writer gets SQLITE_BUSY with no indication why.
//
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sqlite3.h>

namespace lmp::pcc {

// Thrown for programming errors against the database: a malformed statement, a schema
// that does not match the query, a closed handle. NOT thrown for "no rows" or for a
// missing artifact, which are ordinary results and are typed as such.
class SqlError : public std::runtime_error {
  public:
    explicit SqlError(const std::string& what) : std::runtime_error(what) {}
};

class Stmt;

// An open database handle. Move-only: two owners closing one handle is a double free,
// and sqlite3_close on a handle with live statements silently leaks instead of failing.
class Db {
  public:
    Db() = default;
    explicit Db(const std::string& path);
    ~Db();

    Db(const Db&) = delete;
    Db& operator=(const Db&) = delete;
    Db(Db&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Db& operator=(Db&& other) noexcept;

    // Statements with no results and no parameters: DDL, PRAGMA, BEGIN/COMMIT.
    void exec(std::string_view sql);

    [[nodiscard]] sqlite3* handle() const noexcept { return handle_; }
    [[nodiscard]] std::int64_t last_insert_rowid() const noexcept {
        return sqlite3_last_insert_rowid(handle_);
    }

  private:
    friend class Transaction;
    sqlite3* handle_ = nullptr;
    // Nesting depth, so Transaction can name its savepoint. See the comment there.
    int txn_depth_ = 0;
};

// A prepared statement, finalized by the destructor.
//
// bind() is 1-indexed to match SQLite; column() is 0-indexed to match SQLite. That
// inconsistency is SQLite's and is deliberately not papered over, because a wrapper that
// renumbers is a wrapper you have to remember is renumbering.
class Stmt {
  public:
    Stmt(const Db& db, std::string_view sql);
    ~Stmt();

    Stmt(const Stmt&) = delete;
    Stmt& operator=(const Stmt&) = delete;
    Stmt(Stmt&& other) noexcept : stmt_(std::exchange(other.stmt_, nullptr)) {}
    Stmt& operator=(Stmt&& other) noexcept;

    void bind(int index, std::int64_t value);
    void bind(int index, double value);
    void bind(int index, std::string_view value);
    void bind_blob(int index, std::span<const unsigned char> value);
    void bind_null(int index);

    // True when a row is available; false at the end of the result set.
    [[nodiscard]] bool step();
    // For statements run for their effect. Throws if the statement yields a row, which
    // means the caller used the wrong one of the two.
    void run();
    void reset();

    [[nodiscard]] std::int64_t column_int(int index) const;
    [[nodiscard]] double column_double(int index) const;
    [[nodiscard]] std::string column_text(int index) const;
    [[nodiscard]] std::vector<unsigned char> column_blob(int index) const;
    [[nodiscard]] bool column_is_null(int index) const;

  private:
    void check(int rc, const char* what) const;
    sqlite3_stmt* stmt_ = nullptr;
};

// A transaction that rolls back unless commit() is called.
//
// The important half is the destructor. Every write path in store.cpp can throw --
// SqlError, bad_alloc, a std::string operation -- and without this an exception midway
// through supersede() would leave a fact both closed and un-replaced, which is the one
// state the bi-temporal model says cannot exist.
//
// NESTABLE, via SAVEPOINT rather than BEGIN. put_artifact() opens one and then calls
// remember(), which opens another; a second BEGIN is "cannot start a transaction within
// a transaction" and takes down the process. The alternative -- a transaction-free
// variant of every writer, to be called only from inside another writer -- doubles the
// write surface and gets the wrong one called eventually.
class Transaction {
  public:
    explicit Transaction(Db& db);
    ~Transaction();

    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;

    void commit();

  private:
    Db* db_;
    std::string name_;
    bool done_ = false;
};

// Convenience for the many single-value lookups: returns nullopt when no row matched.
[[nodiscard]] std::optional<std::int64_t> query_int(const Db& db, std::string_view sql,
                                                    std::string_view arg);

} // namespace lmp::pcc
