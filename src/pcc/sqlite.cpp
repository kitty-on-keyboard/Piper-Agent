#include "src/pcc/sqlite.hpp"

namespace lmp::pcc {
namespace {

std::string describe(sqlite3* db, std::string_view what) {
    std::string msg(what);
    const char* detail = db != nullptr ? sqlite3_errmsg(db) : nullptr;
    if (detail != nullptr) {
        msg += ": ";
        msg += detail;
    }
    return msg;
}

} // namespace

// --- Db ---------------------------------------------------------------------

Db::Db(const std::string& path) {
    const int rc = sqlite3_open_v2(path.c_str(), &handle_,
                                   SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (rc != SQLITE_OK) {
        // sqlite3_open_v2 hands back a handle even on failure, precisely so the error
        // message can be read off it. Closing it here is not optional.
        const std::string msg = describe(handle_, "cannot open " + path);
        sqlite3_close(handle_);
        handle_ = nullptr;
        throw SqlError(msg);
    }
}

Db::~Db() {
    if (handle_ != nullptr) {
        sqlite3_close(handle_);
    }
}

Db& Db::operator=(Db&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            sqlite3_close(handle_);
        }
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

void Db::exec(std::string_view sql) {
    char* err = nullptr;
    const int rc = sqlite3_exec(handle_, std::string(sql).c_str(), nullptr, nullptr, &err);
    if (rc == SQLITE_OK) {
        return;
    }
    std::string msg = "exec failed";
    if (err != nullptr) {
        msg += ": ";
        msg += err;
        sqlite3_free(err);
    }
    throw SqlError(msg);
}

// --- Stmt -------------------------------------------------------------------

Stmt::Stmt(const Db& db, std::string_view sql) {
    const int rc = sqlite3_prepare_v2(db.handle(), sql.data(),
                                      static_cast<int>(sql.size()), &stmt_, nullptr);
    if (rc != SQLITE_OK) {
        throw SqlError(describe(db.handle(), "prepare failed"));
    }
}

Stmt::~Stmt() {
    if (stmt_ != nullptr) {
        sqlite3_finalize(stmt_);
    }
}

Stmt& Stmt::operator=(Stmt&& other) noexcept {
    if (this != &other) {
        if (stmt_ != nullptr) {
            sqlite3_finalize(stmt_);
        }
        stmt_ = std::exchange(other.stmt_, nullptr);
    }
    return *this;
}

void Stmt::check(int rc, const char* what) const {
    if (rc != SQLITE_OK) {
        throw SqlError(describe(sqlite3_db_handle(stmt_), what));
    }
}

void Stmt::bind(int index, std::int64_t value) {
    check(sqlite3_bind_int64(stmt_, index, value), "bind int");
}

void Stmt::bind(int index, double value) {
    check(sqlite3_bind_double(stmt_, index, value), "bind double");
}

void Stmt::bind(int index, std::string_view value) {
    // A default-constructed string_view has a NULL data pointer, and sqlite3_bind_text
    // with NULL binds SQL NULL rather than ''. Every "match all sessions" predicate in
    // store.cpp is written `(? = '' OR session = ?)`, and NULL = '' is NULL, which is
    // not true -- so an omitted session silently matched NOTHING instead of everything.
    // Caught by test_pcc_store; worth the two lines to make unrepresentable.
    static constexpr char kEmpty[] = "";
    const char* data = value.empty() ? kEmpty : value.data();
    // SQLITE_TRANSIENT: SQLite copies. The alternative saves a copy and costs a
    // dangling pointer the first time a caller binds a temporary, which every caller
    // here does.
    check(sqlite3_bind_text(stmt_, index, data, static_cast<int>(value.size()),
                            SQLITE_TRANSIENT),
          "bind text");
}

void Stmt::bind_blob(int index, std::span<const unsigned char> value) {
    check(sqlite3_bind_blob(stmt_, index, value.data(), static_cast<int>(value.size()),
                            SQLITE_TRANSIENT),
          "bind blob");
}

void Stmt::bind_null(int index) {
    check(sqlite3_bind_null(stmt_, index), "bind null");
}

bool Stmt::step() {
    const int rc = sqlite3_step(stmt_);
    if (rc == SQLITE_ROW) {
        return true;
    }
    if (rc == SQLITE_DONE) {
        return false;
    }
    throw SqlError(describe(sqlite3_db_handle(stmt_), "step failed"));
}

void Stmt::run() {
    if (step()) {
        throw SqlError("statement returned a row; use step() instead of run()");
    }
}

void Stmt::reset() {
    sqlite3_reset(stmt_);
    sqlite3_clear_bindings(stmt_);
}

std::int64_t Stmt::column_int(int index) const {
    return sqlite3_column_int64(stmt_, index);
}

double Stmt::column_double(int index) const {
    return sqlite3_column_double(stmt_, index);
}

std::string Stmt::column_text(int index) const {
    const auto* text = sqlite3_column_text(stmt_, index);
    if (text == nullptr) {
        return {};
    }
    // Sized rather than NUL-terminated: a stored body may legitimately contain an
    // embedded NUL, and strlen would truncate it silently.
    return {reinterpret_cast<const char*>(text),
            static_cast<std::size_t>(sqlite3_column_bytes(stmt_, index))};
}

std::vector<unsigned char> Stmt::column_blob(int index) const {
    const auto* data = static_cast<const unsigned char*>(sqlite3_column_blob(stmt_, index));
    const int size = sqlite3_column_bytes(stmt_, index);
    if (data == nullptr || size <= 0) {
        return {};
    }
    return {data, data + size};
}

bool Stmt::column_is_null(int index) const {
    return sqlite3_column_type(stmt_, index) == SQLITE_NULL;
}

// --- Transaction ------------------------------------------------------------

Transaction::Transaction(Db& db) : db_(&db) {
    name_ = "pcc_txn_" + std::to_string(db_->txn_depth_++);
    db_->exec("SAVEPOINT " + name_);
}

Transaction::~Transaction() {
    --db_->txn_depth_;
    if (done_) {
        return;
    }
    // A throwing destructor during unwinding terminates, and this destructor runs
    // exactly when something is already going wrong. The rollback is best-effort and
    // its failure is swallowed on purpose.
    //
    // ROLLBACK TO undoes the work but leaves the savepoint on the stack, so the RELEASE
    // is not optional -- without it a rolled-back inner transaction pins the outer one
    // open and the next writer deadlocks against a transaction nobody can see.
    char* err = nullptr;
    const std::string sql = "ROLLBACK TO " + name_ + "; RELEASE " + name_;
    sqlite3_exec(db_->handle(), sql.c_str(), nullptr, nullptr, &err);
    if (err != nullptr) {
        sqlite3_free(err);
    }
}

void Transaction::commit() {
    // RELEASE on the outermost savepoint is what actually commits; on an inner one it
    // merely merges into the enclosing savepoint, which is the nesting semantic we want.
    db_->exec("RELEASE " + name_);
    done_ = true;
}

// --- helpers ----------------------------------------------------------------

std::optional<std::int64_t> query_int(const Db& db, std::string_view sql,
                                      std::string_view arg) {
    Stmt stmt(db, sql);
    stmt.bind(1, arg);
    if (!stmt.step()) {
        return std::nullopt;
    }
    return stmt.column_int(0);
}

} // namespace lmp::pcc
