#pragma once

// parsephony — a JSON parser for LLM tool-call payloads.
//
// Core representation is a flat *tape* of 12-byte nodes rather than a pointer-linked
// DOM. Containers carry a skip pointer to the node just past their subtree, so
// stepping to the next sibling is O(1) instead of a walk through everything in
// between. Scalars are stored as (offset, length) into the source buffer and are
// only converted when something actually asks for the value.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string_view>
#include <string>
#include <vector>
#include <optional>
#include <utility>

namespace parsephony {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

enum class Error : uint8_t {
    Ok = 0,
    Empty,              // no input
    UnexpectedChar,
    UnexpectedEnd,      // input ran out mid-value
    TrailingContent,    // valid document followed by junk
    DepthExceeded,
    BadNumber,
    BadEscape,
    BadUnicode,         // malformed \u, or an unpaired surrogate
    BadUtf8,            // invalid UTF-8 in the raw source
    ControlChar,        // raw byte < 0x20 inside a string
};

const char* error_message(Error e) noexcept;

// ---------------------------------------------------------------------------
// Tape
// ---------------------------------------------------------------------------

enum class Type : uint8_t {
    Null = 0,
    False,
    True,
    Number,
    String,
    Array,
    Object,
    Key,      // object member name; always immediately followed by its value
};

// 12 bytes. Two meanings, depending on whether this is a scalar or a container:
//
//   scalar     : off = byte offset into source (or into the decode buffer if
//                kEscaped), len = byte length
//   container  : off = tape index one past the end of this subtree (skip pointer),
//                len = number of elements (array) or members (object)
struct Node {
    uint32_t off;
    uint32_t len;
    Type     type;
    uint8_t  flags;
    uint16_t _pad;

    static constexpr uint8_t kEscaped = 1u << 0;  // string lives in the decode buffer
    static constexpr uint8_t kFloat   = 1u << 1;  // number has '.' or an exponent

    bool escaped() const noexcept { return (flags & kEscaped) != 0; }
    bool is_float() const noexcept { return (flags & kFloat) != 0; }
};

static_assert(sizeof(Node) == 12, "tape node should stay compact");

class Document;

// A lightweight cursor into the tape. Copyable, non-owning.
class Value {
public:
    Value() noexcept = default;
    Value(const Document* doc, uint32_t idx) noexcept : doc_(doc), idx_(idx) {}

    bool valid() const noexcept { return doc_ != nullptr; }
    explicit operator bool() const noexcept { return doc_ != nullptr; }

    Type type() const noexcept;

    bool is_null()   const noexcept { return valid() && type() == Type::Null; }
    bool is_bool()   const noexcept { if (!valid()) return false; Type t = type(); return t == Type::True || t == Type::False; }
    bool is_number() const noexcept { return valid() && type() == Type::Number; }
    bool is_string() const noexcept { return valid() && type() == Type::String; }
    bool is_array()  const noexcept { return valid() && type() == Type::Array; }
    bool is_object() const noexcept { return valid() && type() == Type::Object; }

    bool get_bool() const noexcept { return valid() && type() == Type::True; }

    // Zero-copy view into the source when the string had no escapes (the common
    // case for tool-call payloads); otherwise a view into the document's decode
    // buffer, unescaped once at parse time.
    std::string_view get_string() const noexcept;

    double get_double() const noexcept;
    std::optional<int64_t> get_int() const noexcept;

    // Raw, undecoded source text of this value.
    std::string_view raw() const noexcept;

    size_t size() const noexcept;                 // array/object element count

    // O(1) per step — follows skip pointers rather than walking subtrees.
    Value operator[](size_t i) const noexcept;            // array index
    Value operator[](std::string_view key) const noexcept; // object lookup
    Value find(std::string_view key) const noexcept { return (*this)[key]; }

    // Dotted path, e.g. doc.root().at("tool_calls.0.function.name")
    Value at(std::string_view path) const noexcept;

    // Iteration over an array's elements, or an object's values.
    class iterator {
    public:
        iterator(const Document* d, uint32_t i) noexcept : doc_(d), idx_(i) {}
        Value operator*() const noexcept { return Value(doc_, idx_); }
        iterator& operator++() noexcept;
        bool operator!=(const iterator& o) const noexcept { return idx_ != o.idx_; }
        bool operator==(const iterator& o) const noexcept { return idx_ == o.idx_; }
    private:
        const Document* doc_;
        uint32_t idx_;
    };

    iterator begin() const noexcept;
    iterator end() const noexcept;

    // Object member (key, value) iteration.
    class member_iterator {
    public:
        member_iterator(const Document* d, uint32_t i) noexcept : doc_(d), idx_(i) {}
        std::pair<std::string_view, Value> operator*() const noexcept;
        member_iterator& operator++() noexcept;
        bool operator!=(const member_iterator& o) const noexcept { return idx_ != o.idx_; }
    private:
        const Document* doc_;
        uint32_t idx_;
    };

    struct members_range {
        const Document* doc; uint32_t first; uint32_t last;
        member_iterator begin() const noexcept { return member_iterator(doc, first); }
        member_iterator end()   const noexcept { return member_iterator(doc, last); }
    };
    members_range members() const noexcept;

private:
    friend class Document;
    const Document* doc_ = nullptr;
    uint32_t idx_ = 0;
};

// ---------------------------------------------------------------------------
// Document — owns the tape and the decode buffer; borrows the source.
// ---------------------------------------------------------------------------

class Document {
public:
    Document() = default;

    Document(Document&&) noexcept = default;
    Document& operator=(Document&&) noexcept = default;
    Document(const Document&) = delete;
    Document& operator=(const Document&) = delete;

    // Invalid after a failed parse: the tape stays physically allocated, so the
    // logical count is what decides whether there is a root at all.
    Value root() const noexcept { return tape_count_ == 0 ? Value() : Value(this, 0); }

    const Node& node(uint32_t i) const noexcept { return tape_[i]; }
    size_t tape_size() const noexcept { return tape_count_; }
    std::string_view source() const noexcept { return src_; }

    // Reuse across parses: the tape stays allocated at its high-water mark and
    // only the logical count resets, so a stream of tool calls does no
    // per-payload allocation at all.
    void clear() noexcept { tape_count_ = 0; decoded_.clear(); }

private:
    friend class Parser;
    friend class Value;

    // Physically sized to the high-water mark; tape_count_ is the logical size.
    std::vector<Node> tape_;
    uint32_t tape_count_ = 0;
    std::string decoded_;      // unescaped string bodies, appended at parse time
    std::string_view src_;
};

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Options {
    uint32_t max_depth = 512;
    bool validate_utf8 = true;   // reject malformed UTF-8 in the source
};

class Parser {
public:
    Parser() = default;
    explicit Parser(Options o) noexcept : opts_(o) {}

    // `json` must outlive `doc` — unescaped strings are views into it.
    Error parse(std::string_view json, Document& doc);

    // Byte offset where parsing stopped; useful for diagnostics.
    size_t error_offset() const noexcept { return err_off_; }

private:
    struct Frame {
        uint32_t node;    // tape index of the open container
        uint32_t count;   // elements/members closed so far
        bool is_object;
    };

    Options opts_;
    size_t err_off_ = 0;
    // Held across calls so a reused Parser does no per-parse allocation.
    std::vector<Frame> stack_;
};

// Convenience: parse with default options.
inline Error parse(std::string_view json, Document& doc) {
    Parser p;
    return p.parse(json, doc);
}

} // namespace parsephony
