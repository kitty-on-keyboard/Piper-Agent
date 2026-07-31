#pragma once

// Streaming / incremental parsing, and the grammar-as-constraint machinery.
//
// This is the half that matters for an agent loop: tokens arrive one at a time
// from the model, and we want (a) fields as early as possible, and (b) the
// ability to stop the model from emitting invalid JSON in the first place.
//
// Entry #9 built a resumable SAX parser but no constraint side. Entry #11 named
// the grammar/parser "duality" but only ever validated — it never produced a
// mask a sampler could use. Both halves live here, driven by one automaton, so
// the thing that validates output is by construction the same thing that
// constrains generation.

#include "parsephony/parsephony.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace parsephony {

// How constrained the automaton's current position is, from the point of view
// of token-mask construction. See mask.hpp for how each class is exploited.
enum class MaskClass : uint8_t {
    Other,       // narrow grammar position: gate on first byte, then simulate
    JsonString,  // inside a JSON string body: almost everything is legal
    FreeText,    // raw text region: everything non-special is legal
};

// ---------------------------------------------------------------------------
// SAX events
// ---------------------------------------------------------------------------

struct Handler {
    virtual ~Handler() = default;
    virtual void on_null() {}
    virtual void on_bool(bool) {}
    virtual void on_number(std::string_view /*raw*/) {}
    virtual void on_string(std::string_view) {}
    virtual void on_key(std::string_view) {}
    virtual void on_object_begin() {}
    virtual void on_object_end() {}
    virtual void on_array_begin() {}
    virtual void on_array_end() {}
};

// ---------------------------------------------------------------------------
// Legal-next-byte set
// ---------------------------------------------------------------------------

// Which bytes the grammar will accept as the very next byte. Exact, not an
// approximation — it is read straight off the automaton's current state.
struct ByteSet {
    uint64_t lo = 0;   // bytes 0..63
    uint64_t hi = 0;   // bytes 64..127
    bool non_ascii = false;  // any byte >= 0x80 (inside a string body)

    void add(unsigned char c) noexcept {
        if (c < 64) lo |= (1ull << c);
        else if (c < 128) hi |= (1ull << (c - 64));
    }
    void add_range(unsigned char a, unsigned char b) noexcept {
        for (unsigned c = a; c <= b; ++c) add(static_cast<unsigned char>(c));
    }
    bool contains(unsigned char c) const noexcept {
        if (c >= 128) return non_ascii;
        return c < 64 ? (lo >> c) & 1 : (hi >> (c - 64)) & 1;
    }
    bool empty() const noexcept { return lo == 0 && hi == 0 && !non_ascii; }
};

// ---------------------------------------------------------------------------
// StreamParser
// ---------------------------------------------------------------------------

class StreamParser {
public:
    explicit StreamParser(Handler* h = nullptr, Options o = {});

    // Feed any number of bytes; chunk boundaries may fall anywhere, including
    // the middle of an escape sequence or a surrogate pair.
    Error feed(std::string_view bytes);

    // Call once the model has stopped emitting.
    Error finish();

    // True once a complete, well-formed document has been seen.
    bool complete() const noexcept { return done_; }

    // Nesting depth right now (0 at top level).
    size_t depth() const noexcept { return stack_.size(); }

    void reset();

    // --- constraint side ---------------------------------------------------

    // Bytes the grammar accepts next, given everything fed so far.
    ByteSet allowed_bytes() const;

    // Would feeding this byte keep the document valid?
    bool accepts(char c) const { return allowed_bytes().contains(static_cast<unsigned char>(c)); }

    // A compact key for the automaton's current configuration. Two states with
    // the same signature accept exactly the same continuations, which is what
    // makes token-mask caching viable.
    uint64_t state_signature() const noexcept;

    // --- probing interface for TokenMaskT -----------------------------------

    // Advance one byte. Identical to feed() of a single byte; named separately
    // so the mask engine's requirements read clearly.
    Error probe_byte(unsigned char c) { return push_byte(c); }

    // Silence callbacks on a copy used for probing.
    void mute() noexcept { h_ = nullptr; }

    // Mask-construction class of the current position.
    MaskClass mask_class() const noexcept {
        if (sub_ == Sub::String && !expect_low_escape_) return MaskClass::JsonString;
        return MaskClass::Other;
    }

private:
    enum class St : uint8_t {
        Value,           // expecting a value
        ObjKeyOrEnd,     // just after '{'
        ObjKey,          // after ',' inside an object
        ObjColon,
        ObjCommaOrEnd,
        ArrValueOrEnd,   // just after '['
        ArrCommaOrEnd,
        Done,
    };

    // Sub-machine for scalars that can straddle a chunk boundary.
    enum class Sub : uint8_t {
        None,
        String,
        StringEscape,
        StringUnicode,   // collecting 4 hex digits
        Number,
        Literal,         // true / false / null
    };

    Error push_byte(unsigned char c);
    Error begin_value(unsigned char c);
    Error end_value();          // a complete value was just produced
    Error close_container(bool is_object);
    Error flush_number();
    void emit_string_or_key();

    Handler* h_;
    Options opts_;

    St  st_ = St::Value;
    Sub sub_ = Sub::None;
    std::vector<uint8_t> stack_;   // 1 = object, 0 = array
    bool done_ = false;
    bool key_pending_ = false;     // the string being read is an object key

    std::string buf_;              // current string / number / literal text
    std::string_view literal_;     // which literal we are matching
    size_t literal_idx_ = 0;

    uint32_t uni_ = 0;             // \u accumulator
    int uni_digits_ = 0;
    uint32_t high_surrogate_ = 0;  // pending high surrogate, 0 if none
    bool expect_low_escape_ = false; // saw a high surrogate, need "\u" next
};

} // namespace parsephony
