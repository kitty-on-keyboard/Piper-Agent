#pragma once

// SWAR (SIMD-within-a-register) scanning helpers.
//
// No cookoff entry did any bulk scanning — all eleven walked strings one byte at
// a time. These let the string scanner clear 8 bytes per iteration in the common
// case (plain ASCII, no quote, no backslash, no control byte), which is nearly
// every string in a tool-call payload.

#include <cstdint>
#include <cstring>

namespace parsephony::swar {

inline uint64_t load64(const char* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

constexpr uint64_t kLo = 0x0101010101010101ull;
constexpr uint64_t kHi = 0x8080808080808080ull;

// Nonzero iff any byte of x is zero. Classic Mycroft test.
inline uint64_t has_zero(uint64_t x) noexcept {
    return (x - kLo) & ~x & kHi;
}

// Nonzero iff any byte of x equals n.
inline uint64_t has_byte(uint64_t x, uint8_t n) noexcept {
    return has_zero(x ^ (kLo * n));
}

// Nonzero iff any byte of x is < n. Only valid for n <= 0x80.
inline uint64_t has_less(uint64_t x, uint8_t n) noexcept {
    return (x - kLo * n) & ~x & kHi;
}

// Index of the first byte whose marker bit is set, given a mask produced by one
// of the tests above. Assumes little-endian, which every platform we target is.
inline unsigned first_marked(uint64_t mask) noexcept {
    return static_cast<unsigned>(__builtin_ctzll(mask) >> 3);
}

// Byte classification table. Short strings — which is most object keys — never
// reach the 8-byte SWAR path, so the tail loop wants to be one load and one
// test per byte rather than four comparisons.
struct CharTable {
    uint8_t v[256];
};

inline constexpr CharTable make_string_table() {
    CharTable t{};
    for (int i = 0; i < 256; ++i) {
        t.v[i] = (i == '"' || i == '\\' || i < 0x20 || i >= 0x80) ? 1 : 0;
    }
    return t;
}

inline constexpr CharTable kStringSpecial = make_string_table();

inline bool string_special(unsigned char c) noexcept {
    return kStringSpecial.v[c] != 0;
}

// Scan a string body for the first byte that needs attention:
//   '"'  (0x22) end of string
//   '\\' (0x5C) escape sequence
//   any byte < 0x20 (illegal raw control character)
//   any byte >= 0x80 (start of a multi-byte UTF-8 sequence)
//
// Returns the offset of that byte relative to `p`, or `len` if the whole range
// is ordinary ASCII text.
inline size_t scan_string_body(const char* p, size_t len) noexcept {
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t v = load64(p + i);
        uint64_t interesting =
              has_byte(v, '"')
            | has_byte(v, '\\')
            | has_less(v, 0x20)
            | (v & kHi);            // high bit set → non-ASCII
        if (interesting) {
            return i + first_marked(interesting);
        }
        i += 8;
    }
    for (; i < len; ++i) {
        if (string_special(static_cast<unsigned char>(p[i]))) return i;
    }
    return len;
}

// Skip ASCII whitespace (space, tab, CR, LF). JSON permits nothing else.
inline size_t skip_whitespace(const char* p, size_t len) noexcept {
    size_t i = 0;
    while (i < len) {
        char c = p[i];
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { ++i; continue; }
        break;
    }
    return i;
}

} // namespace parsephony::swar
