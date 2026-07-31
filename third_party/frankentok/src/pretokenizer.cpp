#include "mlx_qwen_tokenizer/pretokenizer.h"
#include <stdexcept>
#include <iostream>

namespace mlx_qwen_tokenizer {

namespace {

// Offset of the first byte that does not begin a well-formed UTF-8 sequence, or npos if the whole
// buffer is well formed.
//
// This exists so the match loop can pass PCRE2_NO_UTF_CHECK. PCRE2 re-validates the *entire*
// subject on every pcre2_match call regardless of start_offset, so scanning an N-byte string with
// N successive matches costs O(N^2) in validation alone. Hoisting the check out here makes the
// scan linear; the flag is only sound because this ran first and said yes.
//
// Returning the offset rather than a bool is what lets a malformed buffer be cut into well-formed
// runs instead of being handed to PCRE2 unchecked — see split().
size_t first_invalid_utf8(std::string_view s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len;
        uint32_t cp;
        if (c <= 0x7F) { i++; continue; }
        else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1Fu; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0Fu; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07u; }
        else return i;

        if (i + len > n) return i;
        for (size_t j = 1; j < len; ++j) {
            const unsigned char cc = static_cast<unsigned char>(s[i + j]);
            if ((cc & 0xC0) != 0x80) return i;
            cp = (cp << 6) | (cc & 0x3Fu);
        }
        // Overlong encodings, surrogates and out-of-range values are all things PCRE2's own check
        // would reject; skipping them here would let malformed input past NO_UTF_CHECK.
        if (len == 2 && cp < 0x80) return i;
        if (len == 3 && cp < 0x800) return i;
        if (len == 4 && cp < 0x10000) return i;
        if (cp > 0x10FFFF) return i;
        if (cp >= 0xD800 && cp <= 0xDFFF) return i;
        i += len;
    }
    return std::string_view::npos;
}

// Runs the pattern across one well-formed run, appending each match.
void match_run(const pcre2_code* re, pcre2_match_data* match_data, std::string_view run,
               std::vector<std::string_view>& result) {
    PCRE2_SPTR subject = (PCRE2_SPTR)run.data();
    const PCRE2_SIZE subject_length = run.length();
    PCRE2_SIZE start_offset = 0;

    while (start_offset < subject_length) {
        // Sound because the caller verified this run is well formed.
        const int rc = pcre2_match(re, subject, subject_length, start_offset,
                                   PCRE2_NO_UTF_CHECK, match_data, NULL);

        if (rc < 0) {
            if (rc == PCRE2_ERROR_NOMATCH) {
                // Unreachable for well-formed input: the alternatives cover letters, digits,
                // whitespace and everything else exhaustively.
                result.push_back(run.substr(start_offset));
                break;
            }
            // A resource limit (JIT stack, backtracking) rather than bad input. Give up on this
            // one byte and carry on, instead of swallowing the rest of the run as a single piece.
            result.push_back(run.substr(start_offset, 1));
            start_offset += 1;
            continue;
        }

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);

        // Push the matched part
        if (ovector[1] > ovector[0]) {
            result.push_back(run.substr(ovector[0], ovector[1] - ovector[0]));
        }

        // A zero-width match leaves start_offset unmoved and spins forever; step past it.
        start_offset = (ovector[1] > ovector[0]) ? ovector[1] : ovector[1] + 1;
    }
}

} // namespace

Pretokenizer::Pretokenizer()
    : Pretokenizer("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {}

Pretokenizer::Pretokenizer(const std::string& pattern) : re_(nullptr) {
    int errornumber;
    PCRE2_SIZE erroroffset;

    // Compile with UCP and UTF to support \p{L} properly and unicode.
    re_ = pcre2_compile(
        (PCRE2_SPTR)pattern.c_str(),
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &errornumber,
        &erroroffset,
        NULL
    );

    if (re_ == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errornumber, buffer, sizeof(buffer));
        std::cerr << "PCRE2 compilation failed at offset " << erroroffset << ": " << buffer << std::endl;
        throw std::runtime_error("Failed to compile PCRE2 regex");
    }

    // The build turns on PCRE2_SUPPORT_JIT, which is what gives this call any effect — matching is
    // the largest phase of encode(), and the JIT roughly halves its cost. Still best-effort: a
    // PCRE2 built without JIT matches correctly, just slower, so a failure here is not fatal.
    pcre2_jit_compile(re_, PCRE2_JIT_COMPLETE);
}

Pretokenizer::~Pretokenizer() {
    if (re_) {
        pcre2_code_free(re_);
    }
}

void Pretokenizer::split(std::string_view text, std::vector<std::string_view>& result) const {
    result.clear();
    if (text.empty()) return;

    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re_, NULL);

    // Malformed input is cut into maximal well-formed runs, matched one run at a time, with each
    // offending byte emitted as its own pre-token in between.
    //
    // Handing PCRE2 the whole subject instead makes it fail on the first match call, and the only
    // recovery available at that point is to take everything remaining as one piece — so a single
    // corrupt byte anywhere rewrites the segmentation of the entire rest of the document. Cutting
    // at the bad byte confines the damage to that byte, and lets every run keep NO_UTF_CHECK,
    // which is what keeps malformed input linear rather than quadratic.
    //
    // The stray byte still round-trips: BPE is byte-level, so it encodes and decodes as itself.
    size_t base = 0;
    while (base < text.size()) {
        const std::string_view rest = text.substr(base);
        const size_t bad = first_invalid_utf8(rest);
        const std::string_view run = (bad == std::string_view::npos) ? rest : rest.substr(0, bad);

        if (!run.empty()) match_run(re_, match_data, run, result);
        if (bad == std::string_view::npos) break;

        result.push_back(rest.substr(bad, 1));
        base += bad + 1;
    }

    pcre2_match_data_free(match_data);
}

std::vector<std::string> Pretokenizer::split(std::string_view text) const {
    std::vector<std::string_view> views;
    split(text, views);
    return std::vector<std::string>(views.begin(), views.end());
}

} // namespace mlx_qwen_tokenizer
