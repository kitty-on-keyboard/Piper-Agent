#include "mlx_qwen_tokenizer/normalizer.h"

#include <cstdint>
#include <vector>

#include <utf8proc.h>

namespace mlx_qwen_tokenizer {

namespace {

// True if this codepoint can participate in an NFC change. Everything else — ASCII, CJK, emoji,
// precomposed Latin without following marks — is NFC-invariant and can pass through untouched.
//
//  - a canonical decomposition means the codepoint itself may be rewritten (singletons like
//    U+212B, composition exclusions like U+0958);
//  - nonzero combining class means it is a mark that may compose with, or be reordered after,
//    a preceding base;
//  - Hangul jamo compose algorithmically and carry no decomposition entry, so their ranges are
//    listed explicitly.
bool nfc_can_change(utf8proc_int32_t cp) {
    const utf8proc_property_t* p = utf8proc_get_property(cp);
    if (p->combining_class != 0) return true;
    if (p->decomp_type == 0 && p->decomp_seqindex != UINT16_MAX) return true;   // canonical decomp
    if (cp >= 0x1100 && cp <= 0x11FF) return true;                              // jamo L/V/T
    if (cp >= 0xA960 && cp <= 0xA97F) return true;                              // jamo ext-A
    if (cp >= 0xD7B0 && cp <= 0xD7FF) return true;                              // jamo ext-B
    return false;
}

// NFC-transforms `run` and appends the result to `out`. Falls back to appending the raw bytes if
// utf8proc rejects the run.
void transform_append(std::string_view run, std::string& out) {
    static thread_local std::vector<utf8proc_int32_t> codepoints;
    const auto options = static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE);

    const utf8proc_ssize_t needed = utf8proc_decompose(
        reinterpret_cast<const utf8proc_uint8_t*>(run.data()),
        static_cast<utf8proc_ssize_t>(run.size()), nullptr, 0, options);
    if (needed < 0) { out.append(run); return; }

    codepoints.resize(static_cast<size_t>(needed) + 1);
    const utf8proc_ssize_t written = utf8proc_decompose(
        reinterpret_cast<const utf8proc_uint8_t*>(run.data()),
        static_cast<utf8proc_ssize_t>(run.size()),
        codepoints.data(), needed, options);
    if (written < 0) { out.append(run); return; }

    // Compose and re-encode in place; `written` int32 slots provide 4 bytes per codepoint, the
    // UTF-8 worst case.
    const utf8proc_ssize_t bytes = utf8proc_reencode(codepoints.data(), written, options);
    if (bytes < 0) { out.append(run); return; }

    out.append(reinterpret_cast<const char*>(codepoints.data()), static_cast<size_t>(bytes));
}

} // namespace

bool normalize_nfc(std::string_view text, std::string& out) {
    const size_t n = text.size();

    // Single pass: find maximal runs of codepoints that could change under NFC, transform those,
    // and copy everything between them verbatim. `out` is only materialized when the first such
    // run is found, so the common all-invariant input costs one scan and zero copies.
    //
    // A run is extended one codepoint to the left when it starts with a combining mark, because
    // the mark's composition base — possibly a plain ASCII letter, as in "e" + U+0301 — must be
    // inside the slice utf8proc sees for the pair to compose.
    bool changed = false;
    size_t copied = 0;   // input bytes already appended to out (only meaningful once changed)
    size_t i = 0;
    size_t prev_start = 0;   // byte offset of the previous codepoint

    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            prev_start = i;
            i += 1;
            continue;
        }

        // Decode one UTF-8 sequence. Malformed bytes disable normalization entirely: the raw
        // bytes go to the byte-level tokenizer untouched, matching the pre-NFC behaviour.
        size_t len;
        utf8proc_int32_t cp;
        if ((c & 0xE0) == 0xC0)      { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        else return false;
        if (i + len > n) return false;
        for (size_t j = 1; j < len; ++j) {
            const unsigned char cc = static_cast<unsigned char>(text[i + j]);
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }

        if (!nfc_can_change(cp)) {
            prev_start = i;
            i += len;
            continue;
        }

        // Found the start of a mutable run. Anchor it at the preceding codepoint if this one is
        // a mark needing its base; then extend forward across every codepoint that can change,
        // plus any codepoint a following mark might attach to.
        size_t run_start = i;
        if (utf8proc_get_property(cp)->combining_class != 0) run_start = prev_start;

        size_t run_end = i + len;
        size_t k = run_end;
        size_t last_cp_start = i;
        while (k < n) {
            const unsigned char kc = static_cast<unsigned char>(text[k]);
            size_t klen;
            utf8proc_int32_t kcp;
            if (kc < 0x80)                { klen = 1; kcp = kc; }
            else if ((kc & 0xE0) == 0xC0) { klen = 2; kcp = kc & 0x1F; }
            else if ((kc & 0xF0) == 0xE0) { klen = 3; kcp = kc & 0x0F; }
            else if ((kc & 0xF8) == 0xF0) { klen = 4; kcp = kc & 0x07; }
            else return false;
            if (k + klen > n) return false;
            for (size_t j = 1; j < klen; ++j) {
                const unsigned char cc = static_cast<unsigned char>(text[k + j]);
                if ((cc & 0xC0) != 0x80) return false;
                kcp = (kcp << 6) | (cc & 0x3F);
            }

            if (nfc_can_change(kcp)) {
                run_end = k + klen;
                last_cp_start = k;
                k += klen;
                continue;
            }
            // An invariant starter still belongs in the run if a mark follows it — it may be the
            // base that mark composes with. Look one codepoint ahead before deciding to stop.
            // ASCII never composes as the *second* of a pair, so a following ASCII char ends the
            // run unconditionally.
            if (kcp < 0x80) break;

            size_t peek = k + klen;
            bool mark_follows = false;
            if (peek < n) {
                const unsigned char pc = static_cast<unsigned char>(text[peek]);
                size_t plen = 0;
                utf8proc_int32_t pcp = 0;
                if (pc < 0x80)                { plen = 1; pcp = pc; }
                else if ((pc & 0xE0) == 0xC0) { plen = 2; pcp = pc & 0x1F; }
                else if ((pc & 0xF0) == 0xE0) { plen = 3; pcp = pc & 0x0F; }
                else if ((pc & 0xF8) == 0xF0) { plen = 4; pcp = pc & 0x07; }
                else return false;
                if (peek + plen > n) return false;
                bool ok = true;
                for (size_t j = 1; j < plen; ++j) {
                    const unsigned char cc = static_cast<unsigned char>(text[peek + j]);
                    if ((cc & 0xC0) != 0x80) { ok = false; break; }
                    pcp = (pcp << 6) | (cc & 0x3F);
                }
                if (!ok) return false;
                mark_follows = utf8proc_get_property(pcp)->combining_class != 0;
            }
            if (!mark_follows) break;
            run_end = k + klen;
            last_cp_start = k;
            k += klen;
        }
        (void)last_cp_start;

        if (!changed) {
            changed = true;
            out.clear();
        }
        out.append(text.substr(copied, run_start - copied));
        transform_append(text.substr(run_start, run_end - run_start), out);
        copied = run_end;

        prev_start = run_end;
        i = run_end;
    }

    if (!changed) return false;
    out.append(text.substr(copied));
    return true;
}

} // namespace mlx_qwen_tokenizer
