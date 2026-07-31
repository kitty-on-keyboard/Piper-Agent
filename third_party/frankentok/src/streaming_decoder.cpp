#include "mlx_qwen_tokenizer/streaming_decoder.h"

namespace mlx_qwen_tokenizer {

std::string StreamingDecoder::decode(const std::vector<int32_t>& ids) {
    std::string res;
    for (int32_t id : ids) {
        res += decode(id);
    }
    return res;
}

std::string StreamingDecoder::decode(int32_t id) {
    // The vocab's raw-bytes table already holds what every token decodes to — ByteLevel-unmapped
    // at load, once. The previous path re-unmapped the token text on every call: a string
    // allocation plus a hash lookup per character, per generated token, to recompute a constant.
    std::string_view bytes = vocab_.id_to_bytes_view(id);
    if (bytes.empty()) return "";

    // A special token is emitted whole and never enters the byte buffer — it is not byte-level
    // encoded, so mixing it in would corrupt whatever partial sequence is pending.
    if (vocab_.is_special_id(id)) {
        std::string res = flush();
        res.append(bytes);
        return res;
    }

    // Generation hot path: the buffer is almost always empty and the token's bytes almost always
    // complete UTF-8, in which case they can be returned directly without touching the buffer.
    if (buffer_.empty()) {
        const size_t valid = get_valid_utf8_prefix_sv(bytes);
        if (valid == bytes.size()) return std::string(bytes);
    }

    buffer_.append(bytes);
    size_t valid_len = get_valid_utf8_prefix(buffer_);
    if (valid_len > 0) {
        std::string res = buffer_.substr(0, valid_len);
        buffer_.erase(0, valid_len);
        return res;
    }

    return "";
}

std::string StreamingDecoder::decode_bytes(std::string_view bytes) {
    buffer_.append(bytes);
    size_t valid_len = get_valid_utf8_prefix(buffer_);
    if (valid_len == 0) return "";
    std::string res = buffer_.substr(0, valid_len);
    buffer_.erase(0, valid_len);
    return res;
}

std::string StreamingDecoder::flush() {
    std::string res;
    res.swap(buffer_);
    // Whatever is left is genuinely incomplete — the stream ended mid-sequence. Emitting the raw
    // bytes matches what the reference tokenizer does rather than inventing a replacement char.
    return res;
}

// Longest prefix of `str` that is complete, well-formed UTF-8.
//
// The distinction that matters for streaming: a sequence that is merely *incomplete* must be held
// back so its continuation bytes can arrive, but one already proven *invalid* must not be — the
// bytes to prove it wrong are already in hand and no future byte can rescue it. Deferring that
// check until the full expected length arrives strands the buffer forever on malformed input.
size_t StreamingDecoder::get_valid_utf8_prefix_sv(std::string_view str) const {
    size_t i = 0;
    size_t last_valid = 0;
    const size_t n = str.length();

    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(str[i]);
        size_t len = 0;
        if (c <= 0x7F) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else {
            // Invalid starting byte, skip it to allow recovery
            i++;
            continue;
        }

        // Validate every continuation byte actually present, whether or not the sequence is
        // complete. A bad one here is decisive now; waiting for the rest would change nothing.
        const size_t available = (i + len <= n) ? len : (n - i);
        bool invalid = false;
        for (size_t j = 1; j < available; ++j) {
            if ((static_cast<unsigned char>(str[i + j]) & 0xC0) != 0x80) {
                invalid = true;
                break;
            }
        }

        if (invalid) {
            i++;          // resync from the next byte
            continue;
        }

        if (i + len > n) {
            break;        // genuinely incomplete: hold it, the rest may still arrive
        }

        i += len;
        last_valid = i;
    }
    return last_valid;
}

size_t StreamingDecoder::get_valid_utf8_prefix(const std::string& str) const {
    return get_valid_utf8_prefix_sv(str);
}

} // namespace mlx_qwen_tokenizer
