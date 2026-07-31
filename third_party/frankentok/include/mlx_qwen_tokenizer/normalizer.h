#pragma once

#include <string>
#include <string_view>

namespace mlx_qwen_tokenizer {

// NFC-normalizes `text` into `out` and returns true, or returns false when the input can be used
// as-is — already pure ASCII (NFC-invariant by construction) or not valid UTF-8, where byte-level
// tokenization of the raw bytes is the correct behaviour and normalization would corrupt them.
//
// The false-return contract exists so the encode loop only pays for normalization on the pieces
// that can actually change: prose and code are overwhelmingly ASCII, and scanning for a high byte
// is far cheaper than a decompose/recompose round trip that would return its input.
bool normalize_nfc(std::string_view text, std::string& out);

} // namespace mlx_qwen_tokenizer
