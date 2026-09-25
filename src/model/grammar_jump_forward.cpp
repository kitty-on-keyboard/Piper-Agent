#include "src/model/grammar_jump_forward.hpp"

#include <cstdlib>
#include <optional>

namespace lmp::model {
namespace {

std::optional<bool> g_test_override;

} // namespace

bool grammar_jump_forward_enabled() noexcept {
    if (g_test_override.has_value()) {
        return *g_test_override;
    }
    // Exact `1` only — same discipline as LMP_ENUM_MASK / LMP_SHADOW_COMPACT.
    if (const char* s = std::getenv("LMP_GRAMMAR_JUMP_FORWARD");
        s != nullptr && s[0] == '1' && s[1] == '\0') {
        return true;
    }
    return false;
}

void grammar_jump_forward_set_enabled_for_test(std::optional<bool> enabled) noexcept {
    g_test_override = enabled;
}

std::optional<unsigned char>
unique_forced_byte(const parsephony::ToolCallGuard& guard) noexcept {
    // FreeText (ValueText bodies) is never unique for jump purposes, even if a
    // transient card==1 somehow appears near a terminator.
    if (guard.mask_class() == parsephony::MaskClass::FreeText) {
        return std::nullopt;
    }
    return guard.allowed_bytes().unique_byte();
}

std::string collect_forced_span(const parsephony::ToolCallGuard& guard) {
    parsephony::ToolCallGuard probe(guard);
    probe.mute();
    std::string out;
    while (!probe.complete()) {
        const auto u = unique_forced_byte(probe);
        if (!u.has_value()) {
            break;
        }
        if (probe.probe_byte(*u) != parsephony::Error::Ok) {
            break;
        }
        out.push_back(static_cast<char>(*u));
    }
    return out;
}

std::vector<ForcedSpanRecord>
enumerate_forced_spans(const parsephony::ToolCallGuard& start, std::string_view raw) {
    parsephony::ToolCallGuard g(start);
    std::vector<ForcedSpanRecord> spans;
    ForcedSpanRecord cur;
    bool in_span = false;

    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (g.complete()) {
            break;
        }
        const auto u = unique_forced_byte(g);
        const unsigned char expect = static_cast<unsigned char>(raw[i]);
        if (u.has_value() && *u == expect) {
            if (!in_span) {
                cur = ForcedSpanRecord{};
                cur.offset = i;
                in_span = true;
            }
            cur.bytes.push_back(raw[i]);
        } else if (in_span) {
            spans.push_back(std::move(cur));
            cur = ForcedSpanRecord{};
            in_span = false;
        }
        if (g.feed(raw.substr(i, 1)) != parsephony::Error::Ok) {
            break;
        }
    }
    if (in_span && !cur.bytes.empty()) {
        spans.push_back(std::move(cur));
    }
    return spans;
}

std::vector<TokenId>
tokens_for_forced_span(const QwenTokenizer& tok, std::string_view bytes) {
    if (bytes.empty()) {
        return {};
    }
    // Ordinary content encode (no specials): tool-XML framing is emitted as
    // ordinary tokens under the guard, never as structural specials mid-call.
    std::vector<TokenId> ids = tok.encode_content(bytes);
    if (ids.empty()) {
        return {};
    }
    std::string round;
    round.reserve(bytes.size());
    for (TokenId id : ids) {
        const std::string_view piece = tok.token_bytes(id);
        if (piece.empty() && id != 0) {
            // Unknown / special with no byte payload — refuse the jump.
            return {};
        }
        round.append(piece);
    }
    if (round != bytes) {
        return {};
    }
    return ids;
}

} // namespace lmp::model
