#pragma once
//
// THE RESTORED CONVERSATION, for the pane to repaint -- and the only place in this file
// that has to think about how big a reply may be.
//
// The context store holds what the MODEL needs, which is unbounded by design: one
// observation in the real log is 708,670 bytes, and a long run holds hundreds. Putting
// that on the wire would mean a multi-megabyte JSON reply parsed on the extension host and
// then laid out as DOM in a webview, to show a file the reader will scroll past. So the
// transcript is capped twice -- per entry, then in total -- and BOTH caps are reported
// rather than applied quietly.
//
// The tail is kept, not the head. A conversation is read from where it left off, and the
// turns that explain why a run died are the last ones; dropping from the front costs the
// least and the count comes back so the surface can say the paint is partial.
//
// None of this touches session.ctx. The model resumes with everything; only the picture
// is abridged.
#include <cstddef>
#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/surface/protocol_generated.hpp"

namespace lmp::surface::transcript {

constexpr std::size_t kMaxEntryChars = 4000;
constexpr std::size_t kMaxTotalChars = 256U * 1024U;

[[nodiscard]] inline std::string clip(const std::string& in, bool& truncated) {
    if (in.size() <= kMaxEntryChars) {
        truncated = false;
        return in;
    }
    truncated = true;
    // Cut on a UTF-8 boundary. A JSON string carrying half a codepoint is not a display
    // glitch -- it is a parse error at the other end, which loses the whole transcript to
    // save four bytes of one observation.
    std::size_t cut = kMaxEntryChars;
    while (cut > 0 && (static_cast<unsigned char>(in[cut]) & 0xC0U) == 0x80U) {
        --cut;
    }
    return in.substr(0, cut);
}

[[nodiscard]] inline std::vector<protocol::TranscriptEntry> build(const context::ContextStore& ctx,
                                             int& omitted_leading) {
    std::vector<protocol::TranscriptEntry> all;
    for (const context::TurnRecord& t : ctx.recent()) {
        if (!t.user_text.empty()) {
            protocol::TranscriptEntry e;
            e.role = "user";
            e.text = clip(t.user_text, e.truncated);
            all.push_back(std::move(e));
        }
        if (!t.assistant_text.empty()) {
            protocol::TranscriptEntry e;
            e.role = "assistant";
            e.text = clip(t.assistant_text, e.truncated);
            all.push_back(std::move(e));
        }
        // A tool turn is one entry carrying the call AND its result, because that is how
        // the pane draws it live: one collapsible row whose summary is the call and whose
        // body is what came back. Splitting them here would paint two rows for one event.
        if (!t.tool_name.empty()) {
            protocol::TranscriptEntry e;
            e.role = "tool";
            e.tool = t.tool_name;
            e.args = t.tool_args_summary;
            e.is_error = t.observation_is_error;
            e.text = clip(t.observation, e.truncated);
            all.push_back(std::move(e));
        }
    }

    // Second cap, walking backwards so the survivors are the most recent.
    std::size_t total = 0;
    std::size_t first_kept = all.size();
    while (first_kept > 0) {
        const std::size_t cost = all[first_kept - 1].text.size() + 64; // 64 ~ the envelope
        if (total + cost > kMaxTotalChars) {
            break;
        }
        total += cost;
        --first_kept;
    }
    omitted_leading = static_cast<int>(first_kept);
    return std::vector<protocol::TranscriptEntry>(all.begin() + static_cast<std::ptrdiff_t>(first_kept),
                                                  all.end());
}

} // namespace lmp::surface::transcript

