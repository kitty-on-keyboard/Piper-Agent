#pragma once
//
// EventLog -- the single append-only stream (spec S8.1, S14).
//
// This one log is simultaneously the UI feed, the debugging trace, the replay input and
// the capture. It is not three formats that happen to agree; there is one writer and one
// serialisation, so a hand-found bug becomes a permanent test by replaying the file that
// was already being written when it happened.
//
// It is built in phase 0, before anything that emits into it, because the invariant that
// matters -- "every harness->model append emits an event" -- is what makes "did the model
// actually receive this?" an assertion instead of a guess (S19.2). An invariant added
// after the emitters exist is an invariant with holes.
//
// BYTE FAITHFULNESS. Token-level events carry byte fragments that are deliberately not
// valid standalone UTF-8: 944 of Qwen's ~248k tokens are such fragments (S5.3). JSON
// strings must be valid UTF-8, so a value that is not gets BOTH a lossy display form
// (invalid bytes -> U+FFFD) under its own key AND the exact original bytes under
// "<key>__b64". The replayer reads __b64 when present. Nothing is silently mangled, and
// nothing costs base64 in the common case.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/platform/clock.hpp"

namespace lmp::platform {

struct EventField {
    std::string key;
    std::string value;
};

struct Event {
    std::uint64_t seq = 0;
    std::int64_t wall_ns = 0;
    std::int64_t mono_us = 0;
    std::string kind;
    std::vector<EventField> fields;
};

// ---------------------------------------------------------------------------
// Pure core -- no I/O, no globals, no clock. Directly testable.
// ---------------------------------------------------------------------------

// Appends a complete JSON string literal (including the surrounding quotes) to `out`.
// Returns true if `in` was valid UTF-8. When it returns false, `out` received the lossy
// form and the caller is responsible for also emitting the __b64 sibling.
bool append_json_string(std::string& out, std::string_view in);

// True if `in` is well-formed UTF-8: no overlong encodings, no surrogate halves, no
// truncated sequences, nothing above U+10FFFF.
[[nodiscard]] bool is_valid_utf8(std::string_view in) noexcept;

[[nodiscard]] std::string base64_encode(std::string_view in);
// Returns false on any character outside the base64 alphabet or a bad length, leaving
// `out` unspecified.
//
// Its only caller today is the round-trip test that keeps base64_encode honest -- said
// plainly rather than as "used by the replayer", which was written ahead of a replayer
// that reads traces back and still does not exist. Kept anyway, and this is the whole
// argument: append_field base64-encodes any field value that is not valid UTF-8, so a
// trace containing one is not readable without this. An encoder shipped without a
// decoder is how a trace quietly stops being replayable (S2.1.4).
bool base64_decode(std::string_view in, std::string& out);

// One JSONL line, newline-terminated. Deterministic: field order is exactly as given.
[[nodiscard]] std::string serialize_event(const Event& ev);

// ---------------------------------------------------------------------------
// Adapter -- owns a file descriptor, rotates, bounded.
// ---------------------------------------------------------------------------

struct EventLogOptions {
    // All three are required; there is no "sensible default" for how much of the user's
    // disk a background process may consume.
    std::string path;
    std::size_t max_bytes_per_file;
    std::size_t max_files; // includes the live file; must be >= 1
};

struct OpenResult {
    bool ok = false;
    std::string error;
};

class EventLogWriter {
  public:
    EventLogWriter() = default;
    EventLogWriter(const EventLogWriter&) = delete;
    EventLogWriter& operator=(const EventLogWriter&) = delete;
    ~EventLogWriter();

    // Fails if another EventLogWriter in this process already holds `path`. "One
    // writer" is an enforced invariant, not a convention -- two writers interleaving
    // partial lines would corrupt the replay input, and the corruption would look like
    // a model bug.
    [[nodiscard]] OpenResult open(const EventLogOptions& opts);

    // Assigns the sequence number and both timestamps, then writes. `ev.seq`,
    // `ev.wall_ns` and `ev.mono_us` are OUTPUTS -- whatever the caller put there is
    // overwritten, so a caller cannot forge ordering.
    void append(Event& ev, const Clock& clock);

    void flush();
    void close();

    [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }
    [[nodiscard]] std::uint64_t events_written() const noexcept { return next_seq_ - 1; }
    [[nodiscard]] std::size_t bytes_in_current_file() const noexcept { return cur_bytes_; }
    [[nodiscard]] std::uint64_t rotations() const noexcept { return rotations_; }

  private:
    void rotate();

    int fd_ = -1;
    std::string path_;
    std::size_t max_bytes_ = 0;
    std::size_t max_files_ = 0;
    std::size_t cur_bytes_ = 0;
    std::uint64_t next_seq_ = 1;
    std::uint64_t rotations_ = 0;
};

// `path` with ".<n>" inserted before the final extension: events.jsonl -> events.1.jsonl.
[[nodiscard]] std::string rotated_path(std::string_view path, std::size_t index);

} // namespace lmp::platform
