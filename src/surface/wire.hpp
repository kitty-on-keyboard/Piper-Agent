#pragma once
//
// The outbound half of the transport: everything this process says on stdout.
//
// transport.hpp owns stdin -- one reader thread, frame in the producer. This owns the
// other direction, and exists for the same reason: writing to stdout is a SHARED
// resource with more than one writer, and the rule that keeps it coherent has to live
// somewhere a reader can find it rather than being an anonymous static in main's file.
//
// THE RULE. A line is written under one lock, payload and newline together, and the
// flush happens inside it. stdio locks each FILE* call individually, which is not
// enough: a message and its terminator are two calls, so without this another thread's
// line lands between them and BOTH messages are corrupt. The extension's reader is
// line-oriented, so a half-written line is not a display glitch -- it is a parse
// failure that the client reports as a dead sidecar.
//
// Two threads reach it today (the run thread for turn/verification/perf, the token
// streamer for per-token text) and the model-status notifications made a third.
//
#include <string>
#include <string_view>

#include "src/surface/protocol_generated.hpp"

namespace lmp::surface::wire {

// One whole line, atomically against every other writer. The lock is held across the
// flush deliberately; see above.
void write_line(const std::string& line);

// A JSON string literal, escaped by the platform layer -- so the protocol and the event
// log cannot disagree about what a JSON string is.
[[nodiscard]] std::string json_escape(std::string_view in);

void reply_result(const std::string& id, const std::string& body);
void reply_error(const std::string& id, const std::string& message);

// Every outbound notification goes through the GENERATED serializer for its type, so a
// schema change the sidecar has not caught up with is a compile error rather than a
// field the extension silently reads as its default (S4.4).
template <class N>
void notify(const N& n) {
    std::string out;
    out += R"({"jsonrpc":"2.0","method":")";
    out += N::kMethod;
    out += R"(","params":)";
    out += protocol::to_json(n);
    out += "}";
    write_line(out);
}

} // namespace lmp::surface::wire
