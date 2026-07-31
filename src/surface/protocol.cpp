#include "src/surface/protocol_generated.hpp"

#include "src/platform/event_log.hpp"

namespace lmp::protocol {

// Declared by the generated header, defined here: string escaping is delegated to the
// platform layer so the protocol and the event log cannot disagree about what a JSON
// string is -- and so byte-faithfulness for non-UTF-8 payloads has ONE implementation.
void append_value(std::string& out, const std::string& v) {
    (void)platform::append_json_string(out, v);
}

} // namespace lmp::protocol
