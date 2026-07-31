#pragma once

// ToolCallGuard — a byte automaton for Qwen 3.6's tool-call emission format.
//
// Verified against the model's own chat_template.jinja, the format is XML-ish
// framing, not JSON:
//
//     <tool_call>
//     <function=get_weather>
//     <parameter=city>
//     Denver
//     </parameter>
//     </function>
//     </tool_call>
//
// String parameter values are raw multi-line text. Non-string values are JSON
// (the template runs them through `tojson`), which is where the JSON PDA nests
// inside this grammar as a sub-automaton.
//
// The guard is schema-aware, not just shape-aware. Fed a registry of tools it
// constrains, byte by byte:
//
//   - the function name to exactly the registered tool names
//   - parameter names to the chosen tool's parameters, each used at most once
//   - `</function>` to appear only once every required parameter is present
//   - typed parameter values to well-formed JSON of the declared type
//
// Under a token mask built from this automaton, the model *cannot* call a tool
// that does not exist, misspell a parameter, omit a required one, or produce a
// malformed typed value. It also cannot escape the framing: generation is
// complete exactly at the closing `</tool_call>`.
//
// The guard doubles as the extraction parser: when the automaton reaches
// complete(), tool_name() and params() hold the parsed call, so LM-Pipe needs
// no second pass over the text.

#include "parsephony/stream.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace parsephony {

enum class ParamType : uint8_t {
    Text,      // raw text value (JSON-schema "string")
    Json,      // any JSON value
    Number,    // JSON number
    Boolean,   // true / false
    Object,    // JSON object
    Array,     // JSON array
};

struct ParamSpec {
    std::string name;
    ParamType type = ParamType::Text;
    bool required = false;
};

struct ToolSpec {
    std::string name;
    std::vector<ParamSpec> params;   // at most 64 per tool
};

class ToolCallGuard {
public:
    // `tools` must outlive the guard. Tool and parameter names must be
    // non-empty, distinct, and free of the bytes '>', '=', and '\n'
    // (the format's own delimiters).
    explicit ToolCallGuard(const std::vector<ToolSpec>& tools, Options o = {});

    Error feed(std::string_view bytes);
    Error probe_byte(unsigned char c) { return push_byte(c); }

    bool complete() const noexcept { return ph_ == Ph::Done; }
    void reset();

    // --- extraction (valid once complete) -----------------------------------

    struct Param {
        std::string name;
        std::string value;   // raw text, or JSON text for typed parameters
        ParamType type;
    };

    const std::string& tool_name() const noexcept { return name_; }
    const std::vector<Param>& params() const noexcept { return *params_; }

    // --- mask-engine interface ----------------------------------------------

    ByteSet allowed_bytes() const;
    uint64_t state_signature() const noexcept;
    MaskClass mask_class() const noexcept;
    void mute() noexcept { probing_ = true; json_.mute(); }

private:
    enum class Ph : uint8_t {
        Open,        // literal "<tool_call>\n<function="
        Name,        // tool name (candidate prefix match)
        AfterName,   // literal "\n"
        Branch,      // "<parameter=" vs "</function>\n</tool_call>"
        PName,       // parameter name (candidate prefix match)
        AfterPName,  // literal "\n"
        ValueText,   // raw text until "\n</parameter>\n"
        ValueJsonFirst, // first byte of a typed value (type-gated)
        ValueJson,   // inside the JSON sub-automaton
        JsonTerm,    // terminator literal after a typed value
        Done,
    };

    Error push_byte(unsigned char c);
    Error advance_literal(unsigned char c, std::string_view lit, Ph next);
    Error finish_param();
    void  value_append(unsigned char c);
    ByteSet type_start_set(ParamType t) const;
    bool json_completes_on_newline() const;

    const std::vector<ToolSpec>& tools_;
    Options opts_;

    Ph ph_ = Ph::Open;
    uint32_t lit_pos_ = 0;

    // Branch state: which of the two continuations is still alive.
    bool br_param_alive_ = true;
    bool br_close_alive_ = true;

    // Candidate prefix matching (tool names, then parameter names).
    std::string prefix_;
    uint64_t prefix_hash_ = 1469598103934665603ull;

    int tool_ = -1;           // chosen tool index
    int param_ = -1;          // parameter being read
    uint64_t seen_ = 0;       // bitmask of parameters already provided

    uint32_t term_pos_ = 0;   // progress through "\n</parameter>\n"

    StreamParser json_;       // sub-automaton for typed values

    // Extraction. Both accumulators are copy-on-write so the mask engine's
    // per-candidate probe copies cost O(1) instead of O(recorded content).
    std::string name_;
    std::shared_ptr<std::vector<Param>> params_ = std::make_shared<std::vector<Param>>();
    std::shared_ptr<std::string> value_ = std::make_shared<std::string>();
    bool probing_ = false;
};

} // namespace parsephony
