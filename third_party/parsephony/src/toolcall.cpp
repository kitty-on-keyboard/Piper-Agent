#include "parsephony/toolcall.hpp"

#include <algorithm>
#include <cstring>

namespace parsephony {

namespace {

constexpr std::string_view kOpen  = "<tool_call>\n<function=";
constexpr std::string_view kParam = "<parameter=";
constexpr std::string_view kClose = "</function>\n</tool_call>";
constexpr std::string_view kTerm  = "\n</parameter>\n";

inline uint64_t mix(uint64_t h, uint64_t v) noexcept {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

} // namespace

ToolCallGuard::ToolCallGuard(const std::vector<ToolSpec>& tools, Options o)
    : tools_(tools), opts_(o), json_(nullptr, o) {
    value_->reserve(128);
}

void ToolCallGuard::reset() {
    ph_ = Ph::Open;
    lit_pos_ = 0;
    br_param_alive_ = br_close_alive_ = true;
    prefix_.clear();
    prefix_hash_ = 1469598103934665603ull;
    tool_ = -1;
    param_ = -1;
    seen_ = 0;
    term_pos_ = 0;
    json_.reset();
    name_.clear();
    params_ = std::make_shared<std::vector<Param>>();
    value_ = std::make_shared<std::string>();
    value_->reserve(128);
    probing_ = false;
}

Error ToolCallGuard::feed(std::string_view bytes) {
    while (!bytes.empty()) {
        // Enum-constrained text must stay byte-at-a-time: a bulk scan would
        // accept interiors that are not prefixes of any enum_values entry.
        if (ph_ == Ph::ValueText && !enum_active()) {
            if (term_pos_ > 0) {
                unsigned char c = static_cast<unsigned char>(bytes[0]);
                if (c == static_cast<unsigned char>(kTerm[term_pos_])) {
                    bytes.remove_prefix(1);
                    if (++term_pos_ == kTerm.size()) {
                        Error e = finish_param();
                        if (e != Error::Ok) return e;
                    }
                    continue;
                }
                // The partial terminator match was actually value content.
                if (c == static_cast<unsigned char>(kTerm[0])) {   // '\n' restarts
                    value_append(kTerm.substr(0, term_pos_));
                    term_pos_ = 1;
                    bytes.remove_prefix(1);
                    continue;
                }
                if (c < 0x20 && c != '\t' && c != '\r') {
                    value_append(kTerm.substr(0, term_pos_));
                    term_pos_ = 0;
                    bytes.remove_prefix(1);
                    return Error::ControlChar;
                }
                size_t i = 0;
                while (i < bytes.size()) {
                    unsigned char bc = static_cast<unsigned char>(bytes[i]);
                    if (bc == static_cast<unsigned char>(kTerm[0]) || (bc < 0x20 && bc != '\t' && bc != '\r')) {
                        break;
                    }
                    ++i;
                }
                if (i <= 64 - term_pos_) {
                    char buf[64];
                    std::memcpy(buf, kTerm.data(), term_pos_);
                    std::memcpy(buf + term_pos_, bytes.data(), i);
                    value_append(std::string_view(buf, term_pos_ + i));
                } else {
                    value_append(kTerm.substr(0, term_pos_));
                    value_append(bytes.substr(0, i));
                }
                bytes.remove_prefix(i);
                term_pos_ = 0;
                continue;
            }
            // Fast path when term_pos_ == 0: scan for next byte that requires
            // special handling ('\n' or illegal control character).
            size_t i = 0;
            while (i < bytes.size()) {
                unsigned char c = static_cast<unsigned char>(bytes[i]);
                if (c == static_cast<unsigned char>(kTerm[0]) || (c < 0x20 && c != '\t' && c != '\r')) {
                    break;
                }
                ++i;
            }
            if (i > 0) {
                value_append(bytes.substr(0, i));
                bytes.remove_prefix(i);
                if (bytes.empty()) break;
            }
            unsigned char c = static_cast<unsigned char>(bytes[0]);
            bytes.remove_prefix(1);
            if (c == static_cast<unsigned char>(kTerm[0])) {   // '\n' restarts
                term_pos_ = 1;
            } else if (c < 0x20 && c != '\t' && c != '\r') {
                return Error::ControlChar;
            } else {
                value_append(c);
            }
        } else {
            unsigned char c = static_cast<unsigned char>(bytes[0]);
            bytes.remove_prefix(1);
            Error e = push_byte(c);
            if (e != Error::Ok) return e;
        }
    }
    return Error::Ok;
}

void ToolCallGuard::value_append(unsigned char c) {
    if (probing_) return;   // probe copies never need extraction fidelity
    if (value_.use_count() > 1) {
        // A probe copy still shares the buffer; write to a private one.
        auto new_val = std::make_shared<std::string>();
        new_val->reserve(std::max(value_->capacity(), size_t(128)));
        new_val->assign(*value_);
        value_ = std::move(new_val);
    }
    value_->push_back(char(c));
}

void ToolCallGuard::value_append(std::string_view bytes) {
    if (probing_ || bytes.empty()) return;
    if (value_.use_count() > 1) {
        auto new_val = std::make_shared<std::string>();
        new_val->reserve(std::max(value_->capacity(), value_->size() + bytes.size() + 64));
        new_val->assign(*value_);
        value_ = std::move(new_val);
    }
    value_->append(bytes);
}

Error ToolCallGuard::finish_param() {
    // Fail closed: a value that somehow left the enum mask still must not
    // extract as a completed parameter (feed-without-mask / probe drift).
    if (opts_.enforce_enum_values && tool_ >= 0 && param_ >= 0) {
        const auto& ev = tools_[size_t(tool_)].params[size_t(param_)].enum_values;
        if (!ev.empty()) {
            bool ok = false;
            bool any_usable = false;
            for (const auto& e : ev) {
                if (!enum_value_usable(e)) continue;
                any_usable = true;
                if (e == *value_) {
                    ok = true;
                    break;
                }
            }
            if (any_usable && !ok) return Error::UnexpectedChar;
        }
    }
    if (!probing_) {
        if (params_.use_count() > 1) {
            params_ = std::make_shared<std::vector<Param>>(*params_);
        }
        params_->push_back(Param{tools_[size_t(tool_)].params[size_t(param_)].name,
                                 *value_,
                                 tools_[size_t(tool_)].params[size_t(param_)].type});
        value_ = std::make_shared<std::string>();
        value_->reserve(128);
    }
    seen_ |= (1ull << param_);
    param_ = -1;
    term_pos_ = 0;
    ph_ = Ph::Branch;
    lit_pos_ = 0;
    br_param_alive_ = br_close_alive_ = true;
    return Error::Ok;
}

bool ToolCallGuard::enum_value_usable(std::string_view e) const noexcept {
    // Newlines are the parameter terminator's first byte; an enum that embeds
    // one cannot be distinguished from closing the value. Fail open for that
    // entry rather than risk an empty mask.
    return e.find('\n') == std::string_view::npos;
}

bool ToolCallGuard::enum_active() const noexcept {
    if (!opts_.enforce_enum_values || tool_ < 0 || param_ < 0) return false;
    const auto& ev = tools_[size_t(tool_)].params[size_t(param_)].enum_values;
    if (ev.empty()) return false;
    for (const auto& e : ev) {
        if (enum_value_usable(e)) return true;
    }
    return false;
}

bool ToolCallGuard::enum_prefix_match(std::string_view cand) const noexcept {
    const auto& ev = tools_[size_t(tool_)].params[size_t(param_)].enum_values;
    for (const auto& e : ev) {
        if (!enum_value_usable(e)) continue;
        if (e.size() >= cand.size() && e.compare(0, cand.size(), cand) == 0) return true;
    }
    return false;
}

bool ToolCallGuard::enum_exact_match(std::string_view cand) const noexcept {
    const auto& ev = tools_[size_t(tool_)].params[size_t(param_)].enum_values;
    for (const auto& e : ev) {
        if (!enum_value_usable(e)) continue;
        if (e == cand) return true;
    }
    return false;
}

ByteSet ToolCallGuard::enum_allowed_bytes() const {
    ByteSet s;
    if (term_pos_ > 0) {
        s.add(static_cast<unsigned char>(kTerm[term_pos_]));
        return s;
    }
    const auto& ev = tools_[size_t(tool_)].params[size_t(param_)].enum_values;
    const std::string_view cur(*value_);
    for (const auto& e : ev) {
        if (!enum_value_usable(e)) continue;
        if (e.size() > cur.size() && e.compare(0, cur.size(), cur) == 0) {
            s.add(static_cast<unsigned char>(e[cur.size()]));
        }
        if (e == cur) {
            s.add('\n');   // start of "\n</parameter>\n"
        }
    }
    return s;
}

ByteSet ToolCallGuard::type_start_set(ParamType t) const {
    ByteSet s;
    switch (t) {
        case ParamType::Number:  s.add('-'); s.add_range('0', '9'); break;
        case ParamType::Boolean: s.add('t'); s.add('f'); break;
        case ParamType::Object:  s.add('{'); break;
        case ParamType::Array:   s.add('['); break;
        case ParamType::Json:
            s.add('{'); s.add('['); s.add('"'); s.add('-');
            s.add_range('0', '9'); s.add('t'); s.add('f'); s.add('n');
            break;
        case ParamType::Text: break;   // never reaches the JSON path
    }
    return s;
}

// Would feeding '\n' to the JSON sub-automaton complete the value?
// True exactly when a root-level number is validly terminated by it, or the
// value is already complete. Cheap: the probe copy is small.
bool ToolCallGuard::json_completes_on_newline() const {
    if (json_.complete()) return false;   // already ended on a content byte
    StreamParser probe = json_;
    probe.mute();
    if (probe.probe_byte('\n') != Error::Ok) return false;
    return probe.complete();
}

Error ToolCallGuard::push_byte(unsigned char c) {
    switch (ph_) {
        // ---- fixed literals -------------------------------------------------
        case Ph::Open:
            if (c != static_cast<unsigned char>(kOpen[lit_pos_])) return Error::UnexpectedChar;
            if (++lit_pos_ == kOpen.size()) {
                ph_ = Ph::Name;
                prefix_.clear();
                prefix_hash_ = 1469598103934665603ull;
            }
            return Error::Ok;

        case Ph::AfterName:
            if (c != '\n') return Error::UnexpectedChar;
            ph_ = Ph::Branch;
            lit_pos_ = 0;
            br_param_alive_ = br_close_alive_ = true;
            return Error::Ok;

        case Ph::AfterPName: {
            if (c != '\n') return Error::UnexpectedChar;
            const auto& ps = tools_[size_t(tool_)].params[size_t(param_)];
            // Enum literals are matched as exact spellings (XML raw text for
            // strings; JSON number/bool text as stored in enum_values). Route
            // through ValueText so one automaton covers both forms.
            if (enum_active() || ps.type == ParamType::Text) {
                ph_ = Ph::ValueText;
                term_pos_ = 0;
            } else {
                ph_ = Ph::ValueJsonFirst;
            }
            return Error::Ok;
        }

        // ---- candidate prefix matching --------------------------------------
        case Ph::Name: {
            if (c == '>') {
                for (size_t i = 0; i < tools_.size(); ++i) {
                    if (tools_[i].name == prefix_) {
                        tool_ = int(i);
                        if (!probing_) name_ = prefix_;
                        ph_ = Ph::AfterName;
                        return Error::Ok;
                    }
                }
                return Error::UnexpectedChar;   // prefix is not a complete name
            }
            for (const auto& t : tools_) {
                if (t.name.size() > prefix_.size() &&
                    t.name.compare(0, prefix_.size(), prefix_) == 0 &&
                    static_cast<unsigned char>(t.name[prefix_.size()]) == c) {
                    prefix_.push_back(char(c));
                    prefix_hash_ = (prefix_hash_ ^ c) * 1099511628211ull;
                    return Error::Ok;
                }
            }
            return Error::UnexpectedChar;
        }

        case Ph::PName: {
            const auto& ps = tools_[size_t(tool_)].params;
            if (c == '>') {
                for (size_t i = 0; i < ps.size(); ++i) {
                    if ((seen_ >> i) & 1) continue;
                    if (ps[i].name == prefix_) {
                        param_ = int(i);
                        ph_ = Ph::AfterPName;
                        return Error::Ok;
                    }
                }
                return Error::UnexpectedChar;
            }
            for (size_t i = 0; i < ps.size(); ++i) {
                if ((seen_ >> i) & 1) continue;
                if (ps[i].name.size() > prefix_.size() &&
                    ps[i].name.compare(0, prefix_.size(), prefix_) == 0 &&
                    static_cast<unsigned char>(ps[i].name[prefix_.size()]) == c) {
                    prefix_.push_back(char(c));
                    prefix_hash_ = (prefix_hash_ ^ c) * 1099511628211ull;
                    return Error::Ok;
                }
            }
            return Error::UnexpectedChar;
        }

        // ---- the two-way branch ----------------------------------------------
        case Ph::Branch: {
            const auto& ps = tools_[size_t(tool_)].params;
            uint64_t all = ps.size() >= 64 ? ~0ull : ((1ull << ps.size()) - 1);
            bool params_left = (seen_ & all) != all;
            bool required_met = true;
            for (size_t i = 0; i < ps.size(); ++i) {
                if (ps[i].required && !((seen_ >> i) & 1)) { required_met = false; break; }
            }

            bool a = br_param_alive_ && params_left &&
                     lit_pos_ < kParam.size() &&
                     c == static_cast<unsigned char>(kParam[lit_pos_]);
            bool b = br_close_alive_ && required_met &&
                     lit_pos_ < kClose.size() &&
                     c == static_cast<unsigned char>(kClose[lit_pos_]);
            if (!a && !b) return Error::UnexpectedChar;
            br_param_alive_ = a;
            br_close_alive_ = b;
            ++lit_pos_;

            if (a && lit_pos_ == kParam.size()) {
                ph_ = Ph::PName;
                prefix_.clear();
                prefix_hash_ = 1469598103934665603ull;
            } else if (b && lit_pos_ == kClose.size()) {
                ph_ = Ph::Done;
            }
            return Error::Ok;
        }

        // ---- raw text value (also enum literals when enum_values set) ----------
        case Ph::ValueText: {
            if (enum_active()) {
                if (term_pos_ > 0) {
                    if (c != static_cast<unsigned char>(kTerm[term_pos_])) {
                        // Already closed the enum value; do not absorb a bad
                        // terminator byte back into the value (would empty-mask).
                        return Error::UnexpectedChar;
                    }
                    if (++term_pos_ == kTerm.size()) return finish_param();
                    return Error::Ok;
                }
                if (c == static_cast<unsigned char>(kTerm[0])) {
                    if (!enum_exact_match(*value_)) return Error::UnexpectedChar;
                    term_pos_ = 1;
                    return Error::Ok;
                }
                if (c < 0x20 && c != '\t' && c != '\r') return Error::ControlChar;
                // Tentative append: must remain a prefix of some usable enum.
                // Always mutate value_ even when probing_ — TokenMask feeds
                // multi-byte tokens to a muted copy and needs the running prefix
                // (Name/PName use prefix_ the same way). COW keeps the parent safe.
                if (value_.use_count() > 1) {
                    auto new_val = std::make_shared<std::string>();
                    new_val->reserve(std::max(value_->capacity(), size_t(128)));
                    new_val->assign(*value_);
                    value_ = std::move(new_val);
                }
                value_->push_back(char(c));
                if (!enum_prefix_match(*value_)) {
                    value_->pop_back();
                    return Error::UnexpectedChar;
                }
                return Error::Ok;
            }
            if (c == static_cast<unsigned char>(kTerm[term_pos_])) {
                if (++term_pos_ == kTerm.size()) return finish_param();
                return Error::Ok;
            }
            // The partial terminator match was actually value content.
            if (c == static_cast<unsigned char>(kTerm[0])) {   // '\n' restarts
                if (term_pos_ > 0) {
                    value_append(kTerm.substr(0, term_pos_));
                }
                term_pos_ = 1;
                return Error::Ok;
            }
            if (c < 0x20 && c != '\t' && c != '\r') {
                if (term_pos_ > 0) {
                    value_append(kTerm.substr(0, term_pos_));
                    term_pos_ = 0;
                }
                return Error::ControlChar;
            }
            if (term_pos_ > 0) {
                char buf[16];
                std::memcpy(buf, kTerm.data(), term_pos_);
                buf[term_pos_] = static_cast<char>(c);
                value_append(std::string_view(buf, term_pos_ + 1));
                term_pos_ = 0;
                return Error::Ok;
            }
            value_append(c);
            return Error::Ok;
        }

        // ---- typed (JSON) value -------------------------------------------------
        case Ph::ValueJsonFirst: {
            ParamType t = tools_[size_t(tool_)].params[size_t(param_)].type;
            if (!type_start_set(t).contains(c)) return Error::UnexpectedChar;
            json_.reset();
            Error e = json_.probe_byte(c);
            if (e != Error::Ok) return e;
            value_append(c);
            ph_ = Ph::ValueJson;
            return Error::Ok;
        }

        case Ph::ValueJson: {
            if (c == '\n') {
                // Only legal as the value terminator's first byte: a raw newline
                // can never be JSON content, and interior whitespace is not part
                // of the template's compact single-line encoding.
                if (json_.mask_class() == MaskClass::JsonString) return Error::ControlChar;
                if (json_.complete()) {
                    ph_ = Ph::JsonTerm;
                    lit_pos_ = 1;   // this byte is kTerm[0]
                    return Error::Ok;
                }
                if (!json_completes_on_newline()) return Error::UnexpectedChar;
                Error e = json_.probe_byte('\n');   // flushes a root-level number
                if (e != Error::Ok) return e;
                ph_ = Ph::JsonTerm;
                lit_pos_ = 1;
                return Error::Ok;
            }
            if (c == '\t' || c == '\r') return Error::UnexpectedChar;
            if (c == ' ' && json_.mask_class() != MaskClass::JsonString) {
                // Interior whitespace as `tojson` emits it (": " and ", "), but
                // never trailing whitespace after the value completes.
                StreamParser probe = json_;
                probe.mute();
                if (probe.probe_byte(' ') != Error::Ok || probe.complete())
                    return Error::UnexpectedChar;
            }
            if (json_.complete()) return Error::TrailingContent;
            Error e = json_.probe_byte(c);
            if (e != Error::Ok) return e;
            value_append(c);
            if (json_.complete()) {   // ended on a content byte: '}', ']', '"', 'e'…
                ph_ = Ph::JsonTerm;
                lit_pos_ = 0;
            }
            return Error::Ok;
        }

        case Ph::JsonTerm:
            if (c != static_cast<unsigned char>(kTerm[lit_pos_])) return Error::UnexpectedChar;
            if (++lit_pos_ == kTerm.size()) return finish_param();
            return Error::Ok;

        case Ph::Done:
            return Error::TrailingContent;
    }
    return Error::UnexpectedChar;
}

// ---------------------------------------------------------------------------
// Constraint side
// ---------------------------------------------------------------------------

ByteSet ToolCallGuard::allowed_bytes() const {
    ByteSet s;
    switch (ph_) {
        case Ph::Open:
            s.add(static_cast<unsigned char>(kOpen[lit_pos_]));
            return s;

        case Ph::AfterName:
        case Ph::AfterPName:
            s.add('\n');
            return s;

        case Ph::Name: {
            for (const auto& t : tools_) {
                if (t.name.size() > prefix_.size() &&
                    t.name.compare(0, prefix_.size(), prefix_) == 0)
                    s.add(static_cast<unsigned char>(t.name[prefix_.size()]));
                if (t.name == prefix_) s.add('>');
            }
            return s;
        }

        case Ph::PName: {
            const auto& ps = tools_[size_t(tool_)].params;
            for (size_t i = 0; i < ps.size(); ++i) {
                if ((seen_ >> i) & 1) continue;
                if (ps[i].name.size() > prefix_.size() &&
                    ps[i].name.compare(0, prefix_.size(), prefix_) == 0)
                    s.add(static_cast<unsigned char>(ps[i].name[prefix_.size()]));
                if (ps[i].name == prefix_) s.add('>');
            }
            return s;
        }

        case Ph::Branch: {
            const auto& ps = tools_[size_t(tool_)].params;
            uint64_t all = ps.size() >= 64 ? ~0ull : ((1ull << ps.size()) - 1);
            bool params_left = (seen_ & all) != all;
            bool required_met = true;
            for (size_t i = 0; i < ps.size(); ++i) {
                if (ps[i].required && !((seen_ >> i) & 1)) { required_met = false; break; }
            }
            if (br_param_alive_ && params_left && lit_pos_ < kParam.size())
                s.add(static_cast<unsigned char>(kParam[lit_pos_]));
            if (br_close_alive_ && required_met && lit_pos_ < kClose.size())
                s.add(static_cast<unsigned char>(kClose[lit_pos_]));
            return s;
        }

        case Ph::ValueText:
            if (enum_active()) return enum_allowed_bytes();
            // Any byte can be value content; the terminator is recognized, not
            // forced. Control bytes other than \t \n \r stay illegal.
            s.add('\t'); s.add('\n'); s.add('\r');
            s.add_range(0x20, 0x7F);
            s.non_ascii = true;
            return s;

        case Ph::ValueJsonFirst:
            return type_start_set(tools_[size_t(tool_)].params[size_t(param_)].type);

        case Ph::ValueJson: {
            if (json_.complete()) {
                s.add('\n');   // only the terminator may follow
                return s;
            }
            s = json_.allowed_bytes();
            // The sub-grammar's whitespace freedoms shrink to the template's
            // canonical single-line form.
            ByteSet out;
            out.non_ascii = s.non_ascii;
            for (int c = 0; c < 128; ++c) {
                if (!s.contains(static_cast<unsigned char>(c))) continue;
                if (c == '\t' || c == '\r' || c == '\n') continue;
                if (c == ' ' && json_.mask_class() != MaskClass::JsonString) {
                    StreamParser probe = json_;
                    probe.mute();
                    if (probe.probe_byte(' ') != Error::Ok || probe.complete()) continue;
                }
                out.add(static_cast<unsigned char>(c));
            }
            if (json_completes_on_newline()) out.add('\n');
            return out;
        }

        case Ph::JsonTerm:
            s.add(static_cast<unsigned char>(kTerm[lit_pos_]));
            return s;

        case Ph::Done:
            return s;   // nothing: generation stops here
    }
    return s;
}

MaskClass ToolCallGuard::mask_class() const noexcept {
    // Enum-constrained text is narrow (candidate literals), not FreeText — the
    // token mask engine's FreeText fast path would admit illegal tokens.
    if (ph_ == Ph::ValueText && term_pos_ == 0 && !enum_active()) return MaskClass::FreeText;
    if (ph_ == Ph::ValueJson && !json_.complete() &&
        json_.mask_class() == MaskClass::JsonString)
        return MaskClass::JsonString;
    return MaskClass::Other;
}

uint64_t ToolCallGuard::state_signature() const noexcept {
    uint64_t h = 0x243f6a8885a308d3ull;
    h = mix(h, uint64_t(ph_));
    h = mix(h, lit_pos_);
    h = mix(h, prefix_hash_);
    h = mix(h, uint64_t(prefix_.size()));
    h = mix(h, uint64_t(tool_ + 1));
    h = mix(h, uint64_t(param_ + 1));
    h = mix(h, seen_);
    h = mix(h, term_pos_);
    h = mix(h, uint64_t(br_param_alive_) * 2 + uint64_t(br_close_alive_));
    if (ph_ == Ph::ValueJson) h = mix(h, json_.state_signature());
    // Enum value masks depend on how much of a literal has been emitted.
    if (ph_ == Ph::ValueText && enum_active()) {
        uint64_t vh = 1469598103934665603ull;
        for (unsigned char c : *value_) vh = (vh ^ c) * 1099511628211ull;
        h = mix(h, vh);
        h = mix(h, uint64_t(value_->size()));
    }
    return h;
}

} // namespace parsephony
