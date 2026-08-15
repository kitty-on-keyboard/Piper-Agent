#include "src/tools/registry.hpp"

#include "src/pcc/store.hpp"
#include "src/platform/fs.hpp"

#include <algorithm>
#include <cctype>

// The cross-session memory file (spec S8). Split out of registry.cpp because it is the one
// tool whose state outlives the run: every other handler there acts on the workspace
// within a single mission, and this one writes a file the NEXT session reads. That is a
// different lifetime and a different failure mode, and it wants its own tests.

namespace lmp::tools {

namespace fsx = lmp::platform;

bool would_overwrite_existing(const std::string& root, const std::string& rel) {
    const fsx::WorkspaceFs workspace(root);
    if (!workspace.valid()) {
        return false;
    }
    // Read-based rather than stat-based, so it uses the same door every other path check
    // uses. A file too large to read is certainly not empty, so it counts as content.
    const fsx::FileContents f = workspace.read_file_whole(rel, 1);
    return f.status == fsx::FsStatus::TooLarge || (f.ok() && !f.bytes.empty());
}

bool Registry::would_overwrite_existing(const std::string& rel) const {
    const fsx::FileContents f = workspace_fs_.read_file_whole(rel, 1);
    return f.status == fsx::FsStatus::TooLarge || (f.ok() && !f.bytes.empty());
}

// The bytes on disk are already the bytes asked for.
//
// Read-based rather than a hash or a size compare, and it takes the SAME door
// would_overwrite_existing takes. A size compare is the tempting cheap version and it is
// wrong in the direction that matters: the run this was built for wrote 5327 bytes to one
// file four times, and the interesting question is never "is it the same length".
//
// A file too large to read is not unchanged as far as this can prove, so it is written.
// Being wrong here costs one redundant write; being wrong the other way silently drops an
// edit, which is the one outcome a write tool may never have.
static bool already_holds(const fsx::WorkspaceFs& workspace, std::string_view path,
                          std::string_view content,
                          std::size_t max_bytes) {
    const fsx::FileContents f = workspace.read_file_whole(path, max_bytes);
    return f.ok() && f.bytes.size() == content.size() && f.bytes == content;
}

void Registry::note_read_version(const std::string& abs_path, std::string_view bytes) {
    read_versions_[abs_path] = platform::content_sha256_hex(bytes);
}

// The digest out of whatever the model copied off the screen.
//
// THE TOOL TOLD THE MODEL ITS OWN EDITS WERE CORRUPT. read_file ends every result with
// `[content_version sha256=<hex>]`, so a model asked for `expected_version` copies the
// value it was shown -- `sha256=<hex>`, prefix and all. Nothing stripped it, the compare
// is a plain string compare against bare hex, and the conflict it raised printed both
// sides: `expected sha256=123ae005..., current 123ae005...`. Identical digests, reported
// as a mismatch. The model read that correctly ("hashes match but the tool detects a
// difference"), concluded the edit tools were broken, and spent the rest of the run
// writing Swift through shell heredocs and sed -- which is how a run loses every
// preimage check the claim exists to provide.
//
// MEASURED, run 4 of 2026-08-12: eight conflicts across replace_in_file and write_file,
// every one of them expected == current modulo this prefix, starting with the run's very
// first edit.
//
// Postel on the way in, and it belongs HERE rather than at either comparison site: both
// WritePrecondition::expected_version and EditIntent::expected_version are documented as
// lowercase hex, and this is the single door a model-supplied value comes through. The
// headless rename check and the editor's buffer check then both get what they promise.
//
// Deliberately NOT a validity gate. Something that normalises to a non-digest is left
// alone and fails the compare as a conflict, which is the safe direction: refusing to
// write beats writing over a preimage nobody checked.
std::string normalize_content_version(std::string_view supplied) {
    std::string_view v = supplied;
    const auto trim = [&v] {
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t' || v.front() == '\r' ||
                              v.front() == '\n' || v.front() == '"' || v.front() == '\'')) {
            v.remove_prefix(1);
        }
        while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '\r' ||
                              v.back() == '\n' || v.back() == '"' || v.back() == '\'')) {
            v.remove_suffix(1);
        }
    };
    trim();
    // `[content_version sha256=abc]` pasted back whole, brackets and label included.
    if (!v.empty() && v.front() == '[' && v.back() == ']') {
        v.remove_prefix(1);
        v.remove_suffix(1);
        trim();
    }
    const auto strip_prefix = [&v, &trim](std::string_view prefix) {
        if (v.size() >= prefix.size() &&
            std::equal(prefix.begin(), prefix.end(), v.begin(), [](char a, char b) {
                return a == static_cast<char>(std::tolower(static_cast<unsigned char>(b)));
            })) {
            v.remove_prefix(prefix.size());
            trim();
            return true;
        }
        return false;
    };
    strip_prefix("content_version");
    // `sha256=`, `sha256:` and a bare `sha256 ` are all the same claim.
    if (strip_prefix("sha256")) {
        if (!v.empty() && (v.front() == '=' || v.front() == ':')) {
            v.remove_prefix(1);
            trim();
        }
    }
    std::string out(v);
    // The digest is hex, so case carries no information -- but the compare is bytewise.
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

std::string Registry::resolve_expected_version(
    const std::string& abs_path, const std::vector<ToolParamValue>& params) const {
    if (const std::string* supplied = get(params, "expected_version");
        supplied != nullptr && !supplied->empty()) {
        return normalize_content_version(*supplied);
    }
    const auto it = read_versions_.find(abs_path);
    return it == read_versions_.end() ? std::string() : it->second;
}

CommitOutcome Registry::commit_write(const fsx::ContainedPath& path,
                                     std::string_view content,
                                     const platform::WritePrecondition& pre) {
    CommitOutcome out;
    // BEFORE the sink and before the directory is made, because neither is worth doing for
    // a write that is not going to happen -- and routing an empty edit through the editor
    // would put an undo step in the operator's history for a change that does not exist.
    //
    // Deliberately NOT conditioned on the file being one this run wrote. A model that
    // re-writes the operator's file verbatim has made exactly as little progress as one
    // that re-writes its own, and the write gate upstream has already had its say about
    // whether the call was allowed to run at all.
    //
    // A no-op does not need a prior-read claim: nothing is replaced. It also does not
    // mint write authority for a different later content -- the ledger is left alone.
    if (already_holds(workspace_fs_, path.relative, content, ctx_.max_read_bytes)) {
        out.write.status = platform::FsStatus::Ok;
        out.unchanged = true;
        return out;
    }
    if (!edit_sink_) {
        // Parent directories are made by the descriptor-rooted primitive. Writing
        // `src/store.py` into an empty workspace is the most ordinary thing an agent
        // does, and every created component is opened with O_NOFOLLOW before continuing.
        //
        // MEASURED, and it is the whole reason this file exists in this form: a real run
        // regenerated 8 KB of correct code SIX times against that error (turns 15-21),
        // then degenerated into sending empty content, and spent 20 of its 40 turns there
        // before stumbling onto `mkdir -p src` in a shell call. Nothing was wrong with
        // the model's work; it could not create a directory.
        out.write = workspace_fs_.write_file_atomic(path.relative, content, true, pre);
        if (out.write.ok()) {
            // Invalidate: a whole-file rewrite must read again before the next overwrite.
            // Failed writes leave the prior observation in place.
            read_versions_.erase(path.absolute);
        }
        return out;
    }
    // The editor is an external sink and receives an absolute path, but it never receives
    // one whose current components include a symlink.
    const fsx::ContainedPath current = workspace_fs_.contained_path(path.relative);
    if (!current.ok()) {
        out.write.status = current.status;
        out.write.error = current.error;
        return out;
    }
    EditIntent intent;
    intent.abs_path = current.absolute;
    intent.new_content = std::string(content);
    intent.expected_version = pre.expected_version;
    intent.expected_absent = pre.expected_absent;
    const EditOutcome o = edit_sink_(intent);
    if (o.applied) {
        out.write.status = platform::FsStatus::Ok;
        read_versions_.erase(current.absolute);
    } else {
        // Prefer Conflict when the editor names a version mismatch; otherwise IoError so
        // the existing write_failure mapping stays honest for refuse/IO cases.
        const bool conflict =
            o.error.find("version") != std::string::npos ||
            o.error.find("expected_absent") != std::string::npos ||
            o.error.find("already exists") != std::string::npos ||
            o.error.find("conflict") != std::string::npos;
        out.write.status =
            conflict ? platform::FsStatus::Conflict : platform::FsStatus::IoError;
        out.write.error = "the editor did not apply the edit: " + o.error;
    }
    return out;
}

namespace {

// One line's worth of text: newlines and tabs folded to spaces, ends trimmed. Empty when
// there was nothing but whitespace.
std::string one_line(const std::string& raw) {
    std::string s;
    s.reserve(raw.size());
    for (const char c : raw) {
        s += (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
    }
    const std::size_t first = s.find_first_not_of(' ');
    if (first == std::string::npos) {
        return {};
    }
    return s.substr(first, s.find_last_not_of(' ') - first + 1);
}

// The key, reduced to something that can be a stable marker in the markdown mirror.
//
// `]` is removed because the mirror writes `- [key] fact` and a key containing the closing
// bracket would make the marker unparseable -- and this marker is what supersession
// matches on, so an ambiguous one silently stops replacing and starts appending, which is
// the exact failure the key exists to prevent.
constexpr std::size_t kMaxKeyBytes = 64;

std::string sanitize_key(const std::string& raw) {
    std::string k;
    for (const char c : one_line(raw)) {
        if (c != ']' && c != '[') {
            k += c;
        }
    }
    if (k.size() > kMaxKeyBytes) {
        k.resize(kMaxKeyBytes);
    }
    return one_line(k);
}

// Replaces the `- [key] ...` line in `body`, or reports that there was none.
// Position is PRESERVED on replacement rather than moving the note to the end: the file is
// trimmed from the front when it overflows, so re-appending on every update would make a
// frequently-corrected note immortal and quietly evict everything else.
bool replace_keyed_line(std::string& body, const std::string& marker,
                        const std::string& line) {
    std::size_t at = body.rfind(marker, 0) == 0 ? 0 : body.find("\n" + marker);
    if (at == std::string::npos) {
        return false;
    }
    if (body.compare(0, marker.size(), marker) != 0) {
        ++at; // step over the newline the search anchored on
    }
    const std::size_t end = body.find('\n', at);
    body.replace(at, end == std::string::npos ? std::string::npos : end - at + 1, line);
    return true;
}

} // namespace

ToolResult Registry::remember_fact(const std::string& raw, const std::string& raw_key) {
    // Folded to ONE LINE, not refused. Dedupe and trimming both work line-wise, so a
    // multi-line note would silently defeat both; folding keeps every character the
    // model wrote and keeps the file's one invariant.
    const std::string fact = one_line(raw);
    if (fact.empty()) {
        return ToolResult::error(ErrorClass::Malformed, false,
                                 "a note cannot be blank");
    }
    const std::string key = sanitize_key(raw_key);

    const fsx::ContainedPath path = workspace_fs_.contained_path(kMemoryFileName);
    if (!path.ok()) {
        return ToolResult::refused(path.error);
    }
    // KEYED NOTES CARRY THEIR KEY IN THE LINE, and that marker is the whole mechanism.
    //
    // Without it the mirror can only append, so a corrected note leaves the stale one in
    // place and the next session's prompt carries BOTH -- which is precisely the failure
    // the durable store exists to end, reintroduced one layer up. The marker is also what
    // a human sees when they open the file, so the supersession is legible rather than
    // implicit.
    const std::string marker = key.empty() ? std::string() : "- [" + key + "] ";
    const std::string line = key.empty() ? "- " + fact + "\n" : marker + fact + "\n";

    // Read wider than the cap so a file that is already oversized can be trimmed back
    // rather than refused -- otherwise one oversized write would wedge the tool forever.
    const fsx::FileContents cur =
        workspace_fs_.read_file_whole(path.relative, kMemoryMaxBytes * 4);
    std::string body = cur.ok() ? cur.bytes : std::string();

    // Exact repeats are the common case: a model that re-reads its own notes re-derives
    // the same conclusion. An unbounded pile of duplicates is how this section stops
    // being worth the tokens it costs.
    if (body.find(line) != std::string::npos) {
        return ToolResult::okay("already noted: " + fact);
    }
    const bool superseded = !key.empty() && replace_keyed_line(body, marker, line);
    if (!superseded) {
        body += line;
    }

    // Trimmed from the FRONT, so a full file drops its OLDEST notes. Truncating the tail
    // instead would freeze the memory at whatever the project learned first and silently
    // discard everything it learned since.
    while (body.size() > kMemoryMaxBytes) {
        const std::size_t nl = body.find('\n');
        if (nl == std::string::npos) {
            break;
        }
        body.erase(0, nl + 1);
    }

    const fsx::WriteResult w =
        workspace_fs_.write_file_atomic(path.relative, body);
    if (!w.ok()) {
        return ToolResult::error(ErrorClass::Transient, true, w.error);
    }

    // MIRRORED INTO THE DURABLE STORE, where a note is searchable and dated.
    //
    // The markdown file is the weaker of the two homes and the header above says why:
    // it is 16 KiB of undated one-liners with no search, trimmed from the front, and a
    // note that stopped being true last week is indistinguishable from one written this
    // morning. PCC's fact kind is strictly better on every one of those counts, and
    // context_recall can reach it.
    //
    // BOTH, not one. The file is what gets prepended to the next session's prompt whether
    // or not anything queries it, and it is the only half a human can open in an editor;
    // switching to the store alone would trade a working feature for a better-designed
    // one. So the file stays the mirror and the store gets the searchable copy.
    //
    // SUPERSEDED WHEN KEYED, APPENDED WHEN NOT, and the store is told the same thing the
    // file was.
    //
    // Store::remember() closes the previous row under a key; Store::append() adds a row
    // that supersedes nothing. A note with no key must take the second path -- inventing a
    // key from the prose would make two unrelated notes that happen to start alike
    // overwrite each other, which is worse than a duplicate.
    //
    // A keyed fact is stored with NO SESSION, and that is the point of keying it. The
    // supersession query is `WHERE session = ? AND key = ?`, so a key recorded under this
    // run's session would only ever supersede notes from this same run -- and the whole
    // reason to correct a note is that it was written by an EARLIER session. Provenance is
    // the price, and it is the right one to pay: an unkeyed note keeps its session and
    // stays a dated observation, a keyed one is a standing claim about the project that
    // belongs to the workspace rather than to whoever last touched it.
    if (context_source_) {
        const Registry::ContextSource src = context_source_();
        if (src.store != nullptr) {
            pcc::Record rec;
            rec.kind = pcc::kind::kFact;
            rec.body = fact;
            rec.key = key;
            rec.session = key.empty() ? src.session : std::string();
            try {
                if (key.empty()) {
                    (void)src.store->append(std::move(rec));
                } else {
                    (void)src.store->remember(std::move(rec));
                }
            } catch (const std::exception&) {
                // The file write already succeeded, so the note is not lost and the
                // model's next session still sees it. Failing the tool here would report
                // a loss that did not happen.
            }
        }
    }
    // Says WHICH of the two things happened, because they are different facts about the
    // project's memory and the model cannot see the file.
    return ToolResult::okay((superseded ? "replaced the earlier note under '" + key +
                                              "': "
                                        : "noted for later sessions: ") +
                            fact);
}

} // namespace lmp::tools
