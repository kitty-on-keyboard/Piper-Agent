// PCC over MCP: the context store as a server any MCP client can drive.
//
// WHY A SERVER AND NOT JUST A LIBRARY
//   The store is most useful when the agent that wrote a fact is not the only one that
//   can read it. Run this from Cursor or Antigravity against the same database the
//   sidecar uses and a conclusion reached in one is available in the other a second
//   later. That is also why the tools are named for what they DO to context rather than
//   for this codebase -- a client that has never heard of LM_Pipe should still be able to
//   tell what `context_recall` is for.
//
// Nothing is written to stdout except by the transport; diagnostics go to stderr. A
// stray printf here desyncs the client's framer.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/mcp/server.hpp"
#include "src/mcp/transport.hpp"
#include "src/pcc/recall.hpp"
#include "src/pcc/store.hpp"

namespace {

using lmp::mcp::RequestContext;
using lmp::mcp::Resource;
using lmp::mcp::ResourceContents;
using lmp::mcp::ResourceTemplate;
using lmp::mcp::Server;
using lmp::mcp::Tool;
using lmp::mcp::ToolResult;
using lmp::pcc::Item;
using lmp::pcc::Record;
using lmp::pcc::Store;

std::string arg_str(const nlohmann::json& args, const char* key, std::string fallback = {}) {
    if (args.is_object() && args.contains(key) && args[key].is_string()) {
        return args[key].get<std::string>();
    }
    return fallback;
}

std::int64_t arg_int(const nlohmann::json& args, const char* key, std::int64_t fallback) {
    if (args.is_object() && args.contains(key) && args[key].is_number()) {
        return args[key].get<std::int64_t>();
    }
    return fallback;
}

nlohmann::json schema(nlohmann::json properties, std::vector<std::string> required) {
    return {{"type", "object"},
            {"properties", std::move(properties)},
            {"required", required}};
}

const nlohmann::json kStringProp = {{"type", "string"}};
const nlohmann::json kIntProp = {{"type", "integer"}};

nlohmann::json item_json(const Item& item) {
    return {{"uri", lmp::pcc::item_uri(item)}, {"id", item.id},
            {"kind", item.kind},               {"key", item.key},
            {"title", item.title},             {"valid_from", item.valid_from},
            {"valid_to", item.valid_to},       {"superseded", item.valid_to != lmp::pcc::kOpenEnded},
            {"hash", item.hash}};
}

ToolResult recall_result(const lmp::pcc::Recall& r) {
    ToolResult out = ToolResult::text(r.text.empty() ? "no matching context" : r.text);
    // Structured alongside the text: a model reads the packed prose, a program reads the
    // URIs and decides what to fetch next. Both from one call.
    nlohmann::json entries = nlohmann::json::array();
    for (const lmp::pcc::RecallEntry& e : r.entries) {
        nlohmann::json j = item_json(e.item);
        j["included"] = e.included;
        j["score"] = e.score;
        entries.push_back(std::move(j));
    }
    out.structured = {{"tokens_used", r.tokens_used},
                      {"included", r.included},
                      {"pointers_only", r.pointers_only},
                      {"entries", std::move(entries)}};
    return out;
}

void add_recall_tools(Server& server, Store& store) {
    Tool recall;
    recall.name = "context_recall";
    recall.title = "Recall context";
    recall.description =
        "Search everything this agent has stored -- past turns, durable facts, artifact "
        "revisions -- and return as much as fits in a token budget, plus a URI for "
        "anything that did not fit. Ranked by relevance fused with recency. Superseded "
        "facts are excluded unless as_of_valid asks for an earlier moment.";
    recall.input_schema = schema({{"query", kStringProp},
                                  {"token_budget", kIntProp},
                                  {"session", kStringProp},
                                  {"as_of_valid", kIntProp}},
                                 {"query"});
    server.add_tool(std::move(recall), [&store](const nlohmann::json& args, RequestContext&) {
        lmp::pcc::RecallRequest req;
        req.query = arg_str(args, "query");
        req.session = arg_str(args, "session");
        req.token_budget = static_cast<std::size_t>(arg_int(args, "token_budget", 1500));
        req.as_of.valid = arg_int(args, "as_of_valid", lmp::pcc::kNow);
        return recall_result(lmp::pcc::recall(store, req));
    });

    Tool rehydrate;
    rehydrate.name = "context_rehydrate";
    rehydrate.title = "Rehydrate a compacted span";
    rehydrate.description =
        "Given the event range printed in a compacted summary ('events 40-91'), return "
        "the full text of the turns it was made from, newest first, packed to a token "
        "budget. Use this when a summary mentions something you now need the detail of.";
    rehydrate.input_schema = schema({{"first_event", kIntProp},
                                     {"last_event", kIntProp},
                                     {"token_budget", kIntProp},
                                     {"session", kStringProp}},
                                    {"first_event", "last_event"});
    server.add_tool(std::move(rehydrate), [&store](const nlohmann::json& args, RequestContext&) {
        const auto first = static_cast<std::uint64_t>(arg_int(args, "first_event", 0));
        const auto last = static_cast<std::uint64_t>(arg_int(args, "last_event", 0));
        const auto budget = static_cast<std::size_t>(arg_int(args, "token_budget", 4000));
        return recall_result(
            lmp::pcc::rehydrate(store, first, last, budget, arg_str(args, "session")));
    });
}

void add_memory_tools(Server& server, Store& store) {
    Tool remember;
    remember.name = "context_remember";
    remember.title = "Remember a fact";
    remember.description =
        "Store a durable fact under a stable key. Writing the same key again supersedes "
        "the old value rather than overwriting it: the previous value stays queryable "
        "with its own valid time, so a fact that later turns out to be wrong can be "
        "traced instead of vanishing. Re-storing an identical body is a no-op.";
    remember.input_schema = schema({{"key", kStringProp},
                                    {"body", kStringProp},
                                    {"title", kStringProp},
                                    {"session", kStringProp}},
                                   {"key", "body"});
    server.add_tool(std::move(remember), [&store](const nlohmann::json& args, RequestContext&) {
        Record rec;
        rec.key = arg_str(args, "key");
        rec.body = arg_str(args, "body");
        rec.title = arg_str(args, "title", rec.key);
        rec.session = arg_str(args, "session");
        rec.kind = lmp::pcc::kind::kFact;
        if (rec.key.empty() || rec.body.empty()) {
            return ToolResult::failure("key and body are both required");
        }
        return ToolResult::text("pcc://item/" + std::to_string(store.remember(std::move(rec))));
    });

    Tool forget;
    forget.name = "context_forget";
    forget.title = "Mark a fact no longer true";
    forget.description =
        "Close a fact's validity without replacing it -- 'this stopped being true'. The "
        "history stays queryable; only the present changes. This is not a delete, and "
        "there is deliberately no delete.";
    forget.input_schema = schema({{"key", kStringProp}, {"session", kStringProp}}, {"key"});
    server.add_tool(std::move(forget), [&store](const nlohmann::json& args, RequestContext&) {
        const bool closed = store.forget(arg_str(args, "key"), arg_str(args, "session"));
        return closed ? ToolResult::text("closed")
                      : ToolResult::failure("no current fact under that key");
    });

    Tool history;
    history.name = "context_history";
    history.title = "History of a key";
    history.description =
        "Every value ever stored under a key, oldest first, each with the window it was "
        "believed true. Answers 'what did I think at the time, and when did I change my "
        "mind' -- the question you need when a run went wrong an hour ago.";
    history.input_schema = schema({{"key", kStringProp}, {"session", kStringProp}}, {"key"});
    server.add_tool(std::move(history), [&store](const nlohmann::json& args, RequestContext&) {
        const std::vector<Item> items =
            store.history(arg_str(args, "key"), arg_str(args, "session"));
        std::string text;
        nlohmann::json rows = nlohmann::json::array();
        for (const Item& item : items) {
            text += lmp::pcc::item_uri(item) + "  " +
                    (item.valid_to == lmp::pcc::kOpenEnded ? "[current] " : "[superseded] ") +
                    item.body + "\n";
            rows.push_back(item_json(item));
        }
        ToolResult out = ToolResult::text(text.empty() ? "no history for that key" : text);
        out.structured = {{"revisions", std::move(rows)}};
        return out;
    });
}

void add_artifact_tools(Server& server, Store& store) {
    Tool put;
    put.name = "context_put_artifact";
    put.title = "Store an artifact revision";
    put.description =
        "Store a revision of a named artifact (a file path, a generated document). "
        "Identical content is stored once however often it arrives, and a revision is "
        "delta-compressed against its predecessor automatically -- pass the same path "
        "each time and the store handles it.";
    put.input_schema = schema({{"path", kStringProp},
                               {"content", kStringProp},
                               {"session", kStringProp}},
                              {"path", "content"});
    server.add_tool(std::move(put), [&store](const nlohmann::json& args, RequestContext&) {
        const std::string path = arg_str(args, "path");
        const std::string content = arg_str(args, "content");
        if (path.empty()) {
            return ToolResult::failure("path is required");
        }
        const std::string session = arg_str(args, "session");
        // The predecessor is looked up rather than asked for: a caller that has to track
        // the base hash will eventually pass a stale one, and the delta chain is the
        // store's business, not the model's.
        const std::optional<Item> previous = store.current(path, {}, session);
        Record rec;
        rec.key = path;
        rec.title = path;
        rec.session = session;
        rec.kind = lmp::pcc::kind::kArtifact;
        const std::int64_t id = store.put_artifact(
            std::move(rec), content, previous.has_value() ? previous->hash : std::string());
        const std::optional<Item> stored = store.get(id);
        ToolResult out = ToolResult::text("pcc://item/" + std::to_string(id));
        if (stored.has_value()) {
            out.structured = {{"uri", lmp::pcc::item_uri(*stored)},
                              {"hash", stored->hash},
                              {"unchanged", previous.has_value() &&
                                                previous->hash == stored->hash}};
        }
        return out;
    });

    Tool diff;
    diff.name = "context_artifact_diff";
    diff.title = "Diff two artifact revisions";
    diff.description =
        "Unified diff between two stored revisions, by content hash. Cheaper than "
        "recalling both and comparing them, and it is the answer to 'what did I change' "
        "when the edit happened before the last compaction.";
    diff.input_schema = schema({{"from_hash", kStringProp}, {"to_hash", kStringProp}},
                               {"from_hash", "to_hash"});
    server.add_tool(std::move(diff), [&store](const nlohmann::json& args, RequestContext&) {
        const std::optional<std::string> text = store.artifacts().diff(
            arg_str(args, "from_hash"), arg_str(args, "to_hash"));
        if (!text.has_value()) {
            return ToolResult::failure("one or both hashes are not in the store");
        }
        return ToolResult::text(text->empty() ? "identical" : *text);
    });
}

void add_resources(Server& server, Store& store) {
    ResourceTemplate item_tmpl;
    item_tmpl.uri_template = "pcc://item/{id}";
    item_tmpl.name = "context item";
    item_tmpl.description = "One stored item by id, as returned by context_recall.";
    item_tmpl.mime_type = "text/plain";
    server.add_resource_template(
        std::move(item_tmpl), [&store](const std::string& uri, RequestContext&) {
            const std::size_t slash = uri.rfind('/');
            const std::int64_t id =
                slash == std::string::npos ? 0 : std::atoll(uri.c_str() + slash + 1);
            const std::optional<Item> item = store.get(id);
            if (!item.has_value()) {
                return std::vector<ResourceContents>{};
            }
            // An artifact's body is its searchable projection, not its content, so the
            // resource read resolves the CAS. A caller reading pcc://item/N for an
            // artifact wants the file, not a line naming its hash.
            const std::optional<std::string> content = store.artifact_content(*item);
            return std::vector<ResourceContents>{
                ResourceContents::from_text(uri, content.value_or(item->body))};
        });

    Resource stats;
    stats.uri = "pcc://stats";
    stats.name = "store statistics";
    stats.description = "Row and blob counts, and how much the delta chains are saving.";
    stats.mime_type = "application/json";
    server.add_resource(std::move(stats), [&store](const std::string& uri, RequestContext&) {
        const lmp::pcc::StoreStats s = store.stats();
        const nlohmann::json j = {{"items", s.items},
                                  {"current_items", s.current_items},
                                  {"sessions", s.sessions},
                                  {"blobs", s.blobs.blobs},
                                  {"logical_bytes", s.blobs.logical_bytes},
                                  {"stored_bytes", s.blobs.stored_bytes},
                                  {"whole_blobs", s.blobs.whole_blobs},
                                  {"delta_blobs", s.blobs.delta_blobs}};
        return std::vector<ResourceContents>{
            ResourceContents::from_text(uri, j.dump(2), "application/json")};
    });
}

} // namespace

int main(int argc, char** argv) {
    std::string path = "pcc.db";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--db" && i + 1 < argc) {
            path = argv[++i];
        }
    }

    Server::Info info;
    info.name = "pcc";
    info.version = "1.0.0";
    info.instructions =
        "Durable context for an agent: past turns, superseding facts, and versioned "
        "artifacts. Prefer context_recall over guessing -- it costs one call and returns "
        "only what fits the budget you give it. Facts are never deleted, only superseded, "
        "so context_history will tell you what you used to believe and when that changed.";

    try {
        Store store(path);
        Server server(std::move(info));
        add_recall_tools(server, store);
        add_memory_tools(server, store);
        add_artifact_tools(server, store);
        add_resources(server, store);
        server.run(std::make_unique<lmp::mcp::StdioTransport>());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pcc: %s\n", e.what());
        return 1;
    }
    return 0;
}
