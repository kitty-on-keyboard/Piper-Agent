#pragma once
//
// In-process transport: a linked pair of endpoints that hand JSON values straight to
// each other across a queue, with no serialisation and no syscall.
//
// WHY THIS EXISTS, AND WHAT IT IS NOT FOR
//
// stdio cannot be replaced by a shared-memory ring for real MCP servers, and the reason
// is not performance. An MCP server is a separate process, usually written in
// TypeScript or Python and launched through npx or uvx. A lock-free queue is only
// reachable by a peer that was compiled against the same queue, so the moment we invent
// a transport, every server that speaks it is a server we wrote. That trades away the
// entire point of MCP -- the ecosystem of servers we did not write -- to save time that
// measurement says is not being spent (bakeoff/mcp/bench_transport.cpp: a stdio round
// trip is tens of microseconds against tool calls that take milliseconds).
//
// So the ring goes where both ends genuinely are ours: LM_Pipe's own tools, exposed
// through the same Server API as everything else, but reached without leaving the
// process. Same protocol, same handlers, same tests; no pipe, no dump(), no parse().
// A tool registered once is reachable both ways, and the agent above does not know
// which it got.
//
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "src/mcp/transport.hpp"

namespace lmp::mcp {

class InProcessTransport final : public Transport {
public:
    struct Pair {
        std::unique_ptr<InProcessTransport> client;
        std::unique_ptr<InProcessTransport> server;
    };

    // Two endpoints wired to each other. Neither owns the other; both must outlive the
    // session, and stopping either closes both.
    //
    // Note the asymmetry, which is easy to get wrong: Client::connect takes ownership
    // of its half, but Server::run(Transport&) does not. Move the server half into
    // Server::run(std::unique_ptr<Transport>), or keep it in a member that outlives the
    // server thread -- holding a raw pointer to a `Pair` local is a use-after-free.
    [[nodiscard]] static Pair make_pair();

    ~InProcessTransport() override;

    void start(Handlers handlers) override;
    bool send(const nlohmann::json& message) override;
    void stop() override;
    [[nodiscard]] bool is_open() const override;

private:
    InProcessTransport() = default;

    // Delivery happens on the endpoint's own thread rather than inline on the sender's.
    // Inline delivery would make send() re-entrant into the peer's handler, so a server
    // handler replying to a client that is still inside its own dispatch would recurse
    // through both stacks -- and would deadlock the moment either side holds a lock
    // across a send.
    void deliver_loop();
    void enqueue(const nlohmann::json& message);
    void close_peer();

    InProcessTransport* peer_ = nullptr;

    Handlers handlers_;
    std::thread delivery_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<nlohmann::json> queue_;
    bool open_ = false;
    bool stopping_ = false;
    bool started_ = false;
};

} // namespace lmp::mcp
