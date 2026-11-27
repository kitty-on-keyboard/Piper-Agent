#include "src/mcp/inproc_transport.hpp"

namespace lmp::mcp {

InProcessTransport::Pair InProcessTransport::make_pair() {
    Pair p;
    p.client = std::unique_ptr<InProcessTransport>(new InProcessTransport());
    p.server = std::unique_ptr<InProcessTransport>(new InProcessTransport());
    p.client->peer_ = p.server->state_;
    p.server->peer_ = p.client->state_;
    return p;
}

InProcessTransport::~InProcessTransport() {
    stop();
}

void InProcessTransport::start(Handlers handlers) {
    {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        state_->handlers = std::move(handlers);
        state_->open = true;
    }
    const std::lock_guard<std::mutex> jlock(join_mutex_);
    delivery_ = std::thread([state = state_] { deliver_loop(state); });
}

bool InProcessTransport::is_open() const {
    const std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->open;
}

bool InProcessTransport::send(const nlohmann::json& message) {
    std::shared_ptr<State> peer;
    {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->open) {
            return false;
        }
        peer = peer_.lock();
    }
    if (!peer) {
        return false;
    }
    enqueue(peer, message);
    return true;
}

void InProcessTransport::enqueue(const std::shared_ptr<State>& target,
                                 const nlohmann::json& message) {
    const std::lock_guard<std::mutex> lock(target->mutex);
    if (target->stopping) {
        return;
    }
    // The copy here is the whole cost of the transport: one JSON value, no text form.
    target->queue.push_back(message);
    target->cv.notify_one();
}

void InProcessTransport::stop() {
    {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->stopping) {
            state_->stopping = true;
            state_->open = false;
            state_->cv.notify_all();
        }
    }

    {
        const std::lock_guard<std::mutex> jlock(join_mutex_);
        if (delivery_.joinable() && delivery_.get_id() != std::this_thread::get_id()) {
            delivery_.join();
        }
    }

    close_peer();
}

void InProcessTransport::close_peer() {
    const std::shared_ptr<State> peer = peer_.lock();
    if (!peer) {
        return;
    }
    // Wake the peer so its owner's run() returns instead of waiting on a session whose
    // other half is gone.
    const std::lock_guard<std::mutex> lock(peer->mutex);
    if (!peer->stopping) {
        peer->stopping = true;
        peer->open = false;
        peer->cv.notify_all();
    }
}

void InProcessTransport::deliver_loop(const std::shared_ptr<State>& state) {
    for (;;) {
        nlohmann::json msg;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->cv.wait(lock, [&state] { return state->stopping || !state->queue.empty(); });

            // Deliver what is already queued even while stopping: a request that was
            // accepted deserves to reach its handler.
            if (state->queue.empty()) {
                if (state->stopping) {
                    break;
                }
                continue;
            }
            msg = std::move(state->queue.front());
            state->queue.pop_front();
        }

        if (state->handlers.on_message) {
            state->handlers.on_message(msg);
        }
    }

    if (state->handlers.on_closed) {
        state->handlers.on_closed();
    }
}

} // namespace lmp::mcp
