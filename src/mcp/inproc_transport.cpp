#include "src/mcp/inproc_transport.hpp"

namespace lmp::mcp {

InProcessTransport::Pair InProcessTransport::make_pair() {
    Pair p;
    p.client = std::unique_ptr<InProcessTransport>(new InProcessTransport());
    p.server = std::unique_ptr<InProcessTransport>(new InProcessTransport());
    p.client->peer_ = p.server.get();
    p.server->peer_ = p.client.get();
    return p;
}

InProcessTransport::~InProcessTransport() {
    stop();
    // Unlink, so a peer that outlives us cannot enqueue into freed state.
    if (peer_ != nullptr) {
        const std::lock_guard<std::mutex> lock(peer_->mutex_);
        peer_->peer_ = nullptr;
    }
}

void InProcessTransport::start(Handlers handlers) {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        handlers_ = std::move(handlers);
        open_ = true;
        started_ = true;
    }
    delivery_ = std::thread([this] { deliver_loop(); });
}

bool InProcessTransport::is_open() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return open_;
}

bool InProcessTransport::send(const nlohmann::json& message) {
    InProcessTransport* peer = nullptr;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!open_) {
            return false;
        }
        peer = peer_;
    }
    if (peer == nullptr) {
        return false;
    }
    peer->enqueue(message);
    return true;
}

void InProcessTransport::enqueue(const nlohmann::json& message) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
        return;
    }
    // The copy here is the whole cost of the transport: one JSON value, no text form.
    queue_.push_back(message);
    cv_.notify_one();
}

void InProcessTransport::stop() {
    bool join_needed = false;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            join_needed = delivery_.joinable();
        } else {
            stopping_ = true;
            open_ = false;
            join_needed = delivery_.joinable();
            cv_.notify_all();
        }
    }

    if (join_needed && delivery_.get_id() != std::this_thread::get_id()) {
        delivery_.join();
    }

    close_peer();
}

void InProcessTransport::close_peer() {
    InProcessTransport* peer = nullptr;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        peer = peer_;
    }
    if (peer == nullptr) {
        return;
    }
    // Wake the peer so its owner's run() returns instead of waiting on a session whose
    // other half is gone.
    const std::lock_guard<std::mutex> lock(peer->mutex_);
    if (!peer->stopping_) {
        peer->stopping_ = true;
        peer->open_ = false;
        peer->cv_.notify_all();
    }
}

void InProcessTransport::deliver_loop() {
    for (;;) {
        nlohmann::json msg;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });

            // Deliver what is already queued even while stopping: a request that was
            // accepted deserves to reach its handler.
            if (queue_.empty()) {
                if (stopping_) {
                    break;
                }
                continue;
            }
            msg = std::move(queue_.front());
            queue_.pop_front();
        }

        if (handlers_.on_message) {
            handlers_.on_message(msg);
        }
    }

    if (handlers_.on_closed) {
        handlers_.on_closed();
    }
}

} // namespace lmp::mcp
