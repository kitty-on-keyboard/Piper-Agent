// Tests for the in-process transport mechanism.

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "src/mcp/inproc_transport.hpp"
#include "tests/check.hpp"

using namespace lmp::mcp;

namespace {

// Helper to block until a message is received
struct Receiver {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<nlohmann::json> messages;
    bool closed = false;

    Transport::Handlers handlers() {
        Transport::Handlers h;
        h.on_message = [this](const nlohmann::json& msg) {
            std::lock_guard<std::mutex> lock(mutex);
            messages.push_back(msg);
            cv.notify_one();
        };
        h.on_closed = [this]() {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true;
            cv.notify_all();
        };
        return h;
    }

    nlohmann::json wait_for_message() {
        std::unique_lock<std::mutex> lock(mutex);
        bool success = cv.wait_for(lock, std::chrono::seconds(2), [this] { return !messages.empty(); });
        if (!success) {
            return nlohmann::json::object();
        }
        nlohmann::json msg = std::move(messages.front());
        messages.erase(messages.begin());
        return msg;
    }

    bool wait_for_close() {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(2), [this] { return closed; });
    }
};

} // namespace

TEST(make_pair_creates_linked_transports) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();
    CHECK(p.client != nullptr);
    CHECK(p.server != nullptr);
}

TEST(sending_and_receiving_messages) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();

    Receiver client_rx;
    Receiver server_rx;

    p.client->start(client_rx.handlers());
    p.server->start(server_rx.handlers());

    CHECK(p.client->is_open());
    CHECK(p.server->is_open());

    nlohmann::json ping = {{"method", "ping"}};
    nlohmann::json pong = {{"method", "pong"}};

    // Client to server
    CHECK(p.client->send(ping));
    nlohmann::json received = server_rx.wait_for_message();
    CHECK_EQ(received.dump(), ping.dump());

    // Server to client
    CHECK(p.server->send(pong));
    received = client_rx.wait_for_message();
    CHECK_EQ(received.dump(), pong.dump());

    p.client->stop();
    p.server->stop();
}

TEST(stopping_one_side_closes_both) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();

    Receiver client_rx;
    Receiver server_rx;

    p.client->start(client_rx.handlers());
    p.server->start(server_rx.handlers());

    p.client->stop();

    // Client stopped itself, should be closed.
    CHECK(!p.client->is_open());
    // Server should be notified and closed.
    CHECK(!p.server->is_open());

    CHECK(client_rx.wait_for_close());
    CHECK(server_rx.wait_for_close());
}

TEST(send_fails_after_stop) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();

    Receiver client_rx;
    Receiver server_rx;

    p.client->start(client_rx.handlers());
    p.server->start(server_rx.handlers());

    p.client->stop();

    nlohmann::json msg = {{"test", 1}};
    CHECK(!p.client->send(msg));
    CHECK(!p.server->send(msg));

    p.server->stop();
}

TEST(send_fails_if_peer_destroyed) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();

    Receiver client_rx;
    p.client->start(client_rx.handlers());

    // Destroy server
    p.server.reset();

    nlohmann::json msg = {{"test", 1}};
    CHECK(!p.client->send(msg));

    p.client->stop();
}
