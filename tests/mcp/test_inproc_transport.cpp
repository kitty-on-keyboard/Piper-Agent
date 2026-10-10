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

TEST(fifo_message_ordering) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();

    Receiver server_rx;
    p.client->start({});
    p.server->start(server_rx.handlers());

    CHECK(p.client->send({{"seq", 1}}));
    CHECK(p.client->send({{"seq", 2}}));
    CHECK(p.client->send({{"seq", 3}}));

    nlohmann::json msg1 = server_rx.wait_for_message();
    nlohmann::json msg2 = server_rx.wait_for_message();
    nlohmann::json msg3 = server_rx.wait_for_message();

    CHECK_EQ(msg1["seq"].get<int>(), 1);
    CHECK_EQ(msg2["seq"].get<int>(), 2);
    CHECK_EQ(msg3["seq"].get<int>(), 3);

    p.client->stop();
    p.server->stop();
}

TEST(queued_messages_delivered_before_close) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();

    Receiver server_rx;
    p.client->start({});
    p.server->start(server_rx.handlers());

    // Send messages then immediately stop client
    CHECK(p.client->send({{"msg", "first"}}));
    CHECK(p.client->send({{"msg", "second"}}));
    p.client->stop();

    // Server should receive both queued messages before getting closed notification
    nlohmann::json msg1 = server_rx.wait_for_message();
    nlohmann::json msg2 = server_rx.wait_for_message();

    CHECK_EQ(msg1["msg"].get<std::string>(), "first");
    CHECK_EQ(msg2["msg"].get<std::string>(), "second");
    CHECK(server_rx.wait_for_close());
}

TEST(multiple_stop_calls_and_unstarted_stop) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();

    // Stopping unstarted transports should be safe
    p.client->stop();
    p.server->stop();

    // Multiple calls to stop should be idempotent
    p.client->stop();
    p.server->stop();
}

TEST(send_fails_before_start) {
    InProcessTransport::Pair p = InProcessTransport::make_pair();

    nlohmann::json msg = {{"hello", "world"}};
    // Neither client nor server started yet
    CHECK(!p.client->send(msg));
    CHECK(!p.server->send(msg));

    p.client->stop();
    p.server->stop();
}
