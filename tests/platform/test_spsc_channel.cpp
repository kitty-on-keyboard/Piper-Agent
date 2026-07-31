#include "src/platform/spsc_channel.hpp"

#include <string>
#include <thread>
#include <vector>

#include "tests/check.hpp"

using lmp::platform::SpscChannel;

TEST(capacity_is_rounded_up_and_reported_back) {
    SpscChannel<std::string> ch(5);
    // The rounded value is what callers get. A caller that assumed 5 and pushed 8 would
    // otherwise be "wrong" against a number nobody published.
    CHECK_EQ(ch.capacity(), std::size_t{8});
}

TEST(fifo_order_is_preserved) {
    SpscChannel<std::string> ch(4);
    CHECK(ch.try_push("a"));
    CHECK(ch.try_push("b"));
    CHECK(ch.try_push("c"));
    std::string out;
    CHECK(ch.try_pop(out));
    CHECK_EQ(out, std::string("a"));
    CHECK(ch.try_pop(out));
    CHECK_EQ(out, std::string("b"));
    CHECK(ch.try_pop(out));
    CHECK_EQ(out, std::string("c"));
    CHECK(!ch.try_pop(out));
}

TEST(full_channel_refuses_rather_than_overwrites) {
    SpscChannel<int> ch(2);
    CHECK(ch.try_push(1));
    CHECK(ch.try_push(2));
    CHECK(!ch.try_push(3));
    CHECK_EQ(ch.size_approx(), std::size_t{2});
    int v = 0;
    CHECK(ch.try_pop(v));
    CHECK_EQ(v, 1);
    CHECK(ch.try_push(3)); // space freed
}

TEST(closed_is_not_empty_and_drained_needs_both) {
    // The last request before stdin EOF lives here. Treating closed as "stop reading"
    // drops it, and the drop looks like the extension never sent the message.
    SpscChannel<std::string> ch(4);
    CHECK(ch.try_push("last-request"));
    ch.close();
    CHECK(ch.closed());
    CHECK(!ch.empty());
    CHECK(!ch.drained());

    std::string out;
    CHECK(ch.try_pop(out));
    CHECK_EQ(out, std::string("last-request"));
    CHECK(ch.drained());
}

TEST(values_are_owned_not_viewed) {
    SpscChannel<std::string> ch(4);
    std::string src(200, 'x'); // long enough to defeat the small-string buffer
    CHECK(ch.try_push(std::move(src)));
    src = "clobbered";
    std::string out;
    CHECK(ch.try_pop(out));
    CHECK_EQ(out.size(), std::size_t{200});
    CHECK_EQ(out[0], 'x');
}

TEST(concurrent_producer_and_consumer_lose_nothing) {
    constexpr int kN = 20000;
    SpscChannel<std::string> ch(64);
    std::vector<std::string> received;
    received.reserve(kN);

    std::thread producer([&] {
        for (int i = 0; i < kN; ++i) {
            std::string msg = "m" + std::to_string(i);
            while (!ch.try_push(std::move(msg))) {
                std::this_thread::yield();
            }
        }
        ch.close();
    });

    std::string out;
    while (!ch.drained()) {
        if (ch.try_pop(out)) {
            received.push_back(out);
        }
    }
    producer.join();

    CHECK_EQ(received.size(), static_cast<std::size_t>(kN));
    bool in_order = true;
    for (int i = 0; i < kN; ++i) {
        if (received[static_cast<std::size_t>(i)] != "m" + std::to_string(i)) {
            in_order = false;
            break;
        }
    }
    CHECK(in_order);
}
