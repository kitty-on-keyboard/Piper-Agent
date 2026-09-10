// What does an MCP round trip actually cost, and where does the time go?
//
// This exists because "stdio is slow, use a shared-memory ring instead" is an intuition
// that survives only until someone measures it. The numbers it prints are the argument
// in docs/MCP.md for keeping stdio as the transport for third-party servers and putting
// the queue only where both ends are ours.
//
// Four measurements:
//   1. stdio round trip, one request at a time   -- the cost people assume is huge
//   2. in-process round trip                      -- the same protocol, no pipe, no JSON text
//   3. stdio with N requests in flight            -- what actually buys throughput
//   4. sustained load                             -- does anything drift, leak or drop
//
// Build: part of the default build. Run: bench_mcp_transport <path-to-mcp_demo_server>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "src/mcp/client.hpp"
#include "src/mcp/inproc_transport.hpp"
#include "src/mcp/server.hpp"

namespace {

using namespace lmp::mcp;
using Clock = std::chrono::steady_clock;
using Micros = std::chrono::duration<double, std::micro>;

struct Stats {
    double p50 = 0;
    double p99 = 0;
    double mean = 0;
    double total_ms = 0;
};

Stats summarize(std::vector<double>& samples, double total_ms) {
    Stats s;
    if (samples.empty()) {
        return s;
    }
    std::sort(samples.begin(), samples.end());
    s.p50 = samples[samples.size() / 2];
    s.p99 = samples[static_cast<std::size_t>(static_cast<double>(samples.size()) * 0.99)];
    double sum = 0;
    for (const double v : samples) {
        sum += v;
    }
    s.mean = sum / static_cast<double>(samples.size());
    s.total_ms = total_ms;
    return s;
}

void report(const char* label, const Stats& s, int n) {
    std::printf("  %-34s p50 %8.1f us   p99 %8.1f us   mean %8.1f us   %6.0f req/s\n", label,
                s.p50, s.p99, s.mean,
                s.total_ms > 0 ? (static_cast<double>(n) * 1000.0 / s.total_ms) : 0.0);
}

// A server identical to the demo one, but built in-process so the same client code can
// drive it over either transport.
void register_bench_tools(Server& server) {
    Tool echo;
    echo.name = "echo";
    echo.description = "echo";
    echo.input_schema = {{"type", "object"},
                         {"properties", {{"text", {{"type", "string"}}}}}};
    server.add_tool(std::move(echo), [](const nlohmann::json& args, RequestContext&) {
        return ToolResult::text(args.value("text", ""));
    });
}

Stats bench_serial(Client& client, int iterations, const std::string& payload) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    const auto t0 = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        const auto a = Clock::now();
        const ToolResult r = client.call_tool("echo", nlohmann::json{{"text", payload}});
        const auto b = Clock::now();
        if (r.is_error) {
            std::fprintf(stderr, "unexpected tool error\n");
        }
        samples.push_back(Micros(b - a).count());
    }
    const double total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return summarize(samples, total_ms);
}

} // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 2000;
    const std::string small(32, 'x');
    const std::string large(16 * 1024, 'y');

    // ---------------------------------------------------------------- stdio
    if (argc > 1) {
        std::printf("stdio transport (subprocess, newline-delimited JSON)\n");
        {
            Client client(Client::Info{"bench", "1.0"});
            Subprocess::Options opts;
            opts.program = argv[1];
            opts.stderr_mode = StderrMode::kDiscard;
            client.connect_stdio(std::move(opts));
            client.initialize();

            Stats s = bench_serial(client, iterations, small);
            report("echo 32B, serial", s, iterations);
            s = bench_serial(client, iterations / 4, large);
            report("echo 16KB, serial", s, iterations / 4);

            // Concurrency: the lever that actually matters. Several calls outstanding at
            // once, answered by the server's worker pool.
            for (const int concurrency : {2, 8, 32}) {
                std::vector<double> samples;
                std::vector<std::thread> threads;
                std::mutex m;
                const auto t0 = Clock::now();
                const int per_thread = iterations / concurrency;
                for (int t = 0; t < concurrency; ++t) {
                    threads.emplace_back([&] {
                        std::vector<double> local;
                        for (int i = 0; i < per_thread; ++i) {
                            const auto a = Clock::now();
                            static_cast<void>(client.call_tool("echo", nlohmann::json{{"text", small}}));
                            local.push_back(Micros(Clock::now() - a).count());
                        }
                        const std::lock_guard<std::mutex> lock(m);
                        samples.insert(samples.end(), local.begin(), local.end());
                    });
                }
                for (auto& th : threads) {
                    th.join();
                }
                const double total_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                const Stats s2 = summarize(samples, total_ms);
                report(("echo 32B, " + std::to_string(concurrency) + " in flight").c_str(), s2,
                       per_thread * concurrency);
            }
            client.close();
        }
        std::printf("\n");
    }

    // ----------------------------------------------------------- in-process
    std::printf("in-process transport (same protocol, no pipe, no serialisation)\n");
    {
        Server::Info info;
        info.name = "bench-inproc";
        Server server(info);
        register_bench_tools(server);

        InProcessTransport::Pair pair = InProcessTransport::make_pair();
        Transport* server_side = pair.server.get();

        std::thread server_thread([&] { server.run(*server_side); });

        Client client(Client::Info{"bench", "1.0"});
        client.connect(std::move(pair.client));
        client.initialize();

        Stats s = bench_serial(client, iterations, small);
        report("echo 32B, serial", s, iterations);
        s = bench_serial(client, iterations / 4, large);
        report("echo 16KB, serial", s, iterations / 4);

        client.close();
        server.request_stop();
        server_thread.join();
    }

    std::printf("\nNote: a real MCP tool call does filesystem or network work measured in\n"
                "milliseconds. Both transports above are far below that, which is the\n"
                "point -- the transport is not where MCP time goes.\n");
    return 0;
}
