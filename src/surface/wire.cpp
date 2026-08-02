#include "src/surface/wire.hpp"

#include <cstdio>
#include <mutex>

#include "src/platform/event_log.hpp"

namespace lmp::surface::wire {
namespace {

std::mutex& stdout_lock() {
    static std::mutex m;
    return m;
}

} // namespace

void write_line(const std::string& line) {
    const std::lock_guard<std::mutex> guard(stdout_lock());
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string json_escape(std::string_view in) {
    std::string out;
    (void)platform::append_json_string(out, in);
    return out;
}

void reply_result(const std::string& id, const std::string& body) {
    write_line(R"({"jsonrpc":"2.0","id":")" + id + R"(","result":)" + body + "}");
}

void reply_error(const std::string& id, const std::string& message) {
    write_line(R"({"jsonrpc":"2.0","id":")" + id + R"(","error":{"code":-32000,)" +
               R"("message":)" + json_escape(message) + "}}");
}

} // namespace lmp::surface::wire
