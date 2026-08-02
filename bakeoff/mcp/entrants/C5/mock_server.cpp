#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void handle_request(const std::string& line) {
    try {
        auto req = json::parse(line);
        if (req.contains("id") && req.contains("method")) {
            std::string method = req["method"];
            json response = {
                {"jsonrpc", "2.0"},
                {"id", req["id"]}
            };

            if (method == "initialize") {
                response["result"] = {
                    {"protocolVersion", "2024-11-05"},
                    {"capabilities", json::object()},
                    {"serverInfo", {
                        {"name", "mock-mcp-server"},
                        {"version", "1.0.0"}
                    }}
                };
            } else if (method == "tools/list") {
                response["result"] = {
                    {"tools", {
                        {
                            {"name", "echo"},
                            {"description", "Echoes a message"},
                            {"inputSchema", {
                                {"type", "object"},
                                {"properties", {
                                    {"message", {
                                        {"type", "string"},
                                        {"description", "Message to echo"}
                                    }}
                                }},
                                {"required", {"message"}}
                            }}
                        }
                    }}
                };
            } else if (method == "tools/call") {
                std::string name = req["params"]["name"];
                if (name == "echo") {
                    std::string msg = req["params"]["arguments"]["message"];
                    response["result"] = {
                        {"content", {
                            {
                                {"type", "text"},
                                {"text", "Echo: " + msg}
                            }
                        }},
                        {"isError", false}
                    };
                } else {
                    response["error"] = {
                        {"code", -32601},
                        {"message", "Method not found"}
                    };
                }
            } else {
                response["error"] = {
                    {"code", -32601},
                    {"message", "Method not found"}
                };
            }
            std::cout << response.dump() << "\n" << std::flush;
        } else if (req.contains("method") && !req.contains("id")) {
            // Notification
            if (req["method"] == "notifications/initialized") {
                // Ignore or log
            }
        }
    } catch (...) {
        // Parse error or invalid request, ignore for simple mock
    }
}

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        // Basic Content-Length skip if sent
        if (line.starts_with("Content-Length:")) {
            std::getline(std::cin, line); // empty line
            std::getline(std::cin, line); // actual payload
        }
        handle_request(line);
    }
    return 0;
}
