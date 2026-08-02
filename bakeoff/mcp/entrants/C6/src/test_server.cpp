#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        try {
            json req = json::parse(line);

            if (req.contains("method")) {
                std::string method = req["method"];
                json res;

                if (req.contains("id")) {
                    res["jsonrpc"] = "2.0";
                    res["id"] = req["id"];

                    if (method == "initialize") {
                        res["result"] = {
                            {"protocolVersion", "2024-11-05"},
                            {"serverInfo", {
                                {"name", "test-server"},
                                {"version", "1.0.0"}
                            }},
                            {"capabilities", {
                                {"tools", {}}
                            }}
                        };
                    } else if (method == "tools/list") {
                        res["result"] = {
                            {"tools", {
                                {
                                    {"name", "echo"},
                                    {"description", "Echoes the message back"},
                                    {"inputSchema", {
                                        {"type", "object"},
                                        {"properties", {
                                            {"message", {{"type", "string"}}}
                                        }},
                                        {"required", {"message"}}
                                    }}
                                }
                            }}
                        };
                    } else if (method == "tools/call") {
                        std::string tool_name = req["params"]["name"];
                        if (tool_name == "echo") {
                            std::string message = req["params"]["arguments"]["message"];
                            res["result"] = {
                                {"content", {
                                    {
                                        {"type", "text"},
                                        {"text", "Echo: " + message}
                                    }
                                }},
                                {"isError", false}
                            };
                        } else {
                            res["error"] = {
                                {"code", -32601},
                                {"message", "Method not found"}
                            };
                        }
                    } else {
                        res["error"] = {
                            {"code", -32601},
                            {"message", "Method not found"}
                        };
                    }

                    std::cout << res.dump() << std::endl;
                }
            }
        } catch (const std::exception& e) {
            // Ignore parse errors or send an error response if needed
        }
    }
    return 0;
}
