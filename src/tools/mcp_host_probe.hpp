#pragma once

#include <string>

#include "src/tools/mcp_host.hpp"

namespace lmp::tools {

// Parses CLI arguments for mcp_host_probe. Returns false on a usage error.
inline bool parse_mcp_host_probe_args(int argc, char** argv, McpServerConfig& cfg,
                                      std::string& call_tool, std::string& call_args) {
    int i = 1;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--") {
            ++i;
            break;
        }
        if (a == "--trusted") {
            cfg.trusted = true;
        } else if (a == "--call" && i + 1 < argc) {
            call_tool = argv[++i];
        } else if (a == "--args" && i + 1 < argc) {
            call_args = argv[++i];
        } else {
            return false;
        }
    }
    if (i >= argc) {
        return false;
    }
    cfg.command = argv[i++];
    for (; i < argc; ++i) {
        cfg.args.emplace_back(argv[i]);
    }
    return true;
}

} // namespace lmp::tools
