# Interop checks against real MCP implementations

Our own conformance harness (`../conform.py`) can only find bugs we thought to look for.
These two checks use software we did not write, in both directions. They are not part of
`ctest -L gate` because they need npm and a network fetch; run them by hand when the
protocol layer changes.

    npm install @modelcontextprotocol/sdk \
                @modelcontextprotocol/server-everything \
                @modelcontextprotocol/server-filesystem

**Official TypeScript SDK client -> our server.** 18 assertions: handshake,
instructions, tool schemas, structuredContent, tool-failure-vs-protocol-error, progress,
resources, prompts, completion, ping, logging, error codes.

    node drive_our_server.mjs ../../../build/src/mcp/mcp_demo_server

**Our client -> official servers.**

    ../../../build/src/mcp/mcp_probe -- node node_modules/@modelcontextprotocol/server-everything/dist/index.js
    ../../../build/src/mcp/mcp_probe --call read_text_file --args '{"path":"/tmp/x"}' \
        -- node node_modules/@modelcontextprotocol/server-filesystem/dist/index.js /tmp

Last run 2026-08-02 against sdk 1.30.0, server-everything 2026.7.4,
server-filesystem 0.2.0, server-memory 0.6.3: 18/18 and clean both ways.
See docs/MCP.md.
