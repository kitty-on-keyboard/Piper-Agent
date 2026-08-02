#!/bin/bash

# Simple mock MCP server that reads JSON-RPC requests from stdin and writes responses to stdout

while read -r line; do
    # Skip Content-Length header for mock processing simplicity
    if [[ "$line" == Content-Length:* ]]; then
        # Read the empty line
        read -r empty_line
        # Read the actual JSON content
        read -r line
    fi

    if [[ -z "$line" ]]; then
        continue
    fi

    # Extract ID (might be absent for notifications)
    id=$(echo "$line" | grep -o '"id": *[0-9]*' | grep -o '[0-9]*')
    method=$(echo "$line" | grep -o '"method":"[^"]*"' | cut -d'"' -f4)

    if [[ -z "$id" ]]; then
        # Notification, don't respond
        continue
    fi

    if [[ "$method" == "initialize" ]]; then
        response='{"jsonrpc":"2.0","id":'$id',"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock-server","version":"1.0.0"}}}'
    elif [[ "$method" == "tools/list" ]]; then
        response='{"jsonrpc":"2.0","id":'$id',"result":{"tools":[{"name":"echo_tool","description":"Echoes input"}]}}'
    elif [[ "$method" == "tools/call" ]]; then
        msg=$(echo "$line" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)
        response='{"jsonrpc":"2.0","id":'$id',"result":{"content":[{"type":"text","text":"Echo: '$msg'"}]}}'
    else
        response='{"jsonrpc":"2.0","id":'$id',"error":{"code":-32601,"message":"Method not found"}}'
    fi

    # Add Content-Length framing to the response
    len=$(echo -n "$response" | wc -c)
    echo -ne "Content-Length: $len\r\n\r\n$response"
done
