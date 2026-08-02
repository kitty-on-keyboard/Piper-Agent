#!/bin/bash

# A simple mock MCP server reading from stdin and writing to stdout line by line.

while read -r line; do
    if [[ -z "$line" ]]; then
        continue
    fi

    # Extract ID and method using simple grep/sed for demonstration
    id=$(echo "$line" | grep -o '"id":[0-9]*' | cut -d':' -f2)
    method=$(echo "$line" | grep -o '"method":"[^"]*"' | cut -d':' -f2 | tr -d '"')

    if [[ "$method" == "initialize" ]]; then
        echo '{"jsonrpc":"2.0","id":'"$id"',"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock-server","version":"1.0.0"}}}'
    elif [[ "$method" == "tools/list" ]]; then
        echo '{"jsonrpc":"2.0","id":'"$id"',"result":{"tools":[{"name":"echo","description":"Echoes the message"}]}}'
    elif [[ "$method" == "tools/call" ]]; then
        msg=$(echo "$line" | grep -o '"message":"[^"]*"' | cut -d':' -f2 | tr -d '"')
        echo '{"jsonrpc":"2.0","id":'"$id"',"result":{"content":[{"type":"text","text":"Mock echo: '"$msg"'"}]}}'
    elif [[ "$method" == "notifications/initialized" ]]; then
        # Just ignore
        :
    fi
done
