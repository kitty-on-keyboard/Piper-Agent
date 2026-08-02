cmake_minimum_required(VERSION 3.20)
project(McpClient VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")

add_library(mcp_client src/mcp_client.cpp)
target_include_directories(mcp_client PUBLIC include)

add_executable(mcp_example examples/main.cpp)
target_link_libraries(mcp_example PRIVATE mcp_client)
