cmake_minimum_required(VERSION 3.14)
project(mcp-server-cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# macOS ARM64 optimization (though it typically defaults based on the host architecture on macOS, it's good practice to specify or allow defaults)
if (APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS")
endif()

include(FetchContent)

# Fetch nlohmann/json
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_library(mcp_server mcp_server.cpp)
target_include_directories(mcp_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcp_server PUBLIC nlohmann_json::nlohmann_json)

add_executable(mcp_server_demo main.cpp)
target_link_libraries(mcp_server_demo PRIVATE mcp_server)
