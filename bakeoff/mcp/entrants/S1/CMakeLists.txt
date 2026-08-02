cmake_minimum_required(VERSION 3.20)
project(MCPServerCPP VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
endif()

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Threading (std::jthread)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(mcp_server mcp_server.cpp)
target_link_libraries(mcp_server PUBLIC nlohmann_json::nlohmann_json Threads::Threads)
target_include_directories(mcp_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(mcp_server_driver main.cpp)
target_link_libraries(mcp_server_driver PRIVATE mcp_server)
