cmake_minimum_required(VERSION 3.20)
project(mcp_server_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for macOS ARM64 per requirements
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
endif()

# Fetch nlohmann_json
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# Create core library
add_library(mcp_server mcp_server.cpp)
target_link_libraries(mcp_server PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(mcp_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create driver executable
add_executable(mcp_server_exe main.cpp)
target_link_libraries(mcp_server_exe PRIVATE mcp_server)
# Output binary named "mcp_server"
set_target_properties(mcp_server_exe PROPERTIES OUTPUT_NAME "mcp_server")
