cmake_minimum_required(VERSION 3.14)
project(mcp-server-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# macOS ARM64 optimization
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
endif()

# Fetch nlohmann_json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Add our library
add_library(mcp_server mcp_server.cpp)
target_include_directories(mcp_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcp_server PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(mcp_server PUBLIC cxx_std_20)

# Add our example driver
add_executable(mcp_server_example main.cpp)
target_link_libraries(mcp_server_example PRIVATE mcp_server)