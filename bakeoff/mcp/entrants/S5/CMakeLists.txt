cmake_minimum_required(VERSION 3.14)
project(mcp-server-cpp VERSION 1.0.0 LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimize for macOS ARM64 execution if on Apple
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for Mac OS X" FORCE)
endif()

include(FetchContent)

# Fetch nlohmann_json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Create the mcp_server static library
add_library(mcp_server STATIC mcp_server.cpp)
target_include_directories(mcp_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcp_server PUBLIC nlohmann_json::nlohmann_json)

# Add thread support explicitly
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(mcp_server PUBLIC Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(mcp_server PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Create the executable mcp_server_demo
add_executable(mcp_server_demo main.cpp)
target_link_libraries(mcp_server_demo PRIVATE mcp_server)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(mcp_server_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
