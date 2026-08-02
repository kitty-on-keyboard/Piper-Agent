cmake_minimum_required(VERSION 3.14)
project(McpServerCpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# macOS ARM64 target (if on macOS)
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for Mac OS X" FORCE)
endif()

# Threading
find_package(Threads REQUIRED)

# Fetch nlohmann_json
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# Executable
add_executable(mcp_server
    main.cpp
    mcp_server.cpp
)

# Compile options for warnings
if(MSVC)
    target_compile_options(mcp_server PRIVATE /W4 /WX)
else()
    target_compile_options(mcp_server PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Includes and Link Libraries
target_include_directories(mcp_server PRIVATE .)
target_link_libraries(mcp_server PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
