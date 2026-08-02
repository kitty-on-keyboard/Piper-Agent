cmake_minimum_required(VERSION 3.20)
project(mcp-server-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimize for performance where possible, though users might override
if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Fetch nlohmann_json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

find_package(Threads REQUIRED)

# Library
add_library(mcp_server mcp_server.cpp)
target_include_directories(mcp_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcp_server PUBLIC nlohmann_json::nlohmann_json Threads::Threads)

# Optimization config for ARM64 if on macOS
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported OUTPUT lto_output)
if(lto_supported)
    set_property(TARGET mcp_server PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Executable
add_executable(mcp_server_main main.cpp)
target_link_libraries(mcp_server_main PRIVATE mcp_server)
