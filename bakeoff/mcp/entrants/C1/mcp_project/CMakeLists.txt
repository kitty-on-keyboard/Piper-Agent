cmake_minimum_required(VERSION 3.14)
project(mcp_client CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure macOS ARM64 build when compiled on macOS, otherwise it will just compile for the native host
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_library(mcp_client src/mcp_client.cpp)
target_include_directories(mcp_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mcp_client PUBLIC nlohmann_json::nlohmann_json)

find_package(Threads REQUIRED)
target_link_libraries(mcp_client PUBLIC Threads::Threads)

add_executable(mcp_demo src/main.cpp)
target_link_libraries(mcp_demo PRIVATE mcp_client)
