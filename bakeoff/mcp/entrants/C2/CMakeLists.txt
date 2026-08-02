cmake_minimum_required(VERSION 3.14)
project(mcp_client VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS ARM64 specific flags if needed
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for Mac OS X" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_library(mcp_client mcp_client.cpp)
target_include_directories(mcp_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcp_client PUBLIC nlohmann_json::nlohmann_json)

add_executable(mcp_demo main.cpp)
target_link_libraries(mcp_demo PRIVATE mcp_client)
