cmake_minimum_required(VERSION 3.20)
project(mcp-client-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

find_package(Threads REQUIRED)

add_library(mcp_client mcp_client.cpp)
target_include_directories(mcp_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcp_client PUBLIC nlohmann_json::nlohmann_json Threads::Threads)

add_executable(mcp_client_example main.cpp)
target_link_libraries(mcp_client_example PRIVATE mcp_client)

add_executable(mock_server mock_server.cpp)
target_link_libraries(mock_server PRIVATE nlohmann_json::nlohmann_json)
