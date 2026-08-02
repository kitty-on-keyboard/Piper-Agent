cmake_minimum_required(VERSION 3.14)
project(mcp-client-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fetch nlohmann_json
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(mcp_client src/mcp_client.cpp)
target_include_directories(mcp_client PUBLIC include)
target_link_libraries(mcp_client PUBLIC nlohmann_json::nlohmann_json)
# Threads
find_package(Threads REQUIRED)
target_link_libraries(mcp_client PRIVATE Threads::Threads)

# Executable for demo
add_executable(mcp_demo src/main.cpp)
target_link_libraries(mcp_demo PRIVATE mcp_client)

# Executable for test server
add_executable(test_server src/test_server.cpp)
target_link_libraries(test_server PRIVATE nlohmann_json::nlohmann_json)
