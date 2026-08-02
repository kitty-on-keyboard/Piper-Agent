cmake_minimum_required(VERSION 3.20)
project(mcp-client-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_library(mcp_client src/mcp_client.cpp)
target_include_directories(mcp_client PUBLIC include)
target_link_libraries(mcp_client PUBLIC nlohmann_json::nlohmann_json)

add_executable(mcp_example src/main.cpp)
target_link_libraries(mcp_example PRIVATE mcp_client)
