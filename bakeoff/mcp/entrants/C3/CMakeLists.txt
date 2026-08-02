cmake_minimum_required(VERSION 3.10)
project(mcp_client_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add nlohmann json
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(mcp_client mcp_client.cpp)
target_include_directories(mcp_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(mcp_client_example main.cpp)
target_link_libraries(mcp_client_example PRIVATE mcp_client pthread)
