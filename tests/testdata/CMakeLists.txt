# Gate corpora and the minimal scoreboards that pin the shipped engines.
# Answer keys live here (not under src/) so product code and fixtures stay separate.

add_library(lmp_testdata INTERFACE)
target_include_directories(lmp_testdata SYSTEM INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/blast_radius"
  "${CMAKE_CURRENT_SOURCE_DIR}/log_triage"
  # shipped.hpp includes production engines by their src-relative paths
  "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(lmp_testdata INTERFACE simdjson)
target_compile_definitions(lmp_testdata INTERFACE
  LMP_TESTDATA_BR_DIR="${CMAKE_CURRENT_SOURCE_DIR}/blast_radius"
  LMP_TESTDATA_LT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/log_triage")

# One scoreboard per shipped engine. The bridge TUs keep the scorer's contract types
# separate from the production header (same seam the historical cook-off used).
add_executable(blast_radius_score_shipped
  "${CMAKE_CURRENT_SOURCE_DIR}/blast_radius/score.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/blast_radius/entrant_tu.cpp")
target_link_libraries(blast_radius_score_shipped PRIVATE lmp_testdata)
target_compile_definitions(blast_radius_score_shipped PRIVATE
  ENTRANT_HEADER="shipped.hpp"
  ENTRANT_NAME="shipped"
  CORPUS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/blast_radius/corpus.jsonl")
target_compile_options(blast_radius_score_shipped PRIVATE -w)

add_executable(log_triage_score_shipped
  "${CMAKE_CURRENT_SOURCE_DIR}/log_triage/score.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/log_triage/entrant_tu.cpp")
target_link_libraries(log_triage_score_shipped PRIVATE lmp_testdata)
target_compile_definitions(log_triage_score_shipped PRIVATE
  ENTRANT_HEADER="shipped.hpp"
  ADAPTER_HEADER="adapters/shipped.hpp"
  ENTRANT_NAME="shipped"
  CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/log_triage")
target_compile_options(log_triage_score_shipped PRIVATE -w)
