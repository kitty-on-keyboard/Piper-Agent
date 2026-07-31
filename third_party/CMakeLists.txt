# Vendored dependencies. Copied in, not fetched: the benchmark corpora are unrunnable
# without their parsers, and v1's lesson (bakeoff/blast_radius/README.md) is that a
# benchmark you cannot re-run today is a benchmark you have lost. A network fetch between
# a fresh clone and a runnable gate is a way to lose it.
#
#   simdjson 4.6.1   -- bakeoff corpus loaders (verbatim v1 code depends on it)
#   pcre2 10.43      -- frankentok's pretokenizer (Split regex read from tokenizer.json)
#   utf8proc 2.x     -- frankentok's NFC normalizer (tokenizer.json declares NFC)
#   nlohmann_json    -- frankentok's tokenizer.json parse (header-only)
#   frankentok       -- mlx_qwen_tokenizer: the Qwen 3.6 tokenizer, HF-parity verified
#                       (15,045 cases id-identical, exact streaming decode, binary cache)
#   parsephony       -- JSON PDA + Qwen 3.6 XML ToolCallGuard + token mask engine,
#                       measured 17.7 ns/step, 1000/1000 valid constrained generations

# --- simdjson ---------------------------------------------------------------
add_library(simdjson STATIC simdjson/src/simdjson.cpp)
target_include_directories(simdjson
  SYSTEM PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/simdjson/include"
  PRIVATE        "${CMAKE_CURRENT_SOURCE_DIR}/simdjson/src")
target_compile_features(simdjson PUBLIC cxx_std_20)
set_target_properties(simdjson PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- pcre2 (8-bit, JIT on -- frankentok calls pcre2_jit_compile) ------------
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_JIT ON CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
add_subdirectory(pcre2 EXCLUDE_FROM_ALL)

# --- utf8proc ---------------------------------------------------------------
set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(utf8proc EXCLUDE_FROM_ALL)

# --- nlohmann (header-only) -------------------------------------------------
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/nlohmann/include")

# --- frankentok -------------------------------------------------------------
add_library(frankentok STATIC
  frankentok/src/bpe.cpp
  frankentok/src/normalizer.cpp
  frankentok/src/pretokenizer.cpp
  frankentok/src/streaming_decoder.cpp
  frankentok/src/tokenizer.cpp
  frankentok/src/trie.cpp
  frankentok/src/vocab.cpp
)
target_include_directories(frankentok
  SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/frankentok/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/frankentok"
                # pretokenizer.h includes <pcre2.h>, generated into pcre2's build dir
                "${CMAKE_CURRENT_BINARY_DIR}/pcre2")
target_link_libraries(frankentok PUBLIC pcre2-8 PRIVATE nlohmann_json utf8proc)
target_compile_definitions(frankentok PRIVATE PCRE2_CODE_UNIT_WIDTH=8 UTF8PROC_STATIC)
target_compile_features(frankentok PUBLIC cxx_std_20)
target_compile_options(frankentok PRIVATE -O2 -w)

# --- parsephony (std-only core) ---------------------------------------------
add_library(parsephony STATIC
  parsephony/src/parsephony.cpp
  parsephony/src/stream.cpp
  parsephony/src/vocab.cpp
  parsephony/src/toolcall.cpp
)
target_include_directories(parsephony SYSTEM PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/parsephony/include")
target_compile_features(parsephony PUBLIC cxx_std_20)
target_compile_options(parsephony PRIVATE -O2 -w)
