# Vendored dependencies. Copied in, not submoduled: the log-triage benchmark is
# unrunnable without simdjson, and v1's lesson (bakeoff/blast_radius/README.md) is that
# a benchmark you cannot re-run today is a benchmark you have lost. A network fetch
# between a fresh clone and a runnable gate is a way to lose it.
#
# simdjson 4.6.1 -- include/ + src/ from the upstream tree, LICENSE carried.

add_library(simdjson STATIC simdjson/src/simdjson.cpp)
target_include_directories(simdjson
  SYSTEM PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/simdjson/include"
  PRIVATE        "${CMAKE_CURRENT_SOURCE_DIR}/simdjson/src")
# Third-party code is exempt from lmp_warnings by construction: it is not linked here.
# The exemption is this comment plus the absence of the target, not a -Wno- list that
# would also silence our own code in any TU that includes the header.
target_compile_features(simdjson PUBLIC cxx_std_20)
set_target_properties(simdjson PROPERTIES POSITION_INDEPENDENT_CODE ON)
