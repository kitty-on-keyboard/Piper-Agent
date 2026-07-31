# Six layers (spec S3). Each depends only on those above it; scripts/gate_layers.py
# enforces the include direction, and CMake enforces the link direction here. Both,
# because a header-only violation links fine and a link-only violation compiles fine.
#
#   L0 platform  L1 model  L2 tools  L3 context  L4 loop  L5 surface

add_subdirectory(platform)
add_subdirectory(model)
add_subdirectory(tools)
