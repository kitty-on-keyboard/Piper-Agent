# Six layers (spec S3). Each depends only on those above it; scripts/gate_layers.py
# enforces the include direction, and CMake enforces the link direction here. Both,
# because a header-only violation links fine and a link-only violation compiles fine.
#
#   L0 platform  L1 model  L1 pcc  L2 tools  L3 context  L4 loop  L5 surface
#
# pcc shares L1 with model rather than sitting between them: the two are siblings that
# know nothing of each other, and both context (L3) and tools (L2) need to reach the
# store without inverting anything.

add_subdirectory(platform)
add_subdirectory(model)
add_subdirectory(pcc)
# mcp is off to the side of the six layers on purpose: it links nothing from LM_Pipe, so
# it has no layer to violate and stays liftable into its own repo.
add_subdirectory(mcp)
add_subdirectory(tools)
add_subdirectory(context)
add_subdirectory(loop)
add_subdirectory(surface)
