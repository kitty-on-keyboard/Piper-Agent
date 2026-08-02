lmp_add_test(test_transport SOURCES test_transport.cpp LABELS gate LINK lmp_surface)

# The MCP settings parser (M1). Its `trusted` field decides whether an unsandboxed tool
# may run without an approval card, so it is tested against every shape a hand-edited
# settings file can produce that is not a literal boolean true (S7.5).
lmp_add_test(test_mcp_settings SOURCES test_mcp_settings.cpp LABELS gate LINK lmp_surface)
