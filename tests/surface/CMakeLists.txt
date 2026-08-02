lmp_add_test(test_transport SOURCES test_transport.cpp LABELS gate LINK lmp_surface)

# The MCP settings parser (M1). Its `trusted` field decides whether an unsandboxed tool
# may run without an approval card, so it is tested against every shape a hand-edited
# settings file can produce that is not a literal boolean true (S7.5).
lmp_add_test(test_mcp_settings SOURCES test_mcp_settings.cpp LABELS gate LINK lmp_surface)

# The model lifecycle (M2). Loading is an act the operator asks for by name now, so it
# has a state, a refusal and an unload -- and the half-loaded session is the shape worth
# a test, because it fails by generating fluent nonsense rather than by stopping.
lmp_add_test(test_session SOURCES test_session.cpp LABELS gate LINK lmp_surface)
