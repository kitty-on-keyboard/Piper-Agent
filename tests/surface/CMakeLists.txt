lmp_add_test(test_transport SOURCES test_transport.cpp LABELS gate LINK lmp_surface)

# The MCP settings parser (M1). Its `trusted` field decides whether an unsandboxed tool
# may run without an approval card, so it is tested against every shape a hand-edited
# settings file can produce that is not a literal boolean true (S7.5).
lmp_add_test(test_mcp_settings SOURCES test_mcp_settings.cpp LABELS gate LINK lmp_surface)

# The model lifecycle (M2). Loading is an act the operator asks for by name now, so it
# has a state, a refusal and an unload -- and the half-loaded session is the shape worth
# a test, because it fails by generating fluent nonsense rather than by stopping.
lmp_add_test(test_session SOURCES test_session.cpp LABELS gate LINK lmp_surface)

# The context/pcc adapter, and specifically the run where it cannot open a database. The
# failure is invisible at the time and only matters twenty turns later, when a detail the
# trim dropped is wanted back -- so the reason it failed has to survive the moment.
lmp_add_test(test_context_journal SOURCES test_context_journal.cpp LABELS gate
             LINK lmp_surface)

# The run's identity. `run_id` was the JSON-RPC request counter until 2026-08-19, which
# collided 63-to-1 in the real log and would have made resume splice unrelated runs
# together -- a failure no single-run test can see, so the properties are asserted here.
lmp_add_test(test_run_id SOURCES test_run_id.cpp LABELS gate LINK lmp_surface)

# The transcript that goes on the wire when a run is resumed. Every case here is a BOUND:
# one real observation in the log is 708 KB, so the failures are a reply big enough to
# stall the host and a cut through the middle of a codepoint -- which loses the whole
# transcript to a parse error, not just the entry it mangled.
lmp_add_test(test_transcript SOURCES test_transcript.cpp LABELS gate LINK lmp_surface)
