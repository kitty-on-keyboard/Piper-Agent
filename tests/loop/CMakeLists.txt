lmp_add_test(test_loop SOURCES test_loop.cpp LABELS gate LINK lmp_loop)

# Agent::step end to end IN THE GATE, on the miniature vocabulary. PHASES.md named
# agent.cpp:168 and :287 as surviving mutants with one shared blocker -- "not driven
# end-to-end in the gate, vocab-fixture blocker" -- so shipping the fixture without this
# would have cleared the blocker on paper only.
lmp_add_test(test_agent_step SOURCES test_agent_step.cpp LABELS gate LINK lmp_loop TIMEOUT 120)
add_dependencies(test_agent_step lmp_mini_vocab)
target_compile_definitions(test_agent_step PRIVATE
  LMP_MINI_VOCAB_JSON="${CMAKE_BINARY_DIR}/tests/model/mini_qwen/tokenizer.json")

# Registered twice off one source, same rule as tests/model/test_grammar.cpp (R1). This
# needs a vocabulary and never the weights -- it drives ScriptedBackend -- so the gate
# variant runs on the miniature vocabulary and reaches CI. The fixture's merges are
# ASCII-only, which makes every token of a multi-byte string a fragment: the streaming
# hazard is exercised harder there than on the real checkpoint's 944 fragment tokens.
lmp_add_test(test_token_stream_gate SOURCES test_token_stream.cpp LABELS gate LINK lmp_loop TIMEOUT 300)
add_dependencies(test_token_stream_gate lmp_mini_vocab)
target_compile_definitions(test_token_stream_gate PRIVATE
  LMP_MINI_VOCAB_JSON="${CMAKE_BINARY_DIR}/tests/model/mini_qwen/tokenizer.json")

lmp_add_test(test_token_stream SOURCES test_token_stream.cpp LABELS realmodel LINK lmp_loop TIMEOUT 300)

# Pure timing + ordering over the dispatch primitive: no tokenizer, no model, so gate.
lmp_add_test(test_parallel_calls SOURCES test_parallel_calls.cpp LABELS gate LINK lmp_loop TIMEOUT 60)

# The whole loop, with eyes: the model calls view_image on a file in its workspace and
# names a colour that appears nowhere in the prompt. The only test that proves the tool
# chain and the embedding splice are connected end to end.
lmp_add_test(test_vision_agent SOURCES test_vision_agent.cpp LABELS realmodel
             LINK lmp_loop lmp_model lmp_tools lmp_context TIMEOUT 1800)
lmp_stage_metallib(test_vision_agent)
