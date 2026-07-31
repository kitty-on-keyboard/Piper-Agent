lmp_add_test(test_corpus_loads SOURCES test_corpus_loads.cpp LABELS gate LINK bakeoff_corpora)
lmp_add_test(test_log_triage_engine SOURCES test_log_triage_engine.cpp LABELS gate
             LINK bakeoff_corpora)
