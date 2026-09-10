#pragma once
//
// Thin wrapper: score the production engine on the committed corpus.
// Includes src/tools/log_triage.hpp so the number is what the agent actually gets.
// Holdout.jsonl is the blind comparison; corpus scores were tuned with the key open.
//
#include "tools/log_triage.hpp"
