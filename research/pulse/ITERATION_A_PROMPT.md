# Pulse Iteration A — prompt + features (labels frozen)

**Date:** 2026-09-21  
**Status:** Research-confirmed. Re-run Phase 1 micro on same N=34.  
**Goal:** Break force_tool collapse while keeping high force recall on force golds.

## Confirmed (Benchbot dig + Research amend)

### Host text (required)
1. **Strip hindsight:** remove `next=` / outcome trails / any post-decision fields from `context_summary` before decode.
2. **Structured features only** (no free prose essay). Emit a fixed block, e.g.:
   ```
   consec=<int> streak=<int> prompt_tok=<int> reread_max=<int>
   think=<int> text=<int> tool_tok=<int> why=<loop_cut|no_progress|degenerate|none>
   ```
3. Optional one-line mission stub ≤20 tokens if already in label; else omit.

### Forced prefix (replace hostile “text-instead-of-tool” opener)

Use **neutral** cue + **one-line defs** + **letter codes** (kill name/order prior):

```
Pick exactly one harness next-step. Reply with only the code letter.

A = force_tool — task clearly needs a tool next (read/edit/test/shell); stop pure think/text.
B = nudge — one more recovery chance; mild stuck, not exhausted.
C = stall — further nudges futile; end cleanly.
D = compact — context bloated or re-read storm; compact before another nudge.

Features:
<structured block>
```

### Decode
- Mask over token ids for `A`/`B`/`C`/`D` **or** full option strings — prefer **single-letter** ids if vocab-stable (document which).
- Map letter → enum for journal compatibility.
- **Shuffle option order** across items (seeded by label id) so force is not always first — required for Iteration A.
- Keep p_min / policy unchanged for this iter (measure argmax first).

### Success gate for Iteration A (before B)
Must show **all**:
1. Accuracy **>** Phase-1 Pulse 29.4% **and** preferably approaching heuristic 61.8% without needing to beat it yet.
2. False-force **≪ 24/34** (target ≤10/34 as soft bar).
3. Force recall on force golds **≥ 8/10** (keep unique upside).
4. Still **0** mask escapes.

If mass moves (stall/compact/nudge appear with non-trivial P) but accuracy soft → proceed to Iteration B (hybrid).  
If still 100% force collapse with honest defs + no hindsight + shuffle → category path weak; one hybrid B then KILL if still dead.

### Non-changes
- Labels frozen.  
- `LMP_PULSE` default stays off.  
- No training on these 34.  
- No TypeSafe.
