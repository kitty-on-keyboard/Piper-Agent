# Tiny gate (second-process typed Choice)

**Status:** Flag-off stub behind `LMP_TINY_GATE` (default **off / unset**). **Not** product default. **Not** same-weights Pulse. **Not** MTP.

**Hypothesis (non-binding):** Sean — 27B overthought Choice on its own stuckness; a tiny Qwen3 may be sharper. Verify on live proves; do not treat offline accuracy alone as KEEP.

## Why a second process

`MlxBackend::load` is one full causal LM per process (S5.11). `load_mtp` merges an MTP draft head into the same weight store — it is **not** a second chat model. Co-resident tiny gate → sibling process (`scripts/lmp_tiny_gate.py`) + HTTP/stdio JSON.

## Policy when `LMP_TINY_GATE=1`

1. **Stage-0** (features only, no LLM):
   - **stall** if `consec ≥ 3` OR `streak ≥ 4`
   - else **compact** if `prompt_tok ≥ 16000` OR `reread_max ≥ 3`
2. **Else binary gate:** 2-way letter mask **force_tool vs nudge** only (structured features ~100–200 tok; strip hindsight `next=` / `outcome=`). Emit **force_tool** iff `p_force ≥ p_min`, else **nudge**.
3. Never ask the model for stall/compact. Never revive 4-way Pulse.

Default `p_min` = **0.55**. Override with `LMP_TINY_GATE_P_MIN`.

## Enable (Benchbot)

```bash
# 1) Download a Qwen3-family tiny MLX 4-bit (target)
#    e.g. mlx-community/Qwen3-0.6B-4bit → $HOME/Models/Qwen3-0.6B-4bit

export LMP_GATE_MODEL_DIR="$HOME/Models/Qwen3-0.6B-4bit"
export LMP_GATE_URL="http://127.0.0.1:18765"   # optional; this is the default

# 2) Start the helper (second process; keep it up for the prove)
python3 scripts/lmp_tiny_gate.py --listen --model-dir "$LMP_GATE_MODEL_DIR"

# 3) Run Piper / sidecar with the flag on
export LMP_TINY_GATE=1
# optional
export LMP_TINY_GATE_P_MIN=0.55
```

Unset or `LMP_TINY_GATE=0` → baseline #150 / stall heuristics only (flag-off == baseline). No gate process required when off.

### Env summary

| Env | Default | Meaning |
|-----|---------|---------|
| `LMP_TINY_GATE` | unset/0 | Master switch |
| `LMP_GATE_MODEL_DIR` | (none) | Tiny MLX checkpoint path (helper) |
| `LMP_GATE_URL` | `http://127.0.0.1:18765` | Helper base URL (sidecar client) |
| `LMP_TINY_GATE_P_MIN` | `0.55` | force iff `p_force ≥ p_min` |
| `LMP_GATE_HOST` / `LMP_GATE_PORT` | `127.0.0.1` / `18765` | Helper listen bind |

## Journal

Events kind `gate` include `choice`, `p` / `p_force` / `p_vec`, `p_min`, `policy`, `stage0`, `latency_ms`, `model`, `encoding`, `ok`, `error`. Nudge events gain `gate_policy`.

## Prove plan (bounded)

1. **Smoke:** load 0.6B; `POST /v1/choose` with a tiny forced-prefix; measure letter-mask latency.
2. **Offline:** same N=34 host texts → tiny model; compare to 27B same-weights binary + Stage-0 heuristic (archaeology: `pulse-force-binary`).
3. **Live:** if offline clears force-recall/ff bars → bowling seed7 A/B (gate process on vs off), same KEEP bar as before.
4. **Only then:** Godoer one enum questionnaire (narrow) — **not in this PR**.

## T1 seam

Same T1 / #150 adjacency as Pulse archaeology (#194 KEEP-seed / closed #193): after `degenerate_text` / TextOnly where #150 would nudge. Spun tool no-progress stays on heuristics.

## Future Godoer

Same transport (second-process letter-mask Choice), different enum sets (e.g. continue / retarget / bake / stop). Out of scope for this Piper T1 PR.

## Not in this PR

Fine-tune on N=34 · second full model inside the main sidecar · 4-way Pulse reopen · Godoer enums · merge-as-default / flag-on.
