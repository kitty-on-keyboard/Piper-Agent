# Pulse

**Status:** KEEP-seed stub behind `LMP_PULSE` (default **off / unset**). **Not** product default. **Not** a 4-way reopen (4-way stays KILL).

## Policy when `LMP_PULSE=1`

1. **Stage-0** (features only, no LLM):
   - **stall** if `consec ≥ 3` OR `streak ≥ 4`
   - else **compact** if `prompt_tok ≥ 16000` OR `reread_max ≥ 3`
2. **Else binary Pulse:** 2-way letter mask **force_tool vs nudge** only (structured features; strip hindsight `next=` / `outcome=`). Emit **force_tool** iff `p_force ≥ p_min`, else **nudge**.
3. Never ask the model for stall/compact.

## `p_min` defaults (per model)

| Path | Default `p_min` | Why |
|------|----------------:|-----|
| **27B dense** (primary product) | **0.55** | Best joint acc/ff/forceR on N=34 |
| **A3B / MoE** (transfer only) | **0.50** | 0.55 fails force R 6/10; 0.50 passes 9/10 |
| Env override | `LMP_PULSE_P_MIN` | Wins over family default when set |

Family is sniffed from the checkpoint path/name (`A3B` / `MoE` markers). Unknown paths use the 27B default (0.55).

## Enable (Benchbot prove)

```bash
export LMP_PULSE=1
# optional; overrides family default
export LMP_PULSE_P_MIN=0.55   # 27B
# export LMP_PULSE_P_MIN=0.50 # A3B recall-max / transfer
```

Unset or `LMP_PULSE=0` → baseline #150 / stall heuristics only (flag-off == baseline).

## T1 seam

Same T1 / #150 adjacency as closed #193 archaeology: after `degenerate_text` / TextOnly where #150 would nudge. **Replaces** any 4-way questionnaire with binary force/nudge only.

## Not in this seed

TypeSafe / second model / train on N=34 / T2–T3 Godoer / merge-as-default / revive 4-way enum.
