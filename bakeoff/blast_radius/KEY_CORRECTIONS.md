# Corrections applied to the answer key in v2

`corpus.jsonl` arrived in this repo **byte-identical** to v1. This file records every
change made to it since, with the argument, so that "the key changed" is never something
you have to discover by diffing.

`tests/bakeoff/test_corpus_loads.cpp` asserts that exactly the corrections listed here are
present. Applying a fourth without writing it down fails that test; writing one down
without applying it fails too.

---

## 2026-07-30 — three cases labelled `parsed` that rule 7 makes `partial`

**Carried forward from the v2 build spec §18, which flagged them as key defects.**
Confirmed against the key itself before acting, rather than accepted on the spec's word
(§19.4: a handed-down diagnosis is a hypothesis with an author).

| id | cmd | was | now |
|---|---|---|---|
| `benign_ctest` | `ctest --output-on-failure` | `parsed` | `partial` |
| `look_cargo_offline` | `cargo build --offline` | `parsed` | `partial` |
| `chain_mkdir_cmake` | `mkdir -p build && cd build && cmake ..` | `parsed` | `partial` |

### The argument

Labelling rule 7, published in `README.md` and unchanged:

> **`partial` means the effect depends on bytes not in the string** — a script file, an
> unexpanded variable, a substitution, a downloaded payload.

The same key applies that rule to project-code-running commands everywhere else:

- `chain_make_clean` (`make clean; make`) → `partial`, *"the rule is in a Makefile we
  cannot see"*
- `indir_make_install` (`make install`) → `partial`, *"the Makefile is not in the string"*
- `indir_npm_build` (`npm run build`) → `partial`, *"package.json scripts are unseen"*
- `look_npm_format` (`npm run format`) → `partial`, *"Script body unseen"*
- `net_npm_install` (`npm install`) → `partial`, relabelled in round 1 for exactly this

And the contract header `blast_radius.hpp` names `npm run build` and `make install` as the
partial exemplars, in the doc comment on `PartiallyParsed`.

The three cases above run project-supplied code whose bytes are equally not in the string:

- `ctest` executes the test binaries the project built. A test binary is a locally built
  executable; §7.1 of the build spec is explicit that "any script file, any locally built
  binary" has effects that are simply not in the string. This is a *stronger* case for
  `partial` than `make`, because `make` at least has to read a Makefile first.
- `cargo build --offline` executes `build.rs` and any procedural macro in the dependency
  graph, both of which are ordinary Rust programs run at compile time. `--offline` removes
  the *network* capability — correctly, and that part of the old label stands — but it
  says nothing about code execution, and the old `why` reasoned only about the network.
- `cmake ..` executes `CMakeLists.txt`, which is a full imperative language with
  `execute_process()` and `file(WRITE)`. The old `why` — "three stages, all benign" — is
  an assertion about a file the classifier cannot see.

**The flags are unchanged.** All three keep `"caps": []`, because rule 7's second sentence
still holds: "The visible flags still stand." Nothing dangerous is *visible*. Only the
status moves, and the status is the field whose entire job is to say "sandbox this one
regardless of its flags."

### What this invalidates

`README.md`'s round-1 results table and the incumbent baseline numbers were measured on
the pre-correction key, on 2026-07-30. **Three headline cases changed status, so every
`status miss` column in that table is now stale**, and the `exact` column with it. The
table is left in place unedited because it is the historical record of a blind
comparison; it is not a current measurement and must not be quoted as one (§16: no metric
quoted without checking what it counts).

Re-score before quoting:

```bash
cmake --build build --target blast_radius_score_incumbent && ./build/blast_radius_score_incumbent
```

### Separately: README.md's round-1 table was already stale on arrival

Found while attributing the effect of the three corrections above, by scoring `e12`
against a reconstructed pre-correction key. `README.md` is a verbatim v1 port and is not
edited (§18), so the correction is recorded here instead.

| e12, headline | wmiss | alarms | status miss | exact |
|---|---|---|---|---|
| `README.md`'s table | 169 | 9 | 7 | 105 |
| ported key, before the 3 corrections | 169 | **8** | 7 | **106** |
| ported key, after the 3 corrections | 169 | 8 | **9** | **105** |

The first row disagrees with the second by one false alarm and one exact case, and the
port is byte-identical to v1 — so the disagreement is between the v1 README and the v1
corpus, not between v1 and v2.

The cause is in the README's own text. It describes `effective_truth()` as the round-1 fix
for a scorer that "was paying entrants to be wrong": `adv_unterminated_quote` charged eight
false alarms to anyone who answered it correctly, and "9 of 11 entrants answered exactly
right and all 9 were penalised". `corpus.hpp` as shipped **contains** that fix, and `e12`
answers that case exactly and unpenalised today. **The table was measured before the fix
the same document describes.** Reports age faster than code (§19.6), including this one.

Consequence: the table's `alarms` and `exact` columns are pre-fix, and its `status miss`
column is additionally pre-correction. Only `wmiss` survives all of it — which is the
primary metric, so the ranking still stands. Re-score before quoting any other column.

### What this does not settle

Three cases is where the argument stops. `contested_npm_dev` (`npm run dev`) is a live
disagreement and stays in the contested bucket, excluded from the headline. If a future
reader thinks a fourth case is wrong, the rule is unchanged: argue it here, name the case,
change the key — never edit the key to match an implementation (§11.3).
