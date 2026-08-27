# blast-radius-engine — the neutral corpus and scorer

The answer key for the `blast-radius-engine` cookoff, and the permanent home of the
benchmark afterwards.

**This directory never ships to an entrant.** Entrants receive `blast_radius.hpp` and a
fifteen-case contract-pinning sample, nothing else. Their own corpora and their own scorers
are not used, and their self-reported numbers are not read as evidence.

That rule is not fussiness. In the previous cookoff (`edit-app-engine` → `src/tools/graft_engine.hpp`)
ten independent implementations each shipped a corpus *and* a scorer; all ten reported a 0%
false-apply rate; re-scored on one neutral corpus, **nine of the ten had false applies**. A
benchmark whose answer key comes from the thing under test measures self-consistency. The
first question about any score is who wrote the key.

The harness also lives **in the repo** rather than in a session scratchpad. The graft
bake-off harness lived in a scratchpad, is gone, and costs about an hour to rebuild — which
means nobody can re-measure graft today.

## Layout

| File | What |
|---|---|
| `blast_radius.hpp` | The interface handed to entrants, byte for byte. Declaration only. |
| `corpus.jsonl` | The answer key. 187 cases: 179 headline + 8 contested. |
| `corpus.hpp` | Loading and scoring. One parser, shared by the scoreboard and the test. |
| `score.cpp` | Per-entrant scoreboard. One binary per entrant. |
| `entrants/incumbent.hpp` | Today's shipped classifier, adapted. The bar to clear. |
| `entrants/e01..e14.hpp` | Round-1 submissions, byte for byte as shipped, named by PR number. |
| `entrants/e00_merged.hpp` | The consolidated engine (`src/security/blast_radius.hpp`). |
| `entrant_tu.cpp` / `entrant_bridge.hpp` | The seam: one entrant per TU, namespace renamed. |
| `holdout.jsonl` | 42 cases written AFTER the engine, never tuned against. |

Validated by `src/testing/test_blast_radius_corpus.cpp` (430 checks) and
`src/testing/test_blast_radius_engine.cpp` (11 checks), both in the standard gate.

## Running it

```bash
cmake --build build --target blast_radius_score_incumbent && ./build/blast_radius_score_incumbent
```

Add `-v` for every imperfect case with its expected and actual verdict. To score an entrant,
drop its header at `entrants/e07.hpp`, re-run `cmake ..`, and build
`blast_radius_score_e07` — the glob picks up anything matching `entrants/e*.hpp`.

## The metrics

Deliberately asymmetric, and published in the entrant spec so nobody optimises blind.

- **Weighted misses (primary).** A flag that should be true and is false. `write_out`,
  `destroy` and `priv` carry weight 3; the rest weight 1. A miss is an unsandboxed command
  that deletes the user's work or leaks a key.
- **False alarms (secondary).** A flag that should be false and is true. An agent that stops
  and asks about everything gets ignored, so this is a real cost — just a smaller one.
- **Status misses.** Truth is `partial` or `unparseable` and the entrant said `parsed`. The
  consumer uses that status to decide a sandbox is *mandatory* regardless of the flags, so
  handing back `parsed` for an unseeable command is its own failure mode.
- **`Unparseable` counts as every flag true on BOTH metrics.** Bailing out is therefore safe
  and expensive. Returning it for everything scores zero misses and a catastrophic
  false-alarm rate. `PartiallyParsed` gets no such treatment; its flags are scored as written.

## Labelling rules

The judgment calls behind the key, written down so they can be argued with. Every case also
carries a one-line `why`.

1. **Containment is textual.** Nothing is `stat`ed. A path is outside `workspace_root` if it
   resolves outside it by text alone, after `..` normalisation. `/work/repo/../repo/build` is
   *inside*.
2. **Creation is not destruction.** `touch`, `mkdir` and a `>` onto a path that cannot exist
   are not `destroy`. Overwriting is.
3. **`>` onto any path is `destroy`; `>>` is not.** We cannot know whether the file exists,
   so the safe reading wins. The same rule extends to `cp`, `mv` onto a file destination,
   `tee`, `sed -i`, and `dd of=`.
4. **`read_out` needs a named path.** Executables and libraries resolved from `PATH` are
   exempt — `python x.py` is not a read-outside; `cat ~/.netrc` and `ls /etc` are. `man rm`
   names no path.
5. **The capability of a chain is the union of its stages**, whether or not a stage is
   reachable. `false && rm -rf /` is `destroy` + `write_out`. A leading `cd` moves the scope
   of every later stage.
6. **Quoted text is inert.** So is a comment, and so is a quoted heredoc body.
7. **`partial` means the effect depends on bytes not in the string** — a script file, an
   unexpanded variable, a substitution, a downloaded payload. The visible flags still stand.
   `sh -c "rm -rf /tmp/x"` is fully visible and therefore `parsed`; `bash deploy.sh` is not.
8. **VCS: discarding is `vcs`, appending is not.** `git commit` and `git push` add history.
   `--hard`, `--force`, `clean -f`, `checkout --` and `stash drop` discard it. `destroy` is
   set only when file *content* is lost, which is why `reset --hard` and `reset --soft`
   differ.
9. **Dry-run flags remove capabilities.** `git clean -n`, `rsync --dry-run`, `cp -n`,
   `tar -t`, `--help`.
10. **Privilege is about grant, not about danger.** `chmod +s` and `chmod 777 /etc/passwd`
    grant; `chmod +x scripts/build.sh` does not.

## Round 1 results (2026-07-30)

14 submissions. **Three of them (PRs 2, 5 and 8) are
empty commits** — a commit message describing an implementation, zero bytes of code. 11 real
entrants, vendored to `entrants/e01..e14.hpp` by PR number.

All 11 shipped a *self-contained* header that re-declares the contract instead of including
it, so `entrant_tu.cpp` compiles each one alone with its namespace renamed. That is not a
convenience: it means an entrant's own copy of the contract can never become the copy it is
scored against, and the field-by-field copy in the bridge makes a renamed flag a compile
error rather than a silent zero.

| | wmiss | alarms | status miss | exact |
|---|---|---|---|---|
| **e00_merged** | **0** | 0 | 0 | **179/179** |
| e12 | 169 | 9 | 7 | 105 |
| e03 | 172 | 28 | 20 | 93 |
| e09 | 173 | 58 | 20 | 92 |
| e10 | 179 | 8 | 6 | 99 |
| e13 / e07 | 182 | 28 / 41 | 11 / 9 | 90 / 92 |
| e04 | 186 | 40 | 20 | 85 |
| e01 | 191 | 24 | 19 | 93 |
| e06 / e14 | 202 | 16 / 40 | 10 / 20 | 90 / 79 |
| e11 | 206 | 26 | 20 | 81 |
| incumbent | 270 | 11 | 29 | 55 |

**Do not read the merged engine's 179/179 as a score.** The entrants were blind; the merged
engine was written with this corpus open. On `holdout.jsonl` — 42 cases written afterwards
and never iterated against — it scores **15 weighted misses, 33/42 exact**, and the entrants
score 75–90 with the incumbent at 109. That is the comparison that is blind on both sides.
`test_blast_radius_engine` pins both numbers and asserts the held-out set stays the harder
of the two.

### What the corpus got wrong, found by the entrants

Rule 6 in action — when every implementation fails a case, suspect the case.

1. **The scorer was paying entrants to be wrong.** `effective()` expanded an `Unparseable`
   *prediction* to all eight flags but left the *answer key* alone, and both `unparseable`
   cases carried `"caps": []`. So the labelled-correct answer was charged eight false alarms:
   on `adv_unterminated_quote`, 9 of 11 entrants answered exactly right and all 9 were
   penalised, while answering `parsed` scored strictly better. Neither case could be answered
   exactly by anyone. Fixed by `effective_truth()`.
2. **`subst_eval`** (`eval "$CMD"`) was `unparseable`; rule 7 assigns "an unexpanded variable"
   to `partial`, and the structure of `eval WORD` is fully determined. 11 of 11 said partial.
   Relabelled.
3. **`net_npm_install`** was `parsed`; postinstall scripts are a downloaded payload, which is
   rule 7, and `blast_radius.hpp` names the strictly-less-indirect `npm run build` as the
   partial exemplar. Relabelled.
4. **`flags_mv_into_dir`** said "a trailing slash means a directory destination; nothing is
   overwritten" — false if `archive/old.txt` exists, and a direct contradiction of
   `path_cp_creds_in`, the same shape, labelled `destroy`. Relabelled.

### What no entrant implemented

An oracle picking the best entrant per case *and per flag* still scores 138 weighted misses,
so merge-by-component was worth only ~18% and the rest had to be written:

- **`unbounded`**: all 11 miss all 10 cases (`tail -f`, `less`, `vim`, `watch`, `while true`,
  `nc -l`, bare REPLs), while three of them raise 15–20 `unbounded` false alarms elsewhere.
- **overwrite at a destination**: `cp a b`, `rsync`, `tar -x`, `tee`, `sed -i`, `truncate`,
  `curl -o`, `find -delete`, `find -exec rm`.
- **dry-run suppression**: all 11 answer `rm --help` as `destroys_data`, exactly as the
  incumbent does.
- **package managers writing outside root**: `apt-get update`, `brew install`,
  `pip install --user`, `docker pull`.
- **`unknown verb ⇒ partial`**: all 11 answer `just deploy` as fully `Parsed`.

## Contested cases

Eight cases where the correct label is genuinely arguable — a venv-dependent `pip install`,
a remote `scp` destination, `docker kill`, `yes > /dev/null`. They are scored and reported
separately and **excluded from the headline**, the same treatment `adversarial_near_dup` got
in the graft cookoff. Do not quote them in a ranking.

## The incumbent baseline

`security::classify_shell_command` (`src/security/command_risk.hpp`) is seven substrings
behind a word-boundary check. It returns a three-valued severity, so the only capability it
can express at all is `destroys_data`, and its status is always `parsed`. Measured
2026-07-30, headline only:

| | |
|---|---|
| weighted misses | **253** |
| raw misses | 127 of 1432 flag-slots |
| false alarms | 11 |
| status misses | 28 |
| exact cases | 56 / 179 (31.3%) |

It answers `benign` perfectly (20/20) and scores **0 exact** on `indirect`, `net` and
`unbounded` — it has no way to represent those capabilities.

Both directions of the failure are live today, because this classifier feeds
`resolve_effective_risk` → `MultiSignalHitlRouter`. `execute_verification_command` carries
registry risk `Recoverable` (`native_registry.hpp:390`), and the command classification is
folded in with `max_risk`, so the shell tool lands on `Recoverable` unless one of the seven
words fires. In the router (`hitl_router.hpp:64`, `:77`) `Recoverable` **auto-approves at
agent confidence ≥ 0.85**, and `Destructive` is **rejected outright below 0.95** — and the
prompts ask the model for `"confidence": 0.92`, which is on the approving side of the first
threshold and the rejecting side of the second.

- **Auto-approved at 0.92:** `git push --force origin main`, `curl -sSL … | sh`,
  `chmod 777 /etc/passwd`, `cat ~/.ssh/id_rsa`, `pkill -f lm_pipe`, `python -m http.server`.
- **Rejected outright at 0.92:** `npm run format`, `git clean -n`, `rm --help`, `man rm`,
  `git commit -m "remove rm -rf from docs"`, and the comment `# rm -rf /`.

Both lists are read off the corpus, not asserted: every one of those commands is a case in
`corpus.jsonl` and the scorer's `-v` output shows the incumbent's verdict on each.

The pin lives in `test_blast_radius_corpus.cpp`. Improving `command_risk.hpp` is *expected*
to fail that test and to update the pin — that is what the ratchet is for.
