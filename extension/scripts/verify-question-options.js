// A question card must offer the choices the model meant, and no others.
//
//   node scripts/verify-question-options.js     # reads src/webview.ts directly
//
// WHAT IT PINS. `ask_question` declares `options` as "newline-separated choices, one per
// line", and the view split on /\n+/ to honour that. Models write richer than the
// description asks -- a typical two-way choice with argument bullets under each header:
//
//     Keep the current layout
//       - Familiar to existing users
//       - Fewer moving parts
//
//     Switch to a split view
//       - Side-by-side comparison
//
// A blank line separates the two real choices and the bullets argue for the one above
// them. Splitting on newlines collapsed the blank line and produced extra clickable rows
// for a two-way choice. Reported from the installed extension as "a choice of 4 options
// is becoming 12".
//
// Sliced out of webview.ts by its own markers rather than copied, for the reason
// verify-transcript-order.js is: a harness that tests a stale duplicate is worse than no
// harness at all. webview.ts is a TypeScript template literal, so the file's `\\n` is the
// script's `\n`; unescaping here is what makes this test the real code.

const fs = require("fs");
const path = require("path");

const SRC = fs.readFileSync(path.join(__dirname, "..", "src", "webview.ts"), "utf8");

const START = "const Q_BULLET =";
const END = "// --- inbound ---";
const a = SRC.indexOf(START);
const b = SRC.indexOf(END, a);
if (a < 0 || b < 0 || b <= a) {
  console.error("FAIL: could not slice questionOptions; markers moved");
  process.exit(1);
}

// The view script lives inside a template literal, so every backslash in the source is
// doubled. Undo exactly that to recover the code the browser actually runs.
const block = SRC.slice(a, b).replace(/\\\\/g, "\\");

let questionOptions;
let questionFromText;
let questionCard;
try {
  const exported = new Function(
    `${block}; return { questionOptions: questionOptions, questionFromText: questionFromText, questionCard: questionCard };`
  )();
  questionOptions = exported.questionOptions;
  questionFromText = exported.questionFromText;
  questionCard = exported.questionCard;
} catch (err) {
  console.error("FAIL: sliced block does not parse: " + err.message);
  process.exit(1);
}

let failures = 0;
const check = (name, actual, expected) => {
  const got = JSON.stringify(actual);
  const want = JSON.stringify(expected);
  if (got !== want) {
    failures++;
    console.error(`FAIL: ${name}\n  expected ${want}\n  got      ${got}`);
  }
};

const labels = (opts) => opts.map((o) => o.label);

// 1. THE REPORTED DEFECT. Two choices, not eight.
const camera =
  "Keep the current layout\n" +
  "  - Familiar to existing users\n" +
  "  - Fewer moving parts\n" +
  "  - No migration cost\n" +
  "\n" +
  "Switch to a split view\n" +
  "  - Side-by-side comparison\n" +
  "  - Easier to review diffs\n" +
  "  - Takes more screen space";
check("blank-line blocks are two options", labels(questionOptions({ options: camera })), [
  "Keep the current layout",
  "Switch to a split view",
]);
check(
  "detail lines survive on the option they belong to",
  questionOptions({ options: camera })[1].detail,
  "Side-by-side comparison · Easier to review diffs · Takes more screen space"
);

// 2. Three blocks: the "12 rows" the report described.
const combat =
  "Smallest change that compiles\n  - Touch one file\n  - Easy to revert\n  - May leave the real bug\n\n" +
  "Refactor the shared helper\n  - One fix, many call sites\n  - Larger diff\n  - Better long-term\n\n" +
  "Hybrid approach\n  - Patch now, refactor next\n  - Two-step landing";
check("twelve lines are three options", labels(questionOptions({ options: combat })), [
  "Smallest change that compiles",
  "Refactor the shared helper",
  "Hybrid approach",
]);

// 3. The plain form the description asks for still works, and is the common case.
const plain =
  "A narrow, dramatic rock tunnel with towering sheer walls (claustrophobic & intense)\n" +
  "A wide, sweeping desert canyon with natural stone arches (open & scenic)\n" +
  "A ruined industrial canyon with broken bridges & scaffolding (fits the rustlands theme)";
check("one per line is unchanged", questionOptions({ options: plain }).length, 3);

// 4. A flat bulleted list is options, NOT one option with three detail lines. This is the
//    case that makes the "is there a header line?" guard necessary rather than tidy.
check(
  "a flat bulleted list is the options themselves",
  labels(questionOptions({ options: "- Postgres\n- SQLite\n- MySQL" })),
  ["Postgres", "SQLite", "MySQL"]
);

// 5. Indented detail with no blank line between the choices.
check(
  "unindented lines start options when there are no blank lines",
  labels(questionOptions({ options: "Rewrite it\n  cleaner, slower\nPatch it\n  faster, uglier" })),
  ["Rewrite it", "Patch it"]
);

// 6. Everything on one line with inline markers.
check(
  "inline markers on a single line still split",
  labels(questionOptions({ options: "Option A: keep it  Option B: drop it" })),
  ["keep it", "drop it"]
);

// 7. The two-character escape, which survives a round trip as text.
check(
  "literal backslash-n is a separator",
  labels(questionOptions({ options: "alpha\\nbeta\\ngamma" })),
  ["alpha", "beta", "gamma"]
);

// 8. An array argument, and the malformed cases that must render as a plain tool row.
check("an array is taken as given", labels(questionOptions({ options: ["x", "y"] })), ["x", "y"]);
check("no options is no card", questionOptions({}), []);
check("blank options is no card", questionOptions({ options: "   " }), []);

// 9. Ordinals, stripped of the marker because the card draws its own badge.
check(
  "ordinal markers are stripped from the label",
  labels(questionOptions({ options: "1. First\n2. Second\n3. Third" })),
  ["First", "Second", "Third"]
);

// --- options written INSIDE the question text, with no `options` argument -------------
//
// `ask_user` carried an optional `options` its description never mentioned, so a model
// picking that name wrote the choices into the prose instead. Verbatim from the run that
// was reported: four real choices, rendered as a plain tool row and a text box.
const embedded =
  "Which approach should we take for the shared helper?\n" +
  "\n" +
  "Option 1: Extract it now (one PR, larger diff)\n" +
  "Option 2: Patch the call site (smallest change)\n" +
  "Option 3: Add a wrapper and migrate later\n" +
  "Option 4: Leave it; document the quirk";
const parsed = questionFromText(embedded);
check("an embedded Option list is four options", labels(parsed.options), [
  "Extract it now (one PR, larger diff)",
  "Patch the call site (smallest change)",
  "Add a wrapper and migrate later",
  "Leave it; document the quirk",
]);
check(
  "the question keeps only the text above the list",
  parsed.question,
  "Which approach should we take for the shared helper?"
);

check(
  "numbered lists embedded in prose are read",
  labels(questionFromText("Pick one:\n1) Rewrite\n2) Patch").options),
  ["Rewrite", "Patch"]
);

// Prose must NOT become a card. Inventing choices around ordinary sentences is the same
// mistake as the old fallback that offered the loop's status string as the only button.
check(
  "ordinary prose yields no options",
  questionFromText("I read the file. So. Then I checked the tests and they pass.").options,
  []
);
check(
  "a single marker is not a list",
  questionFromText("Here is the plan:\n1. Do the thing and then report back").options,
  []
);
check("empty text yields no options", questionFromText("").options, []);

// --- the whole decision: what does the card actually offer? ---------------------
//
// questionOptions and questionFromText were both correct on 2026-08-24 and the human
// still got a broken card, because the code that CHOSE between them lived inline in the
// render block where nothing could reach it. These assertions are on questionCard, which
// is that choice.

// THE REPORTED BUG, verbatim from ~/Library/Logs/LM_Pipe/events.jsonl seq 58. The model
// wrote four numbered choices into the question and passed 'options' as a one-line
// summary of them. That summary parses to exactly ONE option -- correctly, there is
// nothing in it that says "four" -- and one option used to beat the question text, so a
// four-way design question reached the human as a single button reading
// "Katana / Tachi / Dual Swords / Long Sword".
const SWORDS_Q =
  "Before I finalize the combat specifics, which sword style should our player use?\n\n" +
  "1. **Katana** (Traditional Japanese - Quick draws, precise strikes, high skill ceiling)\n" +
  "2. **Tachi** (Longer reach, slower but heavier hits, more dramatic animations)\n" +
  "3. **Dual Swords** (Ninja-style, rapid combos, lower individual damage but faster pace)\n" +
  "4. **Oarobi/Monohoshizao** (Long sword, balanced speed/reach, versatile moveset)";

check(
  "a one-line options summary loses to four enumerated choices in the question",
  labels(questionCard({ options: "Katana / Tachi / Dual Swords / Long Sword" }, SWORDS_Q).options),
  [
    "Katana (Traditional Japanese - Quick draws, precise strikes, high skill ceiling)",
    "Tachi (Longer reach, slower but heavier hits, more dramatic animations)",
    "Dual Swords (Ninja-style, rapid combos, lower individual damage but faster pace)",
    "Oarobi/Monohoshizao (Long sword, balanced speed/reach, versatile moveset)",
  ]
);
check(
  "and the card is titled with the question, not with the list",
  questionCard({ options: "Katana / Tachi / Dual Swords / Long Sword" }, SWORDS_Q).question,
  "Before I finalize the combat specifics, which sword style should our player use?"
);

// Markdown emphasis is not part of the choice: the label is both the button text and the
// answer sent back, so "**Katana**" would be visible in both. Covered above by the
// expected labels; asserted here on its own so a change to the stripper is not silent.
check(
  "emphasis is stripped from labels",
  labels(questionFromText("Pick:\n1. **Bold** thing\n2. __Under__ thing").options),
  ["Bold thing", "Under thing"]
);

// A REAL options argument still wins. The fallback must not start second-guessing a model
// that did exactly what the tool description asked for.
check(
  "three real options are used as-is, and the question is left alone",
  labels(questionCard(
    { options: "Dark Knight (Samurai armor)\nHell Warrior (Dark fantasy)\nSamurai (Classic)" },
    "What mixamo character should we use for the player?"
  ).options),
  ["Dark Knight (Samurai armor)", "Hell Warrior (Dark fantasy)", "Samurai (Classic)"]
);

// Measured, r-18cf17272d7adbb0-223bccea seq 127: four newline-separated choices, no
// bullets, no Option-N markers. This is a real card; if it parses below 2 the view
// skips the card and the human only sees the composer.
const ARISSA_OPTS =
  "Use Arissa (female samurai from existing FBX files)\n" +
  "Find a different male samurai character from Mixamo\n" +
  "Use Arissa but modify her appearance via materials\n" +
  "Keep placeholder boxes but wire up real animations";
check(
  "four plain newline choices from the Mixamo ask_user are a card",
  labels(questionCard({ options: ARISSA_OPTS }, "Which character model?").options),
  [
    "Use Arissa (female samurai from existing FBX files)",
    "Find a different male samurai character from Mixamo",
    "Use Arissa but modify her appearance via materials",
    "Keep placeholder boxes but wire up real animations",
  ]
);

// Measured, r-18cf39935cc24ac8-2ef88a6b: Plan mode wrote four lettered questions into the
// answer channel and never called ask_user. The loop now promotes that text to ask_user
// with the whole block as `question` and no `options` argument -- so the card must come
// from questionFromText, same as an ask_user that stuffed choices into the question.
const STRANGE =
  "This is a creative design question, so let me narrow down the vision:\n\n" +
  "**Question 1: What's the core experience?**\n\n" +
  "A) **Exploration toy** - A sandbox where users zoom into fractals.\n\n" +
  "B) **Abstract game** - Players navigate through shifting landscapes.\n\n" +
  "C) **Generative art tool** - Users create their own fractal art.\n\n" +
  "D) **Rhythmic/kinetic experience** - Math visualizations respond to music.\n\n" +
  "**Question 2: What's the primary interaction model?**\n\n" +
  "A) **Mouse-driven exploration** - Click/drag to zoom.\n\n" +
  "B) **Keyboard-driven manipulation** - Keys change equations.\n\n" +
  "C) **Hybrid** - Mouse for navigation, keyboard for parameters.";
check(
  "promoted ask_user with no options argument still draws a card from A)/B) lines",
  questionCard({}, STRANGE).options.length >= 2,
  true
);
check(
  "and the first two labels are the first question's choices, not the headings",
  labels(questionCard({}, STRANGE).options).slice(0, 2),
  [
    "Exploration toy - A sandbox where users zoom into fractals.",
    "Abstract game - Players navigate through shifting landscapes.",
  ]
);

// Measured, r-18cfb8403c4b4d90-9bd00186 seq 69: six real choices, then `</parameter>` glued
// to the last line and seven `</function>` lines the guard swallowed as value text. Those
// lines used to become clickable rows G–M. They are framing, not choices.
const FRACTAL_OPTS =
  "Audio resonance system — use the resonance_hz values already defined in every landmark to generate spatialized sound as you approach discoveries, turning exploration into an auditory experience\n" +
  "Fractal lifeforms — procedural creatures that inhabit specific coordinate zones, flee when you get close, and drop collectible \"data fragments\" when tracked down\n" +
  "Multiplayer co-op exploration — two probes sharing one fractal space, communicating landmark discoveries in real-time, with shared mission objectives\n" +
  "Challenge mode — time-limited zoom races, parameter constraint puzzles (e.g. \"reach this coordinate using only Julia set + orbit trap\"), and a tournament leaderboard\n" +
  "Procedural narrative — story fragments unlocked by discovering specific landmark combinations, building a lore about what these mathematical spaces actually are\n" +
  "Export/share ecosystem — generate shareable image URLs with embedded state codes, embeddable fractal viewers, and a community gallery server</parameter>\n" +
  "</function>\n</function>\n</function>\n</function>\n</function>\n</function>\n</function>";
check(
  "leaked tool-call closers are not options",
  labels(questionCard({ options: FRACTAL_OPTS }, "Which direction for a new feature?").options).length,
  6
);
check(
  "and the last real choice keeps its text, without the glued closer",
  labels(questionCard({ options: FRACTAL_OPTS }, "Which direction for a new feature?").options)[5],
  "Export/share ecosystem — generate shareable image URLs with embedded state codes, embeddable fractal viewers, and a community gallery server"
);

// NO CHOICE TO OFFER. The caller draws a card only at >= 2, so both of these render as the
// plain tool row and the composer box -- which is the honest answer to "we could not find
// a choice set", and strictly better than one button the human never chose.
check(
  "a single option with no list in the question stays below the card threshold",
  questionCard({ options: "Just the one thing" }, "Which way?").options.length < 2,
  true
);
check(
  "an open question with no options offers nothing to click",
  questionCard({}, "What should the enemy AI prioritise?").options.length,
  0
);

if (failures > 0) {
  console.error(`verify-question-options: ${failures} failure(s)`);
  process.exit(1);
}
console.log("verify-question-options: ok, all assertions");
