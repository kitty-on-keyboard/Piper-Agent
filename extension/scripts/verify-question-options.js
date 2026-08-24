// A question card must offer the choices the model meant, and no others.
//
//   node scripts/verify-question-options.js     # reads src/webview.ts directly
//
// WHAT IT PINS. `ask_question` declares `options` as "newline-separated choices, one per
// line", and the view split on /\n+/ to honour that. Models write richer than the
// description asks -- these are verbatim from ~/Library/Logs/LM_Pipe/events.jsonl:
//
//     Fixed cinematic angles (PS1 original)
//       - Dramatic, controlled views per encounter
//       - Limited player freedom, strong storytelling
//       - Closer to the original experience
//
//     Free third-person camera (modern action games)
//       - Full player control over camera
//       ...
//
// A blank line separates the two real choices and the bullets argue for the one above
// them. Splitting on newlines collapsed the blank line and produced EIGHT clickable rows
// for a two-way choice -- six of them fragments that answer nothing. Reported from the
// installed extension as "a choice of 4 options is becoming 12".
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
try {
  questionOptions = new Function(`${block}; return questionOptions;`)();
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

// 1. THE REPORTED DEFECT, verbatim from the event log. Two choices, not eight.
const camera =
  "Fixed cinematic angles (PS1 original)\n" +
  "  - Dramatic, controlled views per encounter\n" +
  "  - Limited player freedom, strong storytelling\n" +
  "  - Closer to the original experience\n" +
  "\n" +
  "Free third-person camera (modern action games)\n" +
  "  - Full player control over camera\n" +
  "  - More freedom in combat positioning\n" +
  "  - Feels more like Devil May Cry / Sekiro";
check("blank-line blocks are two options", labels(questionOptions({ options: camera })), [
  "Fixed cinematic angles (PS1 original)",
  "Free third-person camera (modern action games)",
]);
check(
  "detail lines survive on the option they belong to",
  questionOptions({ options: camera })[1].detail,
  "Full player control over camera · More freedom in combat positioning · Feels more like Devil May Cry / Sekiro"
);

// 2. Three blocks, also from the log: the "12 rows" the report described.
const combat =
  "Fast, agile strikes (Onimusha original)\n  - Quick sword combos, dodge rolls\n  - Spirit absorption as core mechanic\n  - Closer to the source material\n\n" +
  "Heavy, deliberate swings (Sekiro/Dark Souls)\n  - Weighty impacts, stamina management\n  - Parry/defend mechanics\n  - More punishing, methodical combat\n\n" +
  "Hybrid approach\n  - Light attacks for speed, heavy for damage\n  - Mix of quick combos and powerful finishing moves\n  - Best of both worlds";
check("twelve lines are three options", labels(questionOptions({ options: combat })), [
  "Fast, agile strikes (Onimusha original)",
  "Heavy, deliberate swings (Sekiro/Dark Souls)",
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

if (failures > 0) {
  console.error(`verify-question-options: ${failures} failure(s)`);
  process.exit(1);
}
console.log("verify-question-options: ok, 12 assertion(s)");
