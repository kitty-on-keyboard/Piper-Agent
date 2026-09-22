import assert from 'node:assert/strict';
import { apply, emptyState, offloadPercent, expandLive, expandLog, estimateTokens } from './feed_model.js';

let s = emptyState();
s = apply(s, { kind: 'delta', channel: 'thinking', text: 'plan the ' });
s = apply(s, { kind: 'delta', channel: 'thinking', text: 'edit' });
s = apply(s, { kind: 'delta', channel: 'answer', text: 'done' });
s = apply(s, { kind: 'tool_call', seq: 1, tool: 'read_file', path: 'src/a.js' });
s = apply(s, { kind: 'tool_result', seq: 1, tool: 'read_file', status: 'ok', summary: 'x'.repeat(400), read_bytes: 12, edit_bytes: 0 });
s = apply(s, { kind: 'write', seq: 2, path: 'src/b.js', edit_bytes: 40, tool: 'write_file' });
s = apply(s, { kind: 'turn', think_tokens: 10, text_tokens: 4, tool_tokens: 6 });
s = apply(s, { kind: 'prompt', tokens: 9000 });
s = apply(s, { kind: 'orch', text: 'slice running' });

// After the first turn, thinking is sealed into thoughts.
assert.equal(s.thinking, '');
assert.deepEqual(s.thoughts, [{ seq: undefined, text: 'plan the edit' }]);
assert.equal(s.answer, 'done');
assert.equal(s.actions.length, 2);
assert.equal(s.actions[0].tool, 'read_file');
assert.equal(s.actions[0].path, 'src/a.js');
assert.equal(s.actions[0].status, 'ok');
assert.equal(s.actions[0].read_bytes, 12);
assert.equal(s.actions[0].summary.length, 240);
assert.equal(s.actions[1].path, 'src/b.js');
assert.equal(s.actions[1].edit_bytes, 40);
assert.equal(s.generatedTokens, 20);
assert.equal(s.contextTokens, 9000);
assert.equal(s.cloudTokens, null);
assert.equal(offloadPercent(s), null);
assert.equal(s.cloudNotes.length, 1);
assert.equal(s.cloudNotes[0].text, 'slice running');

s = apply(s, { kind: 'result', generated_tokens: 100 });
assert.equal(s.generatedTokens, 100);
s = apply(s, { kind: 'turn', think_tokens: 5, text_tokens: 5, tool_tokens: 5 });
assert.equal(s.generatedTokens, 100);

s = apply(s, { kind: 'orch', text: 'reviewed', tokens: 25 });
assert.equal(s.cloudTokens, 25);
assert.equal(offloadPercent(s), 80);

s = apply(s, { kind: 'tool_result', seq: 9, tool: 'read_file', status: 'ok', summary: 'kept', read_bytes: 3, edit_bytes: 0 });
assert.equal(s.actions.find((a) => a.seq === 1).path, 'src/a.js');
assert.equal(s.actions.find((a) => a.seq === 9).summary, 'kept');
assert.equal(s.actions.length, 3);

// A delta after the turn lands on thinking (the next open block), not inside sealed thoughts.
s = apply(s, { kind: 'delta', channel: 'thinking', text: '<think>no' });
assert.ok(s.thinking.endsWith('<think>no'));

// A second thinking delta plus a second turn seals a second thought and does not erase the first.
s = apply(s, { kind: 'turn', think_tokens: 1, text_tokens: 1, tool_tokens: 1 });
assert.equal(s.thoughts.length, 2);
assert.equal(s.thoughts[0].text, 'plan the edit');
assert.equal(s.thoughts[1].text, '<think>no');

let liveState = emptyState();
const liveLines = [
  { seq: 1, kind: 'delta', channel: 'thinking', text: 'hmm' },
  { seq: 2, kind: 'turn', tool: 'read_file', path: 'src/a.js', status: 'ok', summary: 'body', read_bytes: 4, edit_bytes: 0, think_tokens: 3, text_tokens: 1, tool_tokens: 2 },
  { seq: 3, kind: 'write', tool: 'write_file', path: 'src/b.js', edit_bytes: 8 },
];
for (const line of liveLines) {
  for (const ev of expandLive(line)) liveState = apply(liveState, ev);
}
// After the turn, thinking is sealed and seq is preserved.
assert.equal(liveState.thinking, '');
assert.equal(liveState.thoughts[0].text, 'hmm');
assert.equal(liveState.thoughts[0].seq, 2);
assert.equal(liveState.actions.length, 2);
assert.equal(liveState.actions[0].tool, 'read_file');
assert.equal(liveState.actions[0].path, 'src/a.js');
assert.equal(liveState.actions[0].status, 'ok');
assert.equal(liveState.actions[0].read_bytes, 4);
assert.equal(liveState.actions[1].path, 'src/b.js');
assert.equal(liveState.actions[1].edit_bytes, 8);
assert.equal(liveState.generatedTokens, 6);
liveState = apply(liveState, expandLog({ kind: 'prompt', tokens: '9000' })[0]);
assert.equal(liveState.generatedTokens, 6);
assert.equal(liveState.contextTokens, 9000);
assert.equal(offloadPercent(liveState), null);
const logTurn = expandLog({ kind: 'turn', think_tokens: '2', text_tokens: '2', tool_tokens: '1' });
liveState = apply(liveState, logTurn[0]);
assert.equal(liveState.generatedTokens, 11);

// Checklist test.
let cs = emptyState();
cs = apply(cs, { kind: 'checklist', open: '1', count: '2', items: '[x] Read player.gd | [ ] Write item_drop.gd' });
assert.equal(cs.checklist.items.length, 2);
assert.equal(cs.checklist.items[0].done, true);
assert.equal(cs.checklist.items[1].done, false);
assert.equal(cs.checklist.items[1].text, 'Write item_drop.gd');
assert.equal(cs.checklist.open, 1);

// A second checklist event replaces the list in place.
cs = apply(cs, { kind: 'checklist', open: '0', count: '2', items: '[x] Read player.gd | [x] Write item_drop.gd' });
assert.equal(cs.checklist.items.length, 2);
assert.equal(cs.checklist.items[0].done, true);
assert.equal(cs.checklist.items[1].done, true);
assert.equal(cs.checklist.open, 0);

// estimateTokens tests.
assert.equal(estimateTokens('abcd'), 1);
assert.equal(estimateTokens(''), 0);
assert.equal(estimateTokens(null), 0);

// Cloud tokens test.
let cloudState = emptyState();
cloudState = apply(cloudState, { kind: 'cloud', tokens: estimateTokens('a'.repeat(8)) });
assert.equal(cloudState.cloudTokens, 2);
cloudState = apply(cloudState, { kind: 'orch', text: 'note' });
assert.equal(cloudState.cloudTokens, 2); // unchanged by orch without tokens

console.log('ok');
