// feed_model.js — pure state machine for the orchestrator visualizer feed.
// ESM. No dependencies. No DOM.

/**
 * @returns the initial feed state.
 */
export function emptyState() {
  return {
    actions: [],
    thinking: '',
    thoughts: [],
    answer: '',
    generatedTokens: 0,
    generatedFinal: false,
    contextTokens: null,
    cloudTokens: null,
    cloudNotes: [],
    checklist: null,
  };
}

// --- helpers -----------------------------------------------------------

function finiteNum(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function strOrEmpty(v) {
  return typeof v === 'string' ? v : '';
}

function numOrZero(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/**
 * Coerce any value to a finite number via `Number`; non-finite -> 0.
 * Used by expandLive/expandLog where wire values may be strings or missing.
 */
function coerceNum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Create a fresh action row with defaults, then overlay fields from `src`
 * that are present and of the right type.
 */
function makeRow(seq, src) {
  const row = {
    seq: typeof seq === 'number' ? seq : 0,
    tool: '',
    path: '',
    command: '',
    status: '',
    summary: '',
    read_bytes: 0,
    edit_bytes: 0,
  };
  if (src) {
    if ('tool' in src) row.tool = strOrEmpty(src.tool);
    if ('path' in src) row.path = strOrEmpty(src.path);
    if ('command' in src) row.command = strOrEmpty(src.command);
    if ('status' in src) row.status = strOrEmpty(src.status);
    if ('summary' in src) row.summary = strOrEmpty(src.summary);
    if ('read_bytes' in src) row.read_bytes = numOrZero(src.read_bytes);
    if ('edit_bytes' in src) row.edit_bytes = numOrZero(src.edit_bytes);
  }
  return row;
}

/**
 * Find the index of the row with the given seq, or -1.
 */
function findRowIndex(actions, seq) {
  for (let i = 0; i < actions.length; i++) {
    if (actions[i].seq === seq) return i;
  }
  return -1;
}

/**
 * Estimate token count from text length. Non-string or empty returns 0.
 */
export function estimateTokens(text) {
  if (typeof text !== 'string' || text.length === 0) return 0;
  return Math.max(1, Math.round(text.length / 4));
}

// --- main reducer ------------------------------------------------------

/**
 * Apply one event to `state` and return a NEW state object.
 * The input is never mutated.
 */
export function apply(state, event) {
  const kind = event && event.kind;

  switch (kind) {
    case 'delta': {
      const next = { ...state };
      if (event.channel === 'thinking') {
        next.thinking = state.thinking + (typeof event.text === 'string' ? event.text : '');
      } else if (event.channel === 'answer') {
        next.answer = state.answer + (typeof event.text === 'string' ? event.text : '');
      }
      // any other channel: ignored (still return a copy so identity differs)
      return next;
    }

    case 'tool_call': {
      const seq = event.seq;
      const idx = findRowIndex(state.actions, seq);
      let actions;
      if (idx === -1) {
        actions = [...state.actions, makeRow(seq, event)];
      } else {
        actions = state.actions.slice();
        const row = { ...actions[idx] };
        if ('tool' in event) row.tool = strOrEmpty(event.tool);
        if ('path' in event) row.path = strOrEmpty(event.path);
        if ('command' in event) row.command = strOrEmpty(event.command);
        // do NOT clear status, summary, read_bytes, edit_bytes
        actions[idx] = row;
      }
      return { ...state, actions };
    }

    case 'tool_result': {
      const seq = event.seq;
      const idx = findRowIndex(state.actions, seq);
      let actions;
      if (idx === -1) {
        const row = makeRow(seq, null);
        row.status = strOrEmpty(event.status);
        row.read_bytes = numOrZero(event.read_bytes);
        row.edit_bytes = numOrZero(event.edit_bytes);
        row.summary = String(event.summary).slice(0, 240);
        if ('tool' in event) row.tool = strOrEmpty(event.tool);
        actions = [...state.actions, row];
      } else {
        actions = state.actions.slice();
        const row = { ...actions[idx] };
        if ('tool' in event) row.tool = strOrEmpty(event.tool);
        row.status = strOrEmpty(event.status);
        row.read_bytes = numOrZero(event.read_bytes);
        row.edit_bytes = numOrZero(event.edit_bytes);
        row.summary = String(event.summary).slice(0, 240);
        actions[idx] = row;
      }
      return { ...state, actions };
    }

    case 'write': {
      const seq = event.seq;
      const idx = findRowIndex(state.actions, seq);
      let actions;
      if (idx === -1) {
        const row = makeRow(seq, null);
        row.path = strOrEmpty(event.path);
        row.edit_bytes = numOrZero(event.edit_bytes);
        if ('tool' in event) row.tool = strOrEmpty(event.tool);
        actions = [...state.actions, row];
      } else {
        actions = state.actions.slice();
        const row = { ...actions[idx] };
        row.path = strOrEmpty(event.path);
        row.edit_bytes = numOrZero(event.edit_bytes);
        if ('tool' in event) row.tool = strOrEmpty(event.tool);
        actions[idx] = row;
      }
      return { ...state, actions };
    }

    case 'turn': {
      let next = { ...state };
      // Seal non-empty thinking as a thought when a turn arrives.
      if (state.thinking !== '') {
        const seq = typeof event.seq === 'number' ? event.seq : undefined;
        next.thoughts = [...state.thoughts, { seq, text: state.thinking }];
        next.thinking = '';
      }
      // Keep the existing generatedFinal early return (no token update).
      if (state.generatedFinal) {
        return next;
      }
      const add =
        finiteNum(event.think_tokens) +
        finiteNum(event.text_tokens) +
        finiteNum(event.tool_tokens);
      next.generatedTokens = state.generatedTokens + add;
      return next;
    }

    case 'prompt': {
      if (typeof event.tokens === 'number' && Number.isFinite(event.tokens)) {
        return { ...state, contextTokens: event.tokens };
      }
      return { ...state };
    }

    case 'result': {
      if (typeof event.generated_tokens === 'number' && Number.isFinite(event.generated_tokens)) {
        return { ...state, generatedTokens: event.generated_tokens, generatedFinal: true };
      }
      return { ...state };
    }

    case 'orch': {
      const next = { ...state, cloudNotes: [...state.cloudNotes] };
      next.cloudNotes.push({ text: String(event.text || '') });
      if (Object.hasOwn(event, 'tokens') && typeof event.tokens === 'number' && Number.isFinite(event.tokens)) {
        next.cloudTokens = event.tokens;
      }
      return next;
    }

    case 'cloud': {
      // If tokens is a finite number, replace cloudTokens; otherwise leave it unchanged.
      if (typeof event.tokens === 'number' && Number.isFinite(event.tokens)) {
        return { ...state, cloudTokens: event.tokens };
      }
      return { ...state };
    }

    case 'checklist': {
      let items;
      if (Array.isArray(event.items)) {
        items = event.items.map((it) => ({
          text: typeof it.text === 'string' ? it.text : '',
          done: it.done === true,
        }));
      } else if (typeof event.items === 'string') {
        // Parse pipe-separated items: " [x] One | [ ] Two "
        items = event.items.split('|').map((raw) => {
          const trimmed = raw.trim();
          let done = false;
          let text = trimmed;
          if (/^\[x\]/i.test(trimmed)) {
            done = true;
            text = trimmed.replace(/^\[x\]\s*/, '');
          } else if (/^\[\s*\]/.test(trimmed)) {
            text = trimmed.replace(/^\[\s*\]\s*/, '');
          }
          return { text, done };
        });
      } else {
        items = [];
      }
      const open = items.filter((it) => it.done !== true).length;
      return { ...state, checklist: { open, total: items.length, items } };
    }

    default:
      // unknown kind: ignored, but still return a new object identity
      return { ...state };
  }
}

// --- wire -> event expanders --------------------------------------------
// These turn raw journal / event-log lines into reducer events. They are pure
// and side-effect free; `apply` consumes their output.

/**
 * Expand one live.jsonl line (kind delta | turn | write) into reducer events.
 * Unknown kinds return [].
 */
export function expandLive(line) {
  if (!line || typeof line !== 'object') return [];
  const kind = line.kind;

  if (kind === 'delta') {
    return [{ kind: 'delta', channel: line.channel, text: typeof line.text === 'string' ? line.text : '' }];
  }

  if (kind === 'turn') {
    const out = [];
    if (typeof line.tool === 'string' && line.tool !== '') {
      out.push({ kind: 'tool_call', seq: line.seq, tool: line.tool, path: line.path, command: line.command });
      out.push({ kind: 'tool_result', seq: line.seq, tool: line.tool, status: line.status, summary: line.summary, read_bytes: coerceNum(line.read_bytes), edit_bytes: coerceNum(line.edit_bytes) });
    }
    out.push({ kind: 'turn', seq: line.seq, think_tokens: coerceNum(line.think_tokens), text_tokens: coerceNum(line.text_tokens), tool_tokens: coerceNum(line.tool_tokens) });
    return out;
  }

  if (kind === 'write') {
    return [{ kind: 'write', seq: line.seq, tool: line.tool, path: line.path, edit_bytes: coerceNum(line.edit_bytes) }];
  }

  return [];
}

/**
 * Expand one events.jsonl object into reducer events. Event-log numbers arrive
 * as strings, so they are coerced with Number; a non-finite value drops that
 * event from the returned array.
 */
export function expandLog(ev) {
  if (!ev || typeof ev !== 'object') return [];
  const kind = ev.kind;

  if (kind === 'prompt') {
    const tokens = Number(ev.tokens);
    return Number.isFinite(tokens) ? [{ kind: 'prompt', tokens }] : [];
  }

  if (kind === 'turn') {
    const think_tokens = Number(ev.think_tokens);
    const text_tokens = Number(ev.text_tokens);
    const tool_tokens = Number(ev.tool_tokens);
    if (!Number.isFinite(think_tokens) || !Number.isFinite(text_tokens) || !Number.isFinite(tool_tokens)) return [];
    return [{ kind: 'turn', think_tokens, text_tokens, tool_tokens }];
  }

  if (kind === 'tool_call') {
    return [{ kind: 'tool_call', seq: ev.seq, tool: ev.tool || '', path: ev.path || ev['arg.path'] || '', command: ev.command || '' }];
  }

  if (kind === 'tool_result') {
    const read_bytes = Number(ev.read_bytes);
    const edit_bytes = Number(ev.edit_bytes);
    if (!Number.isFinite(read_bytes) && !Number.isFinite(edit_bytes)) return [];
    return [{ kind: 'tool_result', seq: ev.seq, tool: ev.tool, status: ev.status, summary: ev.summary, read_bytes: Number.isFinite(read_bytes) ? read_bytes : 0, edit_bytes: Number.isFinite(edit_bytes) ? edit_bytes : 0 }];
  }

  if (kind === 'write') {
    const edit_bytes = Number(ev.edit_bytes);
    if (!Number.isFinite(edit_bytes)) return [];
    return [{ kind: 'write', seq: ev.seq, tool: ev.tool, path: ev.path, edit_bytes }];
  }

  if (kind === 'checklist') {
    if (typeof ev.items === 'string' || Array.isArray(ev.items)) {
      return [{ kind: 'checklist', items: ev.items, open: ev.open, count: ev.count }];
    }
    return [];
  }

  // generation, token, and anything else: no feed event.
  return [];
}

// --- derived -----------------------------------------------------------

/**
 * Percentage of tokens that were generated locally vs. total (local + cloud).
 * Returns null when the computation is not meaningful.
 */
export function offloadPercent(state) {
  if (state.cloudTokens == null) return null;
  const total = state.generatedTokens + state.cloudTokens;
  if (total <= 0) return null;
  return Math.round((state.generatedTokens / total) * 100);
}
