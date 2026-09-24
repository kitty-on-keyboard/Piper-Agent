import { apply, emptyState, expandLive, expandLog } from './feed_model.js';

class PiperWatch {
  constructor() {
    this.feed = emptyState();
    this.seenLive = new Set();
    this.seenOrch = new Set();
    this.liveActive = false;
    this.turnCount = 0;
    this.sliceId = null;
    this.startedAt = null;
    this.running = false;
    this.conn = document.getElementById('conn');
    this.cwd = document.getElementById('cwd');
    this.sliceName = document.getElementById('sliceName');
    this.status = document.getElementById('status');
    this.elapsed = document.getElementById('elapsed');
    this.turn = document.getElementById('turn');
    this.rail = document.getElementById('rail');
    this.brief = document.getElementById('brief');
    this.feedEl = document.getElementById('feed');
    this.review = document.getElementById('review');
    setInterval(() => this.tick(), 1000);
    this.connect();
  }

  connect() {
    const source = new EventSource('/api/events');
    source.onopen = () => {
      this.setConn(true);
      this.load();
    };
    source.onerror = () => this.setConn(false);
    source.addEventListener('slice_changed', (event) => {
      this.applySnapshot(JSON.parse(event.data));
    });
    source.addEventListener('task_updated', (event) => {
      this.showTask(JSON.parse(event.data));
    });
    source.addEventListener('result_updated', (event) => {
      this.showResult(JSON.parse(event.data));
    });
    source.addEventListener('live_event', (event) => {
      this.handleLive(JSON.parse(event.data));
    });
    source.addEventListener('orch_event', (event) => {
      this.handleOrch(JSON.parse(event.data));
    });
    source.addEventListener('log_event', (event) => {
      this.handleLog(JSON.parse(event.data));
    });
  }

  setConn(up) {
    this.conn.textContent = up ? 'Live' : 'Reconnecting';
    this.conn.className = 'pill ' + (up ? 'on' : 'off');
  }

  async load() {
    const response = await fetch('/api/status');
    this.applySnapshot(await response.json());
  }

  resetFeed() {
    this.feed = emptyState();
    this.seenLive = new Set();
    this.seenOrch = new Set();
    this.liveActive = false;
    this.turnCount = 0;
    this.startedAt = null;
    this.running = false;
    this.review.hidden = true;
    this.review.replaceChildren();
    this.turn.textContent = 'turn 0';
  }

  applySnapshot(data) {
    if (!data || typeof data !== 'object') return;
    const id = (data.active && data.active.id) || (data.task && data.task.id) || null;
    if (id && id !== this.sliceId) this.resetFeed();
    this.sliceId = id;
    if (data.cwd) {
      const parts = String(data.cwd).split('/');
      this.cwd.textContent = parts[parts.length - 1] || data.cwd;
    }
    this.renderRail(data.slices || [], data.progress || []);
    if (data.task) this.showTask(data.task);
    for (const line of data.recent_live || []) this.handleLive(line);
    for (const line of data.recent_orch || []) this.handleOrch(line);
    if (!this.liveActive) {
      for (const event of data.recent_events || []) this.handleLog(event);
    }
    if (data.result) this.showResult(data.result);
    this.render();
  }

  showTask(task) {
    if (!task || !task.id) return;
    if (task.id !== this.sliceId) {
      this.sliceId = task.id;
      this.resetFeed();
      this.sliceId = task.id;
    }
    this.sliceName.textContent = task.id;
    const text = typeof task.prompt === 'string' ? task.prompt.trim() : '';
    this.brief.hidden = !text;
    this.brief.replaceChildren();
    if (text) {
      const kicker = document.createElement('div');
      kicker.className = 'kicker';
      kicker.textContent = 'Brief';
      const body = document.createElement('div');
      body.className = 'brief-text';
      body.textContent = text;
      this.brief.append(kicker, body);
    }
    if (!this.startedAt) {
      this.startedAt = Date.now();
      this.running = true;
    }
    if (this.status.textContent === 'idle' || this.status.textContent === 'running') {
      this.setStatus('running');
    }
  }

  showResult(result) {
    if (!result) return;
    this.running = false;
    const ok = result.status === 'ok';
    this.setStatus(ok ? 'ok' : (result.status || 'stalled'));
    if (!this.review.hidden) return;
    const card = document.createElement('div');
    const kicker = document.createElement('div');
    kicker.className = 'kicker';
    kicker.textContent = ok ? 'Passed' : 'Review';
    const body = document.createElement('div');
    body.className = 'review-text';
    const files = Array.isArray(result.files_touched) ? result.files_touched.join(', ') : '';
    body.textContent = [result.message || '', files].filter(Boolean).join('\n');
    card.append(kicker, body);
    this.review.hidden = false;
    this.review.replaceChildren(card);
  }

  handleLive(line) {
    if (!line || typeof line !== 'object') return;
    if (line.seq != null) {
      if (this.seenLive.has(line.seq)) return;
      this.seenLive.add(line.seq);
    }
    if (line.kind === 'delta' || line.kind === 'turn' || line.kind === 'write') {
      this.liveActive = true;
    }
    if (line.kind === 'turn') {
      this.turnCount += 1;
      this.turn.textContent = `turn ${this.turnCount}`;
      this.setStatus('running');
    }
    for (const event of expandLive(line)) this.feed = apply(this.feed, event);
    this.render();
  }

  handleOrch(line) {
    if (!line || typeof line !== 'object') return;
    const key = JSON.stringify(line);
    if (this.seenOrch.has(key)) return;
    this.seenOrch.add(key);
    const text = line.text || line.note || '';
    this.feed = apply(this.feed, { kind: 'orch', text });
    if (line.kind === 'review' || line.verdict) {
      const card = document.createElement('div');
      const kicker = document.createElement('div');
      kicker.className = 'kicker';
      kicker.textContent = line.verdict ? `Review · ${line.verdict}` : 'Review';
      const body = document.createElement('div');
      body.className = 'review-text';
      body.textContent = text;
      card.append(kicker, body);
      this.review.hidden = false;
      this.review.replaceChildren(card);
    }
    this.render();
  }

  handleLog(event) {
    if (!event || typeof event !== 'object') return;
    const structural = event.kind === 'turn' || event.kind === 'tool_call' ||
      event.kind === 'tool_result' || event.kind === 'write';
    if (structural && this.liveActive) return;
    for (const item of expandLog(event)) this.feed = apply(this.feed, item);
    this.render();
  }

  setStatus(state) {
    this.status.textContent = state;
    this.status.className = 'pill ' + (state === 'ok' ? 'ok' : state === 'running' ? 'run' : state === 'idle' ? '' : 'bad');
  }

  tick() {
    if (!this.startedAt) return;
    const secs = Math.floor((Date.now() - this.startedAt) / 1000);
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    this.elapsed.textContent = `${m}:${String(s).padStart(2, '0')}`;
  }

  renderRail(slices, progress) {
    const notes = new Map();
    for (const line of progress) {
      const parts = line.split('|').map((part) => part.trim());
      if (parts.length >= 2) notes.set(parts[0], parts[1]);
    }
    this.rail.replaceChildren();
    if (!slices.length) {
      const empty = document.createElement('div');
      empty.className = 'empty';
      empty.textContent = 'No slices yet';
      this.rail.append(empty);
      return;
    }
    for (const slice of slices) {
      const item = document.createElement('div');
      item.className = 'rail-item' + (slice.id === this.sliceId ? ' active' : '');
      const id = document.createElement('span');
      id.className = 'rail-id';
      id.textContent = slice.id;
      const status = document.createElement('span');
      status.className = 'rail-status';
      status.textContent = notes.get(slice.id) || slice.status || '';
      item.append(id, status);
      this.rail.append(item);
    }
  }

  render() {
    const order = [];
    const thoughts = this.feed.thoughts || [];
    const bySeq = new Map();
    for (const thought of thoughts) {
      if (thought.seq != null) bySeq.set(String(thought.seq), thought);
    }
    const used = new Set();
    for (const action of this.feed.actions) {
      const thought = bySeq.get(String(action.seq));
      if (thought && !used.has(thought)) {
        order.push({ type: 'thought', thought });
        used.add(thought);
      }
      order.push({ type: 'action', action });
    }
    for (const thought of thoughts) {
      if (!used.has(thought)) order.push({ type: 'thought', thought });
    }
    if (this.feed.thinking) order.push({ type: 'live' });
    if (this.feed.answer) order.push({ type: 'answer' });

    this.feedEl.replaceChildren();
    if (!order.length && this.brief.hidden) {
      const empty = document.createElement('div');
      empty.className = 'empty';
      empty.textContent = 'Waiting for a slice';
      this.feedEl.append(empty);
      return;
    }
    for (const item of order) {
      if (item.type === 'thought' || item.type === 'live') {
        const details = document.createElement('details');
        details.className = 'thought';
        details.open = item.type === 'live';
        const summary = document.createElement('summary');
        summary.textContent = 'Reasoning';
        const pre = document.createElement('pre');
        pre.textContent = item.type === 'live' ? this.feed.thinking : item.thought.text;
        details.append(summary, pre);
        this.feedEl.append(details);
      } else if (item.type === 'answer') {
        const block = document.createElement('div');
        block.className = 'answer';
        block.textContent = this.feed.answer;
        this.feedEl.append(block);
      } else {
        this.feedEl.append(this.actionRow(item.action));
      }
    }
    const log = document.getElementById('log');
    if (log) log.scrollTop = log.scrollHeight;
  }

  actionRow(action) {
    const row = document.createElement('div');
    row.className = 'tool';
    const head = document.createElement('div');
    head.className = 'tool-head';
    const name = document.createElement('span');
    name.className = 'tool-name';
    name.textContent = action.tool || 'action';
    const meta = document.createElement('span');
    meta.className = 'tool-meta';
    const bits = [];
    if (action.path) bits.push(action.path);
    else if (action.command) bits.push(action.command);
    if (action.status) bits.push(action.status);
    if (action.summary) bits.push(action.summary);
    meta.textContent = bits.join(' · ');
    head.append(name, meta);
    row.append(head);
    return row;
  }
}

new PiperWatch();
