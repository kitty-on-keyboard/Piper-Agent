// Main Frontend Logic for Piper Orchestrator Visualizer
import { initOrb } from './orb.js';
import { ConduitBus } from './conduit.js';
import { apply, emptyState, offloadPercent, expandLive, expandLog, estimateTokens } from './feed_model.js';

class PiperVisualizerApp {
  constructor() {
    this.brandOrb = initOrb(document.getElementById('brandOrb'));
    this.piperOrb = initOrb(document.getElementById('piperOrbMount'));
    this.conduit = new ConduitBus(
      document.getElementById('conduitCanvas'),
      document.getElementById('packetOverlay')
    );

    this.piperFeed = document.getElementById('piperFeed');
    this.orchFeed = document.getElementById('orchFeed');
    this.sliceRail = document.getElementById('sliceRail');
    this.connStatus = document.getElementById('connStatus');
    this.connLabel = document.getElementById('connLabel');
    this.cwdLabel = document.getElementById('cwdLabel');
    this.piperStatusBadge = document.getElementById('piperStatusBadge');
    this.piperTurnBadge = document.getElementById('piperTurnBadge');
    this.orchMissionBadge = document.getElementById('orchMissionBadge');

    // Token tracking elements
    this.piperTokensTotal = document.getElementById('piperTokensTotal');
    this.orchTokensTotal = document.getElementById('orchTokensTotal');
    this.offloadPct = document.getElementById('offloadPct');
    this.piperTokensPaneBadge = document.getElementById('piperTokensPaneBadge');
    this.orchTokensPaneBadge = document.getElementById('orchTokensPaneBadge');

    this.piperTokens = 0;
    this.orchTokens = 0;
    this.orchTokensKnown = false; // true when we have real cloud token data

    // Activity & progress tracking
    this.activityEl = document.getElementById('piperActivity');
    this.elapsedEl = document.getElementById('elapsedTimer');
    this.filesTouchedBadge = document.getElementById('filesTouchedBadge');
    this.taskStartTime = null;
    this.elapsedInterval = null;
    this.filesTouched = new Set();

    this.slices = new Map();
    this.currentSliceId = null;
    this.currentAssistantMsg = null;
    this.activeThoughtBlock = null;
    this.activeThoughtContent = null;
    this.lastTaskHash = null;
    this.lastResultHash = null;
    this.seenSeqs = new Set();

    // Feed reducer state
    this.feed = emptyState();
    this.liveActive = false;
    this.seenLive = new Set();
    this.seenOrch = new Set();
    this.turnCount = 0;
    this.piperSection = null;
    this.orchRunning = null;
    this.lastAction = '';
    this.thoughtEl = null;
    this.answerEl = null;

    // Thinking mode: collapsed | expanded | hidden
    this.thinkingMode = localStorage.getItem('piperThinkingMode') || 'collapsed';
    this.thinkingToggleEl = document.getElementById('thinkingToggle');
    this.applyThinkingMode();
    if (this.thinkingToggleEl) {
      this.thinkingToggleEl.addEventListener('click', () => this.cycleThinkingMode());
    }

    this.initSSE();
  }

  // ---------------------------------------------------------------------------
  // Thinking Mode
  // ---------------------------------------------------------------------------

  cycleThinkingMode() {
    const order = ['collapsed', 'expanded', 'hidden'];
    const idx = order.indexOf(this.thinkingMode);
    this.thinkingMode = order[(idx + 1) % order.length];
    localStorage.setItem('piperThinkingMode', this.thinkingMode);
    this.applyThinkingMode();
  }

  applyThinkingMode() {
    document.body.classList.remove('thinking-collapsed', 'thinking-expanded', 'thinking-hidden');
    document.body.classList.add(`thinking-${this.thinkingMode}`);

    // Update toggle button
    if (this.thinkingToggleEl) {
      const labels = { collapsed: '◇ Collapsed', expanded: '◆ Expanded', hidden: '○ Hidden' };
      this.thinkingToggleEl.querySelector('.toggle-label').textContent = labels[this.thinkingMode];
      this.thinkingToggleEl.setAttribute('data-mode', this.thinkingMode);
    }

    // Apply open/close state to all existing thought blocks
    const blocks = document.querySelectorAll('.thought-block');
    blocks.forEach(b => {
      if (this.thinkingMode === 'expanded') b.open = true;
      else if (this.thinkingMode === 'collapsed') b.open = false;
    });
  }

  // ---------------------------------------------------------------------------
  // Activity & Elapsed
  // ---------------------------------------------------------------------------

  setActivity(text) {
    if (this.activityEl) {
      this.activityEl.textContent = text || '';
      this.activityEl.classList.toggle('active', !!text);
    }
  }

  startElapsed() {
    this.taskStartTime = Date.now();
    this.stopElapsed();
    this.updateElapsed();
    this.elapsedInterval = setInterval(() => this.updateElapsed(), 1000);
    if (this.elapsedEl) this.elapsedEl.classList.add('running');
  }

  stopElapsed() {
    if (this.elapsedInterval) {
      clearInterval(this.elapsedInterval);
      this.elapsedInterval = null;
    }
    if (this.elapsedEl) this.elapsedEl.classList.remove('running');
  }

  updateElapsed() {
    if (!this.taskStartTime || !this.elapsedEl) return;
    const secs = Math.floor((Date.now() - this.taskStartTime) / 1000);
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    this.elapsedEl.textContent = `${m}:${s.toString().padStart(2, '0')}`;
  }

  trackFile(path) {
    if (!path) return;
    // Normalize to basename for display count
    const base = path.split('/').pop();
    if (base && !this.filesTouched.has(path)) {
      this.filesTouched.add(path);
      if (this.filesTouchedBadge) {
        this.filesTouchedBadge.textContent = `${this.filesTouched.size} files`;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // SSE
  // ---------------------------------------------------------------------------

  initSSE() {
    const evtSource = new EventSource('/api/events');

    evtSource.onopen = () => {
      this.connStatus.style.background = 'var(--ok-bg)';
      this.connStatus.style.color = 'var(--ok)';
      this.connLabel.textContent = 'Connected';
      this.fetchInitialState();
    };

    evtSource.onerror = () => {
      this.connStatus.style.background = 'var(--fail-bg)';
      this.connStatus.style.color = 'var(--fail)';
      this.connLabel.textContent = 'Reconnecting';
    };

    evtSource.addEventListener('status', (e) => {
      const data = JSON.parse(e.data);
      if (data.cwd) this.cwdLabel.textContent = data.cwd.split('/').pop() || data.cwd;
    });

    evtSource.addEventListener('task_updated', (e) => {
      this.handleTask(JSON.parse(e.data));
    });

    evtSource.addEventListener('result_updated', (e) => {
      this.handleResult(JSON.parse(e.data));
    });

    evtSource.addEventListener('gate_requested', (e) => {
      this.handleGate(JSON.parse(e.data));
    });

    evtSource.addEventListener('answer_updated', (e) => {
      this.handleAnswer(JSON.parse(e.data));
    });

    evtSource.addEventListener('log_event', (e) => {
      this.handleLogEvent(JSON.parse(e.data));
    });

    evtSource.addEventListener('live_event', (e) => {
      this.handleLive(JSON.parse(e.data));
    });

    evtSource.addEventListener('orch_event', (e) => {
      this.handleOrch(JSON.parse(e.data));
    });

    evtSource.addEventListener('wake_event', (e) => {
      this.handleWake(JSON.parse(e.data));
    });
  }

  async fetchInitialState() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.cwd) this.cwdLabel.textContent = data.cwd.split('/').pop() || data.cwd;
      if (data.task) this.handleTask(data.task, false);
      // Live lines share no seq space with the event log. Apply them first so
      // a replayed log cannot add a second row or a second token total.
      if (data.recent_live && data.recent_live.length) {
        for (const line of data.recent_live) this.handleLive(line);
      }
      if (data.recent_orch && data.recent_orch.length) {
        for (const line of data.recent_orch) this.handleOrch(line);
      }
      if (data.recent_events && data.recent_events.length) {
        for (const ev of data.recent_events) {
          this.handleLogEvent(ev);
        }
      }
      if (data.result) this.handleResult(data.result, false);
      if (data.awaiting_user) this.handleGate(data.awaiting_user);
    } catch (err) {
      console.error('Failed to load initial state', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  handleTask(task, animate = true) {
    if (!task || !task.id) return;
    const taskHash = JSON.stringify(task);
    if (this.lastTaskHash === taskHash) return;
    this.lastTaskHash = taskHash;

    this.currentSliceId = task.id;
    this.slices.set(task.id, { task, status: 'running' });
    this.updateSliceRail();

    // Reset progress tracking for new task
    this.filesTouched.clear();
    if (this.filesTouchedBadge) this.filesTouchedBadge.textContent = '0 files';
    this.startElapsed();
    this.setActivity('Preparing environment...');

    // No heuristic token counting for orchestrator.
    // Cloud tokens only populated when real data arrives.
    this.updateTokenMetrics();

    if (animate) {
      this.conduit.flyPacket('right-to-left', 'task', task.id, task.check || '');
    }

    this.piperOrb.setState('running');
    this.brandOrb.setState('running');
    this.piperStatusBadge.textContent = 'Working';
    this.orchMissionBadge.textContent = task.id;

    // Render Orchestrator Card
    const card = document.createElement('div');
    card.className = 'msg-card accent-orch';
    card.innerHTML = `
      <div class="msg-card-head">
        <span class="msg-card-tag tag-orch">Slice Dispatched · ${task.id}</span>
        <span class="tag-time">${this.ts()}</span>
      </div>
      <div class="msg-text">${this.formatTaskPrompt(task.prompt || '')}</div>
      ${task.check ? `
        <div class="tool-tile" style="margin-top: 10px;">
          <div class="tool-tile-head">
            <span class="tool-chip check">Acceptance Check</span>
            <span class="tag-time">operator gate</span>
          </div>
          <div class="tool-cmd">${this.esc(task.check)}</div>
        </div>
      ` : ''}
    `;
    this.orchFeed.appendChild(card);
    this.scrollBottom(this.orchFeed);

    // Render Piper Acceptance Card on Piper pane
    const piperCard = document.createElement('div');
    piperCard.className = 'msg-card accent-piper';
    piperCard.innerHTML = `
      <div class="msg-card-head">
        <span class="msg-card-tag tag-piper">Slice Accepted · ${this.esc(task.id)}</span>
        <span class="tag-time">${this.ts()}</span>
      </div>
      <div class="msg-text">Worker assigned. Preparing environment and model prefill...</div>
    `;
    this.piperFeed.appendChild(piperCard);
    this.scrollBottom(this.piperFeed);

    this.currentAssistantMsg = null;
    this.activeThoughtBlock = null;
    this.beginSection(task.id);
    const cloud = estimateTokens(task.prompt || '');
    if (cloud > 0) {
      this.feed = apply(this.feed, { kind: 'cloud', tokens: cloud });
      this.renderFeed();
    }
  }

  handleResult(result, animate = true) {
    if (!result || !result.task_id) return;
    const resultHash = JSON.stringify(result);
    if (this.lastResultHash === resultHash) return;
    this.lastResultHash = resultHash;

    const slice = this.slices.get(result.task_id) || {};
    slice.result = result;
    slice.status = result.status === 'ok' ? 'passed' : 'stalled';
    this.slices.set(result.task_id, slice);
    this.updateSliceRail();

    // Stop elapsed timer
    this.stopElapsed();
    this.setActivity('');

    const genTok = Number(result.generated_tokens);
    if (Number.isFinite(genTok)) {
      this.feed = apply(this.feed, { kind: 'result', generated_tokens: genTok });
    }
    this.renderFeed();

    if (animate) {
      this.conduit.flyPacket('left-to-right', 'result', result.task_id, result.diff_stat || result.status);
    }

    const isOk = result.status === 'ok';
    this.piperOrb.setState(isOk ? 'ok' : 'error');
    this.brandOrb.setState(isOk ? 'ok' : 'error');
    this.piperStatusBadge.textContent = isOk ? 'Completed' : 'Stalled';

    // Piper Result Card
    const piperCard = document.createElement('div');
    piperCard.className = 'msg-card accent-piper';
    piperCard.innerHTML = `
      <div class="msg-card-head">
        <span class="msg-card-tag tag-piper">${isOk ? 'Slice Completed' : 'Slice Stalled'} · ${result.task_id}</span>
        <span class="tag-time">${this.ts()}</span>
      </div>
      <div class="msg-text"><strong>${this.esc(result.message || 'Slice concluded.')}</strong></div>
      <div class="diff-stats">
        <span>Diff:</span>
        <strong style="color: ${isOk ? 'var(--ok)' : 'var(--warn)'}">${this.esc(this.formatDiff(result.diff_stat))}</strong>
      </div>
      ${result.files_touched && result.files_touched.length ? `
        <div class="files-list">
          ${result.files_touched.map(f => `<span class="file-pill">${this.esc(f)}</span>`).join('')}
        </div>
      ` : ''}
    `;
    this.piperFeed.appendChild(piperCard);
    this.scrollBottom(this.piperFeed);

    // Orchestrator Review Card
    const orchCard = document.createElement('div');
    orchCard.className = 'msg-card accent-orch';
    orchCard.innerHTML = `
      <div class="msg-card-head">
        <span class="msg-card-tag tag-orch">Slice Verified · ${result.task_id}</span>
        <span class="tag-time">${this.ts()}</span>
      </div>
      <div class="msg-text">
        Status: <strong>${isOk ? 'Passed' : 'Needs Attention'}</strong><br/>
        ${this.esc(result.message || '')}
      </div>
    `;
    this.orchFeed.appendChild(orchCard);
    this.scrollBottom(this.orchFeed);
  }

  handleGate(gate) {
    if (!gate) return;
    this.conduit.flyPacket('left-to-right', 'gate', 'Gate', gate.tool || '');

    const card = document.createElement('div');
    card.className = 'gate-card';
    card.id = 'gateCard';
    card.innerHTML = `
      <div class="gate-header">
        <span>Gate Escalation: ${this.esc(gate.gate || 'irreversible')}</span>
      </div>
      <div class="msg-text">
        <strong>Tool:</strong> <code>${this.esc(gate.tool || '')}</code><br/>
        <strong>Command:</strong> <code>${this.esc(gate.command || gate.preview || '')}</code>
      </div>
      <div class="gate-actions">
        <button class="btn btn-approve" id="gateApproveBtn">Approve</button>
        <button class="btn btn-deny" id="gateDenyBtn">Deny</button>
      </div>
    `;

    this.piperFeed.appendChild(card);
    this.scrollBottom(this.piperFeed);

    card.querySelector('#gateApproveBtn').onclick = () => this.sendAnswer("approved");
    card.querySelector('#gateDenyBtn').onclick = () => this.sendAnswer("denied");
  }

  async sendAnswer(choice) {
    try {
      await fetch('/api/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer: choice })
      });
      const gateCard = document.getElementById('gateCard');
      if (gateCard) {
        gateCard.style.opacity = '0.5';
        gateCard.querySelector('.gate-actions').innerHTML = `<em>Responded: ${choice}</em>`;
      }
      this.conduit.flyPacket('right-to-left', 'task', 'Answer', choice);
    } catch (e) {
      console.error('Failed to post answer', e);
    }
  }

  handleAnswer(ans) {
    const card = document.createElement('div');
    card.className = 'msg-card accent-orch';
    card.innerHTML = `
      <div class="msg-card-head">
        <span class="msg-card-tag tag-orch">Orchestrator Decision</span>
        <span class="tag-time">${this.ts()}</span>
      </div>
      <div class="msg-text">Gate answer: <strong>${this.esc(JSON.stringify(ans))}</strong></div>
    `;
    this.piperFeed.appendChild(card);
    this.scrollBottom(this.piperFeed);

    // No heuristic token counting for orchestrator
    this.updateTokenMetrics();
  }

  getEventField(ev, key, altKeys = []) {
    if (ev[key] !== undefined && ev[key] !== null) return ev[key];
    for (const alt of altKeys) {
      if (ev[alt] !== undefined && ev[alt] !== null) return ev[alt];
    }
    if (Array.isArray(ev.fields)) {
      const found = ev.fields.find(f => f.key === key || altKeys.includes(f.key));
      if (found) return found.value;
    }
    return undefined;
  }

  handleLogEvent(ev) {
    if (!ev || !ev.kind) return;

    if (ev.seq !== undefined) {
      if (this.seenSeqs.has(ev.seq)) return;
      this.seenSeqs.add(ev.seq);
    }

    if (ev.kind === 'phase') {
      const at = this.getEventField(ev, 'at') || '';
      if (at === 'render_begin') {
        this.piperStatusBadge.textContent = 'Rendering';
        this.piperOrb.setState('running');
        this.setActivity('Rendering context...');
      } else if (at === 'generate_begin') {
        this.piperStatusBadge.textContent = 'Generating';
        this.piperOrb.setState('running');
        this.setActivity('Generating response...');
      }
    }

    if (ev.kind === 'prompt') {
      const pTok = parseInt(this.getEventField(ev, 'tokens') || 0, 10);
      if (pTok > 0) {
        this.piperStatusBadge.textContent = `Prefill ${pTok.toLocaleString()} tok`;
        this.setActivity(`Loading ${pTok.toLocaleString()} prompt tokens...`);
      }
    }

    if (ev.kind === 'generation') {
      const speed = this.getEventField(ev, 'decode_tok_per_s');
      const ttft = this.getEventField(ev, 'ttft_ms');
      this.piperStatusBadge.textContent = speed ? `${parseFloat(speed).toFixed(1)} tok/s` : 'Working';
      this.setActivity(speed
        ? `Generating at ${parseFloat(speed).toFixed(1)} tok/s`
        : (ttft ? `First token ${(parseFloat(ttft) / 1000).toFixed(1)}s` : 'Generating...'));
    }

    if (ev.kind === 'turn') {
      const turnNum = this.getEventField(ev, 'n', ['turn', 'step']);
      if (turnNum) this.piperTurnBadge.textContent = `turn ${turnNum}`;
      const tool = this.getEventField(ev, 'tool');
      if (tool && tool !== '-') this.lastAction = tool;
      this.setActivity(`Turn ${turnNum || '?'} complete`);
    }

    if (ev.kind === 'tool_call' || ev.kind === 'tool' || ev.kind === 'exec') {
      this.piperOrb.setState('executing');
      const toolName = this.getEventField(ev, 'tool', ['name']) || ev.kind;
      const cmd = this.getEventField(ev, 'command', ['path', 'cmd']) || '';
      this.piperStatusBadge.textContent = `Exec ${toolName}`;
      this.lastAction = toolName;
      this.setActivity(`Executing ${toolName}${cmd ? ': ' + cmd.split('/').pop() : ''}...`);
      const path = this.getEventField(ev, 'path', ['file']) || cmd;
      if (path && !path.startsWith('-')) this.trackFile(path);
    }

    if (ev.kind === 'write') {
      const path = this.getEventField(ev, 'path', ['file', 'normalised']) || '';
      const tool = this.getEventField(ev, 'tool') || 'write';
      this.lastAction = tool;
      if (path) this.trackFile(path);
      this.setActivity(`Writing ${path.split('/').pop() || path}...`);
    }

    this.absorbLog(ev);
    this.renderFeed();

    if (ev.kind === 'verification') {
      const contract = this.getEventField(ev, 'contract') || '';
      const passed = this.getEventField(ev, 'passed') === '1' || this.getEventField(ev, 'passed') === 1;

      const tile = document.createElement('div');
      tile.className = 'tool-tile';
      tile.innerHTML = `
        <div class="tool-tile-head">
          <span class="tool-chip check">Contract Verification</span>
          <span style="font-size: 10px; font-weight: 700; color: ${passed ? 'var(--ok)' : 'var(--warn)'};">
            ${passed ? 'PASS ✓' : 'VERIFYING'}
          </span>
          <span class="tag-time">${this.ts()}</span>
        </div>
        ${contract ? `<div class="tool-cmd">${this.esc(contract)}</div>` : ''}
      `;
      this.piperFeed.appendChild(tile);
      this.scrollBottom(this.piperFeed);
    }

    if (ev.kind === 'run_end') {
      const reason = this.getEventField(ev, 'termination_reason') || 'done';
      this.piperStatusBadge.textContent = reason === 'wall_clock' ? 'Timed Out' : 'Finished';
      this.stopElapsed();
      this.setActivity('');
    }
  }

  handleWake(wake) {
    if (wake.kind === 'ask') {
      this.handleGate(wake);
    } else if (wake.kind === 'done') {
      this.handleResult(wake);
    }
  }

  beginSection(sliceId) {
    this.feed = emptyState();
    this.liveActive = false;
    this.seenLive = new Set();
    this.seenOrch = new Set();
    this.turnCount = 0;
    this.lastAction = '';
    this.piperSection = null;
    this.thoughtEl = null;
    this.answerEl = null;
    this.ensureSection();
    this.orchRunning = document.createElement('div');
    this.orchRunning.className = 'msg-card accent-orch';
    this.orchRunning.textContent = `${sliceId || 'slice'} · waiting`;
    this.orchFeed.appendChild(this.orchRunning);
    this.scrollBottom(this.orchFeed);
    this.renderFeed();
  }

  ensureSection() {
    if (this.piperSection) return;
    this.piperSection = document.createElement('div');
    this.piperSection.className = 'slice-section';
    this.piperFeed.appendChild(this.piperSection);
    this.thoughtEl = null;
    this.answerEl = null;
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
      if (line.tool) this.lastAction = line.tool;
      this.piperTurnBadge.textContent = `turn ${this.turnCount}`;
    }
    if (line.kind === 'write' && line.path) this.trackFile(line.path);
    if (line.kind === 'turn' && line.path) this.trackFile(line.path);
    for (const ev of expandLive(line)) this.feed = apply(this.feed, ev);
    this.renderFeed();
  }

  handleOrch(line) {
    if (!line || typeof line !== 'object') return;
    const key = JSON.stringify(line);
    if (this.seenOrch.has(key)) return;
    this.seenOrch.add(key);
    const text = line.text || line.note || '';
    const event = { kind: 'orch', text };
    if (Object.hasOwn(line, 'tokens') && Number.isFinite(Number(line.tokens))) {
      event.tokens = Number(line.tokens);
    }
    this.feed = apply(this.feed, event);
    const card = document.createElement('div');
    card.className = 'msg-card accent-orch';
    const head = document.createElement('div');
    head.className = 'msg-card-head';
    const tag = document.createElement('span');
    tag.className = 'msg-card-tag tag-orch';
    tag.textContent = 'Orchestrator';
    const time = document.createElement('span');
    time.className = 'tag-time';
    time.textContent = this.ts();
    head.append(tag, time);
    const body = document.createElement('div');
    body.className = 'msg-text';
    body.textContent = text;
    card.append(head, body);
    this.orchFeed.appendChild(card);
    this.scrollBottom(this.orchFeed);
    this.renderFeed();
  }

  absorbLog(ev) {
    const structural = ev.kind === 'turn' || ev.kind === 'tool_call' || ev.kind === 'tool' ||
      ev.kind === 'exec' || ev.kind === 'write' || ev.kind === 'tool_result';
    if (structural && this.liveActive) return;
    const event = (ev.kind === 'tool' || ev.kind === 'exec') ? { ...ev, kind: 'tool_call' } : ev;
    for (const item of expandLog(event)) this.feed = apply(this.feed, item);
  }

  renderFeed() {
    this.ensureSection();
    this.renderChecklist();
    this.renderTimeline();

    if (this.orchRunning) {
      const id = this.currentSliceId || 'slice';
      if (this.turnCount === 0 && !this.lastAction) {
        this.orchRunning.textContent = `${id} · waiting`;
      } else {
        const action = this.lastAction ? ` · ${this.lastAction}` : '';
        this.orchRunning.textContent = `${id} · turn ${this.turnCount}${action}`;
      }
    }
    this.updateTokenMetrics();
    this.scrollBottom(this.piperFeed);
  }

  renderChecklist() {
    const list = this.feed.checklist;
    let panel = this.piperSection.querySelector(':scope > .checklist-panel');
    if (!list || !list.items || !list.items.length) {
      if (panel) panel.remove();
      return;
    }
    if (!panel) {
      panel = document.createElement('div');
      panel.className = 'checklist-panel';
      this.piperSection.prepend(panel);
    }
    const done = list.items.filter((item) => item.done).length;
    const head = document.createElement('div');
    head.className = 'checklist-head';
    head.textContent = `Checklist · ${done}/${list.total} done`;
    const ul = document.createElement('ul');
    for (const item of list.items) {
      const li = document.createElement('li');
      if (item.done) li.className = 'done';
      const box = document.createElement('span');
      box.className = 'box';
      box.textContent = item.done ? '✓' : '';
      const label = document.createElement('span');
      label.textContent = item.text;
      li.append(box, label);
      ul.append(li);
    }
    panel.replaceChildren(head, ul);
  }

  renderTimeline() {
    let timeline = this.piperSection.querySelector(':scope > .feed-timeline');
    if (!timeline) {
      timeline = document.createElement('div');
      timeline.className = 'feed-timeline';
      this.piperSection.appendChild(timeline);
    }
    const thoughts = this.feed.thoughts || [];
    const bySeq = new Map();
    for (const thought of thoughts) {
      if (thought.seq != null) bySeq.set(String(thought.seq), thought);
    }
    const used = new Set();
    const order = [];
    for (const action of this.feed.actions) {
      const thought = bySeq.get(String(action.seq));
      if (thought && !used.has(thought)) {
        order.push({ type: 'thought', thought, key: 't-' + String(thought.seq) });
        used.add(thought);
      }
      order.push({ type: 'action', action, key: 'a-' + String(action.seq) });
    }
    thoughts.forEach((thought, i) => {
      if (!used.has(thought)) order.push({ type: 'thought', thought, key: 't-loose-' + i });
    });
    if (this.feed.thinking) order.push({ type: 'live', key: 'live-think' });
    if (this.feed.answer) order.push({ type: 'answer', key: 'answer' });

    const keep = new Set();
    for (const item of order) {
      keep.add(item.key);
      let node = timeline.querySelector(`[data-key="${CSS.escape(item.key)}"]`);
      if (item.type === 'thought' || item.type === 'live') {
        if (!node) {
          node = document.createElement('details');
          node.className = 'thought-block';
          node.dataset.key = item.key;
          node.open = item.type === 'live' || this.thinkingMode === 'expanded';
          const summary = document.createElement('summary');
          summary.textContent = 'Reasoning';
          const body = document.createElement('div');
          body.className = 'thought-content';
          node.append(summary, body);
        }
        node.querySelector('.thought-content').textContent =
          item.type === 'live' ? this.feed.thinking : item.thought.text;
        if (item.type === 'live' || this.thinkingMode === 'expanded') node.open = true;
      } else if (item.type === 'answer') {
        if (!node) {
          node = document.createElement('div');
          node.className = 'msg-card accent-piper';
          node.dataset.key = item.key;
          const head = document.createElement('div');
          head.className = 'msg-card-head';
          const tag = document.createElement('span');
          tag.className = 'msg-card-tag tag-piper';
          tag.textContent = 'Piper';
          head.appendChild(tag);
          const body = document.createElement('div');
          body.className = 'msg-text msg-stream-body';
          node.append(head, body);
        }
        node.querySelector('.msg-text').textContent = this.feed.answer;
      } else {
        node = this.actionTile(timeline, item.action, item.key);
      }
      timeline.appendChild(node);
    }
    for (const child of [...timeline.children]) {
      if (!keep.has(child.dataset.key)) child.remove();
    }
  }

  actionTile(timeline, action, key) {
    let tile = timeline.querySelector(`[data-key="${CSS.escape(key)}"]`);
    if (!tile) {
      tile = document.createElement('div');
      tile.className = 'tool-tile';
      tile.dataset.key = key;
      tile.dataset.seq = String(action.seq);
      const head = document.createElement('div');
      head.className = 'tool-tile-head';
      const chip = document.createElement('span');
      chip.className = 'tool-chip exec';
      const time = document.createElement('span');
      time.className = 'tag-time';
      const cmd = document.createElement('div');
      cmd.className = 'tool-cmd';
      head.append(chip, time);
      tile.append(head, cmd);
    }
    const chip = tile.querySelector('.tool-chip');
    const write = (action.tool || '').includes('write') || (action.tool || '').includes('replace') || action.edit_bytes > 0;
    chip.className = 'tool-chip ' + (write ? 'write' : 'exec');
    chip.textContent = action.tool || 'action';
    const bits = [];
    if (action.path) bits.push(action.path);
    else if (action.command) bits.push(action.command);
    if (action.read_bytes) bits.push(`${action.read_bytes} read`);
    if (action.edit_bytes) bits.push(`${action.edit_bytes} written`);
    if (action.status) bits.push(action.status);
    if (action.summary) bits.push(action.summary);
    tile.querySelector('.tool-cmd').textContent = bits.join(' · ');
    if (action.path) this.trackFile(action.path);
    return tile;
  }

  formatDiff(stat) {
    if (!stat) return 'none';
    if (typeof stat === 'string') return stat;
    if (typeof stat === 'object') {
      const ins = stat.insertions ?? 0;
      const del = stat.deletions ?? 0;
      const files = stat.files ?? 0;
      return `+${ins} -${del} (${files} files)`;
    }
    return String(stat);
  }

  // ---------------------------------------------------------------------------
  // Token streaming
  // ---------------------------------------------------------------------------

  appendToken(text) {
    // No token counter increment here — turn events are authoritative.

    if (!this.currentAssistantMsg) {
      this.currentAssistantMsg = document.createElement('div');
      this.currentAssistantMsg.className = 'msg-card accent-piper';
      this.currentAssistantMsg.innerHTML = `
        <div class="msg-card-head">
          <span class="msg-card-tag tag-piper">Piper Response</span>
          <span class="tag-time">${this.ts()}</span>
        </div>
        <div class="msg-text msg-stream-body"></div>
      `;
      this.piperFeed.appendChild(this.currentAssistantMsg);
    }

    const streamBody = this.currentAssistantMsg.querySelector('.msg-stream-body');

    if (text.includes('<think>')) {
      this.activeThoughtBlock = document.createElement('details');
      this.activeThoughtBlock.className = 'thought-block';
      // Respect thinking mode for new blocks
      this.activeThoughtBlock.open = this.thinkingMode === 'expanded';
      this.activeThoughtBlock.innerHTML = `
        <summary>Reasoning</summary>
        <div class="thought-content"></div>
      `;
      streamBody.appendChild(this.activeThoughtBlock);
      this.activeThoughtContent = this.activeThoughtBlock.querySelector('.thought-content');
      text = text.replace('<think>', '');
      this.setActivity('Thinking...');
    }

    if (text.includes('</think>')) {
      const parts = text.split('</think>');
      if (this.activeThoughtContent) this.activeThoughtContent.textContent += parts[0];
      this.activeThoughtBlock = null;
      this.activeThoughtContent = null;
      text = parts[1] || '';
      this.setActivity('Responding...');
    }

    if (this.activeThoughtContent) {
      this.activeThoughtContent.textContent += text;
    } else if (text) {
      let textNode = streamBody.lastChild;
      if (!textNode || textNode.nodeType !== Node.TEXT_NODE) {
        textNode = document.createTextNode('');
        streamBody.appendChild(textNode);
      }
      textNode.nodeValue += text;
    }

    this.scrollBottom(this.piperFeed);
  }

  // ---------------------------------------------------------------------------
  // Metrics & UI helpers
  // ---------------------------------------------------------------------------

  updateTokenMetrics() {
    const generated = this.feed ? this.feed.generatedTokens : 0;
    const fmtP = generated.toLocaleString();

    if (this.piperTokensTotal) {
      this.piperTokensTotal.textContent = fmtP;
      this.piperTokensTotal.classList.add('counter-bump');
      setTimeout(() => this.piperTokensTotal.classList.remove('counter-bump'), 300);
    }
    if (this.piperTokensPaneBadge) {
      const ctx = this.feed && this.feed.contextTokens != null
        ? ` · ctx ${this.feed.contextTokens.toLocaleString()}`
        : '';
      this.piperTokensPaneBadge.textContent = `${fmtP} tokens${ctx}`;
    }

    const cloud = this.feed ? this.feed.cloudTokens : null;
    if (cloud == null) {
      if (this.orchTokensTotal) this.orchTokensTotal.textContent = '—';
      if (this.orchTokensPaneBadge) this.orchTokensPaneBadge.textContent = '— tokens';
    } else {
      const fmtO = cloud.toLocaleString();
      if (this.orchTokensTotal) this.orchTokensTotal.textContent = fmtO;
      if (this.orchTokensPaneBadge) this.orchTokensPaneBadge.textContent = `${fmtO} tokens`;
    }

    if (this.offloadPct) {
      const pct = this.feed ? offloadPercent(this.feed) : null;
      this.offloadPct.textContent = pct == null ? '—' : `${pct}%`;
    }
  }

  updateSliceRail() {
    this.sliceRail.innerHTML = '';
    for (const [id, data] of this.slices.entries()) {
      const pill = document.createElement('div');
      pill.className = `slice-pill ${id === this.currentSliceId ? 'active' : ''} ${data.status || ''}`;
      pill.innerHTML = `<span class="slice-dot"></span> <span>${this.esc(id)}</span>`;
      this.sliceRail.appendChild(pill);
    }
  }

  formatTaskPrompt(prompt) {
    return this.esc(prompt)
      .replace(/## (.*?)\n/g, '<h4 style="color:var(--orch-accent); margin: 6px 0 2px; font-size: 12px;">$1</h4>')
      .replace(/- EDIT: (.*?)\n/g,         '<div><span class="file-pill" style="border-left: 2px solid var(--orch-accent);">$1</span></div>')
      .replace(/- CREATE: (.*?)\n/g,       '<div><span class="file-pill" style="border-left: 2px solid var(--ok);">$1</span></div>')
      .replace(/- DO NOT TOUCH: (.*?)\n/g, '<div><span class="file-pill" style="border-left: 2px solid var(--fail); opacity:0.6;">$1</span></div>');
  }

  esc(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  ts() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  scrollBottom(el) {
    el.scrollTop = el.scrollHeight;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.__app = new PiperVisualizerApp();
});
