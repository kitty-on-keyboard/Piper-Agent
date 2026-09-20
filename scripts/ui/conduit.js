// Conduit Animation Bus
// Visualizes packet transfer and pulse interactions between Piper and Orchestrator.

export class ConduitBus {
  constructor(canvasEl, overlayEl) {
    this.canvas = canvasEl;
    this.ctx = canvasEl ? canvasEl.getContext('2d') : null;
    this.overlay = overlayEl;
    this.particles = [];
    this.active = true;
    this.width = 0;
    this.height = 0;

    if (this.canvas) {
      this.resize();
      window.addEventListener('resize', () => this.resize());
      this.loop = this.loop.bind(this);
      requestAnimationFrame(this.loop);
    }
  }

  resize() {
    if (!this.canvas) return;
    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
  }

  spawnAmbient() {
    if (this.particles.length > 30) return;
    const fromLeft = Math.random() > 0.5;
    this.particles.push({
      x: fromLeft ? 0 : this.width,
      y: 80 + Math.random() * (this.height - 160),
      targetX: fromLeft ? this.width : 0,
      targetY: 80 + Math.random() * (this.height - 160),
      progress: 0,
      speed: 0.003 + Math.random() * 0.005,
      size: 1.5 + Math.random() * 2,
      color: fromLeft ? '#14B8A6' : '#3B82F6',
      alpha: 0.2 + Math.random() * 0.4
    });
  }

  loop() {
    if (!this.active || !this.ctx) return;
    this.ctx.clearRect(0, 0, this.width, this.height);

    if (Math.random() < 0.2) this.spawnAmbient();

    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.progress += p.speed;
      if (p.progress >= 1) {
        this.particles.splice(i, 1);
        continue;
      }

      // Bezier curve across the bridge
      const cx = this.width / 2;
      const cy = (p.y + p.targetY) / 2 + Math.sin(p.progress * Math.PI) * 40;
      const t = p.progress;
      const curX = (1 - t) * (1 - t) * p.x + 2 * (1 - t) * t * cx + t * t * p.targetX;
      const curY = (1 - t) * (1 - t) * p.y + 2 * (1 - t) * t * cy + t * t * p.targetY;

      this.ctx.beginPath();
      this.ctx.arc(curX, curY, p.size, 0, Math.PI * 2);
      this.ctx.fillStyle = p.color;
      this.ctx.globalAlpha = p.alpha * Math.sin(t * Math.PI);
      this.ctx.shadowBlur = 8;
      this.ctx.shadowColor = p.color;
      this.ctx.fill();
      this.ctx.globalAlpha = 1.0;
      this.ctx.shadowBlur = 0;
    }

    requestAnimationFrame(this.loop);
  }

  flyPacket(direction, type, title, subtitle = '') {
    if (!this.overlay) return;

    const el = document.createElement('div');
    el.className = `packet-flyer ${direction} ${type}`;
    el.innerHTML = `
      <div class="packet-icon"></div>
      <div class="packet-info">
        <div class="packet-title">${title}</div>
        ${subtitle ? `<div class="packet-sub">${subtitle}</div>` : ''}
      </div>
    `;

    this.overlay.appendChild(el);

    // Trigger shockwave / burst on completion
    setTimeout(() => {
      this.triggerBurst(direction === 'right-to-left' ? 60 : this.width - 60, window.innerHeight * 0.4, type);
      setTimeout(() => el.remove(), 400);
    }, 950);
  }

  triggerBurst(x, y, type) {
    if (!this.ctx) return;
    const color = type === 'task' ? '#3B82F6' : type === 'result' ? '#14B8A6' : '#F59E0B';
    for (let i = 0; i < 16; i++) {
      const angle = (i / 16) * Math.PI * 2;
      const dist = 30 + Math.random() * 40;
      this.particles.push({
        x: x,
        y: y,
        targetX: x + Math.cos(angle) * dist,
        targetY: y + Math.sin(angle) * dist,
        progress: 0,
        speed: 0.02 + Math.random() * 0.03,
        size: 2 + Math.random() * 2.5,
        color: color,
        alpha: 0.9
      });
    }
  }
}
