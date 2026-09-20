// Piper Soundwave & Ribbon Logo Engine
// Ported from extension/src/orb.ts for standalone browser execution.

export function initOrb(containerEl) {
  if (!containerEl) return { setState: () => {} };

  containerEl.innerHTML = `
    <div id="orb" class="sw-container" aria-hidden="true">
      <svg class="sw-svg svg-layer-ribbon" viewBox="0 0 100 100" fill="none">
        <defs>
          <linearGradient id="pGrad" x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#14B8A6"/>
            <stop offset="40%" stop-color="#06B6D4"/>
            <stop offset="80%" stop-color="#3B82F6"/>
            <stop offset="100%" stop-color="#8B5CF6"/>
          </linearGradient>
          <filter id="pGlow">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
          </filter>
        </defs>
        <path d="M 32 82 V 26 C 32 26, 32 18, 48 18 C 68 18, 76 30, 68 46 C 60 60, 32 54, 32 54 L 62 54"
              stroke="url(#pGrad)" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" filter="url(#pGlow)"/>
        <circle cx="62" cy="54" r="4" fill="#06B6D4"/>
      </svg>
      <svg class="sw-svg svg-layer-bars" viewBox="0 0 120 120" fill="none">
        <defs>
          <linearGradient id="tbGrad" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stop-color="#14B8A6"/>
            <stop offset="40%" stop-color="#06B6D4"/>
            <stop offset="75%" stop-color="#3B82F6"/>
            <stop offset="100%" stop-color="#8B5CF6"/>
          </linearGradient>
          <filter id="swGlow">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
          </filter>
        </defs>
        <rect class="sw-bar sw-1" x="32" y="22" width="10" height="76" rx="5" fill="url(#tbGrad)" filter="url(#swGlow)"/>
        <rect class="sw-bar sw-2" x="48" y="22" width="10" height="40" rx="5" fill="url(#tbGrad)" filter="url(#swGlow)"/>
        <rect class="sw-bar sw-3" x="64" y="22" width="10" height="40" rx="5" fill="url(#tbGrad)" filter="url(#swGlow)"/>
        <rect class="sw-bar sw-4" x="80" y="30" width="10" height="24" rx="5" fill="url(#tbGrad)" filter="url(#swGlow)"/>
        <path d="M53 62 C80 62, 85 52, 85 38" stroke="url(#tbGrad)" stroke-width="7" stroke-linecap="round" fill="none" filter="url(#swGlow)"/>
      </svg>
    </div>
  `;

  return {
    setState: (state) => {
      const s = (state || '').toString().toLowerCase();
      const orbEl = containerEl.querySelector('#orb');
      if (!orbEl) return;

      orbEl.classList.remove('busy', 'executing', 'error', 'success');
      if (s === 'idle' || s === 'done' || s === 'ok') {
        if (s === 'done' || s === 'ok') orbEl.classList.add('success');
      } else if (s === 'executing' || s === 'tool' || s === 'running') {
        orbEl.classList.add('busy', 'executing');
      } else if (s === 'error' || s === 'died' || s === 'stalled') {
        orbEl.classList.add('error');
      } else {
        orbEl.classList.add('busy');
      }
    }
  };
}
