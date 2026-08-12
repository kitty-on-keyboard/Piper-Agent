// The Dual-Layer Fiber Soundwave Logo Engine for Piper Agent
//
// Dual-Layer Architecture:
// 1. Layer 1 (svg-layer-ribbon): Pristine continuous fluid ribbon 'P' for Idle Standby ONLY.
// 2. Layer 2 (svg-layer-bars): 4 dynamic fiber soundwave equalizer bars for Thinking & Executing states ONLY.

const ORB_SIZE = 34;

export function orbStyles(): string {
  return `
:root { --orb-size: ${ORB_SIZE}px; }

#headRow { display: flex; align-items: center; gap: 11px; padding-right: 56px; }
#headText { flex: 1; min-width: 0; }

.sw-container {
  width: var(--orb-size);
  height: var(--orb-size);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  flex: none;
}

.sw-svg {
  width: var(--orb-size);
  height: var(--orb-size);
  overflow: visible;
  position: absolute;
  top: 0;
  left: 0;
  transition: opacity 0.3s ease;
}

/* Standby Only: Solid Ribbon 'P' */
body:not(.busy) .svg-layer-ribbon {
  opacity: 1 !important;
  display: block !important;
  animation: quietRibbonBreath 2.8s ease-in-out infinite alternate;
}

body:not(.busy) .svg-layer-bars {
  opacity: 0 !important;
  display: none !important;
  animation: none !important;
}

/* Active Only: Dynamic Soundwave Equalizer Bars */
body.busy .svg-layer-ribbon {
  opacity: 0 !important;
  display: none !important;
  animation: none !important;
}

body.busy .svg-layer-bars {
  opacity: 1 !important;
  display: block !important;
}

@keyframes quietRibbonBreath {
  0% {
    filter: drop-shadow(0 0 3px rgba(20, 184, 166, 0.5));
    transform: scale(0.97);
  }
  50% {
    filter: drop-shadow(0 0 10px rgba(6, 182, 212, 0.85));
    transform: scale(1.03);
  }
  100% {
    filter: drop-shadow(0 0 14px rgba(139, 92, 246, 0.95));
    transform: scale(1.0);
  }
}

.sw-bar {
  transform-box: fill-box;
  transform-origin: center bottom;
  transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
}

body.busy .sw-1 { animation: soundwaveFlow 0.75s ease-in-out infinite alternate 0.0s; }
body.busy .sw-2 { animation: soundwaveFlow 0.75s ease-in-out infinite alternate 0.2s; }
body.busy .sw-3 { animation: soundwaveFlow 0.75s ease-in-out infinite alternate 0.4s; }
body.busy .sw-4 { animation: soundwaveFlow 0.75s ease-in-out infinite alternate 0.1s; }

body.executing .sw-1 { animation: rapidPulse 0.3s ease-in-out infinite alternate 0.0s; }
body.executing .sw-2 { animation: rapidPulse 0.3s ease-in-out infinite alternate 0.1s; }
body.executing .sw-3 { animation: rapidPulse 0.3s ease-in-out infinite alternate 0.2s; }
body.executing .sw-4 { animation: rapidPulse 0.3s ease-in-out infinite alternate 0.05s; }

@keyframes soundwaveFlow {
  0% { transform: scaleY(0.3); }
  100% { transform: scaleY(1.45); }
}

@keyframes rapidPulse {
  0% { transform: scaleY(0.2); }
  100% { transform: scaleY(1.6); }
}

@media (prefers-reduced-motion: reduce) {
  .sw-svg, .sw-bar { animation: none !important; }
}
`;
}

export function orbMarkup(): string {
  return `<div id="orb" class="sw-container" aria-hidden="true">
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
      <path d="M 32 82 V 26 C 32 26, 32 18, 48 18 C 68 18, 76 30, 68 46 C 60 60, 32 54, 32 54 L 62 54" stroke="url(#pGrad)" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" filter="url(#pGlow)"/>
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
  </div>`;
}

export function orbScript(): string {
  return `
(function () {
  window.__orb = {
    state: function (s) {
      var st = (s || '').toString().toLowerCase();
      if (st === 'busy' || st === 'thinking') {
        document.body.classList.add('busy');
        document.body.classList.remove('executing');
      } else if (st === 'executing' || st === 'tool' || st === 'running') {
        document.body.classList.add('busy', 'executing');
      } else {
        document.body.classList.remove('busy', 'executing');
      }
    },
    impulse: function () {},
    read: function () {
      return { state: 'ready', settled: true };
    }
  };
})();
`;
}
