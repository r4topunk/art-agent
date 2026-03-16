import { muralZoom } from '../mural/controls.js';
import { toggleSound, setVolume } from '../mural/sonify.js';
import { state } from '../state.js';

function updateVolDisplay() {
  const a = state.audio;
  const pct = a.masterGain ? Math.round(a.masterGain.gain.value * 100) : 60;
  document.getElementById('vol-val').textContent = pct + '%';
}

export function wireMuralToolbar(controls) {
  const $ = (id) => document.getElementById(id);

  $('btn-zoom-out').addEventListener('click', () => muralZoom(-1));
  $('btn-zoom-in').addEventListener('click', () => muralZoom(+1));
  $('btn-transition-down').addEventListener('click', () => controls.muralTransitionTime(-500));
  $('btn-transition-up').addEventListener('click', () => controls.muralTransitionTime(+500));
  $('btn-flip-down').addEventListener('click', () => controls.kaleidoFlipTime(-500));
  $('btn-flip-up').addEventListener('click', () => controls.kaleidoFlipTime(+500));
  $('mural-pause-btn').addEventListener('click', () => controls.toggleMuralPause());
  $('mural-rotate-btn').addEventListener('click', () => controls.toggleTileRotation());
  $('mural-mode-btn').addEventListener('click', () => controls.toggleMuralMode());
  $('mural-sound-btn').addEventListener('click', () => toggleSound());
  $('btn-vol-down').addEventListener('click', () => { setVolume(-0.1); updateVolDisplay(); });
  $('btn-vol-up').addEventListener('click', () => { setVolume(+0.1); updateVolDisplay(); });

  $('mural-canvas-wrap').addEventListener('wheel', e => {
    e.preventDefault(); muralZoom(e.deltaY < 0 ? 1 : -1);
  }, { passive: false });
}
