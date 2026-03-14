import { ZOOM_STEPS } from '../constants.js';
import { state } from '../state.js';
import { renderMural } from './render.js';
import { clearTileCache } from './cache.js';
import {
  startLayoutLoop, stopLayoutLoop,
  startRotationLoop, stopRotationLoop,
  startKaleidoAnim, stopKaleidoAnim,
} from './timers.js';

export function muralZoom(delta) {
  let idx = ZOOM_STEPS.indexOf(state.muralTileSize);
  if (idx === -1) idx = ZOOM_STEPS.indexOf(16);
  idx = Math.max(0, Math.min(ZOOM_STEPS.length - 1, idx + delta));
  state.muralTileSize = ZOOM_STEPS[idx];
  document.getElementById('zoom-val').textContent = state.muralTileSize + 'px';
  renderMural();
}

export function muralTransitionTime(delta) {
  state.TRANSITION_MS = Math.max(500, Math.min(15000, state.TRANSITION_MS + delta));
  document.getElementById('transition-val').textContent = (state.TRANSITION_MS / 1000).toFixed(1) + 's';
  if (state.layoutTimer) { stopLayoutLoop(); startLayoutLoop(); }
  if (state.kaleidoRunning) { stopKaleidoAnim(); startKaleidoAnim(); }
}

export function kaleidoFlipTime(delta) {
  state.KALEIDO_FLIP_MS = Math.max(100, Math.min(10000, state.KALEIDO_FLIP_MS + delta));
  document.getElementById('flip-val').textContent = (state.KALEIDO_FLIP_MS / 1000).toFixed(1) + 's';
  if (state.rotationTimer) { stopRotationLoop(); startRotationLoop(); }
}

export function toggleMuralPause() {
  state.muralPaused = !state.muralPaused;
  document.getElementById('mural-pause-btn').textContent = state.muralPaused ? '\u25B6' : '\u23F8';
}

export function toggleTileRotation() {
  state.muralTileRotation = !state.muralTileRotation;
  const btn = document.getElementById('mural-rotate-btn');
  btn.textContent = state.muralTileRotation ? 'Rotate \u2713' : 'Rotate \u2717';
  btn.classList.toggle('active', state.muralTileRotation);
  clearTileCache();
  renderMural();
}

export function toggleMuralMode() {
  state.muralMode = state.muralMode === 'wallpaper' ? 'kaleidoscope' : 'wallpaper';
  const btn = document.getElementById('mural-mode-btn');
  btn.textContent = state.muralMode === 'kaleidoscope' ? 'Kaleidoscope' : 'Wallpaper';
  btn.classList.toggle('kaleido', state.muralMode === 'kaleidoscope');
  renderMural();
}
