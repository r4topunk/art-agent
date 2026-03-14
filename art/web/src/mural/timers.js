import { renderMural } from './render.js';
import { state } from '../state.js';

function isMuralActive() {
  return document.getElementById('page-mural').classList.contains('active');
}

// Layout reshuffle timer
export function startLayoutLoop() {
  if (state.layoutTimer) return;
  function tick() {
    if (!state.muralPaused && isMuralActive() && state.muralMode === 'wallpaper') {
      state.layoutSeed++;
      renderMural();
    }
    state.layoutTimer = setTimeout(tick, state.TRANSITION_MS);
  }
  state.layoutTimer = setTimeout(tick, state.TRANSITION_MS);
}

export function stopLayoutLoop() {
  if (state.layoutTimer) { clearTimeout(state.layoutTimer); state.layoutTimer = null; }
}

// Tile rotation timer
export function startRotationLoop() {
  if (state.rotationTimer) return;
  function tick() {
    if (!state.muralPaused && state.muralTileRotation && isMuralActive() && state.muralMode === 'wallpaper') {
      renderMural();
    }
    state.rotationTimer = setTimeout(tick, state.KALEIDO_FLIP_MS);
  }
  state.rotationTimer = setTimeout(tick, state.KALEIDO_FLIP_MS);
}

export function stopRotationLoop() {
  if (state.rotationTimer) { clearTimeout(state.rotationTimer); state.rotationTimer = null; }
}

// Kaleidoscope animation
export function startKaleidoAnim() {
  if (state.kaleidoRunning) return;
  state.kaleidoRunning = true;
  function tick() {
    if (!state.kaleidoRunning) return;
    if (!state.muralPaused && isMuralActive()) renderMural();
    state.kaleidoTimer = setTimeout(tick, state.TRANSITION_MS);
  }
  state.kaleidoTimer = setTimeout(tick, state.TRANSITION_MS);
}

export function stopKaleidoAnim() {
  state.kaleidoRunning = false;
  if (state.kaleidoTimer) { clearTimeout(state.kaleidoTimer); state.kaleidoTimer = null; }
}
