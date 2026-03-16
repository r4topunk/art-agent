import { renderMural } from './render.js';
import { golInit, golStep } from './gameoflife.js';
import { clearTileCache } from './cache.js';
import { state } from '../state.js';
import { updateFromGrid, stopSound, buildWavetables } from './sonify.js';

function isMuralActive() {
  return document.getElementById('page-mural').classList.contains('active');
}

// ── Layout reshuffle (wallpaper only) ──

export function startLayoutLoop() {
  if (state.layoutTimer) return;
  function tick() {
    if (!state.muralPaused && isMuralActive()) {
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

// ── Tile rotation re-render (wallpaper only) ──

export function startRotationLoop() {
  if (state.rotationTimer) return;
  function tick() {
    if (!state.muralPaused && state.muralTileRotation && isMuralActive()) {
      renderMural();
    }
    state.rotationTimer = setTimeout(tick, state.KALEIDO_FLIP_MS);
  }
  state.rotationTimer = setTimeout(tick, state.KALEIDO_FLIP_MS);
}

export function stopRotationLoop() {
  if (state.rotationTimer) { clearTimeout(state.rotationTimer); state.rotationTimer = null; }
}

// ── Kaleidoscope animation ──

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

// ── Game of Life ──

function golRound() {
  if (state.muralMode !== 'gameoflife' || !isMuralActive()) return;
  if (state.muralPaused) {
    // Retry later instead of resetting while paused
    state.gol.resetTimer = setTimeout(golRound, 500);
    return;
  }
  clearTileCache();
  golInit();
  state.gol.tickMS = state.KALEIDO_FLIP_MS;
  buildWavetables(state.gol.tileA, state.gol.tileB, state.gol.tileIdxA, state.gol.tileIdxB);
  renderMural();
  state.gol.running = true;
  golTick();
  // Schedule next round (new tile combo)
  state.gol.resetTimer = setTimeout(() => {
    stopGolTick();
    golRound();
  }, state.TRANSITION_MS);
}

function golTick() {
  if (!state.gol.running || state.muralMode !== 'gameoflife') return;
  if (!state.muralPaused) {
    golStep();
    renderMural();
    updateFromGrid(state.gol.grid, state.gol.cols, state.gol.rows);
  }
  state.gol.tickTimer = setTimeout(golTick, state.gol.tickMS);
}

function stopGolTick() {
  state.gol.running = false;
  if (state.gol.tickTimer) { clearTimeout(state.gol.tickTimer); state.gol.tickTimer = null; }
}

export function startGol() {
  stopGol();
  golRound();
}

export function stopGol() {
  stopGolTick();
  if (state.gol.resetTimer) { clearTimeout(state.gol.resetTimer); state.gol.resetTimer = null; }
  stopSound();
}

// Restart just the tick timer with new KALEIDO_FLIP_MS (called when Flip changes)
export function restartGolTick() {
  if (!state.gol.running) return;
  stopGolTick();
  state.gol.running = true;
  golTick();
}
