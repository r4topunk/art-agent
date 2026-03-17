import { renderMural, startCrossfade } from './render.js';
import { golInit, golStep } from './gameoflife.js';
import { clearTileCache } from './cache.js';
import { state } from '../state.js';
import { updateFromGrid, stopSound, buildWavetables } from './sonify.js';

function isMuralActive() {
  return document.getElementById('page-mural').classList.contains('active');
}

// ── Game of Life ──

function golRound() {
  if (!isMuralActive()) return;
  if (state.muralPaused) {
    // Retry later instead of resetting while paused
    state.gol.resetTimer = setTimeout(golRound, 500);
    return;
  }
  startCrossfade(500);
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
  if (!state.gol.running) return;
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
