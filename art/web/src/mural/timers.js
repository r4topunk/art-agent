import { renderMural, startCrossfade } from './render.js';
import { golInit, golStep } from './gameoflife.js';
import { morphoMutateTiles } from './morphogenesis.js';
import { clearTileCache } from './cache.js';
import { state } from '../state.js';
import { updateFromGrid, stopSound, buildWavetables } from './sonify.js';

function isMuralActive() {
  return document.getElementById('page-mural').classList.contains('active');
}

function isMorphogenesis() {
  return state.gol.variant === 'morphogenesis';
}

// ── Game of Life ──

function golRound() {
  if (!isMuralActive()) return;
  if (state.muralPaused) {
    state.gol.resetTimer = setTimeout(golRound, 500);
    return;
  }

  // Morphogenesis: never reset the simulation — just mutate tiles
  if (isMorphogenesis() && state.gol.morpho) {
    startCrossfade(800);
    morphoMutateTiles();
    buildWavetables(state.gol.tileA, state.gol.tileB, state.gol.tileIdxA, state.gol.tileIdxB);
    renderMural();
    // Schedule next tile mutation
    state.gol.resetTimer = setTimeout(golRound, state.TRANSITION_MS);
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

  // Morphogenesis: init once, tick forever, mutate tiles on round timer
  if (isMorphogenesis()) {
    golInit();  // calls morphoInit()
    state.gol.tickMS = state.KALEIDO_FLIP_MS;
    buildWavetables(state.gol.tileA, state.gol.tileB, state.gol.tileIdxA, state.gol.tileIdxB);
    renderMural();
    state.gol.running = true;
    golTick();
    // Schedule tile mutations (no simulation reset)
    state.gol.resetTimer = setTimeout(golRound, state.TRANSITION_MS);
    return;
  }

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
