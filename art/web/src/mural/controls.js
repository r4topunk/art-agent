import { ZOOM_STEPS } from '../constants.js';
import { state } from '../state.js';
import { renderMural } from './render.js';
import { clearTileCache } from './cache.js';
import { saveSettings } from '../persist.js';
import {
  startLayoutLoop, stopLayoutLoop,
  startRotationLoop, stopRotationLoop,
  startKaleidoAnim, stopKaleidoAnim,
  startGol, stopGol, restartGolTick,
} from './timers.js';

// ── Helpers ──

function stopAllTimers() {
  stopLayoutLoop();
  stopRotationLoop();
  stopKaleidoAnim();
  stopGol();
}

function startTimersForMode() {
  switch (state.muralMode) {
    case 'wallpaper':
      startLayoutLoop();
      startRotationLoop();
      break;
    case 'kaleidoscope':
      startKaleidoAnim();
      break;
    case 'gameoflife':
      startGol();
      break;
  }
}

// ── Zoom ──

export function muralZoom(delta) {
  let idx = ZOOM_STEPS.indexOf(state.muralTileSize);
  if (idx === -1) idx = ZOOM_STEPS.indexOf(16);
  idx = Math.max(0, Math.min(ZOOM_STEPS.length - 1, idx + delta));
  state.muralTileSize = ZOOM_STEPS[idx];
  document.getElementById('zoom-val').textContent = state.muralTileSize + 'px';
  saveSettings({ muralTileSize: state.muralTileSize });
  renderMural();
}

// ── Transition time (layout reshuffle / kaleido regen / GoL round reset) ──

export function muralTransitionTime(delta) {
  state.TRANSITION_MS = Math.max(500, Math.min(15000, state.TRANSITION_MS + delta));
  document.getElementById('transition-val').textContent = (state.TRANSITION_MS / 1000).toFixed(1) + 's';
  saveSettings({ TRANSITION_MS: state.TRANSITION_MS });
  // Restart whichever timer uses TRANSITION_MS in the current mode
  switch (state.muralMode) {
    case 'wallpaper':
      if (state.layoutTimer) { stopLayoutLoop(); startLayoutLoop(); }
      break;
    case 'kaleidoscope':
      if (state.kaleidoRunning) { stopKaleidoAnim(); startKaleidoAnim(); }
      break;
    case 'gameoflife':
      startGol(); // startGol calls stopGol internally
      break;
  }
}

// ── Flip time (tile rotation speed / GoL tick speed) ──

export function kaleidoFlipTime(delta) {
  state.KALEIDO_FLIP_MS = Math.max(100, Math.min(10000, state.KALEIDO_FLIP_MS + delta));
  document.getElementById('flip-val').textContent = (state.KALEIDO_FLIP_MS / 1000).toFixed(1) + 's';
  saveSettings({ KALEIDO_FLIP_MS: state.KALEIDO_FLIP_MS });
  switch (state.muralMode) {
    case 'wallpaper':
      if (state.rotationTimer) { stopRotationLoop(); startRotationLoop(); }
      break;
    case 'gameoflife':
      state.gol.tickMS = state.KALEIDO_FLIP_MS;
      restartGolTick();
      break;
    // kaleidoscope: KALEIDO_FLIP_MS not used, no-op
  }
}

// ── Pause ──

export function toggleMuralPause() {
  state.muralPaused = !state.muralPaused;
  document.getElementById('mural-pause-btn').textContent = state.muralPaused ? '\u25B6' : '\u23F8';
  saveSettings({ muralPaused: state.muralPaused });
}

// ── Tile rotation ──

export function toggleTileRotation() {
  state.muralTileRotation = !state.muralTileRotation;
  const btn = document.getElementById('mural-rotate-btn');
  btn.textContent = state.muralTileRotation ? 'Rotate \u2713' : 'Rotate \u2717';
  btn.classList.toggle('active', state.muralTileRotation);
  saveSettings({ muralTileRotation: state.muralTileRotation });
  clearTileCache();
  renderMural();
}

// ── Fullscreen ──

export function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen();
  }
}

// ── Mode toggle ──

const MODES = ['wallpaper', 'kaleidoscope', 'gameoflife'];
const MODE_LABELS = { wallpaper: 'Wallpaper', kaleidoscope: 'Kaleidoscope', gameoflife: 'Game of Life' };

export function toggleMuralMode() {
  stopAllTimers();

  const idx = MODES.indexOf(state.muralMode);
  state.muralMode = MODES[(idx + 1) % MODES.length];
  const btn = document.getElementById('mural-mode-btn');
  btn.textContent = MODE_LABELS[state.muralMode];
  btn.classList.toggle('kaleido', state.muralMode !== 'wallpaper');
  saveSettings({ muralMode: state.muralMode });

  startTimersForMode();
  if (state.muralMode !== 'gameoflife') renderMural(); // GoL renders via its own init
}

// Sync toolbar DOM to match current state (called on startup)
export function syncToolbarUI() {
  document.getElementById('zoom-val').textContent = state.muralTileSize + 'px';
  document.getElementById('transition-val').textContent = (state.TRANSITION_MS / 1000).toFixed(1) + 's';
  document.getElementById('flip-val').textContent = (state.KALEIDO_FLIP_MS / 1000).toFixed(1) + 's';
  document.getElementById('mural-pause-btn').textContent = state.muralPaused ? '\u25B6' : '\u23F8';
  const rotBtn = document.getElementById('mural-rotate-btn');
  rotBtn.textContent = state.muralTileRotation ? 'Rotate \u2713' : 'Rotate \u2717';
  rotBtn.classList.toggle('active', state.muralTileRotation);
  const modeBtn = document.getElementById('mural-mode-btn');
  modeBtn.textContent = MODE_LABELS[state.muralMode];
  modeBtn.classList.toggle('kaleido', state.muralMode !== 'wallpaper');
}

// ── Grid interaction (Game of Life click/drag cell toggle) ──

export function initGridInteraction() {
  const canvas = document.getElementById('mural-canvas');
  let painting = false;
  let paintValue = 0;

  function getCellFromEvent(e) {
    const { gol } = state;
    if (!gol.grid) return null;
    const rect = canvas.getBoundingClientRect();
    const ts = state.muralTileSize;
    const wrap = document.getElementById('mural-canvas-wrap');
    const ox = Math.round((wrap.clientWidth - gol.cols * ts) / 2);
    const oy = Math.round((wrap.clientHeight - gol.rows * ts) / 2);
    const col = Math.floor((e.clientX - rect.left - ox) / ts);
    const row = Math.floor((e.clientY - rect.top - oy) / ts);
    if (col < 0 || col >= gol.cols || row < 0 || row >= gol.rows) return null;
    return { col, row };
  }

  canvas.addEventListener('mousedown', (e) => {
    if (state.muralMode !== 'gameoflife') return;
    const cell = getCellFromEvent(e);
    if (!cell) return;
    const idx = cell.row * state.gol.cols + cell.col;
    paintValue = state.gol.grid[idx] ? 0 : 1;
    state.gol.grid[idx] = paintValue;
    painting = true;
    renderMural();
    e.preventDefault();
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!painting || state.muralMode !== 'gameoflife') return;
    const cell = getCellFromEvent(e);
    if (!cell) return;
    state.gol.grid[cell.row * state.gol.cols + cell.col] = paintValue;
    renderMural();
  });

  window.addEventListener('mouseup', () => { painting = false; });
}

// Called from tabs.js when switching to mural page
export { startTimersForMode };
