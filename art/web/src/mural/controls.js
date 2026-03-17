import { ZOOM_STEPS } from '../constants.js';
import { state } from '../state.js';
import { renderMural, startCrossfade } from './render.js';
import { clearTileCache } from './cache.js';
import { saveSettings } from '../persist.js';
import {
  startGol, stopGol, restartGolTick,
} from './timers.js';
import { morphoResize, morphoApplyPreset, MORPHO_PRESETS } from './morphogenesis.js';
import { RD_PRESETS } from './reactiondiffusion.js';

// ── Helpers ──

function stopAllTimers() {
  stopGol();
}

function startTimersForMode() {
  startGol();
}

// ── Zoom ──

export function muralZoom(delta) {
  let idx = ZOOM_STEPS.indexOf(state.muralTileSize);
  if (idx === -1) idx = ZOOM_STEPS.indexOf(16);
  idx = Math.max(0, Math.min(ZOOM_STEPS.length - 1, idx + delta));
  state.muralTileSize = ZOOM_STEPS[idx];
  document.getElementById('zoom-val').textContent = state.muralTileSize + 'px';
  // Sync range slider
  const slider = document.getElementById('range-tile');
  if (slider) slider.value = idx;
  saveSettings({ muralTileSize: state.muralTileSize });
  // Morphogenesis: resample simulation to new grid size
  if (state.gol.variant === 'morphogenesis' && state.gol.morpho) {
    morphoResize();
    clearTileCache();
  }
  renderMural();
}

// ── Transition time (layout reshuffle / kaleido regen / GoL round reset) ──

export function muralTransitionTime(delta) {
  state.TRANSITION_MS = Math.max(500, Math.min(15000, state.TRANSITION_MS + delta));
  document.getElementById('transition-val').textContent = (state.TRANSITION_MS / 1000).toFixed(1) + 's';
  // Sync range slider
  const slider = document.getElementById('range-transition');
  if (slider) slider.value = state.TRANSITION_MS;
  saveSettings({ TRANSITION_MS: state.TRANSITION_MS });
  startGol(); // startGol calls stopGol internally
}

// ── Flip time (tile rotation speed / GoL tick speed) ──

export function kaleidoFlipTime(delta) {
  state.KALEIDO_FLIP_MS = Math.max(100, Math.min(10000, state.KALEIDO_FLIP_MS + delta));
  document.getElementById('flip-val').textContent = (state.KALEIDO_FLIP_MS / 1000).toFixed(1) + 's';
  // Sync range slider
  const slider = document.getElementById('range-flip');
  if (slider) slider.value = state.KALEIDO_FLIP_MS;
  saveSettings({ KALEIDO_FLIP_MS: state.KALEIDO_FLIP_MS });
  state.gol.tickMS = state.KALEIDO_FLIP_MS;
  restartGolTick();
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
  btn.textContent = state.muralTileRotation ? 'ON' : 'OFF';
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

// ── Variant switch ──

const VARIANTS = ['conway', 'quadlife', 'reaction-diffusion', 'morphogenesis'];

export function switchVariant(variant) {
  if (!VARIANTS.includes(variant)) return;
  if (state.gol.variant === variant) return;

  state.gol.variant = variant;
  // Reset pinned preset when switching variants
  state.gol.pinnedPreset = -1;
  saveSettings({ golVariant: variant, pinnedPreset: -1 });
  syncModeBar();
  syncPresetBar();

  // Restart GoL with new variant
  startCrossfade(300);
  stopGol();
  startGol();
}

export function syncModeBar() {
  document.querySelectorAll('#mode-bar .mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.variant === state.gol.variant);
  });
}

// ── Preset bar (sub-modes for RD / Morphogenesis) ──

function getPresetsForVariant() {
  const v = state.gol.variant;
  if (v === 'reaction-diffusion') return RD_PRESETS;
  if (v === 'morphogenesis') return MORPHO_PRESETS;
  return null;
}

export function syncPresetBar() {
  const bar = document.getElementById('preset-bar');
  if (!bar) return;

  const presets = getPresetsForVariant();
  if (!presets) {
    bar.classList.add('hidden');
    bar.innerHTML = '';
    return;
  }

  bar.classList.remove('hidden');
  bar.innerHTML = '';

  // "Auto" button
  const autoBtn = document.createElement('button');
  autoBtn.className = 'preset-btn' + (state.gol.pinnedPreset < 0 ? ' active' : '');
  autoBtn.textContent = 'Auto';
  autoBtn.addEventListener('click', () => pinPreset(-1));
  bar.appendChild(autoBtn);

  // Preset buttons
  presets.forEach((p, i) => {
    const btn = document.createElement('button');
    btn.className = 'preset-btn' + (state.gol.pinnedPreset === i ? ' active' : '');
    btn.textContent = p.label;
    btn.addEventListener('click', () => pinPreset(i));
    bar.appendChild(btn);
  });
}

export function pinPreset(idx) {
  state.gol.pinnedPreset = idx;
  saveSettings({ pinnedPreset: idx });
  syncPresetBar();

  if (state.gol.variant === 'reaction-diffusion') {
    // RD: restart to apply the new preset immediately
    startCrossfade(300);
    stopGol();
    startGol();
  } else if (state.gol.variant === 'morphogenesis') {
    // Morphogenesis: snap parameters immediately, inject noise
    morphoApplyPreset(idx);
  }
}

// Sync toolbar DOM to match current state (called on startup)
export function syncToolbarUI() {
  // Tile size
  document.getElementById('zoom-val').textContent = state.muralTileSize + 'px';
  const tileSlider = document.getElementById('range-tile');
  if (tileSlider) {
    const idx = ZOOM_STEPS.indexOf(state.muralTileSize);
    tileSlider.value = idx >= 0 ? idx : 5;
  }

  // Transition
  document.getElementById('transition-val').textContent = (state.TRANSITION_MS / 1000).toFixed(1) + 's';
  const transSlider = document.getElementById('range-transition');
  if (transSlider) transSlider.value = state.TRANSITION_MS;

  // Flip
  document.getElementById('flip-val').textContent = (state.KALEIDO_FLIP_MS / 1000).toFixed(1) + 's';
  const flipSlider = document.getElementById('range-flip');
  if (flipSlider) flipSlider.value = state.KALEIDO_FLIP_MS;

  // Pause
  document.getElementById('mural-pause-btn').textContent = state.muralPaused ? '\u25B6' : '\u23F8';

  // Rotate
  const rotBtn = document.getElementById('mural-rotate-btn');
  rotBtn.textContent = state.muralTileRotation ? 'ON' : 'OFF';
  rotBtn.classList.toggle('active', state.muralTileRotation);

  // Mode bar + preset bar
  syncModeBar();
  syncPresetBar();

  // Past generations
  const pastGensSlider = document.getElementById('range-past-gens');
  if (pastGensSlider) pastGensSlider.value = state.pastGenerations || 0;
  const pastGensVal = document.getElementById('past-gens-val');
  if (pastGensVal) {
    const n = state.pastGenerations || 0;
    const avail = Math.min(n, state.pieceHistory.length);
    pastGensVal.textContent = avail > 0 ? `${avail}/${n}` : '0';
  }

  // CRT
  const crtBtn = document.getElementById('mural-crt-btn');
  if (crtBtn) crtBtn.classList.toggle('active', !document.body.classList.contains('no-crt'));

  // Volume
  const volSlider = document.getElementById('range-volume');
  if (volSlider) volSlider.value = state.volumePercent ?? 60;
  document.getElementById('vol-val').textContent = (state.volumePercent ?? 60) + '%';

  // Sound (settings panel + toolbar icon)
  const soundBtn = document.getElementById('mural-sound-btn');
  if (soundBtn) {
    soundBtn.textContent = state.audio.enabled ? 'ON' : 'OFF';
    soundBtn.classList.toggle('active', state.audio.enabled);
  }
  const soundToolbarBtn = document.getElementById('mural-sound-toolbar-btn');
  if (soundToolbarBtn) soundToolbarBtn.classList.toggle('active', state.audio.enabled);
}

// Called from tabs.js when switching to mural page
export { startTimersForMode };
