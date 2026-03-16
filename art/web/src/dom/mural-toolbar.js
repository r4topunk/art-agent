import { muralZoom } from '../mural/controls.js';
import { toggleSound, setVolume } from '../mural/sonify.js';
import { state } from '../state.js';
import { saveSettings } from '../persist.js';
import { ZOOM_STEPS } from '../constants.js';
import { renderMural } from '../mural/render.js';

function updateVolDisplay() {
  const pct = state.volumePercent ?? 60;
  document.getElementById('vol-val').textContent = pct + '%';
}

// Keep toolbar sound icon in sync with audio state
export function syncSoundToolbarBtn() {
  const btn = document.getElementById('mural-sound-toolbar-btn');
  if (btn) btn.classList.toggle('active', state.audio.enabled);
}

// Mode-aware labels for timing controls
const TIMING_LABELS = {
  wallpaper:    { transition: 'Cycle',     flip: 'Rotation' },
  kaleidoscope: { transition: 'Regen',     flip: null },       // flip unused
  gameoflife:   { transition: 'Round',     flip: 'Tick speed' },
};

export function updateTimingLabels() {
  const labels = TIMING_LABELS[state.muralMode] || TIMING_LABELS.wallpaper;
  const tLabel = document.getElementById('transition-label');
  const fLabel = document.getElementById('flip-label');
  const fRow   = document.getElementById('flip-row');

  if (tLabel) tLabel.textContent = labels.transition;
  if (fRow) fRow.style.display = labels.flip ? '' : 'none';
  if (fLabel && labels.flip) fLabel.textContent = labels.flip;
}

export function wireMuralToolbar(controls) {
  const $ = (id) => document.getElementById(id);

  // ── Settings panel toggle ──
  $('mural-settings-toggle').addEventListener('click', () => {
    const panel = $('mural-settings');
    const btn = $('mural-settings-toggle');
    panel.classList.toggle('collapsed');
    btn.classList.toggle('open', !panel.classList.contains('collapsed'));
  });

  // ── Mode button ──
  $('mural-mode-btn').addEventListener('click', () => {
    controls.toggleMuralMode();
    updateTimingLabels();
  });

  // ── Pause ──
  $('mural-pause-btn').addEventListener('click', () => controls.toggleMuralPause());

  // ── Fullscreen button ──
  $('mural-fullscreen-btn').addEventListener('click', () => controls.toggleFullscreen());

  // ── CRT button ──
  $('mural-crt-btn').addEventListener('click', () => {
    document.body.classList.toggle('no-crt');
    const isOff = document.body.classList.contains('no-crt');
    $('mural-crt-btn').classList.toggle('active', !isOff);
    saveSettings({ crtDisabled: isOff });
  });

  // ── Tile size slider ──
  $('range-tile').addEventListener('input', (e) => {
    const idx = parseInt(e.target.value);
    state.muralTileSize = ZOOM_STEPS[idx];
    $('zoom-val').textContent = state.muralTileSize + 'px';
    saveSettings({ muralTileSize: state.muralTileSize });
    renderMural();
  });

  // ── Transition time slider ──
  $('range-transition').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    const oldVal = state.TRANSITION_MS;
    state.TRANSITION_MS = val;
    $('transition-val').textContent = (val / 1000).toFixed(1) + 's';
    saveSettings({ TRANSITION_MS: val });
    controls.muralTransitionTime(0); // restart timers without changing value
  });

  // ── Flip time slider ──
  $('range-flip').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    state.KALEIDO_FLIP_MS = val;
    $('flip-val').textContent = (val / 1000).toFixed(1) + 's';
    saveSettings({ KALEIDO_FLIP_MS: val });
    controls.kaleidoFlipTime(0); // restart timers without changing value
  });

  // ── Rotate toggle ──
  $('mural-rotate-btn').addEventListener('click', () => controls.toggleTileRotation());

  // ── Sound toggle (settings panel) ──
  $('mural-sound-btn').addEventListener('click', () => {
    toggleSound();
    syncSoundToolbarBtn();
    saveSettings({ soundEnabled: state.audio.enabled });
  });

  // ── Sound toggle (toolbar icon) ──
  $('mural-sound-toolbar-btn').addEventListener('click', () => {
    toggleSound();
    syncSoundToolbarBtn();
    saveSettings({ soundEnabled: state.audio.enabled });
  });

  // ── Volume slider ──
  $('range-volume').addEventListener('input', (e) => {
    const pct = parseInt(e.target.value);
    state.volumePercent = pct;
    $('vol-val').textContent = pct + '%';
    // Set actual gain
    const a = state.audio;
    if (a.ctx && a.masterGain) {
      a.masterGain.gain.value = pct / 100;
    }
    saveSettings({ volumePercent: pct });
  });

  // ── Mouse wheel zoom on canvas ──
  $('mural-canvas-wrap').addEventListener('wheel', e => {
    e.preventDefault();
    muralZoom(e.deltaY < 0 ? 1 : -1);
    // Sync slider
    const idx = ZOOM_STEPS.indexOf(state.muralTileSize);
    $('range-tile').value = idx;
  }, { passive: false });
}
