// Dev-only performance monitor overlay.
// Tracks FPS, frame time, memory, grid size, and tile cache stats.

import { state } from '../state.js';
import { getTileCacheSize } from './cache.js';

let panel, fields;
let frames = 0;
let lastFpsTime = 0;
let lastFrameTime = 0;
let frameTimeSamples = [];
let rafId = 0;

const MAX_SAMPLES = 60;

function fmt(n, decimals = 1) {
  return n.toFixed(decimals);
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return fmt(bytes / 1024, 0) + ' KB';
  return fmt(bytes / 1048576) + ' MB';
}

function buildPanel() {
  panel = document.createElement('div');
  panel.id = 'perf-monitor';
  panel.innerHTML = `
    <div class="perf-title">PERF</div>
    <div class="perf-row"><span class="perf-label">FPS</span><span class="perf-val" data-f="fps">--</span></div>
    <div class="perf-row"><span class="perf-label">Frame</span><span class="perf-val" data-f="frame">--</span></div>
    <div class="perf-row"><span class="perf-label">Grid</span><span class="perf-val" data-f="grid">--</span></div>
    <div class="perf-row"><span class="perf-label">Cells</span><span class="perf-val" data-f="cells">--</span></div>
    <div class="perf-row"><span class="perf-label">Cache</span><span class="perf-val" data-f="cache">--</span></div>
    <div class="perf-row"><span class="perf-label">Gen</span><span class="perf-val" data-f="gen">--</span></div>
    <div class="perf-row"><span class="perf-label">Heap</span><span class="perf-val" data-f="heap">--</span></div>
    <canvas id="perf-fps-graph" width="120" height="32"></canvas>
  `;
  document.getElementById('page-mural').appendChild(panel);

  fields = {};
  panel.querySelectorAll('[data-f]').forEach(el => { fields[el.dataset.f] = el; });
}

// Sparkline for FPS history
const fpsHistory = [];

function drawFpsGraph() {
  const canvas = document.getElementById('perf-fps-graph');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (fpsHistory.length < 2) return;

  // 60fps reference line
  ctx.strokeStyle = 'rgba(57,255,20,0.15)';
  ctx.beginPath();
  ctx.moveTo(0, h * (1 - 60 / 80));
  ctx.lineTo(w, h * (1 - 60 / 80));
  ctx.stroke();

  // FPS line
  ctx.strokeStyle = 'rgba(57,255,20,0.7)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const step = w / (fpsHistory.length - 1);
  for (let i = 0; i < fpsHistory.length; i++) {
    const y = h * (1 - Math.min(fpsHistory[i], 80) / 80);
    if (i === 0) ctx.moveTo(0, y);
    else ctx.lineTo(i * step, y);
  }
  ctx.stroke();
}

export function perfTick() {
  if (!panel) return;
  const now = performance.now();
  frames++;

  // Frame time
  if (lastFrameTime > 0) {
    const dt = now - lastFrameTime;
    frameTimeSamples.push(dt);
    if (frameTimeSamples.length > MAX_SAMPLES) frameTimeSamples.shift();
  }
  lastFrameTime = now;

  // FPS (update every 500ms)
  if (now - lastFpsTime >= 500) {
    const elapsed = now - lastFpsTime;
    const fps = Math.round(frames / (elapsed / 1000));
    fields.fps.textContent = fps;
    fields.fps.className = 'perf-val ' + (fps >= 50 ? 'perf-ok' : fps >= 25 ? 'perf-warn' : 'perf-bad');

    fpsHistory.push(fps);
    if (fpsHistory.length > 120) fpsHistory.shift();
    drawFpsGraph();

    frames = 0;
    lastFpsTime = now;
  }

  // Frame time avg
  if (frameTimeSamples.length > 0) {
    const avg = frameTimeSamples.reduce((a, b) => a + b, 0) / frameTimeSamples.length;
    const max = Math.max(...frameTimeSamples);
    fields.frame.textContent = fmt(avg) + ' / ' + fmt(max) + ' ms';
    fields.frame.className = 'perf-val ' + (avg < 20 ? 'perf-ok' : avg < 40 ? 'perf-warn' : 'perf-bad');
  }

  // Grid info
  const { cols, rows, generation } = state.gol;
  if (cols && rows) {
    fields.grid.textContent = cols + 'x' + rows;
    fields.cells.textContent = (cols * rows).toLocaleString();
  }

  // Generation
  fields.gen.textContent = generation ?? '--';

  // Tile cache
  fields.cache.textContent = getTileCacheSize() + ' tiles';

  // JS Heap (Chrome only)
  if (performance.memory) {
    fields.heap.textContent = formatBytes(performance.memory.usedJSHeapSize)
      + ' / ' + formatBytes(performance.memory.jsHeapSizeLimit);
  } else {
    fields.heap.textContent = 'n/a';
  }
}

export function initPerfMon() {
  if (!import.meta.env.DEV) return;
  buildPanel();
  lastFpsTime = performance.now();
}

export function destroyPerfMon() {
  if (panel) { panel.remove(); panel = null; }
}
