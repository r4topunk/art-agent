import { buildGameOfLife, getPooledPieces } from './gameoflife.js';
import { state } from '../state.js';
import { perfTick } from './perfmon.js';

let muralCanvas, muralCtx;
let muralW = 0, muralH = 0;

// ── Visual crossfade state ──
let crossfade = null; // { old: OffscreenCanvas, start: number, duration: number }
let crossfadeRaf = 0;

function getCanvas() {
  if (!muralCanvas) {
    muralCanvas = document.getElementById('mural-canvas');
    muralCtx = muralCanvas.getContext('2d');
  }
  return { muralCanvas, muralCtx };
}

export function startCrossfade(durationMs = 500) {
  const { muralCanvas: canvas } = getCanvas();
  if (!canvas || canvas.width === 0) return;
  // Snapshot current canvas content
  const snap = new OffscreenCanvas(canvas.width, canvas.height);
  snap.getContext('2d').drawImage(canvas, 0, 0);
  crossfade = { old: snap, start: performance.now(), duration: durationMs };
  // Drive smooth blending via rAF
  if (crossfadeRaf) cancelAnimationFrame(crossfadeRaf);
  function loop() {
    if (!crossfade) { crossfadeRaf = 0; return; }
    renderMural();
    crossfadeRaf = requestAnimationFrame(loop);
  }
  crossfadeRaf = requestAnimationFrame(loop);
}

export function renderMural() {
  const pieces = getPooledPieces();
  const overlay = document.getElementById('mural-overlay');
  if (!pieces.length) { overlay.textContent = 'no pieces yet'; return; }
  overlay.textContent = '';
  const { muralCanvas: canvas, muralCtx: ctx } = getCanvas();
  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W; canvas.height = H;
    muralW = W; muralH = H;
  }
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, W, H);
  const src = state.gol.grid ? buildGameOfLife(state.muralTileSize) : null;
  if (!src) return;
  const ox = Math.round((W - src.width) / 2);
  const oy = Math.round((H - src.height) / 2);
  ctx.drawImage(src, ox, oy);

  // Crossfade: overlay old frame fading out
  if (crossfade) {
    const t = Math.min(1, (performance.now() - crossfade.start) / crossfade.duration);
    ctx.globalAlpha = 1 - t;
    ctx.drawImage(crossfade.old, 0, 0);
    ctx.globalAlpha = 1;
    if (t >= 1) crossfade = null;
  }
  perfTick();
}
