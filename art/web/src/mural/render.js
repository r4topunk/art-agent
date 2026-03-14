import { buildWallpaper } from './wallpaper.js';
import { buildKaleido } from './kaleidoscope.js';
import { state } from '../state.js';

let muralCanvas, muralCtx;
let muralW = 0, muralH = 0;

function getCanvas() {
  if (!muralCanvas) {
    muralCanvas = document.getElementById('mural-canvas');
    muralCtx = muralCanvas.getContext('2d');
  }
  return { muralCanvas, muralCtx };
}

export function renderMural() {
  const pieces = state.muralMode === 'kaleidoscope' && state.selectedPieces.length
    ? state.selectedPieces
    : state.allPieces.length ? state.allPieces : [];
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
  const src = state.muralMode === 'kaleidoscope' && state.selectedPieces.length
    ? buildKaleido(state.selectedPieces, state.muralTileSize)
    : buildWallpaper(pieces, state.muralTileSize);
  const ox = Math.round((W - src.width) / 2);
  const oy = Math.round((H - src.height) / 2);
  ctx.drawImage(src, ox, oy);
}
