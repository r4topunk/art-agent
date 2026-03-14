import { getCachedTile } from './cache.js';
import { buildWallpaper } from './wallpaper.js';
import { state } from '../state.js';

export function buildKaleido(pieces, tileSize) {
  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (!pieces.length) return buildWallpaper(state.allPieces.length ? state.allPieces : pieces, tileSize);
  const k = Math.min(pieces.length, 1 + Math.floor(Math.random() * 5));
  const pool = [...pieces].sort(() => Math.random() - 0.5).slice(0, k);
  const hC = Math.ceil(W / 2 / tileSize) + 1, hR = Math.ceil(H / 2 / tileSize) + 1;
  const qW = hC * tileSize, qH = hR * tileSize;
  const quad = new OffscreenCanvas(qW, qH);
  const qctx = quad.getContext('2d');
  qctx.imageSmoothingEnabled = false;
  for (let gy = 0; gy < hR; gy++)
    for (let gx = 0; gx < hC; gx++) {
      const pi = Math.floor(Math.random() * pool.length);
      const rot = state.muralTileRotation ? Math.floor(Math.random() * 4) : 0;
      const flip = state.muralTileRotation ? Math.random() < 0.5 : false;
      const tile = getCachedTile(pool[pi], 'k' + pi, tileSize, rot, flip);
      qctx.drawImage(tile, gx * tileSize, gy * tileSize);
    }
  const full = new OffscreenCanvas(qW * 2, qH * 2);
  const fctx = full.getContext('2d');
  fctx.imageSmoothingEnabled = false;
  fctx.drawImage(quad, 0, 0);
  fctx.save(); fctx.translate(qW * 2, 0); fctx.scale(-1, 1); fctx.drawImage(quad, 0, 0); fctx.restore();
  fctx.save(); fctx.translate(0, qH * 2); fctx.scale(1, -1); fctx.drawImage(quad, 0, 0); fctx.restore();
  fctx.save(); fctx.translate(qW * 2, qH * 2); fctx.scale(-1, -1); fctx.drawImage(quad, 0, 0); fctx.restore();
  return full;
}
