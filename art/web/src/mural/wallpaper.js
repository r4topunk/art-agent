import { getCachedTile } from './cache.js';
import { state } from '../state.js';

export function buildWallpaper(pieces, tileSize) {
  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const cols = Math.ceil(W / tileSize) + 1, rows = Math.ceil(H / tileSize) + 1;
  const n = pieces.length;
  const now = performance.now();
  const rotPhase = state.muralTileRotation ? Math.floor(now / state.KALEIDO_FLIP_MS) : 0;
  const off = new OffscreenCanvas(cols * tileSize, rows * tileSize);
  const octx = off.getContext('2d');
  octx.imageSmoothingEnabled = false;
  for (let gy = 0; gy < rows; gy++)
    for (let gx = 0; gx < cols; gx++) {
      const si = gy * cols + gx;
      const pi = (si + state.layoutSeed * 7) % n;
      const rot = state.muralTileRotation ? ((si * 7 + pi * 13) + rotPhase) & 3 : 0;
      const flip = state.muralTileRotation ? !!(((si * 11 + pi * 5) + rotPhase) & 1) : false;
      const tile = getCachedTile(pieces[pi], 'w' + pi, tileSize, rot, flip);
      octx.drawImage(tile, gx * tileSize, gy * tileSize);
    }
  return off;
}
