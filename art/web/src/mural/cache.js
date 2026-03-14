import { PAL_FLAT } from '../constants.js';

let tileCache = new Map();
let tileCacheSize = -1;
const tileImgData = new ImageData(16, 16);

export function invalidateTileCache() {
  tileCache.clear();
  tileCacheSize = -1;
}

export function clearTileCache() {
  tileCache.clear();
}

export function getCachedTile(grid, key, tileSize, rot, flip) {
  if (tileCacheSize !== tileSize) { tileCache.clear(); tileCacheSize = tileSize; }
  const fullKey = rot || flip ? key + 'r' + rot + (flip ? 'f' : '') : key;
  if (tileCache.has(fullKey)) return tileCache.get(fullKey);
  if (tileCache.size > 800) tileCache.clear();
  // Render 16x16 grid into ImageData with pixel-level rotation/flip
  const d = tileImgData.data;
  for (let r = 0; r < 16; r++)
    for (let c = 0; c < 16; c++) {
      let sr = r, sc = c;
      if (rot === 1)      { sr = c; sc = 15 - r; }
      else if (rot === 2) { sr = 15 - r; sc = 15 - c; }
      else if (rot === 3) { sr = 15 - c; sc = r; }
      if (flip) sc = 15 - sc;
      const i = (r * 16 + c) * 4, pi = ((grid[sr] && grid[sr][sc]) || 0) * 4;
      d[i] = PAL_FLAT[pi]; d[i+1] = PAL_FLAT[pi+1]; d[i+2] = PAL_FLAT[pi+2]; d[i+3] = 255;
    }
  const src = new OffscreenCanvas(16, 16);
  src.getContext('2d').putImageData(tileImgData, 0, 0);
  let tile;
  if (tileSize === 16) {
    tile = src;
  } else {
    tile = new OffscreenCanvas(tileSize, tileSize);
    const tctx = tile.getContext('2d');
    tctx.imageSmoothingEnabled = false;
    tctx.drawImage(src, 0, 0, tileSize, tileSize);
  }
  tileCache.set(fullKey, tile);
  return tile;
}
