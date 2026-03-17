import { getCachedTile } from './cache.js';
import { state } from '../state.js';
import { rebuildPitchTable } from './sonify.js';

// Cellular automata at tile resolution on a toroidal grid.
// Supports multiple rule variants via state.gol.variant.

export const GOL_VARIANTS = ['conway', 'immigration', 'quadlife'];
export const GOL_VARIANT_LABELS = {
  conway: "Conway's GoL",
  immigration: 'Immigration',
  quadlife: 'QuadLife',
};

// Number of tile slots each variant needs (including dead tile)
const VARIANT_TILE_COUNT = {
  conway: 2,        // alive + dead
  immigration: 3,    // 2 species + dead
  quadlife: 5,       // 4 species + dead
};

function pickDistinctTiles(pieces, count) {
  if (pieces.length < count) count = pieces.length;
  function fp(piece) {
    let s = 0;
    for (let r = 0; r < piece.length; r++)
      for (let c = 0; c < piece[r].length; c++) s += piece[r][c];
    return s;
  }
  const picked = [];
  const pickedIdx = [];
  const usedFps = new Set();
  // Shuffle indices
  const indices = Array.from({ length: pieces.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  // First pass: pick tiles with distinct fingerprints
  for (const i of indices) {
    if (picked.length >= count) break;
    const f = fp(pieces[i]);
    if (!usedFps.has(f)) {
      usedFps.add(f);
      picked.push(pieces[i]);
      pickedIdx.push(i);
    }
  }
  // Fill remaining if not enough distinct ones
  for (const i of indices) {
    if (picked.length >= count) break;
    if (!pickedIdx.includes(i)) {
      picked.push(pieces[i]);
      pickedIdx.push(i);
    }
  }
  return { tiles: picked, indices: pickedIdx };
}

export function golInit() {
  const pieces = state.allPieces.length ? state.allPieces : [];
  if (pieces.length < 2) return;
  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const ts = state.muralTileSize;
  const cols = Math.ceil(W / ts) + 1;
  const rows = Math.ceil(H / ts) + 1;

  const variant = state.gol.variant || 'conway';
  const needed = VARIANT_TILE_COUNT[variant] || 2;
  const { tiles, indices } = pickDistinctTiles(pieces, needed);

  // Store tiles array: index 0 = dead, 1+ = alive states
  state.gol.tiles = tiles;
  state.gol.tileIndices = indices;
  // Keep tileA/tileB for backward compat with sonify
  state.gol.tileA = tiles[1] || tiles[0];
  state.gol.tileB = tiles[0];
  state.gol.tileIdxA = indices[1] || indices[0];
  state.gol.tileIdxB = indices[0];
  state.gol.cols = cols;
  state.gol.rows = rows;
  state.gol.maxState = needed - 1; // max cell value (0 = dead)

  // Build kaleidoscope-symmetric initial state
  const hC = Math.ceil(cols / 2);
  const hR = Math.ceil(rows / 2);
  const grid = new Uint8Array(rows * cols);
  const aliveStates = needed - 1; // number of alive states

  for (let r = 0; r < hR; r++)
    for (let c = 0; c < hC; c++) {
      let val = 0;
      if (variant === 'quadlife') {
        // 4 species, ~40% density
        val = Math.random() < 0.4 ? (1 + Math.floor(Math.random() * 4)) : 0;
      } else if (variant === 'immigration') {
        val = Math.random() < 0.4 ? (1 + Math.floor(Math.random() * 2)) : 0;
      } else {
        // For conway: alive = maxState (highest value)
        val = Math.random() < 0.4 ? aliveStates : 0;
      }
      const mc = cols - 1 - c;
      const mr = rows - 1 - r;
      grid[r * cols + c] = val;
      grid[r * cols + mc] = val;
      grid[mr * cols + c] = val;
      grid[mr * cols + mc] = val;
    }

  state.gol.grid = grid;
  state.gol.generation = 0;
  rebuildPitchTable(rows);
}

// Count alive neighbors (any non-zero state counts as alive)
function countNeighbors(grid, cols, rows, r, c) {
  let n = 0;
  for (let dr = -1; dr <= 1; dr++)
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = (r + dr + rows) % rows;
      const nc = (c + dc + cols) % cols;
      if (grid[nr * cols + nc] > 0) n++;
    }
  return n;
}

// Get array of neighbor values (non-zero only)
function neighborValues(grid, cols, rows, r, c) {
  const vals = [];
  for (let dr = -1; dr <= 1; dr++)
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = (r + dr + rows) % rows;
      const nc = (c + dc + cols) % cols;
      const v = grid[nr * cols + nc];
      if (v > 0) vals.push(v);
    }
  return vals;
}

// Majority color from neighbor values
function majorityColor(vals) {
  if (!vals.length) return 1;
  const counts = {};
  for (const v of vals) counts[v] = (counts[v] || 0) + 1;
  let best = vals[0], bestN = 0;
  for (const k in counts) {
    if (counts[k] > bestN) { bestN = counts[k]; best = +k; }
  }
  return best;
}

export function golStep() {
  const { grid, cols, rows, maxState } = state.gol;
  if (!grid) return;
  const variant = state.gol.variant || 'conway';
  const next = new Uint8Array(rows * cols);

  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const cell = grid[idx];

      switch (variant) {
        case 'conway': {
          const n = countNeighbors(grid, cols, rows, r, c);
          if (cell) next[idx] = (n === 2 || n === 3) ? 1 : 0;
          else next[idx] = (n === 3) ? 1 : 0;
          break;
        }

        case 'immigration': {
          // 2 species (1, 2) + dead (0). Standard GoL rules.
          // Newborns take majority color of alive neighbors.
          const n = countNeighbors(grid, cols, rows, r, c);
          if (cell > 0) {
            next[idx] = (n === 2 || n === 3) ? cell : 0;
          } else {
            if (n === 3) {
              next[idx] = majorityColor(neighborValues(grid, cols, rows, r, c));
            } else {
              next[idx] = 0;
            }
          }
          break;
        }

        case 'quadlife': {
          // 4 species (1-4) + dead (0). Standard GoL rules.
          // Newborns take majority color of alive neighbors.
          const n = countNeighbors(grid, cols, rows, r, c);
          if (cell > 0) {
            next[idx] = (n === 2 || n === 3) ? cell : 0;
          } else {
            if (n === 3) {
              next[idx] = majorityColor(neighborValues(grid, cols, rows, r, c));
            } else {
              next[idx] = 0;
            }
          }
          break;
        }
      }
    }

  state.gol.grid = next;
  state.gol.generation++;
}

export function buildGameOfLife(tileSize) {
  const { grid, cols, rows, tiles, tileIndices, maxState } = state.gol;
  if (!grid || !tiles || tiles.length < 2) return null;
  const off = new OffscreenCanvas(cols * tileSize, rows * tileSize);
  const ctx = off.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const val = grid[r * cols + c];
      // Map cell value to tile: 0=dead(tiles[0]), 1..maxState mapped to tiles[1..N]
      let tileIdx;
      if (val === 0) {
        tileIdx = 0;
      } else {
        // Map value to tile slot: for conway val=1→slot1, for multi-state variants map accordingly
        tileIdx = Math.min(val, tiles.length - 1);
      }
      const piece = tiles[tileIdx];
      const pidx = tileIndices[tileIdx];
      const si = r * cols + c;
      const rot = state.muralTileRotation ? ((si * 7 + pidx * 13) + state.gol.generation) & 3 : 0;
      const flip = state.muralTileRotation ? !!(((si * 11 + pidx * 5) + state.gol.generation) & 1) : false;
      const tile = getCachedTile(piece, 'g' + tileIdx, tileSize, rot, flip);
      ctx.drawImage(tile, c * tileSize, r * tileSize);
    }
  return off;
}
