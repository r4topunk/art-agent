import { getCachedTile } from './cache.js';
import { state } from '../state.js';
import { rebuildPitchTable } from './sonify.js';

// Conway's Game of Life at tile resolution on a toroidal grid.
// Two tiles are picked: tileA = alive, tileB = dead.
// Initial state is a kaleidoscope-symmetric random pattern.

export function golInit() {
  const pieces = state.allPieces.length ? state.allPieces : [];
  if (pieces.length < 2) return;
  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const ts = state.muralTileSize;
  const cols = Math.ceil(W / ts) + 1;
  const rows = Math.ceil(H / ts) + 1;

  // Pick 2 tiles with different visual content (fingerprint = sum of all pixel values)
  function fp(piece) {
    let s = 0;
    for (let r = 0; r < piece.length; r++)
      for (let c = 0; c < piece[r].length; c++) s += piece[r][c];
    return s;
  }

  const idxA = Math.floor(Math.random() * pieces.length);
  const fpA = fp(pieces[idxA]);

  // Collect all candidates with a different fingerprint, then pick randomly
  const candidates = [];
  for (let i = 0; i < pieces.length; i++) {
    if (i !== idxA && fp(pieces[i]) !== fpA) candidates.push(i);
  }
  const idxB = candidates.length > 0
    ? candidates[Math.floor(Math.random() * candidates.length)]
    : (idxA + 1) % pieces.length;
  state.gol.tileA = pieces[idxA];
  state.gol.tileB = pieces[idxB];
  state.gol.tileIdxA = idxA;
  state.gol.tileIdxB = idxB;
  state.gol.cols = cols;
  state.gol.rows = rows;

  // Build kaleidoscope-symmetric initial state
  // Work on a quadrant, then mirror
  const hC = Math.ceil(cols / 2);
  const hR = Math.ceil(rows / 2);
  const grid = new Uint8Array(rows * cols);

  // Random quadrant with ~40% alive density
  for (let r = 0; r < hR; r++)
    for (let c = 0; c < hC; c++) {
      const alive = Math.random() < 0.4 ? 1 : 0;
      // Mirror to all 4 quadrants
      const mc = cols - 1 - c;
      const mr = rows - 1 - r;
      grid[r * cols + c] = alive;
      grid[r * cols + mc] = alive;
      grid[mr * cols + c] = alive;
      grid[mr * cols + mc] = alive;
    }

  state.gol.grid = grid;
  state.gol.generation = 0;
  rebuildPitchTable(rows);
}

export function golStep() {
  const { grid, cols, rows } = state.gol;
  if (!grid) return;
  const next = new Uint8Array(rows * cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      let neighbors = 0;
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = (r + dr + rows) % rows;
          const nc = (c + dc + cols) % cols;
          neighbors += grid[nr * cols + nc];
        }
      const alive = grid[r * cols + c];
      // Conway's rules
      if (alive) {
        next[r * cols + c] = (neighbors === 2 || neighbors === 3) ? 1 : 0;
      } else {
        next[r * cols + c] = (neighbors === 3) ? 1 : 0;
      }
    }
  state.gol.grid = next;
  state.gol.generation++;
}

export function buildGameOfLife(tileSize) {
  const { grid, cols, rows, tileA, tileB, tileIdxA, tileIdxB } = state.gol;
  if (!grid || !tileA || !tileB) return null;
  const off = new OffscreenCanvas(cols * tileSize, rows * tileSize);
  const ctx = off.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const alive = grid[r * cols + c];
      const piece = alive ? tileA : tileB;
      const idx = alive ? tileIdxA : tileIdxB;
      const si = r * cols + c;
      const rot = state.muralTileRotation ? ((si * 7 + idx * 13) + state.gol.generation) & 3 : 0;
      const flip = state.muralTileRotation ? !!(((si * 11 + idx * 5) + state.gol.generation) & 1) : false;
      const tile = getCachedTile(piece, 'g' + (alive ? 'a' : 'b'), tileSize, rot, flip);
      ctx.drawImage(tile, c * tileSize, r * tileSize);
    }
  return off;
}
