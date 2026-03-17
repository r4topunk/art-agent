import { getCachedTile } from './cache.js';
import { state } from '../state.js';
import { rebuildPitchTable } from './sonify.js';
import { rdInit, rdStep } from './reactiondiffusion.js';
import { morphoInit, morphoStep } from './morphogenesis.js';

// Cellular automata at tile resolution on a toroidal grid.
// Supports multiple rule variants via state.gol.variant.

export const GOL_VARIANTS = ['conway', 'immigration', 'quadlife', 'reaction-diffusion', 'morphogenesis'];
export const GOL_VARIANT_LABELS = {
  conway: "Conway's GoL",
  immigration: 'Immigration',
  quadlife: 'QuadLife',
  'reaction-diffusion': 'Reaction-Diffusion',
  morphogenesis: 'Morphogenesis',
};

// Number of tile slots each variant needs (including dead tile)
const VARIANT_TILE_COUNT = {
  conway: 2,        // alive + dead
  immigration: 3,    // 2 species + dead
  quadlife: 5,       // 4 species + dead
  'reaction-diffusion': 5,  // dead + 4 concentration levels
  morphogenesis: 5,  // dead + 4 concentration levels
};

// Color histogram for a 16x16 tile (8-palette normalized)
function tileHistogram(piece) {
  const hist = new Float32Array(8);
  let total = 0;
  for (let r = 0; r < piece.length; r++)
    for (let c = 0; c < piece[r].length; c++) {
      hist[piece[r][c]]++;
      total++;
    }
  if (total > 0) for (let i = 0; i < 8; i++) hist[i] /= total;
  return hist;
}

// L2 distance between two histograms
function histDist(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += (a[i] - b[i]) ** 2;
  return d; // skip sqrt — comparing relative magnitudes is enough
}

// Shannon entropy of a histogram (higher = more complex/patterned tile)
function histEntropy(hist) {
  let h = 0;
  for (let i = 0; i < hist.length; i++) {
    if (hist[i] > 0) h -= hist[i] * Math.log2(hist[i]);
  }
  return h;
}

// Weighted random pick: select index with probability proportional to weights
function weightedRandom(weights) {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) sum += weights[i];
  let r = Math.random() * sum;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

// Dominant color of a tile (palette index with highest count)
function dominantColor(hist) {
  let best = 0;
  for (let i = 1; i < hist.length; i++) {
    if (hist[i] > hist[best]) best = i;
  }
  return best;
}

// Top-2 colors of a tile (for secondary diversity check)
function topTwoColors(hist) {
  let first = 0, second = -1;
  for (let i = 1; i < hist.length; i++) {
    if (hist[i] > hist[first]) { second = first; first = i; }
    else if (second === -1 || hist[i] > hist[second]) second = i;
  }
  return [first, second === -1 ? first : second];
}

// Farthest-point sampling that enforces different dominant colors between picks.
// Uses entropy bias + stochastic selection for variety across rounds.
export function pickDistinctTiles(pieces, count) {
  if (pieces.length <= count) {
    return {
      tiles: pieces.slice(0, count),
      indices: Array.from({ length: Math.min(pieces.length, count) }, (_, i) => i),
    };
  }

  const hists = pieces.map(tileHistogram);
  const entropies = hists.map(histEntropy);
  const maxEnt = Math.max(...entropies, 1e-6);
  const entWeights = entropies.map(e => 0.15 + 0.85 * (e / maxEnt));
  const domColors = hists.map(dominantColor);
  const topColors = hists.map(topTwoColors);

  const picked = [];
  const pickedIdx = [];
  const used = new Set();
  const usedDomColors = new Set();

  // First tile: entropy-weighted random pick
  const first = weightedRandom(entWeights);
  picked.push(pieces[first]);
  pickedIdx.push(first);
  used.add(first);
  usedDomColors.add(domColors[first]);

  while (picked.length < count) {
    const candidates = [];
    for (let i = 0; i < pieces.length; i++) {
      if (used.has(i)) continue;

      let minDist = Infinity;
      for (const pi of pickedIdx) {
        const d = histDist(hists[i], hists[pi]);
        if (d < minDist) minDist = d;
      }

      // Penalize if dominant color already used — scale down score heavily
      let colorPenalty = 1.0;
      if (usedDomColors.has(domColors[i])) {
        // Check if at least the secondary color is different
        const [, sec] = topColors[i];
        colorPenalty = usedDomColors.has(sec) ? 0.05 : 0.2;
      }

      candidates.push({
        idx: i,
        score: minDist * entWeights[i] * colorPenalty,
      });
    }
    if (!candidates.length) break;

    // Sort descending, weighted-random from top 25%
    candidates.sort((a, b) => b.score - a.score);
    const topN = Math.max(1, Math.ceil(candidates.length * 0.25));
    const topCandidates = candidates.slice(0, topN);
    const topScores = topCandidates.map(c => c.score);
    const winner = topCandidates[weightedRandom(topScores)];

    picked.push(pieces[winner.idx]);
    pickedIdx.push(winner.idx);
    used.add(winner.idx);
    usedDomColors.add(domColors[winner.idx]);
  }

  return { tiles: picked, indices: pickedIdx };
}

// Pool current pieces with N past generations (controlled by state.pastGenerations)
export function getPooledPieces() {
  const current = state.allPieces.length ? state.allPieces : [];
  const n = state.pastGenerations || 0;
  if (n <= 0 || !state.pieceHistory.length) return current;
  const histLen = state.pieceHistory.length;
  const sliceStart = Math.max(0, histLen - n);
  const past = state.pieceHistory.slice(sliceStart).flat();
  return [...current, ...past];
}

export function golInit() {
  const variant = state.gol.variant || 'conway';

  // Reaction-diffusion and morphogenesis have their own init
  if (variant === 'reaction-diffusion') return rdInit();
  if (variant === 'morphogenesis') return morphoInit();

  const pieces = getPooledPieces();
  if (pieces.length < 2) return;
  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const ts = state.muralTileSize;
  const cols = Math.ceil(W / ts) + 1;
  const rows = Math.ceil(H / ts) + 1;

  const needed = VARIANT_TILE_COUNT[variant] || 2;
  const { tiles, indices } = pickDistinctTiles(pieces, needed);

  state.gol.tiles = tiles;
  state.gol.tileIndices = indices;
  state.gol.tileA = tiles[1] || tiles[0];
  state.gol.tileB = tiles[0];
  state.gol.tileIdxA = indices[1] || indices[0];
  state.gol.tileIdxB = indices[0];
  state.gol.cols = cols;
  state.gol.rows = rows;
  state.gol.maxState = needed - 1;

  // Build kaleidoscope-symmetric initial state
  const hC = Math.ceil(cols / 2);
  const hR = Math.ceil(rows / 2);
  const grid = new Uint8Array(rows * cols);
  const aliveStates = needed - 1;

  for (let r = 0; r < hR; r++)
    for (let c = 0; c < hC; c++) {
      let val = 0;
      if (variant === 'quadlife') {
        val = Math.random() < 0.4 ? (1 + Math.floor(Math.random() * 4)) : 0;
      } else if (variant === 'immigration') {
        val = Math.random() < 0.4 ? (1 + Math.floor(Math.random() * 2)) : 0;
      } else {
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

  // Reaction-diffusion and morphogenesis have their own continuous simulation
  if (variant === 'reaction-diffusion') return rdStep();
  if (variant === 'morphogenesis') return morphoStep();

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
