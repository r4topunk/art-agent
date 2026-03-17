// Gray-Scott reaction-diffusion system.
// Produces organic, Rorschach-like patterns: spots, stripes, labyrinths, coral.

import { state } from '../state.js';
import { pickDistinctTiles } from './gameoflife.js';
import { rebuildPitchTable } from './sonify.js';

export const RD_PRESETS = [
  { name: 'mitosis',    f: 0.0367, k: 0.0649, label: 'Mitosis' },
  { name: 'coral',      f: 0.0545, k: 0.062,  label: 'Coral' },
  { name: 'labyrinth',  f: 0.029,  k: 0.057,  label: 'Labyrinth' },
  { name: 'solitons',   f: 0.03,   k: 0.062,  label: 'Solitons' },
  { name: 'worms',      f: 0.078,  k: 0.061,  label: 'Worms' },
];

const Du = 0.2097;   // U diffusion rate
const Dv = 0.105;    // V diffusion rate
const INIT_STEPS = 1500;  // pre-cook on init for visible structure
const TICK_STEPS = 24;    // per visual tick

// ── Simulation core ──

function simulate(u, v, cols, rows, f, k, steps, bilateral) {
  const total = rows * cols;
  const uTmp = new Float32Array(total);
  const vTmp = new Float32Array(total);

  // Double-buffer to avoid per-step copies
  const uBuf = [u, uTmp];
  const vBuf = [v, vTmp];
  let cur = 0;

  for (let step = 0; step < steps; step++) {
    const uS = uBuf[cur], vS = vBuf[cur];
    const uD = uBuf[1 - cur], vD = vBuf[1 - cur];

    for (let r = 0; r < rows; r++) {
      const rN = r === 0 ? rows - 1 : r - 1;
      const rS = r === rows - 1 ? 0 : r + 1;
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const uVal = uS[idx];
        const vVal = vS[idx];

        const cW = c === 0 ? cols - 1 : c - 1;
        const cE = c === cols - 1 ? 0 : c + 1;

        // 5-point Laplacian (toroidal)
        const lapU = uS[rN * cols + c] + uS[rS * cols + c]
                   + uS[r * cols + cW] + uS[r * cols + cE] - 4 * uVal;
        const lapV = vS[rN * cols + c] + vS[rS * cols + c]
                   + vS[r * cols + cW] + vS[r * cols + cE] - 4 * vVal;

        const uvv = uVal * vVal * vVal;
        uD[idx] = uVal + Du * lapU - uvv + f * (1 - uVal);
        vD[idx] = vVal + Dv * lapV + uvv - (f + k) * vVal;
      }
    }
    cur = 1 - cur;
  }

  // If result ended in temp buffer, copy back
  if (cur === 1) { u.set(uTmp); v.set(vTmp); }

  // Force bilateral symmetry (left → right mirror)
  if (bilateral) {
    const halfC = Math.ceil(cols / 2);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < halfC; c++) {
        const mc = cols - 1 - c;
        u[r * cols + mc] = u[r * cols + c];
        v[r * cols + mc] = v[r * cols + c];
      }
    }
  }
}

// Map V concentration → discrete tile state (0-4)
function vToState(val) {
  if (val < 0.04) return 0;
  if (val < 0.12) return 1;
  if (val < 0.20) return 2;
  if (val < 0.30) return 3;
  return 4;
}

function updateDiscreteGrid() {
  const { cols, rows, rd } = state.gol;
  if (!rd || !rd.v) return;
  const grid = state.gol.grid;
  const total = rows * cols;
  for (let i = 0; i < total; i++) grid[i] = vToState(rd.v[i]);
}

// ── Public API ──

export function rdInit() {
  const pieces = state.allPieces.length ? state.allPieces : [];
  if (pieces.length < 2) return;

  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const ts = state.muralTileSize;
  const cols = Math.ceil(W / ts) + 1;
  const rows = Math.ceil(H / ts) + 1;

  // 5 tile slots: dead + 4 concentration levels
  const { tiles, indices } = pickDistinctTiles(pieces, 5);

  state.gol.tiles = tiles;
  state.gol.tileIndices = indices;
  state.gol.tileA = tiles[1] || tiles[0];
  state.gol.tileB = tiles[0];
  state.gol.tileIdxA = indices[1] || indices[0];
  state.gol.tileIdxB = indices[0];
  state.gol.cols = cols;
  state.gol.rows = rows;
  state.gol.maxState = 4;

  const total = rows * cols;
  const u = new Float32Array(total).fill(1.0);
  const v = new Float32Array(total).fill(0.0);

  // Random preset each round
  const presetIdx = Math.floor(Math.random() * RD_PRESETS.length);
  const bilateral = true; // RD is always bilateral (Rorschach)

  // Seed V: circular patches of chemical V
  const seedCols = bilateral ? Math.ceil(cols / 2) : cols;
  const nSeeds = 3 + Math.floor(Math.random() * 5);
  for (let s = 0; s < nSeeds; s++) {
    const cx = Math.floor(Math.random() * seedCols * 0.6 + seedCols * 0.2);
    const cy = Math.floor(Math.random() * rows * 0.6 + rows * 0.2);
    const radius = 2 + Math.floor(Math.random() * 3);
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        if (dr * dr + dc * dc > radius * radius) continue;
        const r = (cy + dr + rows) % rows;
        const c = (cx + dc + cols) % cols;
        const idx = r * cols + c;
        u[idx] = 0.5;
        v[idx] = 0.25 + Math.random() * 0.05;
        if (bilateral) {
          const mc = cols - 1 - c;
          u[r * cols + mc] = u[idx];
          v[r * cols + mc] = v[idx];
        }
      }
    }
  }

  // Pre-cook: run enough iterations for visible pattern structure
  const preset = RD_PRESETS[presetIdx];
  simulate(u, v, cols, rows, preset.f, preset.k, INIT_STEPS, bilateral);

  state.gol.rd = { u, v, preset: presetIdx };
  state.gol.grid = new Uint8Array(total);
  updateDiscreteGrid();
  state.gol.generation = 0;
  rebuildPitchTable(rows);
}

export function rdStep() {
  const { cols, rows, rd } = state.gol;
  if (!rd || !rd.u || !rd.v) return;

  const preset = RD_PRESETS[rd.preset] || RD_PRESETS[0];
  const bilateral = true; // RD is always bilateral (Rorschach)

  simulate(rd.u, rd.v, cols, rows, preset.f, preset.k, TICK_STEPS, bilateral);
  updateDiscreteGrid();
  state.gol.generation++;
}
