// Morphogenesis — continuous reaction-diffusion with advection flow.
// A slowly evolving vector field carries the chemicals across the grid,
// creating organic, path-tracing movement instead of static spots.
// Noise injection + parameter drift keep it alive forever.

import { state } from '../state.js';
import { pickDistinctTiles, getPooledPieces } from './gameoflife.js';
import { rebuildPitchTable } from './sonify.js';
import { clearTileCache } from './cache.js';

// ── Waypoints biased toward dynamic/moving patterns ──
export const MORPHO_PRESETS = [
  { f: 0.03,   k: 0.062,  label: 'Solitons' },
  { f: 0.078,  k: 0.061,  label: 'Worms' },
  { f: 0.014,  k: 0.054,  label: 'Moving Spots' },
  { f: 0.0367, k: 0.0649, label: 'Mitosis' },
  { f: 0.025,  k: 0.056,  label: 'Labyrinth' },
  { f: 0.034,  k: 0.063,  label: 'Pulsing' },
  { f: 0.022,  k: 0.052,  label: 'Chaos' },
  { f: 0.04,   k: 0.064,  label: 'Splitting' },
];
const WAYPOINTS = MORPHO_PRESETS;

const Du = 0.2097;
const Dv = 0.105;
const INIT_STEPS  = 800;
const TICK_STEPS  = 16;
const NOISE_EVERY = 25;       // inject noise frequently
const ADVECT_STRENGTH = 0.9;  // how strongly the flow carries chemicals

// ── Flow field: slowly evolving vortices ──

// Simple hash-based pseudo-noise for flow field generation
function hash(x, y, seed) {
  let h = seed;
  h = ((h << 5) - h + x) | 0;
  h = ((h << 5) - h + y) | 0;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  h = Math.imul(h ^ (h >>> 13), 0x45d9f3b);
  return ((h ^ (h >>> 16)) & 0x7fffffff) / 0x7fffffff;
}

function buildFlowField(cols, rows, mg) {
  const vx = mg.flowVx;
  const vy = mg.flowVy;
  const t = mg.flowTime;

  // Multiple vortex centers that drift over time
  const nVortices = mg.vortices.length;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      let fx = 0, fy = 0;

      // Sum contributions from each vortex
      for (let vi = 0; vi < nVortices; vi++) {
        const vt = mg.vortices[vi];
        const dx = c - vt.x;
        const dy = r - vt.y;
        const dist2 = dx * dx + dy * dy + 1;
        const strength = vt.strength / (1 + dist2 * 0.005);

        // Curl: perpendicular to radial direction = rotation
        fx += -dy * strength / Math.sqrt(dist2);
        fy +=  dx * strength / Math.sqrt(dist2);
      }

      // Add a slow global drift
      const globalAngle = t * 0.02;
      fx += Math.cos(globalAngle) * 0.3;
      fy += Math.sin(globalAngle) * 0.3;

      // Add some turbulent noise that varies spatially and temporally
      const noiseScale = 0.08;
      const nx = Math.sin(c * noiseScale + t * 0.1) * Math.cos(r * noiseScale * 1.3 + t * 0.07);
      const ny = Math.cos(r * noiseScale + t * 0.13) * Math.sin(c * noiseScale * 0.9 + t * 0.09);
      fx += nx * 0.5;
      fy += ny * 0.5;

      vx[idx] = fx * ADVECT_STRENGTH;
      vy[idx] = fy * ADVECT_STRENGTH;
    }
  }
}

function evolveVortices(mg, cols, rows) {
  mg.flowTime += 0.15;

  for (const vt of mg.vortices) {
    // Slow random walk
    vt.x += vt.vx;
    vt.y += vt.vy;
    vt.vx += (Math.random() - 0.5) * 0.3;
    vt.vy += (Math.random() - 0.5) * 0.3;
    // Dampen velocity
    vt.vx *= 0.98;
    vt.vy *= 0.98;
    // Wrap toroidally
    vt.x = ((vt.x % cols) + cols) % cols;
    vt.y = ((vt.y % rows) + rows) % rows;
    // Slowly oscillate strength
    vt.strength = vt.baseStrength * (0.6 + 0.4 * Math.sin(mg.flowTime * vt.freq + vt.phase));
  }
}

// ── Simulation with advection ──

function simulate(u, v, vx, vy, cols, rows, f, k, steps) {
  const total = rows * cols;
  const uTmp = new Float32Array(total);
  const vTmp = new Float32Array(total);
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

        // Advection: upwind differencing
        const fvx = vx[idx];
        const fvy = vy[idx];

        // ∂u/∂x and ∂u/∂y using upwind scheme (stable for advection)
        const dudx = fvx > 0
          ? uVal - uS[r * cols + cW]
          : uS[r * cols + cE] - uVal;
        const dudy = fvy > 0
          ? uVal - uS[rN * cols + c]
          : uS[rS * cols + c] - uVal;
        const dvdx = fvx > 0
          ? vVal - vS[r * cols + cW]
          : vS[r * cols + cE] - vVal;
        const dvdy = fvy > 0
          ? vVal - vS[rN * cols + c]
          : vS[rS * cols + c] - vVal;

        const advU = fvx * dudx + fvy * dudy;
        const advV = fvx * dvdx + fvy * dvdy;

        const uvv = uVal * vVal * vVal;
        uD[idx] = uVal + Du * lapU - uvv + f * (1 - uVal) - advU;
        vD[idx] = vVal + Dv * lapV + uvv - (f + k) * vVal - advV;

        // Clamp to prevent blowup
        if (uD[idx] < 0) uD[idx] = 0;
        if (uD[idx] > 1) uD[idx] = 1;
        if (vD[idx] < 0) vD[idx] = 0;
        if (vD[idx] > 1) vD[idx] = 1;
      }
    }
    cur = 1 - cur;
  }

  if (cur === 1) { u.set(uTmp); v.set(vTmp); }
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
  const { cols, rows } = state.gol;
  const mg = state.gol.morpho;
  if (!mg || !mg.v) return;
  const grid = state.gol.grid;
  const total = rows * cols;
  for (let i = 0; i < total; i++) grid[i] = vToState(mg.v[i]);
}

// ── Noise injection ──

function injectNoise(u, v, cols, rows) {
  const total = rows * cols;

  // Seed new reaction patches — these become new "organisms"
  const nPatches = 2 + Math.floor(Math.random() * 4);
  for (let p = 0; p < nPatches; p++) {
    const cx = Math.floor(Math.random() * cols);
    const cy = Math.floor(Math.random() * rows);
    const radius = 2 + Math.floor(Math.random() * 4);
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        if (dr * dr + dc * dc > radius * radius) continue;
        const r = (cy + dr + rows) % rows;
        const c = (cx + dc + cols) % cols;
        const idx = r * cols + c;
        u[idx] = 0.5 + Math.random() * 0.1;
        v[idx] = 0.25 + Math.random() * 0.1;
      }
    }
  }

  // Spray fine noise to destabilize static structures
  const noiseCount = Math.floor(total * 0.008);
  for (let i = 0; i < noiseCount; i++) {
    const idx = Math.floor(Math.random() * total);
    v[idx] = Math.max(0, Math.min(1, v[idx] + (Math.random() - 0.3) * 0.15));
    u[idx] = Math.max(0, Math.min(1, u[idx] + (Math.random() - 0.5) * 0.1));
  }
}

// ── Parameter drift: faster and biased toward moving-pattern regions ──

function driftParams(mg) {
  const pinned = state.gol.pinnedPreset;

  if (pinned >= 0 && pinned < WAYPOINTS.length) {
    // Pinned: converge toward the fixed waypoint, allow small jitter for life
    const wp = WAYPOINTS[pinned];
    mg.targetF = wp.f;
    mg.targetK = wp.k;
    mg.f += (wp.f - mg.f) * 0.01;
    mg.k += (wp.k - mg.k) * 0.01;
    // Tiny jitter so it's not perfectly static
    mg.f += (Math.random() - 0.5) * 0.00005;
    mg.k += (Math.random() - 0.5) * 0.00003;
  } else {
    // Auto: drift between random waypoints
    const df = mg.f - mg.targetF;
    const dk = mg.k - mg.targetK;
    if (df * df + dk * dk < 5e-7) {
      const wp = WAYPOINTS[Math.floor(Math.random() * WAYPOINTS.length)];
      mg.targetF = wp.f + (Math.random() - 0.5) * 0.01;
      mg.targetK = wp.k + (Math.random() - 0.5) * 0.005;
    }
    mg.f += (mg.targetF - mg.f) * 0.002;
    mg.k += (mg.targetK - mg.k) * 0.002;
    mg.f += (Math.random() - 0.5) * 0.0002;
    mg.k += (Math.random() - 0.5) * 0.0001;
  }

  mg.f = Math.max(0.01, Math.min(0.09, mg.f));
  mg.k = Math.max(0.045, Math.min(0.07, mg.k));
}

// ── Public API ──

export function morphoInit() {
  const pieces = getPooledPieces();
  if (pieces.length < 2) return;

  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const ts = state.muralTileSize;
  const cols = Math.ceil(W / ts) + 1;
  const rows = Math.ceil(H / ts) + 1;

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

  // Seed with scattered organic patches
  const nSeeds = 6 + Math.floor(Math.random() * 8);
  for (let s = 0; s < nSeeds; s++) {
    const cx = Math.floor(Math.random() * cols);
    const cy = Math.floor(Math.random() * rows);
    const radius = 2 + Math.floor(Math.random() * 4);
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        if (dr * dr + dc * dc > radius * radius) continue;
        const r = (cy + dr + rows) % rows;
        const c = (cx + dc + cols) % cols;
        const idx = r * cols + c;
        u[idx] = 0.5;
        v[idx] = 0.25 + Math.random() * 0.05;
      }
    }
  }

  // Create vortex centers for the flow field
  const nVortices = 3 + Math.floor(Math.random() * 3);
  const vortices = [];
  for (let i = 0; i < nVortices; i++) {
    vortices.push({
      x: Math.random() * cols,
      y: Math.random() * rows,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      baseStrength: (Math.random() < 0.5 ? 1 : -1) * (0.5 + Math.random() * 1.5),
      strength: 0,
      freq: 0.03 + Math.random() * 0.05,
      phase: Math.random() * Math.PI * 2,
    });
  }

  const pinned = state.gol.pinnedPreset;
  const wp = (pinned >= 0 && pinned < WAYPOINTS.length)
    ? WAYPOINTS[pinned]
    : WAYPOINTS[Math.floor(Math.random() * WAYPOINTS.length)];
  const mg = {
    u, v,
    f: wp.f, k: wp.k,
    targetF: wp.f, targetK: wp.k,
    tickCount: 0,
    flowVx: new Float32Array(total),
    flowVy: new Float32Array(total),
    flowTime: Math.random() * 100,
    vortices,
  };

  // Build initial flow field
  buildFlowField(cols, rows, mg);

  // Pre-cook with advection active
  simulate(u, v, mg.flowVx, mg.flowVy, cols, rows, mg.f, mg.k, INIT_STEPS);

  state.gol.morpho = mg;
  state.gol.grid = new Uint8Array(total);
  updateDiscreteGrid();
  state.gol.generation = 0;
  rebuildPitchTable(rows);
}

export function morphoStep() {
  const { cols, rows } = state.gol;
  const mg = state.gol.morpho;
  if (!mg || !mg.u || !mg.v) return;

  // Evolve flow field vortices
  evolveVortices(mg, cols, rows);

  // Rebuild flow field every few ticks (expensive but necessary for movement)
  if (mg.tickCount % 3 === 0) {
    buildFlowField(cols, rows, mg);
  }

  // Parameter drift
  driftParams(mg);

  // Simulate with advection
  simulate(mg.u, mg.v, mg.flowVx, mg.flowVy, cols, rows, mg.f, mg.k, TICK_STEPS);

  // Frequent noise injection
  mg.tickCount++;
  if (mg.tickCount % NOISE_EVERY === 0) {
    injectNoise(mg.u, mg.v, cols, rows);
  }

  updateDiscreteGrid();
  state.gol.generation++;
}

// ── Resize: resample simulation to new grid dimensions on zoom ──

function resampleField(src, oldCols, oldRows, newCols, newRows) {
  const dst = new Float32Array(newCols * newRows);
  const sx = oldCols / newCols;
  const sy = oldRows / newRows;

  for (let r = 0; r < newRows; r++) {
    const srcY = r * sy;
    const r0 = Math.floor(srcY);
    const r1 = Math.min(r0 + 1, oldRows - 1);
    const fy = srcY - r0;

    for (let c = 0; c < newCols; c++) {
      const srcX = c * sx;
      const c0 = Math.floor(srcX);
      const c1 = Math.min(c0 + 1, oldCols - 1);
      const fx = srcX - c0;

      // Bilinear interpolation
      const v00 = src[r0 * oldCols + c0];
      const v10 = src[r0 * oldCols + c1];
      const v01 = src[r1 * oldCols + c0];
      const v11 = src[r1 * oldCols + c1];
      dst[r * newCols + c] = v00 * (1 - fx) * (1 - fy)
                           + v10 * fx * (1 - fy)
                           + v01 * (1 - fx) * fy
                           + v11 * fx * fy;
    }
  }
  return dst;
}

export function morphoResize() {
  const mg = state.gol.morpho;
  if (!mg || !mg.u || !mg.v) return;

  const wrap = document.getElementById('mural-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const ts = state.muralTileSize;
  const newCols = Math.ceil(W / ts) + 1;
  const newRows = Math.ceil(H / ts) + 1;

  const oldCols = state.gol.cols;
  const oldRows = state.gol.rows;

  // Skip if dimensions haven't actually changed
  if (newCols === oldCols && newRows === oldRows) return;

  // Resample u and v fields
  mg.u = resampleField(mg.u, oldCols, oldRows, newCols, newRows);
  mg.v = resampleField(mg.v, oldCols, oldRows, newCols, newRows);

  // Reallocate flow field
  const total = newCols * newRows;
  mg.flowVx = new Float32Array(total);
  mg.flowVy = new Float32Array(total);

  // Scale vortex positions to new grid
  for (const vt of mg.vortices) {
    vt.x = vt.x * newCols / oldCols;
    vt.y = vt.y * newRows / oldRows;
  }

  // Update state dimensions
  state.gol.cols = newCols;
  state.gol.rows = newRows;
  state.gol.grid = new Uint8Array(total);

  // Rebuild flow field for new dimensions
  buildFlowField(newCols, newRows, mg);
  updateDiscreteGrid();
  rebuildPitchTable(newRows);
}

// ── Tile mutation: blend in new tiles without resetting the simulation ──

export function morphoMutateTiles() {
  const mg = state.gol.morpho;
  if (!mg) return;

  const pieces = getPooledPieces();
  if (pieces.length < 2) return;

  const { tiles, indices } = pickDistinctTiles(pieces, 5);

  state.gol.tiles = tiles;
  state.gol.tileIndices = indices;
  state.gol.tileA = tiles[1] || tiles[0];
  state.gol.tileB = tiles[0];
  state.gol.tileIdxA = indices[1] || indices[0];
  state.gol.tileIdxB = indices[0];

  clearTileCache();
}

// ── Immediately apply a pinned preset (called from controls.pinPreset) ──

export function morphoApplyPreset(idx) {
  const mg = state.gol.morpho;
  if (!mg) return;

  const { cols, rows } = state.gol;
  const total = cols * rows;

  if (idx >= 0 && idx < WAYPOINTS.length) {
    const wp = WAYPOINTS[idx];
    mg.f = wp.f;
    mg.k = wp.k;
    mg.targetF = wp.f;
    mg.targetK = wp.k;
  }

  // Fade existing field heavily — dissolve current structures
  for (let i = 0; i < total; i++) {
    mg.u[i] = mg.u[i] * 0.3 + 0.7;   // push u back toward 1.0 (empty)
    mg.v[i] = mg.v[i] * 0.15;          // nearly kill v
  }

  // Re-seed with fresh patches so new f/k can form its characteristic pattern
  const nSeeds = 8 + Math.floor(Math.random() * 6);
  for (let s = 0; s < nSeeds; s++) {
    const cx = Math.floor(Math.random() * cols);
    const cy = Math.floor(Math.random() * rows);
    const radius = 2 + Math.floor(Math.random() * 5);
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        if (dr * dr + dc * dc > radius * radius) continue;
        const r = (cy + dr + rows) % rows;
        const c = (cx + dc + cols) % cols;
        const i = r * cols + c;
        mg.u[i] = 0.5;
        mg.v[i] = 0.25 + Math.random() * 0.05;
      }
    }
  }

  // Pre-cook a bit so the new pattern is visible immediately
  simulate(mg.u, mg.v, mg.flowVx, mg.flowVy, cols, rows, mg.f, mg.k, 400);
}
