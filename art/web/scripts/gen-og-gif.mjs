#!/usr/bin/env node
/**
 * Generate an animated GIF for OG image preview.
 * Reads pieces.json → simulates Conway's GoL → outputs og-preview.gif
 *
 * Usage: node scripts/gen-og-gif.mjs [--frames 60] [--delay 80] [--tile 8] [--out public/og-preview.gif]
 */
import { readFileSync, writeFileSync } from 'fs';
import pkg from 'gifenc';
const { GIFEncoder, quantize, applyPalette } = pkg;
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

// --- Config (CLI overrides) ---
const args = process.argv.slice(2);
function flag(name, def) {
  const i = args.indexOf('--' + name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
}

const FRAMES    = +flag('frames', 60);
const DELAY     = +flag('delay', 80);      // ms per frame
const TILE_SIZE = +flag('tile', 8);         // px per cell
const WIDTH     = +flag('width', 1200);     // OG recommended
const HEIGHT    = +flag('height', 630);     // OG recommended
const OUT       = flag('out', resolve(ROOT, 'public/og-preview.gif'));

// --- Palette (same as constants.js) ---
const PALETTE = [
  [0, 0, 0],
  [255, 241, 232],
  [255, 0, 77],
  [255, 163, 0],
  [255, 236, 39],
  [0, 228, 54],
  [41, 173, 255],
  [255, 119, 168],
];

// --- Load pieces ---
const piecesPath = resolve(ROOT, 'public/data/pieces.json');
const { pieces } = JSON.parse(readFileSync(piecesPath, 'utf8'));
console.log(`Loaded ${pieces.length} pieces`);

// --- Pick distinct tiles for alive/dead ---
function pickTiles(count) {
  const fp = (p) => {
    let s = 0;
    for (const row of p) for (const v of row) s += v;
    return s;
  };
  const indices = Array.from({ length: pieces.length }, (_, i) => i);
  // Shuffle
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const picked = [];
  const usedFps = new Set();
  for (const i of indices) {
    if (picked.length >= count) break;
    const f = fp(pieces[i]);
    if (!usedFps.has(f)) {
      usedFps.add(f);
      picked.push(pieces[i]);
    }
  }
  return picked;
}

const tiles = pickTiles(2); // dead=0, alive=1
const tileDead = tiles[0];
const tileAlive = tiles[1];

// --- Grid setup ---
const cols = Math.ceil(WIDTH / TILE_SIZE);
const rows = Math.ceil(HEIGHT / TILE_SIZE);
const canvasW = cols * TILE_SIZE;
const canvasH = rows * TILE_SIZE;

// Init with 4-way kaleidoscope symmetry (like the app)
let grid = new Uint8Array(rows * cols);
const hC = Math.ceil(cols / 2);
const hR = Math.ceil(rows / 2);
for (let r = 0; r < hR; r++)
  for (let c = 0; c < hC; c++) {
    const val = Math.random() < 0.4 ? 1 : 0;
    const mc = cols - 1 - c;
    const mr = rows - 1 - r;
    grid[r * cols + c] = val;
    grid[r * cols + mc] = val;
    grid[mr * cols + c] = val;
    grid[mr * cols + mc] = val;
  }

// --- GoL step ---
function step() {
  const next = new Uint8Array(rows * cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      let n = 0;
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = (r + dr + rows) % rows;
          const nc = (c + dc + cols) % cols;
          if (grid[nr * cols + nc]) n++;
        }
      const cell = grid[r * cols + c];
      if (cell) next[r * cols + c] = (n === 2 || n === 3) ? 1 : 0;
      else next[r * cols + c] = (n === 3) ? 1 : 0;
    }
  grid = next;
}

// --- Render one frame to RGBA pixel buffer ---
function renderFrame(gen) {
  const data = new Uint8Array(canvasW * canvasH * 4);

  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const val = grid[r * cols + c];
      const piece = val ? tileAlive : tileDead;
      const si = r * cols + c;
      // Rotation based on position + generation (like the app)
      const rot = ((si * 7 + val * 13) + gen) & 3;
      const flip = !!(((si * 11 + val * 5) + gen) & 1);

      // Draw 16x16 piece into tile area
      for (let py = 0; py < TILE_SIZE; py++)
        for (let px = 0; px < TILE_SIZE; px++) {
          // Map tile pixel back to 16x16 source
          const srcY = Math.floor(py * 16 / TILE_SIZE);
          const srcX = Math.floor(px * 16 / TILE_SIZE);

          // Apply rotation and flip
          let sr = srcY, sc = srcX;
          if (rot === 1) { sr = srcX; sc = 15 - srcY; }
          else if (rot === 2) { sr = 15 - srcY; sc = 15 - srcX; }
          else if (rot === 3) { sr = 15 - srcX; sc = srcY; }
          if (flip) sc = 15 - sc;

          const palIdx = (piece[sr] && piece[sr][sc]) || 0;
          const rgb = PALETTE[palIdx] || PALETTE[0];

          const outX = c * TILE_SIZE + px;
          const outY = r * TILE_SIZE + py;
          if (outX >= WIDTH || outY >= HEIGHT) continue;

          const i = (outY * WIDTH + outX) * 4;
          data[i] = rgb[0];
          data[i + 1] = rgb[1];
          data[i + 2] = rgb[2];
          data[i + 3] = 255;
        }
  }

  return data;
}

// --- Generate GIF ---
console.log(`Generating ${FRAMES} frames at ${cols}x${rows} cells (${WIDTH}x${HEIGHT}px)...`);

const gif = GIFEncoder();

for (let f = 0; f < FRAMES; f++) {
  if (f > 0) step();
  const rgba = renderFrame(f);
  // Quantize to 256-color palette
  const palette = quantize(rgba, 256);
  const indexed = applyPalette(rgba, palette);
  gif.writeFrame(indexed, WIDTH, HEIGHT, { palette, delay: DELAY });
  if ((f + 1) % 10 === 0) process.stdout.write(`  frame ${f + 1}/${FRAMES}\r`);
}

gif.finish();
const bytes = gif.bytes();
writeFileSync(OUT, Buffer.from(bytes));
console.log(`\nSaved ${OUT} (${(bytes.length / 1024).toFixed(0)} KB)`);
