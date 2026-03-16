import { loadSettings } from './persist.js';
const _s = loadSettings();

export const state = {
  lossHistory: [],
  gradHistory: [],
  allPieces: [],
  selectedPieces: [],
  muralTileSize: _s.muralTileSize ?? 16,
  muralMode: _s.muralMode ?? 'wallpaper',
  kaleidoRunning: false,
  kaleidoTimer: null,
  TRANSITION_MS: _s.TRANSITION_MS ?? 3500,
  KALEIDO_FLIP_MS: _s.KALEIDO_FLIP_MS ?? 3500,
  muralTileRotation: _s.muralTileRotation ?? true,
  muralPaused: _s.muralPaused ?? false,
  layoutSeed: 0,
  layoutTimer: null,
  rotationTimer: null,
  // Game of Life
  gol: {
    grid: null,
    cols: 0,
    rows: 0,
    tileA: null,
    tileB: null,
    tileIdxA: 0,
    tileIdxB: 0,
    generation: 0,
    tickTimer: null,
    resetTimer: null,
    running: false,
  },
  // Audio
  audio: {
    ctx: null,
    enabled: false,
    masterGain: null,
    voices: [],      // [{osc, gain, pan}] x 8
    droneOsc: [],    // [{osc, gain}] x 2
    filter: null,
    scanTimer: null,
  },
};
