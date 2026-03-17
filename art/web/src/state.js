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
  volumePercent: _s.volumePercent ?? 60,
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
    tiles: [],
    tileIndices: [],
    maxState: 1,
    generation: 0,
    tickTimer: null,
    resetTimer: null,
    running: false,
    tickMS: 250,
    scanCol: 0,
    variant: _s.golVariant ?? 'conway',
    rd: null,  // { u: Float32Array, v: Float32Array, preset: number }
  },
  // Audio
  audio: {
    ctx: null,
    enabled: false,
    masterGain: null,
    synthBus: null,  // persistent oscillator bank bus
    grainBus: null,  // birth/death transient bus
    oscBank: [],     // [{osc, gain, panner}] per row
    prevGrid: null,  // Uint8Array copy for diff
    dryGain: null,   // ref for macro modulation
    wetGain: null,   // ref for macro modulation
    droneOsc: [],    // [{osc, gain}] x 2
    filter: null,
    waveAlive: null, // PeriodicWave for alive-cell oscillators
    waveDrone: null, // PeriodicWave for drone oscillators
  },
};
