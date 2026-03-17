import { state } from '../state.js';

// ── Scales & pitch mapping ──

const SCALES = [
  [0, 3, 5, 7, 10],          // C minor pentatonic
  [0, 2, 3, 5, 7, 9, 10],    // D dorian
  [0, 1, 3, 5, 7, 8, 10],    // E phrygian
  [0, 2, 4, 6, 7, 9, 11],    // F lydian
  [0, 2, 4, 5, 7, 9, 10],    // G mixolydian
  [0, 2, 3, 5, 7, 8, 10],    // A aeolian (natural minor)
];
const ROOT_NOTES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00];

let currentScale = [];
let pitchTable = [];

function buildScaleFreqs(scaleIdx, octaveOffset) {
  const intervals = SCALES[scaleIdx];
  const root = ROOT_NOTES[scaleIdx];
  const baseOctaveShift = Math.pow(2, -2 + (octaveOffset || 0));
  const freqs = [];
  for (let oct = 0; oct < 4; oct++) {
    for (const semi of intervals) {
      freqs.push(root * baseOctaveShift * Math.pow(2, oct) * Math.pow(2, semi / 12));
    }
  }
  freqs.sort((a, b) => a - b);
  currentScale = freqs;
}

export function buildPitchTable(rows) {
  const scale = currentScale.length ? currentScale : ROOT_NOTES;
  pitchTable = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const t = rows > 1 ? (rows - 1 - r) / (rows - 1) : 0;
    const idx = Math.round(t * (scale.length - 1));
    pitchTable[r] = scale[idx];
  }
}

export { buildPitchTable as rebuildPitchTable };

// ── Impulse response (algorithmic reverb) ──

function createImpulseResponse(ctx) {
  const sr = ctx.sampleRate;
  const length = Math.floor(sr * 2.5);
  const buffer = ctx.createBuffer(2, length, sr);

  // Early reflections: discrete taps simulating a medium hall
  const earlyTaps = [
    { time: 0.011, amp: 0.7 },
    { time: 0.019, amp: 0.5 },
    { time: 0.027, amp: 0.4 },
    { time: 0.037, amp: 0.3 },
    { time: 0.053, amp: 0.25 },
    { time: 0.071, amp: 0.2 },
  ];

  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    // Diffuse tail: filtered noise with exponential decay
    for (let i = 0; i < length; i++) {
      const t = i / sr;
      // Faster initial decay, then longer tail (double exponential)
      const env = 0.4 * Math.exp(-3.0 * t) + 0.6 * Math.exp(-1.2 * t);
      // Stereo decorrelation: different random seeds per channel
      data[i] = (Math.random() * 2 - 1) * env * 0.5;
    }
    // Stamp early reflections (alternating stereo emphasis)
    for (const tap of earlyTaps) {
      const idx = Math.floor(tap.time * sr);
      if (idx < length) {
        const stereoAmp = ch === 0 ? tap.amp : tap.amp * 0.7;
        data[idx] += stereoAmp * (Math.random() > 0.5 ? 1 : -1);
      }
    }
  }
  return buffer;
}

// ── Saturation curve (soft-clip tanh) ──

function createSaturationCurve(amount = 0.4) {
  const n = 8192;
  const curve = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i * 2) / n - 1;
    // Blend between linear and tanh for subtle warmth
    curve[i] = x * (1 - amount) + Math.tanh(x * 1.5) * amount;
  }
  return curve;
}

// ── Audio graph ──

function buildGraph() {
  const a = state.audio;
  const ctx = a.ctx;

  // ── Master output chain ──
  // masterGain → saturation → lowpass → highShelf → chorusDry+chorusWet → compressor → reverbDry+reverbWet → destination

  a.masterGain = ctx.createGain();
  a.masterGain.gain.value = 0.55;

  // Subtle saturation for analog warmth
  const saturation = ctx.createWaveShaper();
  saturation.curve = createSaturationCurve(0.35);
  saturation.oversample = '2x';

  // Main lowpass filter (modulated by macro)
  a.filter = ctx.createBiquadFilter();
  a.filter.type = 'lowpass';
  a.filter.frequency.value = 3000;
  a.filter.Q.value = 0.8;

  // High shelf for "air" (+3dB above 8kHz)
  const airShelf = ctx.createBiquadFilter();
  airShelf.type = 'highshelf';
  airShelf.frequency.value = 8000;
  airShelf.gain.value = 3;

  // ── Chorus (stereo modulated delay) ──
  const chorusDry = ctx.createGain();
  chorusDry.gain.value = 0.75;

  const chorusWetL = ctx.createGain();
  chorusWetL.gain.value = 0.25;
  const chorusWetR = ctx.createGain();
  chorusWetR.gain.value = 0.25;

  const chorusDelayL = ctx.createDelay(0.05);
  chorusDelayL.delayTime.value = 0.008;
  const chorusDelayR = ctx.createDelay(0.05);
  chorusDelayR.delayTime.value = 0.012;

  // LFO modulating delay times for movement
  const chorusLfoL = ctx.createOscillator();
  chorusLfoL.type = 'sine';
  chorusLfoL.frequency.value = 0.6;
  const chorusLfoDepthL = ctx.createGain();
  chorusLfoDepthL.gain.value = 0.002; // ±2ms modulation
  chorusLfoL.connect(chorusLfoDepthL);
  chorusLfoDepthL.connect(chorusDelayL.delayTime);
  chorusLfoL.start();

  const chorusLfoR = ctx.createOscillator();
  chorusLfoR.type = 'sine';
  chorusLfoR.frequency.value = 0.8; // slightly different rate for stereo
  const chorusLfoDepthR = ctx.createGain();
  chorusLfoDepthR.gain.value = 0.002;
  chorusLfoR.connect(chorusLfoDepthR);
  chorusLfoDepthR.connect(chorusDelayR.delayTime);
  chorusLfoR.start();

  // Chorus merge node
  const chorusMerge = ctx.createChannelMerger(2);

  // ── Compressor (glue + punch) ──
  const compressor = ctx.createDynamicsCompressor();
  compressor.threshold.value = -18;
  compressor.knee.value = 12;
  compressor.ratio.value = 4;
  compressor.attack.value = 0.006;
  compressor.release.value = 0.15;

  // ── Reverb ──
  const convolver = ctx.createConvolver();
  convolver.buffer = createImpulseResponse(ctx);

  a.dryGain = ctx.createGain();
  a.dryGain.gain.value = 0.6;
  a.wetGain = ctx.createGain();
  a.wetGain.gain.value = 0.4;

  // ── Connect master chain ──
  a.masterGain.connect(saturation);
  saturation.connect(a.filter);
  a.filter.connect(airShelf);

  // airShelf → chorus split
  airShelf.connect(chorusDry);
  airShelf.connect(chorusDelayL);
  airShelf.connect(chorusDelayR);

  // Chorus wet: delay → gain → merger (L/R separated for width)
  const chorusSplitL = ctx.createChannelSplitter(2);
  const chorusSplitR = ctx.createChannelSplitter(2);
  chorusDelayL.connect(chorusWetL);
  chorusDelayR.connect(chorusWetR);

  // Both dry and wet → compressor
  chorusDry.connect(compressor);
  chorusWetL.connect(compressor);
  chorusWetR.connect(compressor);

  // Compressor → reverb dry/wet
  compressor.connect(a.dryGain);
  compressor.connect(convolver);
  convolver.connect(a.wetGain);
  a.dryGain.connect(ctx.destination);
  a.wetGain.connect(ctx.destination);

  // Store refs for modulation
  a._chorus = { chorusLfoL, chorusLfoR, chorusLfoDepthL, chorusLfoDepthR, chorusDry, chorusWetL, chorusWetR };
  a._compressor = compressor;
  a._saturation = saturation;

  // ── Synth bus (persistent oscillator bank) ──
  a.synthBus = ctx.createGain();
  a.synthBus.gain.value = 0.5;
  a.synthBus.connect(a.masterGain);

  // ── Grain bus (birth/death/scan transients) ──
  a.grainBus = ctx.createGain();
  a.grainBus.gain.value = 0.4;
  a.grainBus.connect(a.masterGain);

  // ── Drone bus (drone + sub bass) ──
  const droneBus = ctx.createGain();
  droneBus.gain.value = 0.3;
  droneBus.connect(a.masterGain);

  // Two drone oscillators (root + fifth)
  a.droneOsc = [65.41, 98.00].map((freq) => {
    const osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.value = freq;
    const gain = ctx.createGain();
    gain.gain.value = 0;
    osc.connect(gain);
    gain.connect(droneBus);
    osc.start();
    return { osc, gain, bus: droneBus };
  });

  // Sub bass: pure sine one octave below root drone
  const subOsc = ctx.createOscillator();
  subOsc.type = 'sine';
  subOsc.frequency.value = 32.7; // C1
  const subGain = ctx.createGain();
  subGain.gain.value = 0;
  // Sub bass filter: remove everything above ~120Hz
  const subFilter = ctx.createBiquadFilter();
  subFilter.type = 'lowpass';
  subFilter.frequency.value = 120;
  subFilter.Q.value = 0.5;
  subOsc.connect(subGain);
  subGain.connect(subFilter);
  subFilter.connect(droneBus);
  subOsc.start();
  a._sub = { osc: subOsc, gain: subGain };
}

// ── Public API ──

export function initAudio() {
  const a = state.audio;
  if (a.ctx) return;
  a.ctx = new (window.AudioContext || window.webkitAudioContext)();
  buildGraph();
}

export function toggleSound() {
  const a = state.audio;
  if (!a.ctx) initAudio();

  a.enabled = !a.enabled;
  const btn = document.getElementById('mural-sound-btn');
  if (btn) {
    btn.classList.toggle('active', a.enabled);
    btn.textContent = a.enabled ? 'ON' : 'OFF';
  }

  if (a.enabled) {
    if (a.ctx.state === 'suspended') a.ctx.resume();
    // Apply persisted volume
    if (a.masterGain && state.volumePercent != null) {
      a.masterGain.gain.value = state.volumePercent / 100;
    }
    const now = a.ctx.currentTime;
    a.droneOsc.forEach(({ gain }) => {
      gain.gain.cancelScheduledValues(now);
      gain.gain.setTargetAtTime(0.5, now, 0.3);
    });
    if (a._sub) {
      a._sub.gain.gain.cancelScheduledValues(now);
      a._sub.gain.gain.setTargetAtTime(0.35, now, 0.5);
    }
  } else {
    stopSound();
  }
}

export function setVolume(delta) {
  const a = state.audio;
  if (!a.ctx || !a.masterGain) return;
  a.masterGain.gain.value = Math.max(0, Math.min(1, a.masterGain.gain.value + delta));
}

export function stopSound() {
  const a = state.audio;
  if (!a.ctx) return;

  const now = a.ctx.currentTime;
  a.droneOsc.forEach(({ gain }) => {
    gain.gain.cancelScheduledValues(now);
    gain.gain.setTargetAtTime(0, now, 0.5);
  });
  if (a._sub) {
    a._sub.gain.gain.cancelScheduledValues(now);
    a._sub.gain.gain.setTargetAtTime(0, now, 0.5);
  }

  destroyOscBank();
  a.prevGrid = null;

  if (!a.enabled) return;
  a.enabled = false;
  const btn = document.getElementById('mural-sound-btn');
  if (btn) {
    btn.classList.remove('active');
    btn.textContent = 'OFF';
  }
}

// ── Persistent oscillator bank (2-voice unison per row) ──

function fadeOutOscBank(durationSec = 0.8) {
  const a = state.audio;
  if (!a.oscBank.length || !a.ctx) return;
  const now = a.ctx.currentTime;
  const old = a.oscBank;
  a.oscBank = []; // detach immediately so new bank can be created
  // Fade out all voices
  for (const voice of old) {
    voice.gain.gain.cancelScheduledValues(now);
    voice.gain.gain.setTargetAtTime(0, now, durationSec * 0.25);
  }
  // Clean up after fade completes
  setTimeout(() => {
    for (const voice of old) {
      try { voice.oscA.stop(); } catch (_) {}
      try { voice.oscB.stop(); } catch (_) {}
      voice.oscA.disconnect();
      voice.oscB.disconnect();
      voice.panA.disconnect();
      voice.panB.disconnect();
      voice.gain.disconnect();
      voice.panner.disconnect();
    }
  }, durationSec * 1000 + 100);
}

export function initOscBank(rows) {
  fadeOutOscBank();
  const a = state.audio;
  if (!a.ctx || !a.synthBus) return;

  const UNISON_DETUNE = 6; // cents — classic supersaw-style detuning

  a.oscBank = [];
  for (let r = 0; r < rows; r++) {
    const freq = pitchTable[r] || 440;

    // Voice A: detuned slightly sharp, panned slightly left
    const oscA = a.ctx.createOscillator();
    if (a.waveAlive) oscA.setPeriodicWave(a.waveAlive);
    else oscA.type = 'triangle';
    oscA.frequency.value = freq;
    oscA.detune.value = UNISON_DETUNE;

    // Voice B: detuned slightly flat, panned slightly right
    const oscB = a.ctx.createOscillator();
    if (a.waveAlive) oscB.setPeriodicWave(a.waveAlive);
    else oscB.type = 'triangle';
    oscB.frequency.value = freq;
    oscB.detune.value = -UNISON_DETUNE;

    // Shared gain for this row
    const gain = a.ctx.createGain();
    gain.gain.value = 0;

    // Stereo panner for spatial placement
    const panner = a.ctx.createStereoPanner();
    panner.pan.value = 0;

    // Unison spread: slight stereo offset between voices
    const panA = a.ctx.createStereoPanner();
    panA.pan.value = -0.15;
    const panB = a.ctx.createStereoPanner();
    panB.pan.value = 0.15;

    oscA.connect(panA);
    oscB.connect(panB);
    panA.connect(gain);
    panB.connect(gain);
    gain.connect(panner);
    panner.connect(a.synthBus);

    oscA.start();
    oscB.start();

    a.oscBank.push({ oscA, oscB, gain, panner, panA, panB });
  }
}

function destroyOscBank() {
  const a = state.audio;
  for (const voice of a.oscBank) {
    try { voice.oscA.stop(); } catch (_) {}
    try { voice.oscB.stop(); } catch (_) {}
    voice.oscA.disconnect();
    voice.oscB.disconnect();
    voice.panA.disconnect();
    voice.panB.disconnect();
    voice.gain.disconnect();
    voice.panner.disconnect();
  }
  a.oscBank = [];
}

// ── Per-tick grid sonification ──

export function updateFromGrid(grid, cols, rows) {
  const a = state.audio;
  if (!a.enabled || !a.ctx || !a.synthBus) return;

  const now = a.ctx.currentTime;
  const tickMS = state.gol.tickMS || 250;
  const RAMP = Math.max(0.02, (tickMS / 1000) * 0.4);

  if (a.oscBank.length !== rows) initOscBank(rows);

  const prev = a.prevGrid;
  const hasPrev = prev && prev.length === grid.length;

  // Pre-compute per-row stats
  const rowChanges = new Uint16Array(rows);
  let totalChanged = 0;
  let aliveTotal = 0;
  const totalCells = cols * rows;
  for (let i = 0; i < totalCells; i++) {
    if (grid[i]) aliveTotal++;
    if (hasPrev && grid[i] !== prev[i]) {
      rowChanges[Math.floor(i / cols)]++;
      totalChanged++;
    }
  }
  const density    = totalCells > 0 ? aliveTotal / totalCells : 0;
  const changeRate = totalCells > 0 ? totalChanged / totalCells : 0;

  // Advance scan position
  const scanCol = state.gol.scanCol;
  state.gol.scanCol = (scanCol + 1) % cols;
  const SCAN_WIDTH = Math.max(1, Math.floor(cols * 0.06));
  const tickSec = tickMS / 1000;

  // (A) Per-row: amplitude, panning, scan accent, detune
  for (let r = 0; r < rows; r++) {
    let alive = 0;
    let xSum = 0;
    for (let c = 0; c < cols; c++) {
      if (grid[r * cols + c]) { alive++; xSum += c; }
    }
    const ratio = alive / cols;
    const baseAmp = alive > 0 ? ratio * 0.3 : 0;

    // Scan accent
    let scanBoost = 0;
    for (let d = -SCAN_WIDTH; d <= SCAN_WIDTH; d++) {
      const sc = (scanCol + d + cols) % cols;
      if (grid[r * cols + sc]) {
        const proximity = 1 - Math.abs(d) / (SCAN_WIDTH + 1);
        scanBoost = Math.max(scanBoost, proximity * 0.35);
      }
    }

    const amp = baseAmp + scanBoost;
    const rowPan = alive > 0 ? ((xSum / alive) / Math.max(1, cols - 1)) * 2 - 1 : 0;
    const scanPan = (scanCol / Math.max(1, cols - 1)) * 2 - 1;
    const pan = scanBoost > 0 ? rowPan * 0.4 + scanPan * 0.6 : rowPan;

    const { oscA, oscB, gain, panner } = a.oscBank[r];

    // Rhythmic scan punch vs smooth crossfade
    if (scanBoost > 0.1) {
      gain.gain.cancelScheduledValues(now);
      gain.gain.setValueAtTime(gain.gain.value, now);
      gain.gain.linearRampToValueAtTime(amp, now + 0.008);
      gain.gain.setTargetAtTime(baseAmp, now + 0.008, tickSec * 0.5);
    } else {
      gain.gain.setTargetAtTime(amp, now, RAMP);
    }
    panner.pan.setTargetAtTime(pan, now, RAMP * 0.5);

    // Per-row detune modulation: change rate drives unison spread (6–14 cents)
    const rowActivity = rowChanges[r] / cols;
    const dynamicDetune = 6 + rowActivity * 8;
    oscA.detune.setTargetAtTime(dynamicDetune, now, RAMP);
    oscB.detune.setTargetAtTime(-dynamicDetune, now, RAMP);

    // Frequency micro-wobble from activity
    const baseFreq = pitchTable[r] || 440;
    const wobble = 1 + (rowActivity * 0.015 * (((r * 7 + state.gol.generation) % 3) - 1));
    oscA.frequency.setTargetAtTime(baseFreq * wobble, now, RAMP);
    oscB.frequency.setTargetAtTime(baseFreq * wobble, now, RAMP);
  }

  // (B-pre) Scan column accent grains
  {
    const rp = a.roundParams || {};
    const scanAttack = rp.attack || 0.005;
    const scanDecay  = tickSec * 0.3;
    const scanPan = (scanCol / Math.max(1, cols - 1)) * 2 - 1;
    let scanGrains = 0;
    for (let r = 0; r < rows && scanGrains < 6; r++) {
      if (grid[r * cols + scanCol]) {
        spawnGrain(pitchTable[r] || 440, 0.10, scanPan, scanAttack, scanDecay, now);
        scanGrains++;
      }
    }
  }

  // (B) Birth/death transients
  if (hasPrev) {
    const rp = a.roundParams || {};
    const attack = rp.attack || 0.005;
    const decay  = rp.decay  || 0.027;
    const MAX_GRAINS = 18;
    let grainCount = 0;

    const rowOrder = Array.from({ length: rows }, (_, i) => i);
    rowOrder.sort((rA, rB) => rowChanges[rB] - rowChanges[rA]);

    for (const r of rowOrder) {
      if (grainCount >= MAX_GRAINS || rowChanges[r] === 0) break;
      let foundBirth = false, foundDeath = false;
      const startC = Math.floor(Math.random() * cols);
      for (let i = 0; i < cols && grainCount < MAX_GRAINS; i++) {
        const c = (startC + i) % cols;
        const idx = r * cols + c;
        if (prev[idx] === grid[idx]) continue;
        const freq = pitchTable[r] || 440;
        const pan  = cols > 1 ? (c / (cols - 1)) * 2 - 1 : 0;
        if (grid[idx] && !prev[idx] && !foundBirth) {
          // Birth: octave up, bright
          spawnGrain(freq * 2, 0.12, pan, attack, decay, now);
          foundBirth = true;
          grainCount++;
        } else if (!grid[idx] && prev[idx] && !foundDeath) {
          // Death: octave down, soft thud
          spawnGrain(freq * 0.5, 0.08, pan, attack * 2, decay * 2, now);
          foundDeath = true;
          grainCount++;
        }
        if (foundBirth && foundDeath) break;
      }
    }
  }

  // (C) Macro modulation
  const filterBase = (a.roundParams && a.roundParams.filterBase) || 3000;
  const filterTarget = filterBase * (0.2 + density * 1.2 + changeRate * 1.6);
  a.filter.frequency.setTargetAtTime(
    Math.min(filterBase * 3, Math.max(200, filterTarget)), now, 0.15
  );

  // Drone: inverse relationship with density/change
  const droneAmp = Math.min(0.8, 0.05 + (1 - density) * 0.5 + (1 - changeRate) * 0.35);
  a.droneOsc.forEach(({ gain }) => {
    gain.gain.setTargetAtTime(droneAmp, now, 0.2);
  });

  // Sub bass: follows density (more cells = more weight), capped for clean low end
  if (a._sub) {
    const subAmp = Math.min(0.5, density * 0.6);
    a._sub.gain.gain.setTargetAtTime(subAmp, now, 0.3);
  }

  // Reverb wet/dry
  if (a.wetGain && a.dryGain) {
    const wet = 0.15 + changeRate * 0.7;
    a.wetGain.gain.setTargetAtTime(Math.min(0.85, wet), now, 0.15);
    a.dryGain.gain.setTargetAtTime(Math.max(0.15, 1 - wet), now, 0.15);
  }

  // Chorus depth: more depth during change for wider sound
  if (a._chorus) {
    const depth = 0.001 + changeRate * 0.003;
    a._chorus.chorusLfoDepthL.gain.setTargetAtTime(depth, now, 0.2);
    a._chorus.chorusLfoDepthR.gain.setTargetAtTime(depth, now, 0.2);
    // Wet level: subtle when stable, wider during evolution
    const chorusWet = 0.15 + changeRate * 0.25;
    a._chorus.chorusWetL.gain.setTargetAtTime(chorusWet, now, 0.2);
    a._chorus.chorusWetR.gain.setTargetAtTime(chorusWet, now, 0.2);
    a._chorus.chorusDry.gain.setTargetAtTime(1 - chorusWet, now, 0.2);
  }

  // Grain + synth bus levels
  if (a.grainBus) a.grainBus.gain.setTargetAtTime(0.2 + changeRate * 0.6, now, 0.1);
  if (a.synthBus) a.synthBus.gain.setTargetAtTime(0.5 - density * 0.2, now, 0.2);

  // (D) Store grid copy
  a.prevGrid = grid.slice();
}

// ── Grain synthesis ──

function spawnGrain(freq, amp, pan, attack, decay, now) {
  const a = state.audio;
  if (!a.grainBus) return;

  const osc = a.ctx.createOscillator();
  if (a.waveAlive) osc.setPeriodicWave(a.waveAlive);
  else osc.type = 'triangle';
  osc.frequency.value = freq;

  const gain = a.ctx.createGain();
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(amp, now + attack);
  gain.gain.setTargetAtTime(0, now + attack, decay);

  const panner = a.ctx.createStereoPanner();
  panner.pan.value = pan;

  osc.connect(gain);
  gain.connect(panner);
  panner.connect(a.grainBus);

  osc.start(now);
  osc.stop(now + attack + decay * 5);
}

// ── Tile → wavetable ──

function tileToPeriodicWave(ctx, piece, tiltExponent = 0.3) {
  if (!piece || !piece.length || !piece[0]) return null;
  const tRows = piece.length;
  const tCols = piece[0].length;
  const N = tCols;

  const avg = new Float32Array(N);
  for (let c = 0; c < N; c++) {
    let sum = 0;
    for (let r = 0; r < tRows; r++) sum += piece[r][c];
    avg[c] = sum / tRows;
  }

  let mean = 0;
  for (let i = 0; i < N; i++) mean += avg[i];
  mean /= N;
  for (let i = 0; i < N; i++) avg[i] -= mean;

  let maxAbs = 0;
  for (let i = 0; i < N; i++) if (Math.abs(avg[i]) > maxAbs) maxAbs = Math.abs(avg[i]);
  if (maxAbs === 0) return null;
  for (let i = 0; i < N; i++) avg[i] /= maxAbs;

  const rowAvg = new Float32Array(tRows);
  for (let r = 0; r < tRows; r++) {
    let sum = 0;
    for (let c = 0; c < tCols; c++) sum += piece[r][c];
    rowAvg[r] = sum / tCols;
  }

  const Nc = tCols, Nr = tRows;
  const Ntotal = Nc + Nr;
  const signal = new Float32Array(Ntotal);
  for (let i = 0; i < Nc; i++) signal[i] = avg[i];
  let rMean = 0;
  for (let i = 0; i < Nr; i++) rMean += rowAvg[i];
  rMean /= Nr;
  let rMax = 0;
  for (let i = 0; i < Nr; i++) { rowAvg[i] -= rMean; if (Math.abs(rowAvg[i]) > rMax) rMax = Math.abs(rowAvg[i]); }
  if (rMax > 0) for (let i = 0; i < Nr; i++) signal[Nc + i] = rowAvg[i] / rMax;

  const real = new Float32Array(Ntotal + 1);
  const imag = new Float32Array(Ntotal + 1);
  for (let k = 1; k <= Ntotal; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < Ntotal; n++) {
      const angle = (2 * Math.PI * k * n) / Ntotal;
      re += signal[n] * Math.cos(angle);
      im -= signal[n] * Math.sin(angle);
    }
    const tilt = 1 / Math.pow(k, tiltExponent);
    real[k] = (re / Ntotal) * tilt;
    imag[k] = (im / Ntotal) * tilt;
  }

  return ctx.createPeriodicWave(real, imag, { disableNormalization: true });
}

function tileStats(piece) {
  if (!piece || !piece.length || !piece[0]) return { mean: 0, std: 0 };
  let sum = 0, count = 0;
  for (const row of piece) for (const v of row) { sum += v; count++; }
  const mean = count ? sum / count : 0;
  let variance = 0;
  for (const row of piece) for (const v of row) variance += (v - mean) ** 2;
  const std = count ? Math.sqrt(variance / count) : 0;
  return { mean, std };
}

// ── Per-round wavetable + parameter setup ──

export function buildWavetables(tileA, tileB, tileIdxA = 0, tileIdxB = 0) {
  const a = state.audio;
  if (!a.ctx) return;

  const statsA = tileStats(tileA);
  const statsB = tileStats(tileB);
  const energy   = Math.min(1, statsA.mean / 7);
  const contrast = Math.min(1, statsB.std / 3.5);

  const scaleCount = SCALES.length;
  // Smooth scale movement: step ±1 from current scale instead of jumping
  let scaleIdx;
  if (a.roundParams && a.roundParams.scaleIdx != null) {
    const prev = a.roundParams.scaleIdx;
    const roll = Math.random();
    if (roll < 0.4) scaleIdx = prev; // 40% stay
    else if (roll < 0.7) scaleIdx = (prev + 1) % scaleCount; // 30% step up
    else scaleIdx = (prev - 1 + scaleCount) % scaleCount; // 30% step down
  } else {
    scaleIdx = ((tileIdxA * 7 + tileIdxB * 13) % scaleCount + scaleCount) % scaleCount;
  }
  const octaveOffset = energy > 0.6 ? 1 : energy < 0.3 ? -1 : 0;
  const tiltExp   = 0.15 + contrast * 0.40;
  const attack    = 0.003 + energy * 0.017;
  const decay     = 0.015 + contrast * 0.045;
  const filterBase = 2000 + energy * 3000;

  a.roundParams = { attack, decay, tiltExp, filterBase, scaleIdx };

  buildScaleFreqs(scaleIdx, octaveOffset);
  buildPitchTable(state.gol && state.gol.rows ? state.gol.rows : (a.pitchTableRows || 16));

  if (a.filter) {
    const now = a.ctx.currentTime;
    a.filter.frequency.cancelScheduledValues(now);
    a.filter.frequency.setTargetAtTime(filterBase, now, 0.8);
  }

  // Drone: first two scale notes
  const droneFreqs = currentScale.slice(0, 2);
  if (a.droneOsc) {
    a.droneOsc.forEach(({ osc }, i) => {
      if (droneFreqs[i]) osc.frequency.setTargetAtTime(droneFreqs[i], a.ctx.currentTime, 1.2);
    });
  }

  // Sub bass: one octave below root
  if (a._sub && droneFreqs[0]) {
    a._sub.osc.frequency.setTargetAtTime(droneFreqs[0] * 0.5, a.ctx.currentTime, 1.2);
  }

  a.waveAlive = tileToPeriodicWave(a.ctx, tileA, tiltExp);
  a.waveDrone = tileToPeriodicWave(a.ctx, tileB, tiltExp);

  // Crossfade drone wavetable: dip gain, swap wave, restore
  if (a.droneOsc) {
    const now = a.ctx.currentTime;
    a.droneOsc.forEach(({ osc, gain }) => {
      const prevAmp = gain.gain.value;
      gain.gain.cancelScheduledValues(now);
      gain.gain.setTargetAtTime(0, now, 0.25);
      setTimeout(() => {
        if (a.waveDrone) osc.setPeriodicWave(a.waveDrone);
        else osc.type = 'sine';
        gain.gain.cancelScheduledValues(a.ctx.currentTime);
        gain.gain.setTargetAtTime(prevAmp, a.ctx.currentTime, 0.6);
      }, 500);
    });
  }

  const rows = state.gol && state.gol.rows ? state.gol.rows : 0;
  if (rows > 0) initOscBank(rows);
}
