import { N_LAYERS } from './constants.js';
import { state } from './state.js';
import { setPhase } from './phase.js';
import { drawGrid } from './draw/grid.js';
import { drawHeatmap } from './draw/heatmap.js';
import { drawEmbedSim } from './draw/embed.js';
import { drawConfidence } from './draw/confidence.js';
import { updateSparkline } from './draw/sparkline.js';
import { renderMural } from './mural/render.js';
import { clearTileCache } from './mural/cache.js';
import { startKaleidoAnim, stopLayoutLoop, stopRotationLoop } from './mural/timers.js';

let layerCanvases = [];
let galleryCanvases = [];

export function initHandlers(refs) {
  layerCanvases = refs.layerCanvases;
  galleryCanvases = refs.galleryCanvases;
}

function isMuralActive() {
  return document.getElementById('page-mural').classList.contains('active');
}

export function handle(event, data) {
  switch (event) {

  case 'train_start': setPhase('training'); break;

  case 'train_step': {
    state.lossHistory.push(data.loss);
    state.gradHistory.push(data.grad_norm || 0);
    if (state.lossHistory.length > 500) state.lossHistory = state.lossHistory.slice(-300);
    if (state.gradHistory.length > 500) state.gradHistory = state.gradHistory.slice(-300);
    document.getElementById('stat-loss').textContent = data.loss.toFixed(4);
    updateSparkline('loss-chart', state.lossHistory, 200);
    updateSparkline('grad-chart', state.gradHistory, 200);
    break;
  }

  case 'neural_activity': {
    const { layer_maps, embedding_sim, weight_norms } = data;
    for (let li = 0; li < Math.min(layer_maps.length, N_LAYERS); li++)
      drawHeatmap(layerCanvases[li], layer_maps[li]);
    if (weight_norms) {
      let mx = 0.001;
      for (let i = 0; i < weight_norms.length; i++)
        if (weight_norms[i] > mx) mx = weight_norms[i];
      for (let i = 0; i < weight_norms.length; i++) {
        const el = document.getElementById(`energy-${i}`);
        if (el) el.textContent = (weight_norms[i] / mx * 100).toFixed(0) + '%';
      }
    }
    if (embedding_sim) drawEmbedSim(embedding_sim);
    break;
  }

  case 'gen_start': {
    setPhase('generating');
    document.getElementById('stat-gen').textContent = data.generation;
    document.getElementById('stat-temp').textContent = data.temperature.toFixed(3);
    document.getElementById('meta-gen').textContent = data.generation;
    document.getElementById('meta-temp').textContent = data.temperature.toFixed(3);
    document.getElementById('mural-gen-info').textContent = `gen ${data.generation}  \u00b7  temp ${data.temperature.toFixed(3)}`;
    galleryCanvases.forEach(c => { c.getContext('2d').clearRect(0, 0, 16, 16); c.classList.remove('selected'); });
    break;
  }

  case 'gen_progress': {
    const { grids } = data;
    for (let i = 0; i < Math.min(grids.length, 36); i++) if (grids[i]) drawGrid(galleryCanvases[i], grids[i]);
    break;
  }

  case 'gen_pieces': {
    state.allPieces = data.pieces;
    clearTileCache();
    for (let i = 0; i < Math.min(data.pieces.length, 36); i++) drawGrid(galleryCanvases[i], data.pieces[i]);
    if (isMuralActive()) renderMural();
    break;
  }

  case 'gen_confidences': {
    if (data.confidences && data.confidences[0])
      drawConfidence(document.getElementById('confidence-overlay'), data.confidences[0]);
    break;
  }

  case 'scoring_start': setPhase('scoring'); break;

  case 'gen_scored': {
    const { pieces, scores } = data;
    state.allPieces = pieces;
    clearTileCache();
    let bestIdx = 0, bestVal = -1;
    for (let i = 0; i < scores.length; i++) {
      const s = scores[i].composite || 0;
      if (s > bestVal) { bestVal = s; bestIdx = i; }
    }
    if (pieces[bestIdx]) {
      drawGrid(document.getElementById('best-piece'), pieces[bestIdx]);
      document.getElementById('meta-score').textContent = bestVal.toFixed(3);
      const w = document.getElementById('best-wrap');
      w.classList.remove('flash');
      requestAnimationFrame(() => requestAnimationFrame(() => w.classList.add('flash')));
    }
    if (isMuralActive()) renderMural();
    break;
  }

  case 'gen_selected': {
    galleryCanvases.forEach(c => c.classList.remove('selected'));
    const { indices } = data;
    if (indices) {
      indices.forEach(i => { if (i < galleryCanvases.length) galleryCanvases[i].classList.add('selected'); });
      state.selectedPieces = indices.filter(i => i < state.allPieces.length).map(i => state.allPieces[i]);
    }
    break;
  }

  case 'finetune_start': {
    setPhase('finetuning');
    // Only auto-switch to kaleidoscope if in wallpaper mode
    if (state.muralMode === 'wallpaper' && state.selectedPieces.length) {
      stopLayoutLoop();
      stopRotationLoop();
      state.muralMode = 'kaleidoscope';
      const btn = document.getElementById('mural-mode-btn');
      btn.textContent = 'Kaleidoscope'; btn.classList.add('kaleido');
      startKaleidoAnim();
    }
    break;
  }

  case 'train_end': break;
  case 'saving_complete': break;

  case 'gen_complete': {
    const s = data.summary;
    if (s) {
      document.getElementById('stat-gen').textContent = s.generation;
      document.getElementById('stat-temp').textContent = s.temperature.toFixed(3);
    }
    break;
  }

  case 'init_phase':
    document.getElementById('phase-text').textContent = data.phase || 'initializing';
    document.getElementById('stat-phase').textContent = data.phase || 'init';
    break;
  }
}
