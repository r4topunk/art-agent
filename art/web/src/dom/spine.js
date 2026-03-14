import { N_LAYERS } from '../constants.js';

const layerNames = ['input', 'L1', 'L2', 'L3', 'L4', 'output'];

export function buildSpine() {
  const spineScroll = document.getElementById('spine-scroll');
  const layerCanvases = [];

  for (let i = 0; i < N_LAYERS; i++) {
    const wrap = document.createElement('div');
    wrap.className = 'layer-wrap';
    const hm = document.createElement('div');
    hm.className = 'layer-heatmap';
    const canvas = document.createElement('canvas');
    canvas.width = 16; canvas.height = 16;
    hm.appendChild(canvas);
    layerCanvases.push(canvas);
    const meta = document.createElement('div');
    meta.className = 'layer-meta';
    meta.innerHTML = `<span class="layer-name">${layerNames[i]}</span><span id="energy-${i}" style="color:rgba(57,255,20,0.35)"></span>`;
    wrap.appendChild(hm);
    wrap.appendChild(meta);
    spineScroll.appendChild(wrap);
    if (i < N_LAYERS - 1) {
      const conn = document.createElement('div');
      conn.className = 'layer-connector';
      for (let p = 0; p < 3; p++) {
        const dot = document.createElement('div');
        dot.className = 'particle-track';
        dot.style.left = (18 + Math.random() * 64) + '%';
        dot.style.animationDelay = (Math.random() * 2) + 's';
        dot.style.animationDuration = (1.1 + Math.random() * 1.1) + 's';
        conn.appendChild(dot);
      }
      spineScroll.appendChild(conn);
    }
  }

  return layerCanvases;
}
