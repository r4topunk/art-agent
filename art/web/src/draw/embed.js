import { heatPixel } from './heatmap.js';

export function drawEmbedSim(sim) {
  const canvas = document.getElementById('embed-canvas');
  const ctx = canvas.getContext('2d');
  const n = sim.length;
  canvas.width = n; canvas.height = n;
  const img = ctx.createImageData(n, n);
  const d = img.data;
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      const px = heatPixel((sim[r][c] + 1) / 2), i = (r * n + c) * 4;
      d[i] = px[0]; d[i+1] = px[1]; d[i+2] = px[2]; d[i+3] = 255;
    }
  ctx.putImageData(img, 0, 0);
}
