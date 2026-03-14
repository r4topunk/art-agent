import { heatPixel } from './heatmap.js';

export function drawConfidence(canvas, flatConf) {
  if (!flatConf || !flatConf.length) return;
  const ctx = canvas.getContext('2d');
  canvas.width = 16; canvas.height = 16;
  const img = ctx.createImageData(16, 16);
  const d = img.data;
  for (let i = 0; i < 256 && i < flatConf.length; i++) {
    const px = heatPixel(Math.min(1, flatConf[i])), idx = i * 4;
    d[idx] = px[0]; d[idx+1] = px[1]; d[idx+2] = px[2]; d[idx+3] = 180;
  }
  ctx.putImageData(img, 0, 0);
}
