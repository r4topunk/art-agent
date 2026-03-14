import { HEAT } from '../constants.js';

export function heatPixel(v) {
  const t = Math.min(1, Math.max(0, v)) * 6.99;
  const lo = Math.floor(t), hi = Math.min(7, lo + 1), f = t - lo;
  const a = HEAT[lo], b = HEAT[hi];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export function drawHeatmap(canvas, map) {
  if (!map || !map.length) return;
  const ctx = canvas.getContext('2d');
  const R = map.length, C = map[0].length;
  canvas.width = C; canvas.height = R;
  let mx = 0.001;
  for (let r = 0; r < R; r++)
    for (let c = 0; c < C; c++)
      if (map[r][c] > mx) mx = map[r][c];
  const img = ctx.createImageData(C, R);
  const d = img.data;
  for (let r = 0; r < R; r++)
    for (let c = 0; c < C; c++) {
      const px = heatPixel(map[r][c] / mx), i = (r * C + c) * 4;
      d[i] = px[0]; d[i+1] = px[1]; d[i+2] = px[2]; d[i+3] = 255;
    }
  ctx.putImageData(img, 0, 0);
}
