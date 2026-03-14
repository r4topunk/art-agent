import { PALETTE } from '../constants.js';

export function drawGrid(canvas, grid) {
  if (!grid || !grid.length) return;
  const ctx = canvas.getContext('2d');
  const R = grid.length, C = grid[0].length;
  canvas.width = C; canvas.height = R;
  const img = ctx.createImageData(C, R);
  const d = img.data;
  for (let r = 0; r < R; r++)
    for (let c = 0; c < C; c++) {
      const i = (r * C + c) * 4, p = PALETTE[grid[r][c]] || PALETTE[0];
      d[i] = p[0]; d[i+1] = p[1]; d[i+2] = p[2]; d[i+3] = 255;
    }
  ctx.putImageData(img, 0, 0);
}
