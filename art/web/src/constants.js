export const PALETTE = [
  [0,0,0],[255,241,232],[255,0,77],[255,163,0],
  [255,236,39],[0,228,54],[41,173,255],[255,119,168],
];

export const HEAT = [
  [10,10,46],[26,26,110],[44,95,170],[0,168,158],
  [78,196,78],[200,200,0],[224,102,0],[255,34,34],
];

export const N_LAYERS = 6;

export const ZOOM_STEPS = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64];

// Pre-compute PALETTE as flat Uint8 for fast tile rendering
export const PAL_FLAT = new Uint8Array(PALETTE.length * 4);
for (let i = 0; i < PALETTE.length; i++) {
  PAL_FLAT[i * 4]     = PALETTE[i][0];
  PAL_FLAT[i * 4 + 1] = PALETTE[i][1];
  PAL_FLAT[i * 4 + 2] = PALETTE[i][2];
  PAL_FLAT[i * 4 + 3] = 255;
}
