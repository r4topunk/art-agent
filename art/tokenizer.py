from pathlib import Path
import numpy as np
import PIL.Image
from art.config import ArtConfig, PALETTE_16


class PixelTokenizer:
    def __init__(self, config: ArtConfig):
        self.config = config
        # Build reverse lookup: RGB -> color index (nearest)
        self._palette_arr = np.array(PALETTE_16, dtype=np.float32)

    def encode(self, image: PIL.Image.Image) -> list[int]:
        """Encode an RGB image into a token sequence using nearest palette color."""
        img_rgb = image.convert("RGB").resize(
            (self.config.grid_size, self.config.grid_size),
            PIL.Image.NEAREST,
        )
        pixels = np.array(img_rgb, dtype=np.float32).reshape(-1, 3)  # (256, 3)

        # Find nearest palette color for each pixel
        # Broadcast: (256, 1, 3) - (1, 16, 3) -> (256, 16)
        dists = np.sum((pixels[:, None, :] - self._palette_arr[None, :, :]) ** 2, axis=2)
        color_indices = np.argmin(dists, axis=1).tolist()

        tokens = [self.config.BOS] + color_indices + [self.config.EOS]
        return tokens

    def encode_grid(self, grid: np.ndarray) -> list[int]:
        """Encode a grid of color indices (0..n_colors-1) into a token sequence."""
        pixels = grid.flatten().tolist()
        # Clamp to valid color range
        pixels = [max(0, min(self.config.n_colors - 1, p)) for p in pixels]
        tokens = [self.config.BOS] + pixels + [self.config.EOS]
        return tokens

    def decode(self, tokens: list[int]) -> PIL.Image.Image:
        """Decode token sequence into an RGB PIL image."""
        pixels = []
        for token in tokens:
            if token in (self.config.BOS, self.config.EOS, self.config.PAD):
                continue
            # Clamp to valid color range
            color_idx = max(0, min(self.config.n_colors - 1, token))
            pixels.append(color_idx)

        expected = self.config.grid_size * self.config.grid_size
        pixels = pixels[:expected]
        if len(pixels) < expected:
            pixels.extend([0] * (expected - len(pixels)))

        # Convert color indices to RGB
        rgb_pixels = [PALETTE_16[c] for c in pixels]
        img = PIL.Image.new("RGB", (self.config.grid_size, self.config.grid_size))
        img.putdata(rgb_pixels)
        return img

    def decode_to_grid(self, tokens: list[int]) -> np.ndarray:
        """Decode token sequence into a grid of color indices (uint8)."""
        pixels = []
        for token in tokens:
            if token in (self.config.BOS, self.config.EOS, self.config.PAD):
                continue
            color_idx = max(0, min(self.config.n_colors - 1, token))
            pixels.append(color_idx)

        expected = self.config.grid_size * self.config.grid_size
        pixels = pixels[:expected]
        if len(pixels) < expected:
            pixels.extend([0] * (expected - len(pixels)))

        return np.array(pixels, dtype=np.uint8).reshape(
            self.config.grid_size, self.config.grid_size
        )

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def seq_length(self) -> int:
        return self.config.seq_length
