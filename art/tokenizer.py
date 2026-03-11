from pathlib import Path
import PIL.Image
from art.config import ArtConfig


class PixelTokenizer:
    def __init__(self, config: ArtConfig):
        self.config = config

    def encode(self, image: PIL.Image.Image) -> list[int]:
        img_l = image.convert("L")
        pixels = list(img_l.tobytes())

        tokens = [self.config.BOS]
        for pixel in pixels:
            token = self.config.WHITE if pixel >= 128 else self.config.BLACK
            tokens.append(token)
        tokens.append(self.config.EOS)

        return tokens

    def decode(self, tokens: list[int]) -> PIL.Image.Image:
        pixels = []
        for token in tokens:
            if token == self.config.BOS or token == self.config.EOS or token == self.config.PAD:
                continue
            pixel = 255 if token == self.config.WHITE else 0
            pixels.append(pixel)

        expected = self.config.grid_size * self.config.grid_size
        pixels = pixels[:expected]
        if len(pixels) < expected:
            pixels.extend([0] * (expected - len(pixels)))

        img = PIL.Image.new("L", (self.config.grid_size, self.config.grid_size))
        img.putdata(pixels)
        return img

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def seq_length(self) -> int:
        return self.config.seq_length
