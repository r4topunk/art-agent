import random

import numpy as np
import PIL.Image
import pytest

from art.config import ArtConfig, PALETTE_16
from art.tokenizer import PixelTokenizer


@pytest.fixture
def config():
    return ArtConfig()


@pytest.fixture
def tokenizer(config):
    return PixelTokenizer(config)


@pytest.fixture
def random_image(config):
    """Create a random RGB image using palette colors."""
    rng = random.Random(42)
    size = config.grid_size
    img = PIL.Image.new("RGB", (size, size))
    pixels = [PALETTE_16[rng.randint(0, len(PALETTE_16) - 1)] for _ in range(size * size)]
    img.putdata(pixels)
    return img


def test_encode_decode_roundtrip(tokenizer, random_image):
    tokens = tokenizer.encode(random_image)
    decoded = tokenizer.decode(tokens)

    original_pixels = list(random_image.getdata())
    decoded_pixels = list(decoded.getdata())

    assert original_pixels == decoded_pixels


def test_encode_length(tokenizer, random_image, config):
    tokens = tokenizer.encode(random_image)
    assert len(tokens) == 258  # BOS + 256 pixels + EOS


def test_encode_starts_with_bos(tokenizer, random_image, config):
    tokens = tokenizer.encode(random_image)
    assert tokens[0] == config.BOS


def test_encode_ends_with_eos(tokenizer, random_image, config):
    tokens = tokenizer.encode(random_image)
    assert tokens[-1] == config.EOS


def test_decode_ignores_special_tokens(tokenizer, config):
    # Build a valid pixel sequence with mixed color tokens and PADs scattered in
    pure_pixels = list(range(config.n_colors)) * 16  # 256 pixel tokens
    tokens_with_specials = (
        [config.BOS]
        + [config.PAD]
        + pure_pixels[:128]
        + [config.PAD, config.PAD]
        + pure_pixels[128:]
        + [config.PAD]
        + [config.EOS]
    )
    img = tokenizer.decode(tokens_with_specials)
    assert isinstance(img, PIL.Image.Image)
    assert img.size == (config.grid_size, config.grid_size)
    assert img.mode == "RGB"


def test_decode_to_grid(tokenizer, config):
    tokens = [config.BOS] + [5] * 256 + [config.EOS]
    grid = tokenizer.decode_to_grid(tokens)
    assert grid.shape == (16, 16)
    assert np.all(grid == 5)


def test_encode_grid(tokenizer, config):
    grid = np.full((16, 16), 5, dtype=np.uint8)
    tokens = tokenizer.encode_grid(grid)
    assert tokens[0] == config.BOS
    assert tokens[-1] == config.EOS
    assert all(t == 5 for t in tokens[1:-1])


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 11  # 8 colors + BOS + EOS + PAD


def test_seq_length(tokenizer):
    assert tokenizer.seq_length == 258
