import random

import PIL.Image
import pytest

from art.config import ArtConfig
from art.tokenizer import PixelTokenizer


@pytest.fixture
def config():
    return ArtConfig()


@pytest.fixture
def tokenizer(config):
    return PixelTokenizer(config)


@pytest.fixture
def random_image(config):
    rng = random.Random(42)
    size = config.grid_size
    img = PIL.Image.new("L", (size, size))
    pixels = [rng.choice([0, 255]) for _ in range(size * size)]
    img.putdata(pixels)
    return img


def test_encode_decode_roundtrip(tokenizer, random_image):
    tokens = tokenizer.encode(random_image)
    decoded = tokenizer.decode(tokens)

    original_pixels = list(random_image.convert("L").getdata())
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
    # Build a valid pixel sequence and scatter PAD tokens in
    pure_pixels = [config.BLACK, config.WHITE] * 128  # 256 pixel tokens
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
    pixels = list(img.getdata())
    assert all(p in (0, 255) for p in pixels)


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 5


def test_seq_length(tokenizer):
    assert tokenizer.seq_length == 258
