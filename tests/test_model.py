import torch
import pytest

from art.config import ArtConfig
from art.model import PixelGPT


@pytest.fixture
def config():
    return ArtConfig()


@pytest.fixture
def model(config):
    m = PixelGPT(config)
    m.eval()
    return m


def test_param_count(model):
    n_params = model.count_parameters()
    assert 1_000_000 <= n_params <= 20_000_000, (
        f"Expected parameter count between 1M and 20M, got {n_params}"
    )


def test_forward_shape(model, config):
    batch_size = 2
    seq_len = config.seq_length
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_generate_shape(model, config):
    batch_size = 3
    output = model.generate(batch_size=batch_size, device="cpu")
    assert output.shape == (batch_size, config.seq_length)


def test_generate_starts_with_bos(model, config):
    batch_size = 4
    output = model.generate(batch_size=batch_size, device="cpu")
    for i in range(batch_size):
        assert output[i, 0].item() == config.BOS, (
            f"Sequence {i} does not start with BOS token"
        )


def test_weight_tying(model):
    # The model ties output head weights to token embedding weights
    assert model.head.weight is model.tok_emb.weight, (
        "output head weight should be the same object as token embedding weight"
    )
