import numpy as np
import pytest

from art.config import ArtConfig
from art.critic import (
    ArtCritic,
    aesthetics_score,
    complexity_score,
    structure_score,
    per_image_diversity,
    symmetry_score,
)


@pytest.fixture
def config():
    return ArtConfig()


@pytest.fixture
def critic(config):
    return ArtCritic(config)


# ---------------------------------------------------------------------------
# symmetry_score
# ---------------------------------------------------------------------------

def test_symmetry_perfect():
    # 4-fold symmetric pattern with colors
    grid = np.zeros((16, 16), dtype=np.uint8)
    for i in range(0, 8, 2):
        grid[i:16 - i, i:16 - i] = (i // 2 + 1)
        if i + 1 < 8:
            grid[i + 1:16 - i - 1, i + 1:16 - i - 1] = 0
    score = symmetry_score(grid)
    assert score > 0.9, f"Expected symmetry > 0.9, got {score}"


def test_symmetry_random():
    rng = np.random.RandomState(123)
    scores = [symmetry_score(rng.randint(0, 8, (16, 16)).astype(np.uint8)) for _ in range(20)]
    avg = float(np.mean(scores))
    assert avg < 0.3, f"Random 8-color grids should have low symmetry, got avg={avg}"


# ---------------------------------------------------------------------------
# complexity_score
# ---------------------------------------------------------------------------

def test_complexity_uniform():
    all_zero = np.zeros((16, 16), dtype=np.uint8)
    all_five = np.full((16, 16), 5, dtype=np.uint8)
    assert complexity_score(all_zero) < 0.3, "Single-color should have low complexity"
    assert complexity_score(all_five) < 0.3, "Single-color should have low complexity"


def test_complexity_multicolor():
    # Use 8 colors in a pattern — should have higher complexity
    rng = np.random.RandomState(42)
    grid = rng.randint(0, 8, (16, 16)).astype(np.uint8)
    score = complexity_score(grid)
    assert score > 0.3, f"Multi-color grid should have moderate complexity, got {score}"


# ---------------------------------------------------------------------------
# aesthetics_score
# ---------------------------------------------------------------------------

def test_aesthetics_balanced():
    # All quadrants identical — perfect balance
    q = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    grid = np.tile(q, (8, 8))
    score = aesthetics_score(grid)
    assert score > 0.4, f"Balanced grid should have decent aesthetics, got {score}"


# ---------------------------------------------------------------------------
# ArtCritic.score_single
# ---------------------------------------------------------------------------

def test_score_single_keys(critic):
    grid = np.zeros((16, 16), dtype=np.uint8)
    result = critic.score_single(grid)
    expected_keys = {"symmetry", "complexity", "structure", "aesthetics", "diversity", "composite"}
    assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# ArtCritic.rank
# ---------------------------------------------------------------------------

def test_rank_ordering(critic):
    rng = np.random.RandomState(99)
    grids = [rng.randint(0, 8, (16, 16)).astype(np.uint8) for _ in range(5)]
    ranked = critic.rank(grids)
    composites = [entry[1]["composite"] for entry in ranked]
    assert composites == sorted(composites, reverse=True)


# ---------------------------------------------------------------------------
# diversity_bonus
# ---------------------------------------------------------------------------

def test_diversity_identical():
    grid = np.zeros((16, 16), dtype=np.uint8)
    grids = [grid.copy() for _ in range(5)]
    scores = per_image_diversity(grids)
    assert all(s < 0.05 for s in scores), f"Identical grids should have diversity ~0, got {scores}"


def test_diversity_different():
    g1 = np.zeros((16, 16), dtype=np.uint8)
    g2 = np.full((16, 16), 7, dtype=np.uint8)
    scores = per_image_diversity([g1, g2])
    assert all(s > 0.9 for s in scores), f"Completely different grids should have high diversity, got {scores}"


def test_diversity_outlier_gets_higher_score():
    """An outlier should score higher than clones."""
    clone = np.zeros((16, 16), dtype=np.uint8)
    outlier = np.full((16, 16), 7, dtype=np.uint8)
    grids = [clone.copy(), clone.copy(), clone.copy(), outlier]
    scores = per_image_diversity(grids)
    assert scores[3] > scores[0], f"Outlier should score higher: outlier={scores[3]}, clone={scores[0]}"
