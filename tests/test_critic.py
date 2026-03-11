import numpy as np
import pytest

from art.config import ArtConfig
from art.critic import (
    ArtCritic,
    aesthetics_score,
    complexity_score,
    diversity_bonus,
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
    # Concentric squares are symmetric under all four transformations
    # (horizontal flip, vertical flip, 180° rotation, 90° rotation),
    # so symmetry_score should return 1.0.
    grid = np.zeros((16, 16), dtype=np.float64)
    for i in range(0, 8, 2):
        grid[i : 16 - i, i : 16 - i] = 1
        grid[i + 1 : 16 - i - 1, i + 1 : 16 - i - 1] = 0
    score = symmetry_score(grid)
    assert score > 0.9, f"Expected symmetry > 0.9, got {score}"


def test_symmetry_random():
    rng = np.random.RandomState(123)
    # Average over multiple random grids to get a stable estimate
    scores = [symmetry_score(rng.randint(0, 2, (16, 16)).astype(float)) for _ in range(20)]
    avg = float(np.mean(scores))
    assert 0.3 < avg < 0.7, f"Expected random symmetry ~0.5, got avg={avg}"


# ---------------------------------------------------------------------------
# complexity_score
# ---------------------------------------------------------------------------

def test_complexity_uniform():
    all_black = np.zeros((16, 16), dtype=float)
    all_white = np.ones((16, 16), dtype=float)
    assert complexity_score(all_black) < 0.3, "All-black should have low complexity"
    assert complexity_score(all_white) < 0.3, "All-white should have low complexity"


def test_complexity_edges():
    # A pixel-level checkerboard has maximum edge count
    grid = np.indices((16, 16)).sum(axis=0) % 2  # alternating 0/1
    edge_count = (
        np.sum(np.abs(np.diff(grid, axis=0)))
        + np.sum(np.abs(np.diff(grid, axis=1)))
    )
    max_edges = 2 * 16 * 15
    assert edge_count == max_edges, (
        f"Checkerboard should have max edges ({max_edges}), got {edge_count}"
    )


# ---------------------------------------------------------------------------
# aesthetics_score
# ---------------------------------------------------------------------------

def test_aesthetics_balanced():
    # Equal density in all quadrants: fill each quadrant with 50% white
    rng = np.random.RandomState(7)
    grid = np.zeros((16, 16), dtype=float)
    for ri in [slice(0, 8), slice(8, 16)]:
        for ci in [slice(0, 8), slice(8, 16)]:
            block = rng.randint(0, 2, (8, 8)).astype(float)
            # Force exactly 50% density
            block = np.zeros((8, 8))
            block[:4, :4] = 1
            block[4:, 4:] = 1
            grid[ri, ci] = block
    score = aesthetics_score(grid)
    # Balance component should be high with equal quadrant densities
    q1 = np.mean(grid[:8, :8])
    q2 = np.mean(grid[:8, 8:])
    q3 = np.mean(grid[8:, :8])
    q4 = np.mean(grid[8:, 8:])
    balance = 1 - np.std([q1, q2, q3, q4]) * 4
    assert float(np.clip(balance, 0, 1)) > 0.9, (
        f"Equal quadrant densities should yield high balance, got {balance}"
    )


# ---------------------------------------------------------------------------
# ArtCritic.score_single
# ---------------------------------------------------------------------------

def test_score_single_keys(critic):
    grid = np.zeros((16, 16), dtype=float)
    result = critic.score_single(grid)
    expected_keys = {"symmetry", "complexity", "aesthetics", "diversity", "composite"}
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )


# ---------------------------------------------------------------------------
# ArtCritic.rank
# ---------------------------------------------------------------------------

def test_rank_ordering(critic):
    rng = np.random.RandomState(99)
    grids = [rng.randint(0, 2, (16, 16)).astype(float) for _ in range(5)]
    ranked = critic.rank(grids)
    composites = [entry[1]["composite"] for entry in ranked]
    assert composites == sorted(composites, reverse=True), (
        "rank() should return scores in descending order by composite"
    )


# ---------------------------------------------------------------------------
# diversity_bonus
# ---------------------------------------------------------------------------

def test_diversity_identical():
    grid = np.zeros((16, 16), dtype=float)
    grids = [grid.copy() for _ in range(5)]
    bonus = diversity_bonus(grids)
    assert bonus < 0.05, f"Identical grids should have diversity ~0, got {bonus}"


def test_diversity_different():
    # Two opposite grids: all-zeros vs all-ones
    g1 = np.zeros((16, 16), dtype=float)
    g2 = np.ones((16, 16), dtype=float)
    bonus = diversity_bonus([g1, g2])
    assert bonus > 0.9, f"Completely different grids should have high diversity, got {bonus}"
