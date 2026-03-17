"""
One-time data preparation and fixed evaluation for pixel art autoresearch.
Generates bootstrap patterns, builds tokenizer, provides dataloader and critic.

Usage:
    uv run prepare.py              # full prep (generate bootstrap patterns)
    uv run prepare.py --patterns 2000  # fewer patterns (for testing)

Data is stored in ./data/bootstrap/.

DO NOT MODIFY THIS FILE. It contains the fixed evaluation metric.
"""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

GRID_SIZE = 16
N_COLORS = 8
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_GENERATIONS = 5     # number of generations to evaluate over
EVAL_IMAGES_PER_GEN = 36 # images to generate per eval generation

# Vocab: 8 color tokens (0-7) + BOS(8) + EOS(9) + PAD(10)
VOCAB_SIZE = 11
BOS = 8
EOS = 9
PAD = 10
SEQ_LENGTH = GRID_SIZE * GRID_SIZE + 2  # BOS + 256 pixels + EOS

# PICO-8 palette
PALETTE = [
    (0, 0, 0),          # 0  black
    (255, 241, 232),    # 1  cream
    (255, 0, 77),       # 2  red
    (255, 163, 0),      # 3  orange
    (255, 236, 39),     # 4  yellow
    (0, 228, 54),       # 5  green
    (41, 173, 255),     # 6  blue
    (255, 119, 168),    # 7  pink
]

DATA_DIR = Path("data")
BOOTSTRAP_DIR = DATA_DIR / "bootstrap"

# ---------------------------------------------------------------------------
# Bootstrap pattern generation
# ---------------------------------------------------------------------------

def generate_bootstrap_patterns(n_patterns: int = 5000) -> list[np.ndarray]:
    """Generate bootstrap patterns as grids of color indices (0..N_COLORS-1)."""
    rng = np.random.RandomState(42)
    size = GRID_SIZE
    nc = N_COLORS
    patterns: list[np.ndarray] = []

    def rand_pair():
        bg = rng.randint(0, nc)
        fg = rng.randint(0, nc)
        while fg == bg:
            fg = rng.randint(0, nc)
        return bg, fg

    # Horizontal lines ~80
    for spacing in range(1, 5):
        for thickness in range(1, 3):
            for offset in range(spacing):
                bg, fg = rand_pair()
                grid = np.full((size, size), bg, dtype=np.uint8)
                for row in range(size):
                    if (row - offset) % spacing < thickness:
                        grid[row, :] = fg
                patterns.append(grid.copy())
                if len(patterns) >= 80:
                    break

    # Vertical lines ~80
    count = 0
    for spacing in range(1, 5):
        for thickness in range(1, 3):
            for offset in range(spacing):
                bg, fg = rand_pair()
                grid = np.full((size, size), bg, dtype=np.uint8)
                for col in range(size):
                    if (col - offset) % spacing < thickness:
                        grid[:, col] = fg
                patterns.append(grid.copy())
                count += 1
                if count >= 80:
                    break
            if count >= 80:
                break
        if count >= 80:
            break

    # Diagonal lines ~80
    count = 0
    for spacing in range(1, 5):
        for offset in range(spacing):
            for direction in [1, -1]:
                bg, fg = rand_pair()
                grid = np.full((size, size), bg, dtype=np.uint8)
                for r in range(size):
                    for c in range(size):
                        if (r + direction * c - offset) % spacing == 0:
                            grid[r, c] = fg
                patterns.append(grid.copy())
                count += 1
                if count >= 80:
                    break
            if count >= 80:
                break
        if count >= 80:
            break

    # Horizontal symmetry ~400
    for _ in range(400):
        n_used = rng.randint(2, min(6, nc + 1))
        colors = rng.choice(nc, size=n_used, replace=False)
        half = colors[rng.randint(0, n_used, size=(size, size // 2))]
        grid = np.zeros((size, size), dtype=np.uint8)
        grid[:, :size // 2] = half
        grid[:, size // 2:] = half[:, ::-1]
        patterns.append(grid)

    # Vertical symmetry ~400
    for _ in range(400):
        n_used = rng.randint(2, min(6, nc + 1))
        colors = rng.choice(nc, size=n_used, replace=False)
        half = colors[rng.randint(0, n_used, size=(size // 2, size))]
        grid = np.zeros((size, size), dtype=np.uint8)
        grid[:size // 2, :] = half
        grid[size // 2:, :] = half[::-1, :]
        patterns.append(grid)

    # 4-fold symmetry ~400
    for _ in range(400):
        n_used = rng.randint(2, min(6, nc + 1))
        colors = rng.choice(nc, size=n_used, replace=False)
        q = colors[rng.randint(0, n_used, size=(size // 2, size // 2))]
        grid = np.zeros((size, size), dtype=np.uint8)
        grid[:size // 2, :size // 2] = q
        grid[:size // 2, size // 2:] = q[:, ::-1]
        grid[size // 2:, :size // 2] = q[::-1, :]
        grid[size // 2:, size // 2:] = q[::-1, ::-1]
        patterns.append(grid)

    # Checkerboard variants ~60
    for cell_size in range(1, 5):
        for phase in range(2):
            bg, fg = rand_pair()
            grid = np.full((size, size), bg, dtype=np.uint8)
            for r in range(size):
                for c in range(size):
                    if ((r // cell_size) + (c // cell_size) + phase) % 2 == 0:
                        grid[r, c] = fg
            patterns.append(grid.copy())

    for _ in range(40):
        cell_size = rng.randint(1, 5)
        n_cols = rng.randint(2, 5)
        colors = rng.choice(nc, size=n_cols, replace=False)
        grid = np.zeros((size, size), dtype=np.uint8)
        for r in range(size):
            for c in range(size):
                grid[r, c] = colors[((r // cell_size) + (c // cell_size)) % n_cols]
        patterns.append(grid)

    # Filled rectangles ~200
    for _ in range(200):
        bg, fg = rand_pair()
        grid = np.full((size, size), bg, dtype=np.uint8)
        r1, c1 = rng.randint(0, size - 2), rng.randint(0, size - 2)
        r2, c2 = rng.randint(r1 + 1, size), rng.randint(c1 + 1, size)
        grid[r1:r2, c1:c2] = fg
        patterns.append(grid)

    # Crosses ~150
    for _ in range(150):
        bg, fg = rand_pair()
        grid = np.full((size, size), bg, dtype=np.uint8)
        cx, cy = rng.randint(2, size - 2), rng.randint(2, size - 2)
        arm = rng.randint(1, 4)
        thickness = rng.randint(1, 3)
        t = thickness // 2
        grid[max(0, cx - arm):min(size, cx + arm + 1), max(0, cy - t):min(size, cy + t + 1)] = fg
        grid[max(0, cx - t):min(size, cx + t + 1), max(0, cy - arm):min(size, cy + arm + 1)] = fg
        patterns.append(grid)

    # Diamonds ~150
    for _ in range(150):
        bg, fg = rand_pair()
        grid = np.full((size, size), bg, dtype=np.uint8)
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        radius = rng.randint(2, size // 2 + 1)
        filled = rng.choice([True, False])
        for r in range(size):
            for c in range(size):
                dist = abs(r - cx) + abs(c - cy)
                if (filled and dist <= radius) or (not filled and dist == radius):
                    grid[r, c] = fg
        patterns.append(grid)

    # Concentric squares ~100
    for _ in range(100):
        colors = rng.choice(nc, size=rng.randint(2, 5), replace=False)
        grid = np.full((size, size), colors[0], dtype=np.uint8)
        step = rng.randint(1, 4)
        for i in range(0, size // 2, step):
            c = colors[(i // step) % len(colors)]
            grid[i:size - i, i:size - i] = c
        patterns.append(grid)

    # Gradients ~200
    for _ in range(200):
        n_cols = rng.randint(3, min(8, nc + 1))
        colors = rng.choice(nc, size=n_cols, replace=False)
        axis = rng.randint(0, 2)
        grid = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            ci = min(int(i / size * n_cols), n_cols - 1)
            if axis == 0:
                grid[i, :] = colors[ci]
            else:
                grid[:, i] = colors[ci]
        patterns.append(grid)

    # Random noise ~360
    for n_used in range(2, 8):
        for _ in range(60):
            colors = rng.choice(nc, size=n_used, replace=False)
            grid = colors[rng.randint(0, n_used, size=(size, size))].astype(np.uint8)
            patterns.append(grid)

    # Stripes with 3+ colors ~100
    for _ in range(100):
        n_cols = rng.randint(2, 5)
        colors = rng.choice(nc, size=n_cols, replace=False)
        spacing = rng.randint(1, 5)
        axis = rng.randint(0, 2)
        grid = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            c = colors[(i // spacing) % n_cols]
            if axis == 0:
                grid[i, :] = c
            else:
                grid[:, i] = c
        patterns.append(grid)

    # XOR patterns ~100
    for _ in range(100):
        n_cols = rng.randint(2, 5)
        colors = rng.choice(nc, size=n_cols, replace=False)
        a, b = rng.randint(1, size), rng.randint(1, size)
        grid = np.zeros((size, size), dtype=np.uint8)
        for r in range(size):
            for c in range(size):
                grid[r, c] = colors[((r % a) ^ (c % b)) % n_cols]
        patterns.append(grid)

    # Dot grids ~80
    for spacing in range(2, 6):
        for _ in range(20):
            bg, fg = rand_pair()
            grid = np.full((size, size), bg, dtype=np.uint8)
            off_r, off_c = rng.randint(0, spacing), rng.randint(0, spacing)
            for r in range(size):
                for c in range(size):
                    if r % spacing == off_r and c % spacing == off_c:
                        grid[r, c] = fg
            patterns.append(grid)

    # Fill remaining with random symmetric patterns
    target = n_patterns
    while len(patterns) < target:
        mode = rng.randint(0, 5)
        n_used = rng.randint(2, min(6, nc + 1))
        colors = rng.choice(nc, size=n_used, replace=False)
        if mode == 0:
            half = colors[rng.randint(0, n_used, size=(size, size // 2))]
            grid = np.zeros((size, size), dtype=np.uint8)
            grid[:, :size // 2] = half
            grid[:, size // 2:] = half[:, ::-1]
        elif mode == 1:
            half = colors[rng.randint(0, n_used, size=(size // 2, size))]
            grid = np.zeros((size, size), dtype=np.uint8)
            grid[:size // 2, :] = half
            grid[size // 2:, :] = half[::-1, :]
        elif mode == 2:
            q = colors[rng.randint(0, n_used, size=(size // 2, size // 2))]
            grid = np.zeros((size, size), dtype=np.uint8)
            grid[:size // 2, :size // 2] = q
            grid[:size // 2, size // 2:] = q[:, ::-1]
            grid[size // 2:, :size // 2] = q[::-1, :]
            grid[size // 2:, size // 2:] = q[::-1, ::-1]
        elif mode == 3:
            n_c = rng.randint(2, max(3, n_used + 1))
            grid = np.zeros((size, size), dtype=np.uint8)
            for i in range(size):
                ci = min(int(i / size * n_c), n_c - 1)
                if rng.random() > 0.5:
                    grid[i, :] = colors[ci]
                else:
                    grid[:, i] = colors[ci]
        else:
            grid = colors[rng.randint(0, n_used, size=(size, size))].astype(np.uint8)
        patterns.append(grid)

    return patterns[:target]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def encode_grid(grid: np.ndarray) -> list[int]:
    """Encode a grid of color indices into a token sequence."""
    pixels = grid.flatten().tolist()
    pixels = [max(0, min(N_COLORS - 1, p)) for p in pixels]
    return [BOS] + pixels + [EOS]


def decode_to_grid(tokens: list[int]) -> np.ndarray:
    """Decode token sequence into a grid of color indices."""
    pixels = []
    for t in tokens:
        if t in (BOS, EOS, PAD):
            continue
        pixels.append(max(0, min(N_COLORS - 1, t)))
    expected = GRID_SIZE * GRID_SIZE
    pixels = pixels[:expected]
    if len(pixels) < expected:
        pixels.extend([0] * (expected - len(pixels)))
    return np.array(pixels, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PixelDataset(Dataset):
    def __init__(self, patterns: list[np.ndarray]):
        self.sequences = []
        for p in patterns:
            tokens = encode_grid(p)
            self.sequences.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def make_dataloader(patterns: list[np.ndarray], batch_size: int, shuffle: bool = True):
    """Create a DataLoader from bootstrap patterns."""
    dataset = PixelDataset(patterns)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=len(dataset) > batch_size)


# ---------------------------------------------------------------------------
# Critic scoring (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def symmetry_score(grid: np.ndarray) -> float:
    g = grid.astype(np.int16)
    total = g.size
    h_sym = 1 - np.sum(g != np.fliplr(g)) / total
    v_sym = 1 - np.sum(g != np.flipud(g)) / total
    r180 = 1 - np.sum(g != np.rot90(g, 2)) / total
    r90 = 1 - np.sum(g != np.rot90(g)) / total
    return float(np.mean([h_sym, v_sym, r180, r90]))


def complexity_score(grid: np.ndarray) -> float:
    size = grid.shape[0]
    counts_c = np.bincount(grid.flatten(), minlength=N_COLORS)
    n_unique = int(np.sum(counts_c > 0))
    if n_unique <= 1:
        color_score = 0.0
    else:
        probs_c = counts_c / counts_c.sum()
        probs_nz = probs_c[probs_c > 0]
        h = float(-np.sum(probs_nz * np.log2(probs_nz)))
        color_score = h / np.log2(N_COLORS)
        if n_unique == 2:
            color_score = min(color_score, 0.25)

    h_edges = np.sum(grid[:, :-1] != grid[:, 1:])
    v_edges = np.sum(grid[:-1, :] != grid[1:, :])
    max_edges = 2 * size * (size - 1)
    edge_density = float((h_edges + v_edges) / max_edges)
    if edge_density < 0.10:
        edge_score = edge_density * 3.0
    elif edge_density < 0.30:
        edge_score = 0.30 + (edge_density - 0.10) / 0.20 * 0.50
    elif edge_density <= 0.65:
        edge_score = 1.0
    elif edge_density <= 0.85:
        edge_score = 1.0 - (edge_density - 0.65) / 0.20 * 0.60
    else:
        edge_score = 0.20

    block_size = max(1, size // 4)
    n_blocks = size // block_size
    entropy_vals = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = grid[block_size * i:block_size * (i + 1),
                         block_size * j:block_size * (j + 1)]
            counts = np.bincount(block.flatten(), minlength=N_COLORS)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log2(probs))
            max_h = np.log2(N_COLORS)
            entropy_vals.append(h / max_h if max_h > 0 else 0)
    entropy_score = float(np.mean(entropy_vals)) if entropy_vals else 0.0

    h_pairs = set(map(tuple, np.stack([grid[:, :-1].flatten(), grid[:, 1:].flatten()], axis=1)))
    v_pairs = set(map(tuple, np.stack([grid[:-1, :].flatten(), grid[1:, :].flatten()], axis=1)))
    distinct = len(h_pairs | v_pairs)
    max_trans = N_COLORS * (N_COLORS - 1)
    transition_score = min(1.0, distinct / max(1, max_trans * 0.4))

    return float(np.mean([color_score, edge_score, entropy_score, transition_score]))


def structure_score(grid: np.ndarray) -> float:
    size = grid.shape[0]
    row_autocorr = 0.0
    col_autocorr = 0.0
    for axis in [0, 1]:
        if axis == 0:
            same = np.mean(grid[:, :-1] == grid[:, 1:])
        else:
            same = np.mean(grid[:-1, :] == grid[1:, :])
        coherence = float(same)
        if coherence < 0.2 or coherence > 0.9:
            val = 0.2
        elif coherence < 0.4:
            val = (coherence - 0.2) / 0.2
        elif coherence > 0.7:
            val = (0.9 - coherence) / 0.2
        else:
            val = 1.0
        if axis == 0:
            row_autocorr = val
        else:
            col_autocorr = val

    blocks_2x2 = set()
    for r in range(size - 1):
        for c in range(size - 1):
            blocks_2x2.add((grid[r, c], grid[r, c + 1], grid[r + 1, c], grid[r + 1, c + 1]))
    n_patterns = len(blocks_2x2)
    if n_patterns <= 5:
        pattern_score = n_patterns / 10.0
    elif n_patterns <= 160:
        pattern_score = 1.0
    else:
        pattern_score = max(0.5, 1 - (n_patterns - 160) / 150)

    visited = np.zeros_like(grid, dtype=bool)
    regions = []
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                count = 0
                stack = [(i, j)]
                val = grid[i, j]
                while stack:
                    r2, c2 = stack.pop()
                    if 0 <= r2 < size and 0 <= c2 < size and not visited[r2, c2] and grid[r2, c2] == val:
                        visited[r2, c2] = True
                        count += 1
                        stack.extend([(r2 + 1, c2), (r2 - 1, c2), (r2, c2 + 1), (r2, c2 - 1)])
                regions.append(count)

    if regions:
        median_size = float(np.median(regions))
        n_regions = len(regions)
        max_region_frac = float(max(regions)) / (size * size)
        if n_regions < 4:
            region_quality = 0.1
        elif n_regions > 140:
            region_quality = 0.3
        elif median_size < 1.5:
            region_quality = 0.2
        elif max_region_frac > 0.25:
            region_quality = max(0.1, 1.0 - 3.5 * (max_region_frac - 0.25))
        else:
            target = 4.5
            region_quality = max(0.2, 1.0 - abs(median_size - target) / 12.0)
    else:
        region_quality = 0.0

    return float(np.mean([row_autocorr, col_autocorr, pattern_score, region_quality]))


def aesthetics_score(grid: np.ndarray) -> float:
    size = grid.shape[0]
    h_mid, w_mid = size // 2, size // 2
    quadrants = [
        grid[:h_mid, :w_mid], grid[:h_mid, w_mid:],
        grid[h_mid:, :w_mid], grid[h_mid:, w_mid:],
    ]
    freq_vecs = []
    for q in quadrants:
        counts = np.bincount(q.flatten(), minlength=N_COLORS)[:N_COLORS].astype(float)
        counts /= counts.sum() + 1e-8
        freq_vecs.append(counts)
    freq_arr = np.array(freq_vecs)
    balance = 1.0 - float(np.mean(np.std(freq_arr, axis=0)))
    balance_score = float(np.clip(balance, 0, 1))

    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    interior = grid[1:-1, 1:-1].flatten()
    border_freq = np.bincount(border, minlength=N_COLORS)[:N_COLORS].astype(float)
    border_freq /= border_freq.sum() + 1e-8
    interior_freq = np.bincount(interior, minlength=N_COLORS)[:N_COLORS].astype(float)
    interior_freq /= interior_freq.sum() + 1e-8
    framing_score = float(np.sum(np.abs(border_freq - interior_freq)) / 2.0)

    counts_h = np.bincount(grid.flatten(), minlength=N_COLORS)
    n_uniq = int(np.sum(counts_h > 0))
    if n_uniq <= 1:
        harmony = 0.0
    else:
        probs_h = counts_h / counts_h.sum()
        probs_nz = probs_h[probs_h > 0]
        h_ent = float(-np.sum(probs_nz * np.log2(probs_nz)))
        harmony = h_ent / np.log2(N_COLORS)
        if n_uniq == 2:
            harmony = min(harmony, 0.25)

    return float(np.mean([balance_score, framing_score, harmony]))


def per_image_diversity(grids: list[np.ndarray]) -> list[float]:
    n = len(grids)
    if n < 2:
        return [0.0] * n
    scores = []
    for i in range(n):
        dists = [float(np.sum(grids[i] != grids[j])) / grids[i].size for j in range(n) if j != i]
        scores.append(float(np.mean(dists)))
    return scores


def _stripe_fraction(grid: np.ndarray) -> float:
    rows_uniform = sum(1 for r in range(grid.shape[0]) if len(np.unique(grid[r, :])) <= 2)
    cols_uniform = sum(1 for c in range(grid.shape[1]) if len(np.unique(grid[:, c])) <= 2)
    return max(rows_uniform, cols_uniform) / grid.shape[0]


def _row_repeat_fraction(grid: np.ndarray) -> float:
    size = grid.shape[0]
    if size < 2:
        return 0.0
    same = sum(1 for r in range(size - 1) if np.array_equal(grid[r], grid[r + 1]))
    return same / (size - 1)


def _motif_repeat_score(grid: np.ndarray) -> float:
    size = grid.shape[0]
    motif_rows = 0
    for r in range(size):
        row = grid[r, :]
        for period in range(1, 5):
            tile = row[:period]
            repeated = np.tile(tile, (size + period - 1) // period)[:size]
            if np.sum(row == repeated) >= size - 2:
                motif_rows += 1
                break
    motif_cols = 0
    for c in range(size):
        col = grid[:, c]
        for period in range(1, 5):
            tile = col[:period]
            repeated = np.tile(tile, (size + period - 1) // period)[:size]
            if np.sum(col == repeated) >= size - 2:
                motif_cols += 1
                break
    return max(motif_rows, motif_cols) / size


def score_single(grid: np.ndarray) -> dict:
    """Score a single piece. Returns dict with component scores and composite."""
    unique = len(np.unique(grid))
    if unique <= 1:
        gate = 0.1
    elif unique <= 2:
        gate = 0.4
    else:
        gate = 1.0

    _, counts = np.unique(grid, return_counts=True)
    dominance = float(counts.max()) / grid.size
    if dominance > 0.60:
        gate *= 0.05
    elif dominance > 0.30:
        gate *= max(0.1, 1.0 - 5.0 * (dominance - 0.30))

    size = grid.shape[0]
    visited = np.zeros_like(grid, dtype=bool)
    max_region = 0
    for ri in range(size):
        for ci in range(size):
            if not visited[ri, ci]:
                stack = [(ri, ci)]
                color = int(grid[ri, ci])
                count = 0
                while stack:
                    r2, c2 = stack.pop()
                    if 0 <= r2 < size and 0 <= c2 < size and not visited[r2, c2] and grid[r2, c2] == color:
                        visited[r2, c2] = True
                        count += 1
                        stack.extend([(r2 + 1, c2), (r2 - 1, c2), (r2, c2 + 1), (r2, c2 - 1)])
                max_region = max(max_region, count)
    if max_region / grid.size > 0.15:
        gate *= max(0.1, 1.0 - 4.0 * (max_region / grid.size - 0.15))

    stripe_frac = _stripe_fraction(grid)
    if stripe_frac > 0.30:
        gate *= max(0.05, 1.0 - 3.0 * (stripe_frac - 0.30))

    row_rep = _row_repeat_fraction(grid)
    col_rep = _row_repeat_fraction(grid.T)
    if max(row_rep, col_rep) > 0.20:
        gate *= max(0.05, 1.0 - 3.0 * (max(row_rep, col_rep) - 0.20))

    motif_frac = _motif_repeat_score(grid)
    if motif_frac > 0.25:
        gate *= max(0.10, 1.0 - 2.0 * (motif_frac - 0.25))

    weights = {'symmetry': 0.15, 'complexity': 0.25, 'structure': 0.25, 'aesthetics': 0.20, 'diversity': 0.15}
    sym = symmetry_score(grid)
    cplx = complexity_score(grid)
    struct = structure_score(grid)
    aes = aesthetics_score(grid)

    composite = (weights['symmetry'] * sym + weights['complexity'] * cplx +
                 weights['structure'] * struct + weights['aesthetics'] * aes) * gate

    return {'symmetry': sym, 'complexity': cplx, 'structure': struct,
            'aesthetics': aes, 'diversity': 0.0, 'composite': composite, '_gate': gate}


def score_batch(grids: list[np.ndarray]) -> list[dict]:
    """Score a batch of pieces with diversity."""
    div_scores = per_image_diversity(grids)
    weights = {'diversity': 0.15}
    scores = []
    for i, grid in enumerate(grids):
        s = score_single(grid)
        s['diversity'] = div_scores[i]
        s['composite'] += s['_gate'] * weights['diversity'] * div_scores[i]
        scores.append(s)
    return scores


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_composite(model, n_generations: int = EVAL_GENERATIONS,
                       images_per_gen: int = EVAL_IMAGES_PER_GEN,
                       temperature: float = 0.9) -> dict:
    """
    Fixed evaluation metric for pixel art autoresearch.
    Generates images and scores them with the ArtCritic.
    Returns mean composite score across multiple generations.

    Higher is better (0.0 to ~1.0).

    This function:
    1. Generates n_generations batches of images
    2. Scores each batch with score_batch (includes diversity)
    3. Returns aggregated statistics
    """
    model.eval()
    all_composites = []
    all_scores = []

    for gen in range(n_generations):
        # Generate images
        seq = model.generate(
            batch_size=images_per_gen,
            temperature=temperature,
            device="cpu",
        )
        grids = [decode_to_grid(seq[i].tolist()) for i in range(seq.shape[0])]
        scores = score_batch(grids)
        composites = [s['composite'] for s in scores]
        all_composites.extend(composites)
        all_scores.extend(scores)

    mean_composite = float(np.mean(all_composites))
    max_composite = float(np.max(all_composites))
    mean_symmetry = float(np.mean([s['symmetry'] for s in all_scores]))
    mean_complexity = float(np.mean([s['complexity'] for s in all_scores]))
    mean_structure = float(np.mean([s['structure'] for s in all_scores]))
    mean_aesthetics = float(np.mean([s['aesthetics'] for s in all_scores]))
    mean_diversity = float(np.mean([s['diversity'] for s in all_scores]))

    return {
        'mean_composite': mean_composite,
        'max_composite': max_composite,
        'mean_symmetry': mean_symmetry,
        'mean_complexity': mean_complexity,
        'mean_structure': mean_structure,
        'mean_aesthetics': mean_aesthetics,
        'mean_diversity': mean_diversity,
        'n_images': len(all_composites),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for pixel art autoresearch")
    parser.add_argument("--patterns", type=int, default=5000, help="Number of bootstrap patterns")
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    print()

    # Generate bootstrap patterns
    t0 = time.time()
    print(f"Generating {args.patterns} bootstrap patterns...")
    patterns = generate_bootstrap_patterns(args.patterns)
    t1 = time.time()
    print(f"Generated {len(patterns)} patterns in {t1 - t0:.1f}s")

    # Save as numpy array for fast loading
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BOOTSTRAP_DIR / "patterns.npz",
                        patterns=np.stack(patterns))
    print(f"Saved to {BOOTSTRAP_DIR / 'patterns.npz'}")

    # Sanity check: score a few patterns
    print()
    print("Sanity check — scoring 10 random patterns:")
    sample = random.sample(patterns, 10)
    scores = score_batch(sample)
    for i, s in enumerate(scores):
        print(f"  Pattern {i}: composite={s['composite']:.4f} "
              f"(sym={s['symmetry']:.2f} cplx={s['complexity']:.2f} "
              f"struct={s['structure']:.2f} aes={s['aesthetics']:.2f})")

    print()
    print("Done! Ready to train.")
